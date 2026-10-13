import copy
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from runpod import paired_benchmark as paired
from src.optimized_validation import _TimeLimit

ROOT = Path(__file__).resolve().parents[1]
FINDINGS = ROOT / "docs/validation/fresh-optimization-findings-2026-10-11.json"
PLANS = ROOT / "data/plans/optimization-screen-2026-10-11-v2"


def _event(layer=15, event_id="event-1", input_id="input-1"):
    route = {"expert_ids": [2], "weights": [0.75]}
    return {"event_id": event_id, "input_id": input_id, "layer": layer,
            "routing": copy.deepcopy(route), "baseline_nll": 2.0,
            "effects": [{"expert_id": 2, "ablated_nll": 2.25, "delta_nll": 0.25}]}


def test_frozen_findings_and_sealed_plan_selection():
    _, proposal, selection = paired.load_findings(FINDINGS)
    assert proposal["status"] == "proposal_only_NOT_authorized_or_prepared"
    assert [row["input_id"] for row in selection] == list(paired.FROZEN_INPUT_IDS)
    selected = paired.select_inputs([PLANS / "4096", PLANS / "2048"], selection)
    assert len(selected) == 6
    assert [item["row"]["context_length"] for item in selected] == [4096, 2048] * 3
    assert [item["selected"]["pair_order"] for item in selected] == [
        "baseline_then_candidate", "candidate_then_baseline"] * 3


def test_findings_reject_changed_id_or_pair_order(tmp_path):
    raw = json.loads(FINDINGS.read_text())
    raw["benchmark_proposal"]["selection"][0]["input_id"] = "0" * 64
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="IDs/order"):
        paired.load_findings(path)

    raw = json.loads(FINDINGS.read_text())
    raw["benchmark_proposal"]["selection"][1]["pair_order"] = "baseline_then_candidate"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="alternating AB/BA"):
        paired.load_findings(path)


def test_baseline_alias_requires_byte_identical_shared_dependencies(tmp_path):
    baseline_src = tmp_path / "old" / "src"
    baseline_src.mkdir(parents=True)
    for filename in paired.IMPLEMENTATION_FILES:
        shutil.copy2(ROOT / "src" / filename, baseline_src / filename)
    module, identity = paired.load_baseline(baseline_src, ROOT / "src")
    assert module.__name__.startswith("_dsic3007_pair_baseline_")
    assert identity["baseline_implementation_sha256"] == identity["candidate_implementation_sha256"]
    assert callable(module.measure_input)
    (baseline_src / "verify.py").write_text((baseline_src / "verify.py").read_text() + "\n# drift\n")
    with pytest.raises(ValueError, match="dependency differs.*verify.py"):
        paired.load_baseline(baseline_src, ROOT / "src")


def test_comparison_requires_native_route_and_individual_effects():
    baseline, candidate = [_event()], [_event()]
    native_cases = [{"layer": 15, "expert_id": 2, "native_nll": 2.25}]
    routes = {15: {"expert_ids": [2], "weights": [0.75]}}
    assert paired.compare_measurement_events(baseline, candidate, routes, native_cases,
                                              native_baseline_nll=2.0)

    candidate[0]["routing"]["expert_ids"] = [3]
    with pytest.raises(AssertionError, match="routes differ"):
        paired.compare_measurement_events(baseline, candidate, routes, native_cases,
                                          native_baseline_nll=2.0)

    candidate = [_event()]
    candidate[0]["effects"][0]["ablated_nll"] += paired.NLL_ATOL * 2
    with pytest.raises(AssertionError, match="ablated NLL"):
        paired.compare_measurement_events(baseline, candidate, routes, native_cases,
                                          native_baseline_nll=2.0)


def test_pair_order_progress_and_durable_timing(monkeypatch, tmp_path):
    order = []
    outputs = {"baseline": [_event()], "candidate": [_event()]}

    def measure(name):
        def run(model, inputs, optimized):
            assert optimized is True
            order.append(name)
            return copy.deepcopy(outputs[name])
        return run

    monkeypatch.setattr(paired, "_synchronize", lambda model: None)
    monkeypatch.setattr(paired, "_assert_clean", lambda model: None)
    class DummyModel:
        model = SimpleNamespace(layers=[])
        def named_parameters(self):
            return []

    baseline = SimpleNamespace(measure_input=measure("baseline"))
    candidate = SimpleNamespace(measure_input=measure("candidate"),
                                append_timing_row=__import__("src.run_experiment", fromlist=["append_timing_row"]).append_timing_row)
    item = {"row": {"input_id": "input-1", "context_length": 4096, "domain": "c4",
                    "input_ids": [1], "layers": [15]}, "manifest_sha256": "a" * 64}
    events_path, timings_path, progress_path = (tmp_path / name for name in ("events", "timings", "progress"))
    result = paired._pair_measurement(item, [15], "candidate_then_baseline", DummyModel(), baseline,
        candidate, events_path, timings_path, progress_path, time.time() + 60, time.time() + 50,
        1, "measured")

    assert order == ["candidate", "baseline"]
    assert set(result) == {"baseline", "candidate"}
    assert len(events_path.read_text().splitlines()) == 2
    assert len(timings_path.read_text().splitlines()) == 2
    rows = [json.loads(line) for line in progress_path.read_text().splitlines()]
    assert [row["record_type"] for row in rows] == [
        "paired_arm_started", "paired_arm_complete", "paired_arm_started", "paired_arm_complete"]
    assert [row["benchmark_arm"] for row in rows[::2]] == ["candidate", "baseline"]
    assert all(row["event_count"] == 1 for row in rows[1::2])
    assert all(row["timing_sidecar_fsync_seconds"] >= 0 for row in rows[1::2])
    assert all(row["end_to_end_seconds"] >= row["measurement_seconds"] for row in rows[1::2])
    persisted_timing = json.loads(timings_path.read_text().splitlines()[0])
    assert "timing_sidecar_fsync_seconds" not in persisted_timing
    assert "through event journal fsync" in persisted_timing["end_to_end_scope"]


@pytest.mark.parametrize("failure", ["parameter", "decoder_hook"])
def test_pair_refuses_to_persist_events_after_model_integrity_failure(monkeypatch, tmp_path, failure):
    monkeypatch.setattr(paired, "_synchronize", lambda model: None)
    class Parameter:
        _version = 1

    layer = SimpleNamespace(_forward_pre_hooks={}, _forward_hooks={},
        mlp=SimpleNamespace(experts=SimpleNamespace(_forward_pre_hooks={}, _forward_hooks={}, __dict__={})),
        post_attention_layernorm=SimpleNamespace(_forward_pre_hooks={}, _forward_hooks={}))

    class Model:
        model = SimpleNamespace(layers=[layer])
        parameter = Parameter()
        def named_parameters(self):
            return [("weight", self.parameter)]

    model = Model()
    def bad_measure(model, inputs, optimized):
        if failure == "parameter":
            model.parameter._version += 1
        else:
            layer._forward_hooks[123] = object()
        return [_event()]

    baseline = SimpleNamespace(measure_input=bad_measure)
    candidate = SimpleNamespace(measure_input=bad_measure,
        append_timing_row=__import__("src.run_experiment", fromlist=["append_timing_row"]).append_timing_row)
    item = {"row": {"input_id": "input-1", "context_length": 4096, "domain": "c4",
                    "input_ids": [1], "layers": [15]}, "manifest_sha256": "a" * 64}
    events_path, timings_path, progress_path = (tmp_path / name for name in ("events", "timings", "progress"))
    with pytest.raises(RuntimeError, match="parameter versions changed|decoder hooks remain"):
        paired._pair_measurement(item, [15], "baseline_then_candidate", model, baseline,
            candidate, events_path, timings_path, progress_path, time.time() + 60,
            time.time() + 50, 1, "measured")
    assert not events_path.exists()
    assert not timings_path.exists()


def test_cli_rejects_unbounded_deadline_before_running(monkeypatch, capsys, tmp_path):
    args = ["--plans", str(PLANS / "4096"), str(PLANS / "2048"),
            "--baseline-source", str(tmp_path), "--selection", str(FINDINGS),
            "--model-dir", str(tmp_path), "--deadline-epoch", "inf",
            "--output", str(tmp_path / "out.json")]
    with pytest.raises(SystemExit) as exc:
        paired.main(args)
    assert exc.value.code == 2
    assert "deadline-epoch must be finite" in capsys.readouterr().err


def test_validator_internal_time_limit_is_incomplete_not_failure():
    assert paired._is_incomplete_timeout(_TimeLimit("bounded validator cap"), _TimeLimit)
    assert paired._is_incomplete_timeout(TimeoutError("attempt cap"), _TimeLimit)
    assert not paired._is_incomplete_timeout(RuntimeError("numerical mismatch"), _TimeLimit)


def test_receipt_finalization_reserves_time_and_obeys_phase_cap():
    start = 100.0
    assert paired._finalization_within_budget(start, 200.0, now_monotonic=127.0,
                                               now_epoch=150.0, cap_seconds=30.0)
    assert not paired._finalization_within_budget(start, 200.0, now_monotonic=128.1,
                                                   now_epoch=150.0, cap_seconds=30.0)
    assert not paired._finalization_within_budget(start, 200.0, now_monotonic=110.0,
                                                   now_epoch=198.1, cap_seconds=30.0)
