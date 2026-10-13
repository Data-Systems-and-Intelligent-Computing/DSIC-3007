"""CPU orchestration fixtures exercise receipt/cache/durability integration only."""

from copy import deepcopy

import pytest

from src.artifacts import atomic, canonical, digest, file_digest, read_json, read_rows, seal, write_rows
from src.execution_budget import profiles_from_measurements
from src.preparation import window_for
from src import run_experiment as runner
from test_research_collector import tiny_model


def setup(tmp_path, monkeypatch):
    model = tiny_model()
    executor = {"model_catalog_sha256": digest({"revision": "fixture-model"}),
                "runtime": {"gpu_name": "CPU synthetic fixture"}, "implementation_sha256": "f" * 64}
    monkeypatch.setattr(runner, "load_model", lambda *args: model)
    monkeypatch.setattr(runner, "runtime_identity", lambda: executor["runtime"])
    monkeypatch.setattr(runner, "implementation_identity", lambda: executor["implementation_sha256"])
    settings = read_json("configs/research.json")
    review = {"status": "approved", "fixture_only": True}
    review_hash = __import__("hashlib").sha256(canonical(review) + b"\n").hexdigest()
    receipt = tmp_path / "fixture-mechanics.json"
    atomic(receipt, canonical({"status": "pass", "kind": "native-mechanics",
        "context_lengths": [8], "source_review_sha256": review_hash, "executor": executor}) + b"\n")

    def plan(name, documents):
        path = tmp_path / name
        path.mkdir()
        rows, docs = [], []
        for index in range(documents):
            tokens = [3 + index, 4, 5, 6, 7, 8, 9, 10, 11]
            row = dict(window_for(tokens, 7, 8), document_id=f"document-{index}",
                token_ids_sha256=digest(tokens), dataset="fixture", revision="fixture-revision",
                source_group=f"paper-{index}", domain="scientific", layers=[0],
                sampling={"seed": 3008, "requested_units": documents}, sample_order=0,
                eligible_positions=1)
            row["input_id"] = digest({key: row[key] for key in (
                "document_id", "token_ids_sha256", "window_start", "window_end", "target_id", "context_length")})
            rows.append(row)
            docs.append(dict(document_id=row["document_id"], content_sha256=digest(tokens)))
        write_rows(path / "inputs.jsonl", rows)
        write_rows(path / "documents.jsonl", docs)
        for filename, value in (("research.json", settings), ("source-review.json", review),
                                 ("model-catalog.json", {"revision": "fixture-model"})):
            atomic(path / filename, canonical(value) + b"\n")
        seal(path, "experiment-plan", dict(stage="calibration", window=8,
             tokenizer_sha256="b" * 64, source_review_sha256=review_hash, input_count=len(rows)))
        return path
    return plan, receipt


def test_collector_reuses_nested_raw_routes_and_keeps_new_plan_metadata(tmp_path, monkeypatch):
    plan, receipt = setup(tmp_path, monkeypatch)
    first, larger = plan("one", 1), plan("two", 2)
    source, output = tmp_path / "source", tmp_path / "output"
    runner.collect(first, tmp_path, source, receipt)
    original = runner.measure_input
    measured = []
    def record(model, inputs, **kwargs):
        measured.append(inputs["document_id"])
        return original(model, inputs, **kwargs)
    monkeypatch.setattr(runner, "measure_input", record)
    runner.collect(larger, tmp_path, output, receipt, reuse_runs=[source])
    assert measured == ["document-1"]
    events = list(read_rows(output / "events.jsonl"))
    hit = next(row for row in events if row["document_id"] == "document-0")
    assert hit["sampling"]["requested_units"] == 2
    assert hit["raw_source"]["source_measurement_manifest_sha256"] == file_digest(source / "manifest.json")
    assert hit["window_routing"] == next(read_rows(source / "events.jsonl"))["window_routing"]
    profiles = profiles_from_measurements([output])
    assert len(profiles["profiles"]) == 1 and len(profiles["excluded"]) == 1


def test_timing_write_failure_preserves_raw_events_and_unknown_cost_on_resume(tmp_path, monkeypatch):
    plan, receipt = setup(tmp_path, monkeypatch)
    inputs = plan("one", 1)
    output = tmp_path / "run"
    append = runner.append_timing_row
    monkeypatch.setattr(runner, "append_timing_row", lambda *args: (_ for _ in ()).throw(OSError("injected timing write")))
    with pytest.raises(OSError, match="injected timing write"):
        runner.collect(inputs, tmp_path, output, receipt)
    assert len(list(read_rows(output / "events.jsonl"))) == 1
    monkeypatch.setattr(runner, "append_timing_row", append)
    runner.collect(inputs, tmp_path, output, receipt)
    assert len(list(read_rows(output / "events.jsonl"))) == 1
    profiles = profiles_from_measurements([output])
    assert not profiles["profiles"]
    assert profiles["excluded"][0]["reason"] == "no measured persistence timing sidecar"


def test_compact_journal_reads_complete_rows_without_retaining_raw_payloads(tmp_path):
    from src.artifacts import Journal
    from src.run_experiment import measure_input
    inputs = dict(window_for([3, 4, 5, 6, 7, 8, 9, 10, 11], 7, 8),
                  input_id="offset-fixture", layers=[0])
    row = measure_input(tiny_model(), inputs, calibration=True)[0]
    directory = tmp_path / "journal"
    with Journal(directory, {"fixture": True}, compact=True) as journal:
        journal.append(row)
        assert list(journal.records) == [row["event_id"]]
        assert journal.records[row["event_id"]] == row
        assert list(journal.records.offsets.values()) == [0]
    with Journal(directory, {"fixture": True}, compact=True) as journal:
        assert journal.records[row["event_id"]] == row
        with pytest.raises(ValueError, match="already complete"):
            journal.append(row)
    with (directory / "events.jsonl").open("ab") as stream:
        stream.write(b'{"event_id":')
    with pytest.raises(ValueError, match="torn journal tail"):
        Journal(directory, {"fixture": True}, compact=True)
    assert not (directory / "writer.lock").exists()


@pytest.mark.parametrize("scope,checks,execution_mode", [
    ("screen", {}, runner.OPTIMIZED_EXECUTION_MODE),
    ("full_technical", {"parameters_and_cleanup": "fail"}, runner.OPTIMIZED_EXECUTION_MODE),
    ("full_technical", {
        "baseline_routes_weights_mixture_logits": "pass",
        "all_eight_native_ablation_logits_mixture_nll": "pass",
        "survivors_and_untargeted_rows": "pass",
        "downstream_routes_against_native_ablation": "pass",
        "parameters_and_cleanup": "pass",
    }, "native_shape_suffix_v1"),
])
def test_optimized_collector_rejects_diagnostic_failed_checks_or_legacy_mode(
        tmp_path, monkeypatch, scope, checks, execution_mode):
    plan, receipt = setup(tmp_path, monkeypatch)
    inputs = plan("technical", 1)
    metadata = read_json(inputs / "manifest.json")["metadata"]
    metadata["stage"] = "technical"
    (inputs / "manifest.json").unlink()
    seal(inputs, "experiment-plan", metadata)
    mechanics = read_json(receipt)
    mechanics["technical_plan_sha256"] = "1" * 64
    atomic(receipt, canonical(mechanics) + b"\n")
    optimized_receipt = tmp_path / "synthetic-optimized-receipt.json"
    atomic(optimized_receipt, canonical(dict(status="pass", kind="optimized-validation",
        validation_scope=scope, approval="collection_candidate", full_coverage=True,
        checks=checks, executor=mechanics["executor"], execution_mode=execution_mode,
        context_lengths=[8], layers=[0, 7, 15], source_review_sha256=mechanics["source_review_sha256"],
        technical_plan_sha256=[mechanics["technical_plan_sha256"]],
        tolerances=dict(atol=1e-6, rtol=1e-5, nll_atol=1e-6))) + b"\n")
    with pytest.raises(ValueError, match="does not authorize"):
        runner.collect(inputs, tmp_path, tmp_path / "rejected", receipt,
                       optimized=True, optimized_validation_path=optimized_receipt)
    assert not (tmp_path / "rejected" / "events.jsonl").exists()


def test_cleanup_failure_never_persists_a_complete_event(tmp_path, monkeypatch):
    plan, receipt = setup(tmp_path, monkeypatch)
    inputs = plan("cleanup-failure", 1)
    output = tmp_path / "incomplete-run"
    original_forward = runner.forward
    calls = 0

    def corrupt_baseline(model, row):
        nonlocal calls
        calls += 1
        nll, logits = original_forward(model, row)
        if calls == 1:
            logits = logits.clone()
            logits[-1] += 1e-7
        return nll, logits

    monkeypatch.setattr(runner, "forward", corrupt_baseline)
    with pytest.raises(ValueError, match="baseline did not recover"):
        runner.collect(inputs, tmp_path, output, receipt)

    assert calls == 2
    assert not list(read_rows(output / "events.jsonl"))
    failures = list(read_rows(output / "failures.jsonl"))
    assert len(failures) == 1
    assert failures[0]["status"] == "incomplete"
    assert "baseline did not recover" in failures[0]["message"]
