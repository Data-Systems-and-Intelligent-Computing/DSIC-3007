"""Synthetic deadline/gate regressions; no GPU, network, or real model assets."""

from pathlib import Path

import pytest
import torch

from runpod import path_b_benchmark as benchmark
from runpod.path_b_preflight import prepare
from src import model as loader, optimized_validation as native
from src.artifacts import atomic, canonical, read_json
from src.preparation import make_plan
from test_path_b import fixture, settings


def diagnostic(tmp_path):
    prepared, review, _ = fixture(tmp_path, stage="technical", groups=12, files=1)
    plan = tmp_path/"technical"
    make_plan(prepared, review, plan, settings(), "technical", dict.fromkeys(settings()["domains"], 12))
    package = tmp_path/"package"
    prepare(package, plan)
    return package/"plans/path-b-preflight-v1/4096"


@pytest.mark.parametrize("load_seconds", [570, 1500])
def test_slow_load_native_and_timings_share_deadline_and_one_session(tmp_path, monkeypatch, load_seconds):
    plan = diagnostic(tmp_path)
    clock = [10000.0]
    monkeypatch.setattr(benchmark.time, "time", lambda: clock[0])
    monkeypatch.setattr(benchmark.time, "monotonic", lambda: clock[0])
    model = torch.nn.Linear(2, 2).eval()
    model.config = type("Config", (), {"use_cache": False})()
    loads, cases, measured = [], [], []

    def load(*args, **kwargs):
        assert kwargs == {"verify_weights": False}
        loads.append(args)
        clock[0] += load_seconds  # synthetic verification/loading; exceeds old120/125/190s caps.
        return model

    def case(actual_model, row, *args, **kwargs):
        assert actual_model is model
        clock[0] += 100
        values = [{"layer":15, "expert_id":e, "nll_abs_error":0.,
                   "max_logit_abs_error":0., "max_mixture_abs_error":0.} for e in range(8)]
        for i, value in enumerate(values, 1):
            kwargs["progress"]({"completed_comparisons":i, "latest_case":value})
        cases.append(row["input_id"])
        return {"cases":values, "cleanup_pass":True}

    def measure(actual_model, row, optimized):
        assert actual_model is model and optimized and len(cases) == 3
        measured.append(row["input_id"])
        clock[0] += 80
        return [{"event_id":row["input_id"], "shared_seconds":30.,
                 "shared_cleanup_seconds":30., "eight_interventions_seconds":20.,
                 "input_total_seconds":80.}]

    monkeypatch.setattr(loader, "load_model", load)
    monkeypatch.setattr(loader, "runtime_identity", lambda: {"fixture":"cpu"})
    monkeypatch.setattr(loader, "implementation_identity", lambda: "fixture")
    monkeypatch.setattr(native, "_input_case", case)
    monkeypatch.setattr(native, "_memory_metrics", lambda torch: {})
    monkeypatch.setattr(benchmark, "runtime_identity", lambda: {"fixture":"cpu"})
    monkeypatch.setattr(benchmark, "measure_input", measure)
    result = benchmark.run(plan, tmp_path/"model", tmp_path/"validation.json", 11500.)
    assert len(loads) == 1
    assert result["phases"]["verified_model_loading"]["elapsed_seconds"] == load_seconds
    if load_seconds == 570:
        assert result["status"] == "pass"
        assert result["completed_native_comparisons"] == 24
        assert result["completed_measured_inputs"] == 3 and len(measured) == 4
        assert result["elapsed_seconds"] == 1190
    else:
        assert result["status"] == "incomplete"
        assert result["completed_native_comparisons"] == result["completed_measured_inputs"] == 0
        assert not cases and not measured
    receipt = read_json(tmp_path/"independent-native-validation.json")
    assert receipt["approval"] == "diagnostic_only" and receipt["max_seconds"] == 1495


@pytest.mark.parametrize("status,comparisons,expected", [
    ("incomplete", 0, "incomplete"), ("fail", 24, "fail"), ("pass", 16, "fail")])
def test_native_incomplete_or_failed_coverage_never_starts_measurement(tmp_path, monkeypatch, status, comparisons, expected):
    plan = diagnostic(tmp_path)
    def validate(plans, model_dir, output, **kwargs):
        value = {"status":status, "completed_expert_comparisons":comparisons}
        atomic(output, canonical(value))
        return value
    monkeypatch.setattr(benchmark, "validate_optimized", validate)
    monkeypatch.setattr(benchmark, "measure_input", lambda *a, **k: pytest.fail("native gate must precede timings"))
    result = benchmark.run(plan, tmp_path/"model", tmp_path/"validation.json", benchmark.time.time()+1500)
    assert result["status"] == expected and result["completed_measured_inputs"] == 0


def test_corrupt_cached_asset_is_rejected_by_loader_before_native_model_load(tmp_path, monkeypatch):
    import hashlib
    import transformers
    monkeypatch.setattr(loader.importlib.metadata, "version", lambda name: "5.12.1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained",
                        lambda *a, **k: pytest.fail("corrupt weights must never reach model loading"))
    (tmp_path/"config.json").write_bytes(b"bad!")
    catalog = {"revision":"c6f6954f182ffdd9f6ab4f4d772714b7dcd6836b", "files":[
        {"path":"config.json", "size_bytes":4, "sha256":hashlib.sha256(b"good").hexdigest()}]}
    with pytest.raises(ValueError, match="model byte identity mismatch"):
        loader.load_model(tmp_path, catalog, 3007)


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), 0])
def test_cli_refuses_invalid_deadline_before_loading(deadline):
    with pytest.raises(SystemExit):
        benchmark.main(["--plan","unused", "--model-dir","unused", "--output","unused",
                        "--deadline-epoch",str(deadline)])

