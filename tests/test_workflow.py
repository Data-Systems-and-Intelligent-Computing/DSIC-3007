"""A tiny actual OLMoE exercises fresh collection/calibration/resume, not pretrained acceptance."""
import copy
import json

import numpy as np
import pytest
import torch
from transformers import OlmoeConfig, OlmoeForCausalLM

from scripts.analyze import analyze
from scripts.benchmark import benchmark, require_benchmark
from scripts.calibrate import calibrate
from scripts.common import digest, load_prepared, read_json, records, write_json
from scripts.experiment import collect
from scripts.measurement import deterministic


def fixture_study(tmp_path):
    torch.set_num_threads(1)
    deterministic(3007)
    research = copy.deepcopy(read_json("config/research.json"))
    research.update(context=8, layer=1)
    research["model"].update(layers=2, experts=8, active_experts=8)
    research["sampling"].update(technical_groups=1, probe_documents=1, pilot_ladder=[10, 12], positions=2, probe_positions=2)
    research["calibration"]["ladder"] = [1, 2, 3]
    research["analysis"].update(planning_draws=100, bootstrap_draws=100)
    docs, tokens = [], []
    for domain in research["domains"]:
        for stage, count in [("technical", 1), ("probe", 1), ("calibration", 3), ("pilot", 12), ("main", 20)]:
            if domain == "c4" and stage in ("pilot", "main"):
                continue
            for i in range(count):
                for file in range(2 if domain == "code" and stage in ("pilot", "main") else 1):
                    identity = f"{domain}-{stage}-{i}-{file}"
                    docs.append({"id": identity, "domain": domain, "group": f"{domain}-{stage}-{i}",
                                 "stage": stage, "stage_order": i, "offset": len(tokens), "tokens": 14})
                    tokens.extend(np.random.default_rng(int(digest(identity)[:8], 16)).integers(2, 60, 14).tolist())
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    data = np.asarray(tokens, dtype="<i4")
    data.tofile(prepared / "tokens.bin")
    (prepared / "documents.jsonl").write_text("".join(json.dumps(row) + "\n" for row in docs))
    import hashlib
    info = {"research": research, "tokens_sha256": hashlib.sha256(data.tobytes()).hexdigest(), "documents_id": digest(docs)}
    info["id"] = digest(info)
    write_json(prepared / "prepared.json", info)
    execution = {"prepared_path": str(prepared), "results_path": str(tmp_path / "results"), "max_seconds": None}
    cfg = OlmoeConfig(hidden_size=16, intermediate_size=12, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, num_experts=8,
                      num_local_experts=8, num_experts_per_tok=8, vocab_size=64,
                      max_position_embeddings=32, norm_topk_prob=False, eos_token_id=63, pad_token_id=1)
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"
    model = OlmoeForCausalLM(cfg).eval()
    return research, execution, model


def test_fresh_gpu_stage_functions_on_tiny_cpu_model(tmp_path):
    research, execution, model = fixture_study(tmp_path)
    summary = benchmark(research, execution, model=model)
    assert summary["windows"] == 12
    assert summary["comparisons"] == 96
    frozen = calibrate(research, execution, model=model)
    assert frozen["budget"] == 2
    assert all(value["events"] == 2 for value in frozen["probe"].values())
    np.testing.assert_array_equal(frozen["M1"], np.ones(8))
    assert collect(research, execution, "pilot", 10, model=model) == 60
    path = tmp_path / "results/pilot.sqlite"
    assert len(list(records(path))) == 60
    # Resume repeats no completed input; extending keeps all old outcomes.
    assert collect(research, execution, "pilot", 10, model=model) == 60
    assert len(list(records(path))) == 60
    assert collect(research, execution, "pilot", 12, model=model) == 72
    assert len(list(records(path))) == 72
    outcome = analyze(research, execution, "pilot")
    assert outcome["results"]["delta"]["code"]["groups"] == 12
    assert outcome["results"]["delta"]["code"]["documents"] == 24
    # All experts are active here, so utilization is tied and M1/M2 select identically.
    assert outcome["results"]["delta"]["primary"]["estimate"] == 0
    assert outcome["results"]["delta"]["primary"]["ci"] is None
    assert not outcome["pilot"]["sufficient"]
    assert outcome["results"]["M4"]["role"] == "descriptive"
    assert "holm" not in outcome["results"]["M4"]
    with pytest.raises(ValueError, match="Pilot is insufficient"):
        analyze(research, execution, "pilot", freeze_main=True)
    saved = read_json(tmp_path / "results/pilot-analysis/analysis.json")
    assert saved["pilot"]["units"] == 12 and not saved["pilot"]["sufficient"]


def test_diagnostic_cannot_grant_full_acceptance(tmp_path):
    research, execution, model = fixture_study(tmp_path)
    diagnostic = benchmark(research, execution, model=model, limit=1)
    assert diagnostic["complete"] and diagnostic["diagnostic"]
    write_json(tmp_path / "results/benchmark.json", diagnostic)
    info, _, _ = load_prepared(execution, research)
    with pytest.raises(ValueError, match="Full fresh technical"):
        require_benchmark(research, execution, info, model)


def test_prepared_token_corruption_is_detected(tmp_path):
    research, execution, _ = fixture_study(tmp_path)
    load_prepared(execution, research)
    path = tmp_path / "prepared/tokens.bin"
    data = bytearray(path.read_bytes())
    data[0] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="token bytes changed"):
        load_prepared(execution, research)


def test_analysis_parameters_do_not_force_retokenization(tmp_path):
    research, execution, _ = fixture_study(tmp_path)
    research["analysis"]["planning_draws"] = 200
    load_prepared(execution, research)
    research["context"] += 1
    with pytest.raises(ValueError, match="different research settings"):
        load_prepared(execution, research)
