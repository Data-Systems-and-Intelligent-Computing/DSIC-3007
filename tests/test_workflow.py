"""A tiny actual OLMoE exercises fresh collection/calibration/resume, not pretrained acceptance."""
import copy
import json

import numpy as np
import pytest
import torch
from transformers import OlmoeConfig, OlmoeForCausalLM

from scripts.analyze import analyze
from scripts.benchmark import benchmark, diagnostic_cases, require_benchmark, runtime_projection, technical_cases
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


def test_diagnostic_interleaves_domains_and_independent_groups():
    domains = ["c4", "scientific", "code"]
    cases = [({"domain": domain, "group": f"{domain}-{group}", "id": f"{domain}-{group}"}, position)
             for domain in domains for group in range(4) for position in (4095, 6000, 8000, 10000)]
    selected = diagnostic_cases(cases, domains, 9)
    assert [doc["domain"] for doc, _ in selected] == domains * 3
    for domain in domains:
        assert len({doc["group"] for doc, _ in selected if doc["domain"] == domain}) == 3
    # Selection is deterministic and never rewrites full technical cases.
    assert selected == diagnostic_cases(cases, domains, 9)
    assert len(cases) == 48 and len(diagnostic_cases(cases, domains, 100)) == 48
    with pytest.raises(ValueError, match="positive integer"):
        diagnostic_cases(cases, domains, 0)


def test_small_benchmark_reports_timing_without_collecting_other_stages(tmp_path):
    research, execution, model = fixture_study(tmp_path)
    summary = benchmark(research, execution, model=model, limit=9)
    assert summary["complete"] and summary["diagnostic"] and summary["comparisons"] == 72
    assert summary["warmup_seconds"] > 0
    for domain in research["domains"]:
        assert summary["rates"][domain]["verification"]["count"] == 3
        assert summary["rates"][domain]["verification"]["mean"] > 0
        assert summary["rates"][domain]["routing"]["count"] == 3
        assert summary["rates"][domain]["effects"]["count"] == 3
    rows = list(records(tmp_path / "results/benchmark-diagnostic.sqlite"))
    events = [row for _, row in rows if "verification" in row]
    assert len(events) == 9 and all(row["verification_seconds"] > 0 for row in events)
    assert "not a main allocation" in summary["runtime_projection"]["evidence"]
    assert not (tmp_path / "results/calibration.sqlite").exists()
    assert not (tmp_path / "results/pilot.sqlite").exists()
    assert not (tmp_path / "results/main-plan.json").exists()
    assert not (tmp_path / "results/benchmark.json").exists()


def test_runtime_projection_counts_whole_code_groups_and_separates_native_cost(tmp_path):
    research, execution, _ = fixture_study(tmp_path)
    _, documents, _ = load_prepared(execution, research)
    research["sampling"].update(main_min_scientific_documents=2, main_min_code_groups=2)
    rates = {d: {mode: {"mean": value, "max": value * 2} for mode, value in
                 (("routing", i), ("effects", i * 10), ("verification", i * 100))}
             for i, d in enumerate(research["domains"], 1)}
    projection = runtime_projection(documents, research, rates, load_seconds=7)
    scenarios = {r["scenario"]: r for r in projection["scenarios"]}
    minimum = scenarios["main_minimum"]
    assert minimum["groups"] == {"scientific": 2, "code": 2}
    assert minimum["documents"] == {"scientific": 2, "code": 4}
    assert minimum["windows"] == {"scientific": 4, "code": 8}
    assert minimum["work_seconds_mean"] == 4 * 20 + 8 * 30
    assert minimum["work_seconds_observed_max"] == 2 * minimum["work_seconds_mean"]
    assert projection["load_seconds"] == 7  # Not charged once per window or folded into work.
    assert scenarios["pilot_10"]["windows"] == {"scientific": 20, "code": 40}
    assert scenarios["main_inventory"]["windows"] == {"scientific": 40, "code": 80}
    assert scenarios["full_technical"]["work_seconds_mean"] == 4 * (111 + 222 + 333)
    # The first calibration rung cannot freeze; probe is added only to later rungs.
    assert not scenarios["calibration_1"]["freeze_eligible"]
    assert not scenarios["calibration_1"]["includes_probe"]
    assert scenarios["calibration_2_plus_probe"]["includes_probe"]
    # A single-domain receipt cannot pretend other-domain rates were measured.
    incomplete = runtime_projection(documents, research, {"c4": rates["c4"]}, 7)
    assert all(not r["available"] for r in incomplete["scenarios"])


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
