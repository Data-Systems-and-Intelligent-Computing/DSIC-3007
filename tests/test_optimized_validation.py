"""Independent CPU comparisons for the accelerated validation engine."""

import torch
import pytest
from transformers import OlmoeConfig, OlmoeForCausalLM

from src.artifacts import atomic, canonical, file_digest, read_json, read_rows, seal, write_rows
from src.optimized_validation import _input_case, validate_optimized
from src.preparation import window_for


def tiny_model():
    torch.manual_seed(3007)
    config = OlmoeConfig(vocab_size=32, hidden_size=8, intermediate_size=12,
                         num_hidden_layers=16, num_attention_heads=2,
                         num_key_value_heads=2, num_experts=64,
                         num_experts_per_tok=8, norm_topk_prob=False,
                         max_position_embeddings=16, use_cache=False,
                         attn_implementation="eager", experts_implementation="eager")
    model = OlmoeForCausalLM(config).eval()
    model.config._experts_implementation = "eager"
    return model


def tiny_input(tokens=None, layers=None):
    tokens = tokens or [2, 5, 7, 11, 13, 17, 19, 23, 29]
    return dict(window_for(tokens, len(tokens) - 2, len(tokens) - 1),
                input_id="independent-validation", document_id="tiny-doc",
                domain="scientific", source_group="tiny-paper",
                layers=layers or [0, 7, 15])


def assert_clean(model):
    for layer in model.model.layers:
        experts = layer.mlp.experts
        assert not experts._forward_pre_hooks and not experts._forward_hooks
        assert "forward" not in experts.__dict__


def test_independent_native_logits_mixtures_and_downstream_routes_short_context():
    model = tiny_model()
    result = _input_case(model, tiny_input([2, 5, 7, 11, 13]), lambda: 0.0, 600)
    assert len(result["cases"]) == 24
    assert {case["layer"] for case in result["cases"]} == {0, 7, 15}
    assert all(case["downstream_layers_checked"] == 15 - case["layer"] for case in result["cases"])
    assert max(case["nll_abs_error"] for case in result["cases"]) <= 1e-6
    assert result["cleanup_pass"] and result["parameters_unchanged"]
    timing = result["comparison_timings"]
    assert set(timing["optimized_suffix_seconds_by_layer"]) == {"0", "7", "15"}
    assert all(case["native_full_forward_seconds"] > 0 and
               case["optimized_suffix_seconds"] > 0 for case in result["cases"])
    assert timing["synchronized_comparison_total_seconds"] >= sum(
        timing["native_ablation_seconds_by_layer"].values())
    assert_clean(model)


def test_validation_detects_accelerated_numerical_failure(monkeypatch):
    from src import run_experiment

    original = run_experiment._replay_from_layer

    def wrong(*args, **kwargs):
        nll, logits = original(*args, **kwargs)
        return nll + 0.01, logits

    monkeypatch.setattr(run_experiment, "_replay_from_layer", wrong)
    model = tiny_model()
    with pytest.raises(AssertionError, match="NLL mismatch"):
        _input_case(model, tiny_input(layers=[15]), lambda: 0.0, 600)
    assert_clean(model)


def test_validation_exception_cleans_downstream_hooks(monkeypatch):
    from src import run_experiment

    original = run_experiment._replay_from_layer
    calls = 0

    def fail_later_expert(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 7:
            raise RuntimeError("injected later expert replay failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(run_experiment, "_replay_from_layer", fail_later_expert)
    model = tiny_model()
    with pytest.raises(RuntimeError, match="injected later expert replay failure"):
        _input_case(model, tiny_input(layers=[0]), lambda: 0.0, 600)
    assert calls == 7
    assert_clean(model)


def _technical_plan(path, context, research, catalog):
    path.mkdir()
    atomic(path / "research.json", canonical(research) + b"\n")
    atomic(path / "model-catalog.json", canonical(catalog) + b"\n")
    atomic(path / "source-review.json", canonical({"fixture": "review"}) + b"\n")
    rows = []
    for domain in ("c4", "scientific", "code"):
        for group_index in range(3):
            row = dict(tiny_input([2, 5, 7, 11, 13], [0, 7, 15]))
            row.update(input_id=f"{context}-{domain}-{group_index}",
                       document_id=f"doc-{domain}-{group_index}", domain=domain,
                       source_group=f"{domain}-group-{group_index}", context_length=context,
                       input_ids=[2] * context, local_predictor_position=context - 1, target_id=3)
            rows.append(row)
    write_rows(path / "inputs.jsonl", rows)
    seal(path, "experiment-plan", {"stage": "technical", "window": context,
                                    "source_review_sha256": file_digest(path / "source-review.json"),
                                    "units": {"c4": 12, "scientific": 12, "code": 12}})


def test_late_expert_failure_persists_progress_and_never_passes(tmp_path, monkeypatch, capsys):
    from src import model as loader
    from src import optimized_validation as validation

    research = read_json("configs/research.json")
    catalog = {"fixture": "tiny-native-olmOe"}
    plans = [tmp_path / "technical-2048", tmp_path / "technical-4096"]
    _technical_plan(plans[0], 2048, research, catalog)
    _technical_plan(plans[1], 4096, research, catalog)
    model = tiny_model()
    class Session:
        def acquire(self, *args):
            return model

    monkeypatch.setattr(loader, "runtime_identity", lambda: {"fixture": "cpu"})
    monkeypatch.setattr(loader, "implementation_identity", lambda: "implementation-hash")
    durable_comparisons = 0
    original_record, original_progress = validation._record, validation._progress

    def track_record(path, row):
        nonlocal durable_comparisons
        original_record(path, row)
        if row.get("record_type") == "input_progress":
            durable_comparisons = max(durable_comparisons, row["completed_comparisons"])

    def require_durable_before_visible(message):
        if message.startswith("comparison "):
            visible_comparisons = int(message.split()[1].split("/")[0])
            assert durable_comparisons >= visible_comparisons
        original_progress(message)

    monkeypatch.setattr(validation, "_record", track_record)
    monkeypatch.setattr(validation, "_progress", require_durable_before_visible)

    def simulated_later_expert_failure(model, inputs, elapsed, max_seconds, progress=None):
        for index in range(1, 7):
            progress({"input_id": inputs["input_id"], "context_length": len(inputs["input_ids"]),
                      "completed_comparisons": index,
                      "latest_case": {"layer": 0, "expert_id": index - 1,
                                      "max_logit_abs_error": 0.0, "max_mixture_abs_error": 0.0,
                                      "nll_abs_error": 0.0}, "status": "in_progress"})
        raise RuntimeError("injected later expert numerical path failure")

    monkeypatch.setattr("src.optimized_validation._input_case", simulated_later_expert_failure)
    output = tmp_path / "optimized-validation.json"
    with pytest.raises(ValueError, match="optimized validation failed"):
        validate_optimized(plans, tmp_path / "unused-model-path", output,
                           validation_scope="screen", model_session=Session())
    receipt = read_json(output)
    report = list(read_rows(tmp_path / "optimized-validation.json.cases.jsonl"))
    assert receipt["status"] == "fail" and not receipt["full_coverage"]
    assert receipt["approval"] == "diagnostic_only"
    assert any(row.get("record_type") == "input_progress" and
               row["completed_comparisons"] == 6 for row in report)
    assert any(row.get("status") == "fail" for row in report)
    assert_clean(model)
    stdout = capsys.readouterr().out
    assert "plan preflight pass" in stdout
    assert "model load start" in stdout and "model load complete" in stdout
    assert "comparison 6/432 (1.4%)" in stdout
    assert "domain=c4 context=2048 layer=0 expert=5" in stdout
    assert "input 1/18 fail" in stdout
    assert "validation fail inputs=0/18 comparisons=6/432" in stdout


