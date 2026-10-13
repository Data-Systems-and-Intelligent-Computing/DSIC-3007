"""CPU checks for the opt-in full-shape decoder suffix executor."""

import torch
import pytest
from transformers import OlmoeConfig, OlmoeForCausalLM

from src.preparation import window_for
from src.interventions import RoutingCapture
from src.run_experiment import measure_input


def tiny_16_layer_model():
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


def test_suffix_replay_matches_native_all_eight_effects_and_cleans_up():
    model = tiny_16_layer_model()
    tokens = [2, 5, 7, 11, 13, 17, 19, 23, 29]
    inputs = dict(window_for(tokens, 7, 8), input_id="opt-tiny-input",
                  document_id="opt-tiny-doc", domain="scientific",
                  source_group="tiny-paper", layers=[0, 7, 15])

    native = measure_input(model, inputs)
    calls = 0
    original_forward = model.forward

    def counted_forward(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_forward(*args, **kwargs)

    model.forward = counted_forward
    optimized = measure_input(model, inputs, optimized=True)
    assert calls == 2  # captured baseline plus exact uninstrumented cleanup reference
    assert [row["event_id"] for row in optimized] == [row["event_id"] for row in native]
    assert [row["layer"] for row in optimized] == [0, 7, 15]
    for expected, observed in zip(native, optimized, strict=True):
        assert len(observed["effects"]) == 8
        assert observed["routing"] == expected["routing"]
        assert observed["baseline_nll"] == expected["baseline_nll"]
        assert [effect["expert_id"] for effect in observed["effects"]] == [
            effect["expert_id"] for effect in expected["effects"]]
        for expected_effect, observed_effect in zip(expected["effects"], observed["effects"], strict=True):
            assert abs(observed_effect["ablated_nll"] - expected_effect["ablated_nll"]) <= 1e-6
            assert abs(observed_effect["delta_nll"] - expected_effect["delta_nll"]) <= 1e-6
        assert observed["effect_seconds"] >= 0

    clean = model(input_ids=torch.tensor([tokens[:-1]]),
                  attention_mask=torch.ones(1, 8, dtype=torch.long)).logits
    assert all(not layer.mlp.experts._forward_pre_hooks and not layer.mlp.experts._forward_hooks
               for layer in model.model.layers)
    assert torch.isfinite(clean).all()


@pytest.mark.parametrize("layers", [[15], [15, 0, 7]])
def test_optimized_effect_shapes_keep_eight_effects_with_two_full_forwards(layers):
    model = tiny_16_layer_model()
    tokens = [2, 5, 7, 11, 13, 17, 19, 23, 29]
    inputs = dict(window_for(tokens, 7, 8), input_id="shape-input",
                  document_id="shape-doc", domain="scientific",
                  source_group="shape-paper", layers=layers)
    calls = 0
    original_forward = model.forward

    def counted_forward(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_forward(*args, **kwargs)

    model.forward = counted_forward
    rows = measure_input(model, inputs, optimized=True)

    assert calls == 2
    assert [row["layer"] for row in rows] == layers
    assert all(len(row["effects"]) == 8 for row in rows)
    assert all(row["cleanup_seconds"] >= 0 for row in rows)


@pytest.mark.parametrize("calibration,routing_only", [(True, False), (False, True)])
def test_native_calibration_and_probe_keep_two_full_forwards(calibration, routing_only):
    model = tiny_16_layer_model()
    tokens = [2, 5, 7, 11, 13, 17, 19, 23, 29]
    inputs = dict(window_for(tokens, 7, 8), input_id="native-input",
                  document_id="native-doc", domain="code",
                  source_group="native-repo", layers=[15])
    calls = 0
    original_forward = model.forward

    def counted_forward(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_forward(*args, **kwargs)

    model.forward = counted_forward
    rows = measure_input(model, inputs, calibration=calibration, routing_only=routing_only)

    assert calls == 2  # baseline and native cleanup; no optimized effect replay
    assert len(rows) == 1
    if calibration:
        assert rows[0]["kind"] == "calibration_window"
        assert "window_routing" in rows[0]
    else:
        assert rows[0]["kind"] == "probe_routing"
        assert "effects" not in rows[0]


def test_optimized_cleanup_rejects_subtolerance_corrupted_contribution_baseline(monkeypatch):
    model = tiny_16_layer_model()
    tokens = [2, 5, 7, 11, 13, 17, 19, 23, 29]
    inputs = dict(window_for(tokens, 7, 8), input_id="cleanup-drift-input",
                  document_id="cleanup-drift-doc", domain="scientific",
                  source_group="cleanup-paper", layers=[15])
    runner = __import__("src.run_experiment", fromlist=["forward"])
    original_forward = runner.forward
    observed_logits = []

    def record_forwards(model_arg, input_arg):
        nll, logits = original_forward(model_arg, input_arg)
        observed_logits.append(logits.detach().clone())
        return nll, logits

    monkeypatch.setattr(runner, "forward", record_forwards)
    original_capture = RoutingCapture._native_forward_with_capture

    def corrupt_captured_mixture(capture, hidden, ids, weights):
        mixture = original_capture(capture, hidden, ids, weights)
        # The captured native baseline is perturbed below the accepted 1e-6
        # numerical comparison tolerance; cleanup must still compare exactly.
        mixture[capture.q, 0] += 1e-7
        return mixture

    monkeypatch.setattr(RoutingCapture, "_native_forward_with_capture", corrupt_captured_mixture)
    with pytest.raises(ValueError, match="baseline did not recover"):
        measure_input(model, inputs, optimized=True)
    assert len(observed_logits) == 2
    drift = (observed_logits[0] - observed_logits[1]).abs().max().item()
    assert 0 < drift <= 1e-6
    for layer in model.model.layers:
        experts = layer.mlp.experts
        assert not experts._forward_pre_hooks and not experts._forward_hooks
        assert "forward" not in experts.__dict__


def test_suffix_replay_restores_expert_forward_after_failure(monkeypatch):
    model = tiny_16_layer_model()
    parameter_versions = {name: parameter._version for name, parameter in model.named_parameters()}
    parameter_values = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    tokens = [2, 5, 7, 11, 13, 17, 19, 23, 29]
    inputs = dict(window_for(tokens, 7, 8), input_id="opt-failure-input",
                  document_id="opt-failure-doc", domain="code",
                  source_group="tiny-repo", layers=[15])

    def fail(*args, **kwargs):
        raise RuntimeError("injected recomposition failure")

    monkeypatch.setattr(RoutingCapture, "recompose_for_target", fail)
    with pytest.raises(RuntimeError, match="injected recomposition failure"):
        measure_input(model, inputs, optimized=True)
    for layer in model.model.layers:
        experts = layer.mlp.experts
        assert not experts._forward_pre_hooks and not experts._forward_hooks
        assert "forward" not in experts.__dict__
    assert {name: parameter._version for name, parameter in model.named_parameters()} == parameter_versions
    assert all(torch.equal(parameter, parameter_values[name])
               for name, parameter in model.named_parameters())
