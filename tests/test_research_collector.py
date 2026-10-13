"""Tiny CPU mechanics are regression evidence, never pretrained GPU approval."""

from copy import deepcopy
import struct

import pytest
import torch
from transformers import OlmoeConfig, OlmoeForCausalLM

from src.artifacts import Journal, digest, validate_measurement
from src.interventions import RoutingCapture
from src.preparation import eligible_positions, sample_positions, window_for
from src.run_experiment import measure_input


def tiny_model():
    torch.manual_seed(17)
    config = OlmoeConfig(vocab_size=64, hidden_size=8, intermediate_size=12,
                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                        num_experts=64, num_experts_per_tok=8, norm_topk_prob=False,
                        max_position_embeddings=32, use_cache=False,
                        attn_implementation="eager", experts_implementation="eager")
    model = OlmoeForCausalLM(config).eval()
    model.config._experts_implementation = "eager"
    return model


def test_full_context_target_is_outside_and_nested_sampling():
    tokens = list(range(4110))
    assert eligible_positions(tokens[:4096], 4096) == []
    w = window_for(tokens, 4095, 4096)
    assert len(w["input_ids"]) == 4096
    assert w["input_ids"][-1] == 4095 and w["target_id"] == 4096
    assert w["window_end"] == 4096 and w["target_position"] == 4096
    small = sample_positions(tokens, "document", 4096, 2, 3007, "pilot")
    large = sample_positions(tokens, "document", 4096, 8, 3007, "pilot")
    assert small == large[:2] and len(set(large)) == 8
    sensitivity = window_for(tokens, small[0], 2048)
    primary = window_for(tokens, small[0], 4096)
    assert sensitivity["target_id"] == primary["target_id"]
    assert sensitivity["input_ids"] == primary["input_ids"][-2048:]


def test_actual_packed_experts_eight_nlls_survivors_and_cleanup():
    model = tiny_model()
    tokens = [3, 4, 5, 6, 7, 8, 9, 10, 11]
    inputs = dict(window_for(tokens, 7, 8), input_id="tiny-input", document_id="tiny-doc",
                  domain="scientific", source_group="paper", layers=[0])
    row = measure_input(model, inputs)[0]
    assert len(row["effects"]) == 8
    validate_measurement(row)
    again = measure_input(model, inputs)[0]
    assert row["routing"] == again["routing"]
    assert row["baseline_nll"] == again["baseline_nll"]
    assert row["effects"] == again["effects"]
    with pytest.raises(RuntimeError):
        with RoutingCapture(model, 0, 7):
            raise RuntimeError("injected failure")
    assert not model.model.layers[0].mlp.experts._forward_pre_hooks
    assert not model.model.layers[0].mlp.experts._forward_hooks
    with RoutingCapture(model, 0, 7, capture_window=True,
                        capture_reconstruction=True) as hook:
        model(input_ids=torch.tensor([tokens[:-1]]), attention_mask=torch.ones(1,8,dtype=torch.long))
        reconstructed = hook.reconstruct()
        torch.testing.assert_close(reconstructed, hook.mixture, atol=1e-7, rtol=1e-6)


def test_reconstruction_requires_explicit_capture():
    model = tiny_model()
    with RoutingCapture(model, 0, 3) as hook:
        model(input_ids=torch.tensor([[3, 4, 5, 6, 7, 8]]),
              attention_mask=torch.ones(1, 6, dtype=torch.long))
        with pytest.raises(ValueError, match="native batch reconstruction capture required"):
            hook.reconstruct()


def test_reconstruction_reuses_native_expert_assignment_batch_sizes(monkeypatch):
    model = tiny_model()
    input_ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])
    with RoutingCapture(model, 0, 6, capture_reconstruction=True) as hook:
        model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
        ids = hook.reconstruction_ids
        counts = torch.bincount(ids.flatten(), minlength=64)
        active_counts = [(expert, int(counts[expert])) for expert in range(64) if counts[expert]]

        calls = []
        original_linear = torch.nn.functional.linear

        def record_linear(inputs, weight, bias=None):
            calls.append((inputs.shape[0], tuple(weight.shape)))
            return original_linear(inputs, weight, bias)

        monkeypatch.setattr(torch.nn.functional, "linear", record_linear)
        reconstructed = hook.reconstruct()

    expected = []
    for _, rows in active_counts:
        expected.extend([(rows, (24, 8)), (rows, (8, 12))])
    assert calls == expected
    # Replaying the complete native batching and index_add order must reproduce
    # the captured expert mixture exactly, not just within a relaxed tolerance.
    assert torch.equal(reconstructed, hook.mixture)


def test_excluded_expert_matches_native_local_ablation_and_preserves_other_rows():
    model = tiny_model()
    input_ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])
    attention_mask = torch.ones_like(input_ids)
    q = 6
    with RoutingCapture(model, 0, q, capture_window=True,
                        capture_reconstruction=True) as baseline:
        baseline_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.detach().clone()
        baseline_routing = baseline.routing()

    assert len(baseline_routing["expert_ids"]) == 8
    for expert in baseline_routing["expert_ids"]:
        with RoutingCapture(model, 0, q, expert, capture_window=True,
                            capture_reconstruction=True) as ablation:
            ablated_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.detach().clone()
            assert ablation.routing() == baseline_routing
            assert ablation.mask_is_local and ablation.survivors_unchanged
            assert torch.equal(ablation.window_mixture[:q], baseline.window_mixture[:q])
            assert torch.equal(ablation.window_mixture[q + 1:], baseline.window_mixture[q + 1:])
            assert torch.equal(ablation.reconstruct(expert), ablation.mixture)
        assert not torch.equal(ablated_logits, baseline_logits)
        assert not model.model.layers[0].mlp.experts._forward_pre_hooks
        assert not model.model.layers[0].mlp.experts._forward_hooks

    clean_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    assert torch.equal(clean_logits, baseline_logits)


def test_calibration_keeps_every_input_route_and_probe_has_no_ablation():
    model = tiny_model()
    inputs = dict(window_for([3,4,5,6,7,8,9,10,11], 7, 8), input_id="cal-window",
                  document_id="cal-doc", domain="code", source_group="repo", layers=[0])
    row = measure_input(model, inputs, calibration=True)[0]
    assert row["window_routing"]["shape"] == [8,8]
    assert len(row["window_routing"]["expert_ids"]) == 8
    assert len(bytes.fromhex(row["window_routing"]["weights_fp32_le_hex"])) == 8*8*4
    assert "effects" not in row
    probe = measure_input(model, inputs, routing_only=True)[0]
    assert probe["kind"] == "probe_routing" and "effects" not in probe


def test_journal_rejects_incomplete_expert_set_and_changed_resume(tmp_path):
    model = tiny_model()
    inputs = dict(window_for([3,4,5,6,7,8,9,10,11], 7, 8), input_id="journal-input",
                  document_id="doc", domain="scientific", source_group="paper", layers=[0])
    row = measure_input(model, inputs)[0]
    incomplete = deepcopy(row)
    incomplete["effects"].pop()
    with Journal(tmp_path / "run", {"plan": "a"}) as journal:
        with pytest.raises(ValueError, match="eight"):
            journal.append(incomplete)
        journal.append(row)
    with pytest.raises(ValueError, match="resume"):
        Journal(tmp_path / "run", {"plan": "b"})
    with Journal(tmp_path / "run", {"plan": "a"}) as journal:
        assert journal.records[row["event_id"]]["effects"] == row["effects"]
