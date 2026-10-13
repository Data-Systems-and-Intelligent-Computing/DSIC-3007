import copy

import numpy as np
import pytest
import torch
from transformers import OlmoeConfig, OlmoeForCausalLM

from scripts.common import read_json
from scripts.measurement import capture, deterministic, forward, measure, verify


@pytest.fixture
def tiny():
    torch.set_num_threads(1)
    deterministic(3007)
    research = copy.deepcopy(read_json("config/research.json"))
    research.update(context=16, layer=1)
    research["model"].update(layers=2, experts=16, active_experts=8)
    config = OlmoeConfig(hidden_size=24, intermediate_size=16, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=4, num_experts=16,
                         num_local_experts=16, num_experts_per_tok=8, vocab_size=80,
                         max_position_embeddings=64, norm_topk_prob=False, eos_token_id=79, pad_token_id=1)
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    return OlmoeForCausalLM(config).eval(), research


@pytest.mark.parametrize("seed", [0, 3007, 9871])
def test_eight_effects_match_independent_native(tiny, seed):
    model, research = tiny
    ids = np.random.default_rng(seed).integers(2, 78, 16, dtype=np.int32)
    result = verify(model, ids, 17, research)
    assert result["comparisons"] == 8
    assert result["cleanup_exact"]
    assert result["max_errors"]["nll"] <= 1e-6
    np.testing.assert_allclose(result["native_effects"], result["optimized_effects"], atol=1e-6, rtol=0)


def test_single_target_no_replacement_or_renormalization(tiny):
    model, research = tiny
    ids = np.arange(16, dtype=np.int32) + 2
    with capture(model, 1) as baseline:
        forward(model, ids)
    expert = int(baseline["ids"][-1, 0])
    with capture(model, 1, ablate=expert) as changed:
        forward(model, ids)
    expected = baseline["weights"].clone()
    expected[-1, 0] = 0
    assert torch.equal(changed["used_weights"], expected)
    assert torch.equal(changed["ids"], baseline["ids"])
    assert torch.equal(changed["mixture"][:-1], baseline["mixture"][:-1])


def test_capture_restores_override_and_hooks_on_failure(tiny):
    model, _ = tiny
    experts = model.model.layers[-1].mlp.experts
    assert "forward" not in experts.__dict__
    with pytest.raises(RuntimeError):
        with capture(model, 1, optimized=True):
            raise RuntimeError("injected failure")
    assert "forward" not in experts.__dict__
    assert not experts._forward_hooks and not experts._forward_pre_hooks
    assert not model.model.layers[-1].post_attention_layernorm._forward_pre_hooks


def test_nonfinite_input_fails_and_restores(tiny):
    model, research = tiny
    expert = model.model.layers[-1].mlp.experts
    with torch.no_grad():
        expert.gate_up_proj.fill_(float("nan"))
    with pytest.raises(ValueError, match="Non-finite"):
        measure(model, np.arange(16, dtype=np.int32) + 2, 17, research)
    assert "forward" not in expert.__dict__
    assert not expert._forward_hooks


def test_routing_records_every_input_position(tiny):
    model, research = tiny
    result, state, _, _ = measure(model, np.arange(16, dtype=np.int32) + 2, 17, research, routing=True)
    assert state["ids"].shape == (16, 8)
    assert state["weights"].dtype == torch.float32
    assert result["cleanup_exact"] and "effects" not in result
