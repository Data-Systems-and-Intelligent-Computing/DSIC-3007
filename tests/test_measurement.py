import copy

import numpy as np
import pytest
import torch
from transformers import OlmoeConfig, OlmoeForCausalLM

from scripts.common import read_json
from scripts.measurement import capture, deterministic, forward, measure, suffix, verify


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


@torch.inference_mode()
def test_suffix_preserves_native_full_window_normalization_and_inputs(tiny):
    model, research = tiny
    ids = np.arange(research["context"], dtype=np.int32) + 2
    with capture(model, research["layer"], optimized=True) as state:
        forward(model, ids)
    original_mixture = state["mixture"].clone()
    original_residual = state["residual"].clone()
    inputs = []
    handle = model.model.norm.register_forward_pre_hook(lambda module, args: inputs.append(args[0].detach().clone()))
    try:
        for expert in [None, *state["ids"][-1].tolist()]:
            optimized_logits = suffix(model, state, expert)
            optimized_input = inputs.pop()
            with capture(model, research["layer"], ablate=expert):
                native_logits = forward(model, ids)
            native_input = inputs.pop()
            assert optimized_input.shape == native_input.shape == (1, research["context"], model.config.hidden_size)
            torch.testing.assert_close(optimized_input, native_input, atol=0, rtol=0)
            torch.testing.assert_close(optimized_logits, native_logits, atol=0, rtol=0)
            assert torch.equal(state["mixture"], original_mixture)
            assert torch.equal(state["residual"], original_residual)
    finally:
        handle.remove()


def test_native_nll_mismatch_remains_an_error_with_numeric_evidence(tiny, monkeypatch):
    import scripts.measurement as measurement
    model, research = tiny
    original = measurement.measure

    def corrupted(*args, **kwargs):
        result, state, logits, mixtures = original(*args, **kwargs)
        result["ablated_nll"][0] += 1e-4
        return result, state, logits, mixtures

    monkeypatch.setattr(measurement, "measure", corrupted)
    with pytest.raises(ValueError, match="Optimized ablated NLL exceeds frozen tolerance") as error:
        verify(model, np.arange(16, dtype=np.int32) + 2, 17, research)
    for field in ("expert=", "slot=0", "target=17", "optimized_nll=", "native_nll=", "absolute_error=", "limit=", "max_logit_error=", "max_mixture_error="):
        assert field in str(error.value)
    assert research["numerical"]["nll_atol"] == 1e-6
    assert "forward" not in model.model.layers[-1].mlp.experts.__dict__


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_model_loading_timings_preserve_precision_checks(tmp_path, monkeypatch, dtype):
    import json
    from types import SimpleNamespace
    from transformers import AutoModelForCausalLM
    import scripts.common as common
    from scripts.measurement import load_model
    folder = tmp_path / "data/manifests"
    folder.mkdir(parents=True)
    (folder / "model-fp32.json").write_text(json.dumps({"revision": "fixture", "files": [{"path": "weights", "size_bytes": 4}]}))
    (tmp_path / "weights").write_bytes(b"1234")
    research = {"seed": 3007, "layer": 1, "model": {"revision": "fixture", "layers": 2, "experts": 8, "active_experts": 8}}
    class MockModel:
        config = SimpleNamespace(model_type="olmoe", num_hidden_layers=2, num_local_experts=8, num_experts_per_tok=8, norm_topk_prob=False)
        model = SimpleNamespace(layers=[None, None])
        parameter = torch.nn.Parameter(torch.ones(1, dtype=dtype))
        def parameters(self): return iter([self.parameter])
        def to(self, device):
            assert device == "cuda:0"
            return self
        def eval(self): return self
    model = MockModel()
    def constructor(path, **kwargs):
        assert path == str(tmp_path) and kwargs["local_files_only"]
        assert kwargs["dtype"] == "auto" and kwargs["experts_implementation"] == "eager"
        return model
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", staticmethod(constructor))
    timings = {}
    execution = {"device": "cuda:0", "model_path": str(tmp_path)}
    if dtype == torch.float32:
        assert load_model(research, execution, timings=timings) is model
        assert timings["total_load_seconds"] >= timings["from_pretrained_seconds"] + timings["to_gpu_seconds"]
    else:
        with pytest.raises(ValueError, match="original FP32"):
            load_model(research, execution, timings=timings)
        assert not timings
