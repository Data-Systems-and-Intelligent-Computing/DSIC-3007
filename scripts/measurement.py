"""FP32 OLMoE measurement: native reference and final-layer contribution reuse.

Only the final layer is optimized. Independent references call the library's
unmodified expert implementation. Never subtract from an already rounded mixture.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib.metadata
from pathlib import Path
import resource
import time
import types

import numpy as np
import torch
from torch.nn import functional as F


def deterministic(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def runtime(model):
    device = next(model.parameters()).device
    return {"torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
            "cuda": torch.version.cuda, "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}


def memory(model):
    device = next(model.parameters()).device
    result = {"host_peak_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e9}
    if device.type == "cuda":
        result.update(gpu_allocated_gb=torch.cuda.max_memory_allocated(device) / 1e9,
                      gpu_reserved_gb=torch.cuda.max_memory_reserved(device) / 1e9)
    return result


def synchronize(model):
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_model(research, execution):
    from transformers import AutoModelForCausalLM
    from scripts.common import ROOT, read_json
    if not execution["device"].startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Pretrained Path B execution requires a CUDA GPU; CPU synthetic tests use tiny models separately")
    deterministic(research["seed"])
    started = time.monotonic()
    print(f"Loading pinned original FP32 model from {execution['model_path']}", flush=True)
    catalog = read_json(ROOT / "data/manifests/model-fp32.json")
    if catalog["revision"] != research["model"]["revision"]:
        raise ValueError("Model asset catalog differs from the selected FP32 revision")
    for asset in catalog["files"]:
        path = Path(execution["model_path"]) / asset["path"]
        if not path.is_file() or path.stat().st_size != asset["size_bytes"]:
            raise ValueError(f"Prepared pinned model asset missing/wrong size: {asset['path']}")
    # Local files only: downloading belongs to acquisition, never to GPU startup.
    model = AutoModelForCausalLM.from_pretrained(
        execution["model_path"], local_files_only=True, dtype="auto",
        attn_implementation="eager", experts_implementation="eager",
    ).to(execution["device"]).eval()
    cfg = model.config
    expected = research["model"]
    if (cfg.model_type != "olmoe" or cfg.num_hidden_layers != expected["layers"]
            or cfg.num_local_experts != expected["experts"] or cfg.num_experts_per_tok != expected["active_experts"] or cfg.norm_topk_prob
            or any(p.dtype != torch.float32 for p in model.parameters())):
        raise ValueError("Loaded model is not the selected original FP32 OLMoE architecture")
    if research["layer"] != len(model.model.layers) - 1:
        raise ValueError("Final-layer optimization cannot be used for another layer")
    synchronize(model)
    print(f"Loaded model in {time.monotonic() - started:.2f}s: {runtime(model)}", flush=True)
    return model


def nll(logits, target):
    value = -F.log_softmax(logits.double(), dim=-1)[target]
    if not torch.isfinite(value):
        raise ValueError("Non-finite next-token NLL")
    return float(value)


@torch.inference_mode()
def forward(model, ids):
    device = next(model.parameters()).device
    inputs = torch.tensor(np.asarray(ids).copy(), dtype=torch.long, device=device)[None, :]
    logits = model(input_ids=inputs, use_cache=False, logits_to_keep=1,
                   output_router_logits=False, return_dict=True).logits[0, -1].detach().clone()
    if not torch.isfinite(logits).all():
        raise ValueError("Non-finite logits")
    return logits


@contextmanager
def capture(model, layer, optimized=False, ablate=None, noop=False):
    """All hooks and the temporary expert override are restored on every exit."""
    block = model.model.layers[layer]
    experts = block.mlp.experts
    state = {}
    handles = []
    previous_override = experts.__dict__.get("forward")

    def residual_hook(module, inputs):
        state["residual"] = inputs[0][0, -1].detach().clone()

    def expert_before(module, inputs):
        hidden, ids, weights = inputs
        if not torch.isfinite(hidden).all() or not torch.isfinite(weights).all():
            raise ValueError("Non-finite expert input")
        if torch.any(weights < 0) or any(len(set(row)) != len(row) for row in ids[-1:].tolist()):
            raise ValueError("Invalid native active set or coefficient")
        state["ids"] = ids.detach().clone()
        state["weights"] = weights.detach().clone()
        if ablate is not None or noop:
            changed = weights.clone()
            if ablate is not None:
                slots = torch.where(ids[-1] == ablate)[0]
                if len(slots) != 1:
                    raise ValueError("Ablation must target one naturally active expert")
                changed[-1, slots[0]] = 0
                survivor_mask = torch.ones_like(weights, dtype=torch.bool)
                survivor_mask[-1, slots[0]] = False
                if not torch.equal(changed[survivor_mask], weights[survivor_mask]):
                    raise ValueError("Survivor coefficient changed")
            state["used_weights"] = changed.detach().clone()
            return hidden, ids, changed

    def expert_after(module, inputs, output):
        state["mixture"] = output.detach().clone()

    def captured_forward(module, hidden, ids, weights):
        # Match Transformers 5.12.1 eager expert GEMMs and index_add_ order.
        total = torch.zeros_like(hidden)
        mask = F.one_hot(ids, num_classes=module.num_experts).permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        state["contributions"] = []
        for expert_index in hit:
            expert_index = expert_index[0]
            slots, rows = torch.where(mask[expert_index])
            gate, up = F.linear(hidden[rows], module.gate_up_proj[expert_index]).chunk(2, dim=-1)
            output = F.linear(module.act_fn(gate) * up, module.down_proj[expert_index])
            matches = torch.where(rows == hidden.shape[0] - 1)[0]
            for match in matches:
                state["contributions"].append((int(expert_index), int(slots[match]), output[match:match + 1].detach().clone()))
            weighted = output * weights[rows, slots, None]
            total.index_add_(0, rows, weighted.to(total.dtype))
        return total

    try:
        handles.append(block.post_attention_layernorm.register_forward_pre_hook(residual_hook))
        handles.append(experts.register_forward_pre_hook(expert_before))
        handles.append(experts.register_forward_hook(expert_after))
        if optimized:
            experts.forward = types.MethodType(captured_forward, experts)
        yield state
    finally:
        for handle in handles:
            handle.remove()
        if optimized:
            if previous_override is None:
                experts.__dict__.pop("forward", None)
            else:
                experts.forward = previous_override


def recompose(state, expert=None):
    row = torch.zeros_like(state["mixture"][-1:])
    index = torch.zeros(1, dtype=torch.long, device=row.device)
    for identity, slot, contribution in state["contributions"]:
        coefficient = state["weights"][-1, slot] if identity != expert else torch.zeros_like(state["weights"][-1, slot])
        row.index_add_(0, index, contribution * coefficient)
    return row


def suffix(model, state, expert=None):
    row = recompose(state, expert) + state["residual"][None, :]
    return model.lm_head(model.model.norm(row[None, :, :]))[0, -1].detach().clone()


@torch.inference_mode()
def measure(model, ids, target, research, routing=False):
    synchronize(model)
    started = time.monotonic()
    with capture(model, research["layer"], optimized=not routing) as state:
        baseline_logits = forward(model, ids)
    baseline = nll(baseline_logits, target)
    synchronize(model)
    capture_seconds = time.monotonic() - started
    active = state["ids"][-1].tolist()
    if len(active) != research["model"]["active_experts"]:
        raise ValueError("Incomplete active expert set")
    ablated, branch_logits, branch_mixtures = [], [], []
    if not routing:
        tolerances = research["numerical"]
        torch.testing.assert_close(recompose(state), state["mixture"][-1:], atol=tolerances["atol"], rtol=tolerances["rtol"])
        torch.testing.assert_close(suffix(model, state), baseline_logits, atol=tolerances["atol"], rtol=tolerances["rtol"])
        if abs(nll(suffix(model, state), target) - baseline) > tolerances["nll_atol"]:
            raise ValueError("No-op suffix NLL mismatch")
        for expert in active:
            logits = suffix(model, state, expert)
            ablated.append(nll(logits, target))
            branch_logits.append(logits)
            branch_mixtures.append(recompose(state, expert))
    synchronize(model)
    effects_seconds = time.monotonic() - started - capture_seconds
    cleanup_start = time.monotonic()
    clean_logits = forward(model, ids)
    if not torch.equal(clean_logits, baseline_logits) or nll(clean_logits, target) != baseline:
        raise ValueError("Exact native cleanup failed; measurement will not be persisted")
    synchronize(model)
    result = {"baseline_nll": baseline, "active": active,
              "coefficients": state["weights"][-1].cpu().tolist(), "target": int(target),
              "cleanup_exact": True, "capture_seconds": capture_seconds,
              "effects_seconds": effects_seconds, "cleanup_seconds": time.monotonic() - cleanup_start,
              "measurement_seconds": time.monotonic() - started, **memory(model)}
    if not routing:
        result.update(ablated_nll=ablated, effects=[value - baseline for value in ablated])
    return result, state, branch_logits, branch_mixtures


@torch.inference_mode()
def verify(model, ids, target, research):
    """Independent library-native baseline/repeats/no-op/eight masks/cleanup."""
    versions = [(id(p), p._version) for p in model.parameters()]
    tol = research["numerical"]
    optimized, state, logits, mixtures = measure(model, ids, target, research)
    with capture(model, research["layer"]) as native:
        reference = forward(model, ids)
    torch.testing.assert_close(reference, forward(model, ids), atol=0, rtol=0)
    torch.testing.assert_close(reference, forward(model, ids), atol=0, rtol=0)
    with capture(model, research["layer"], noop=True) as no_op:
        noop_logits = forward(model, ids)
    torch.testing.assert_close(noop_logits, reference, atol=0, rtol=0)
    for key in ("ids", "weights"):
        if not torch.equal(state[key], native[key]) or not torch.equal(no_op[key], native[key]):
            raise ValueError("Native routing differs")
    torch.testing.assert_close(state["mixture"], native["mixture"], atol=tol["atol"], rtol=tol["rtol"])
    if not torch.equal(no_op["mixture"], native["mixture"]) or not torch.equal(state["mixture"][:-1], native["mixture"][:-1]):
        raise ValueError("No-op or optimized capture changed untargeted native mixture rows")
    if abs(optimized["baseline_nll"] - nll(reference, target)) > tol["nll_atol"]:
        raise ValueError("Native baseline NLL mismatch")
    errors = {"mixture": 0.0, "logit": 0.0, "nll": 0.0}
    native_effects = []
    for slot, expert in enumerate(optimized["active"]):
        with capture(model, research["layer"], ablate=expert) as ablation:
            actual = forward(model, ids)
        if not torch.equal(ablation["ids"], native["ids"]) or not torch.equal(ablation["weights"], native["weights"]):
            raise ValueError("Ablation changed natural routing")
        if not torch.equal(ablation["mixture"][:-1], native["mixture"][:-1]):
            raise ValueError("Ablation changed untargeted tokens")
        torch.testing.assert_close(mixtures[slot], ablation["mixture"][-1:], atol=tol["atol"], rtol=tol["rtol"])
        torch.testing.assert_close(logits[slot], actual, atol=tol["atol"], rtol=tol["rtol"])
        actual_nll = nll(actual, target)
        error = abs(actual_nll - optimized["ablated_nll"][slot])
        if error > tol["nll_atol"]:
            raise ValueError("Optimized ablated NLL exceeds frozen tolerance")
        errors["mixture"] = max(errors["mixture"], float((mixtures[slot] - ablation["mixture"][-1:]).abs().max()))
        errors["logit"] = max(errors["logit"], float((logits[slot] - actual).abs().max()))
        errors["nll"] = max(errors["nll"], error)
        native_effects.append(actual_nll - nll(reference, target))
    if not torch.equal(forward(model, ids), reference) or versions != [(id(p), p._version) for p in model.parameters()]:
        raise ValueError("Native cleanup or parameter versions changed")
    return {"comparisons": len(native_effects), "max_errors": errors,
            "native_effects": native_effects, "optimized_effects": optimized["effects"], "cleanup_exact": True}
