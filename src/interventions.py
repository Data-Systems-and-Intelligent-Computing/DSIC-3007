"""Capture native packed eager OLMoE routes; mask exactly one contribution."""

from dataclasses import dataclass
import struct


@dataclass
class AblationTarget:
    layer: int
    token_position: int
    expert_id: int


class RoutingCapture:
    def __init__(self, model, layer, token_position, expert_id=None, capture_window=False,
                 capture_reconstruction=False, capture_contributions=False, reuse_from=None):
        import torch
        from transformers.models.olmoe.modeling_olmoe import OlmoeExperts
        if model.training or model.config.use_cache:
            raise ValueError("eval mode and no KV cache required")
        if model.config._experts_implementation != "eager":
            raise ValueError("inspected eager expert execution required")
        self.experts = model.model.layers[layer].mlp.experts
        if not isinstance(self.experts, OlmoeExperts) or self.experts._forward_pre_hooks or self.experts._forward_hooks:
            raise ValueError("unsupported or already instrumented experts")
        if any(p.dtype != torch.float32 for p in self.experts.parameters()):
            raise ValueError("FP32 experts required")
        self.q, self.expert_id, self.capture_window = token_position, expert_id, capture_window
        self.capture_reconstruction = capture_reconstruction
        self.capture_contributions = capture_contributions
        self.reuse_from = reuse_from
        self.contributions = None
        self.calls = 0
        self.handles = [self.experts.register_forward_pre_hook(self._before),
                        self.experts.register_forward_hook(self._after)]
        self._original_forward = None
        self._forward_was_instance_attribute = False
        if capture_contributions:
            self._original_forward = self.experts.forward
            self._forward_was_instance_attribute = "forward" in self.experts.__dict__
            self.experts.forward = self._native_forward_with_capture
        if reuse_from is not None:
            if not reuse_from.capture_contributions or reuse_from.calls != 1:
                self.remove()
                raise ValueError("complete native contribution capture required for replay")
            self._original_forward = self.experts.forward
            self._forward_was_instance_attribute = "forward" in self.experts.__dict__
            self.experts.forward = lambda hidden, ids, weights: reuse_from.recompose(
                hidden, ids, weights, self.expert_id)

    def _before(self, module, args):
        import torch
        self.calls += 1
        if self.calls != 1 or len(args) != 3:
            raise ValueError("unexpected expert invocation")
        hidden, ids, weights = args
        if (hidden.ndim != 2 or ids.shape != weights.shape or ids.shape[1] != 8 or
                self.q >= len(hidden) or weights.dtype != torch.float32 or
                not torch.isfinite(weights).all() or not torch.isfinite(hidden).all()):
            raise ValueError("invalid native expert input")
        self.ids = ids[self.q].detach().clone()
        self.weights = weights[self.q].detach().clone()
        # Mechanics reconstruction must preserve the native expert batch shapes.
        # Single-token GEMV can round differently from the native batched GEMM.
        # Ordinary collection does not retain these additional GPU tensors.
        if self.capture_reconstruction:
            self.reconstruction_hidden = hidden.detach()
            self.reconstruction_ids = ids.detach()
            self.reconstruction_weights = weights.detach()
        if self.capture_contributions:
            self.reconstruction_hidden = hidden.detach()
            self.reconstruction_ids = ids.detach()
            self.reconstruction_weights = weights.detach()
        self.all_ids = ids.detach().cpu().clone() if self.capture_window else None
        self.all_weights = weights.detach().cpu().clone() if self.capture_window else None
        if self.expert_id is None:
            return None
        active = self.ids.tolist()
        if self.expert_id not in active:
            raise ValueError("only naturally active experts can be ablated")
        changed = weights.clone()
        changed[self.q, active.index(self.expert_id)] = 0
        # The router outputs and membership stay untouched. Only one expert's
        # multiplier is zero; all other token/expert contributions are preserved.
        self.mask_is_local = bool(torch.equal(changed[:self.q], weights[:self.q]) and
                                  torch.equal(changed[self.q + 1:], weights[self.q + 1:]))
        other = [i for i, e in enumerate(active) if e != self.expert_id]
        self.survivors_unchanged = bool(torch.equal(changed[self.q, other], weights[self.q, other]))
        return hidden, ids, changed

    def _native_forward_with_capture(self, hidden, ids, weights):
        """Run one inspected eager loop and retain predictor-row unweighted outputs."""
        import torch
        total = torch.zeros_like(hidden)
        expert_mask = torch.nn.functional.one_hot(ids, num_classes=self.experts.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        contributions = []
        for expert_idx in expert_hit:
            e = expert_idx[0]
            if e == self.experts.num_experts:
                continue
            slots, rows = torch.where(expert_mask[e])
            gate, up = torch.nn.functional.linear(hidden[rows], self.experts.gate_up_proj[e]).chunk(2, dim=-1)
            output = torch.nn.functional.linear(self.experts.act_fn(gate) * up, self.experts.down_proj[e])
            at_predictor = torch.where(rows == self.q)[0]
            if len(at_predictor):
                slot = slots[at_predictor[0]]
                contributions.append((int(e), slot.detach().clone(), output[at_predictor[0]].detach().clone()))
            weighted = output * weights[rows, slots, None]
            total.index_add_(0, rows, weighted.to(total.dtype))
        self.contributions = contributions
        return total

    def recompose_for_target(self, target_expert):
        """Build a full-shape branch from cached unweighted outputs and native weights."""
        import torch
        if self.calls != 1 or self.contributions is None or not hasattr(self, "full_mixture"):
            raise ValueError("complete native contribution capture required")
        matches = torch.where(self.ids == target_expert)[0]
        if len(matches) != 1:
            raise ValueError("target expert is not uniquely active")
        slot_to_zero = matches[0]
        result = self.full_mixture.clone()
        row = torch.zeros_like(result[self.q:self.q + 1])
        for expert, slot, output in self.contributions:
            coefficient = self.weights[slot]
            if slot == slot_to_zero:
                coefficient = torch.zeros_like(coefficient)
            row.index_add_(0, torch.zeros(1, dtype=torch.long, device=row.device),
                           (output * coefficient).unsqueeze(0))
        result[self.q] = row[0]
        return result

    def recompose(self, hidden, ids, weights, target_expert):
        """Return the native full-shape mixture with one local coefficient changed."""
        import torch
        if self.calls != 1 or self.contributions is None:
            raise ValueError("native contribution capture required")
        if (not torch.equal(hidden, self.reconstruction_hidden) or
                not torch.equal(ids, self.reconstruction_ids)):
            raise ValueError("native contribution inputs or routes changed")
        expected = self.reconstruction_weights.clone()
        matches = torch.where(self.ids == target_expert)[0]
        if len(matches) != 1:
            raise ValueError("target expert is not uniquely active")
        expected[self.q, matches[0]] = 0
        if not torch.equal(weights, expected):
            raise ValueError("contribution replay changed unexpected coefficients")
        result = self.full_mixture.clone()
        row = torch.zeros_like(result[self.q:self.q + 1])
        for expert, slot, output in self.contributions:
            coefficient = weights[self.q, slot]
            row.index_add_(0, torch.zeros(1, dtype=torch.long, device=row.device), (output * coefficient).unsqueeze(0))
        result[self.q] = row[0]
        return result

    def _after(self, module, args, output):
        import torch
        if not torch.isfinite(output).all():
            raise ValueError("nonfinite expert output")
        self.mixture = output[self.q].detach().clone()
        if self.capture_contributions:
            self.full_mixture = output.detach().clone()
        if self.capture_window:
            self.window_mixture = output.detach().clone()

    def routing(self):
        if self.calls != 1:
            raise ValueError("routing capture incomplete")
        values = self.weights.tolist()
        return {"expert_ids": self.ids.tolist(), "weights": values,
                "weights_fp32_le_hex": struct.pack("<8f", *values).hex(), "coefficient_dtype": "float32"}

    def window_routing(self):
        if self.all_weights is None:
            raise ValueError("whole-window routing was not requested")
        return {"expert_ids": self.all_ids.tolist(), "shape": list(self.all_weights.shape),
                "weights_fp32_le_hex": self.all_weights.contiguous().numpy().tobytes().hex(),
                "coefficient_dtype": "float32", "byte_order": "little"}

    def reconstruct(self, exclude=None):
        import torch
        if not self.capture_reconstruction or self.calls != 1:
            raise ValueError("native batch reconstruction capture required")
        hidden = self.reconstruction_hidden
        ids, weights = self.reconstruction_ids, self.reconstruction_weights
        total = torch.zeros_like(hidden)
        # Replay Transformers 5.12.1 eager expert arithmetic independently,
        # including row order, matrix shapes, weighting and index_add_ order.
        expert_mask = torch.nn.functional.one_hot(ids, num_classes=self.experts.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            e = expert_idx[0]
            if e == self.experts.num_experts or (exclude is not None and e == exclude):
                continue
            slots, rows = torch.where(expert_mask[e])
            gate, up = torch.nn.functional.linear(hidden[rows], self.experts.gate_up_proj[e]).chunk(2, dim=-1)
            output = torch.nn.functional.linear(self.experts.act_fn(gate) * up, self.experts.down_proj[e])
            output = output * weights[rows, slots, None]
            total.index_add_(0, rows, output.to(total.dtype))
        return total[self.q]

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        if self._original_forward is not None:
            if self._forward_was_instance_attribute:
                self.experts.forward = self._original_forward
            else:
                del self.experts.forward
            self._original_forward = None
        if self._original_forward is not None:
            self.experts.forward = self._original_forward
            self._original_forward = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove()


def install_token_level_zero_ablation(model, target: AblationTarget):
    return RoutingCapture(model, target.layer, target.token_position, target.expert_id)


def remove_hook_safely(handle):
    if handle is not None:
        handle.remove()
