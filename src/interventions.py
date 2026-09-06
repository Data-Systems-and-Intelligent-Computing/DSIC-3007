"""Token-level single-expert interventions.

Important:
- Zero only the selected expert contribution for one token position at one layer.
- Do not reroute.
- Do not replace the expert.
- Do not renormalize routing weights.
- Do not change model parameters.
"""

from dataclasses import dataclass


@dataclass
class AblationTarget:
    layer: int
    token_position: int
    expert_id: int


def install_token_level_zero_ablation(model, target: AblationTarget):
    """Return a removable hook handle.

    This scaffold intentionally leaves architecture-specific tensor access explicit.
    Verify against the frozen OLMoE revision before implementing.
    """
    raise NotImplementedError("Implement after inspecting the exact OLMoE module graph.")


def remove_hook_safely(handle):
    if handle is not None:
        handle.remove()
