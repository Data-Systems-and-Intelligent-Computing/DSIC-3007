"""Verification suite for intervention fidelity."""

import math


def assert_loss_close(reference: float, observed: float, atol: float = 1e-6):
    if not math.isclose(reference, observed, abs_tol=atol, rel_tol=0.0):
        raise AssertionError(f"Loss mismatch: reference={reference}, observed={observed}")


def verify_noop_hook(reference_loss: float, noop_loss: float, atol: float = 1e-6):
    assert_loss_close(reference_loss, noop_loss, atol=atol)


def verify_state_cleanup(reference_loss: float, post_cleanup_loss: float, atol: float = 1e-6):
    assert_loss_close(reference_loss, post_cleanup_loss, atol=atol)


def verify_position_specificity(changed_positions):
    if len(changed_positions) != 1:
        raise AssertionError(f"Ablation must affect exactly one token position, got {changed_positions}")
