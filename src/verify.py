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


def verify_native(plan_dir, model_dir, output):
    """Pretrained mechanics receipt; synthetic tests cannot produce this receipt."""
    import torch
    from pathlib import Path
    from .artifacts import atomic, canonical, digest, read_json, read_rows, require, verify_bundle
    from .interventions import RoutingCapture
    from .model import implementation_identity, load_model, runtime_identity
    from .run_experiment import forward
    from .scope import is_path_b, scope_identity, validate_plan
    manifest = verify_bundle(plan_dir, "experiment-plan")
    if manifest["metadata"]["stage"] != "technical":
        raise ValueError("mechanics requires disjoint technical inputs")
    settings = read_json(Path(plan_dir) / "research.json")
    inputs_rows = list(read_rows(Path(plan_dir) / "inputs.jsonl"))
    validate_plan(settings, manifest["metadata"], inputs_rows)
    if is_path_b(settings):
        from .optimized_validation import _validate_full_technical_pair
        _validate_full_technical_pair([(Path(plan_dir), manifest, settings, inputs_rows)],
                                      manifest["metadata"]["source_review_sha256"])
    require(settings, "mechanics.acceptance_tolerances")
    tolerances = settings["mechanics"]["acceptance_tolerances"]
    model_catalog = read_json(Path(plan_dir) / "model-catalog.json")
    model = load_model(model_dir, model_catalog, settings["sampling"]["master_seed"])
    versions = {name: p._version for name, p in model.named_parameters()}
    checks = []
    for inputs in inputs_rows:
        q = inputs["local_predictor_position"]
        for layer in inputs["layers"]:
            with RoutingCapture(model, layer, q, capture_window=True,
                                capture_reconstruction=True) as baseline:
                nll, logits = forward(model, inputs)
                native = baseline.routing()
                torch.testing.assert_close(baseline.reconstruct(), baseline.mixture, **tolerances)
            for _ in range(2):
                with RoutingCapture(model, layer, q) as repeat:
                    repeat_nll, repeat_logits = forward(model, inputs)
                    if repeat.routing() != native or repeat_nll != nll or not torch.equal(repeat_logits, logits):
                        raise ValueError("native three-repeat determinism failed")
            # Replace the local mixture by independently reconstructed native
            # contributions, then check predictor logits and manual target NLL.
            with RoutingCapture(model, layer, q, capture_reconstruction=True) as noop:
                def replace(module, args, output):
                    changed = output.clone()
                    changed[q] = noop.reconstruct()
                    return changed
                handle = noop.experts.register_forward_hook(replace)
                try:
                    noop_nll, noop_logits = forward(model, inputs)
                finally:
                    handle.remove()
                torch.testing.assert_close(noop_logits, logits, **tolerances)
                assert_loss_close(nll, noop_nll, atol=tolerances["atol"])
            target = torch.tensor([inputs["target_id"]], device=logits.device)
            manual = float(torch.nn.functional.cross_entropy(logits[None], target))
            assert_loss_close(nll, manual, atol=tolerances["atol"])
            for expert in native["expert_ids"]:
                with RoutingCapture(model, layer, q, expert, capture_window=True,
                                    capture_reconstruction=True) as ablation:
                    forward(model, inputs)
                    if ablation.routing() != native or not ablation.mask_is_local or not ablation.survivors_unchanged:
                        raise ValueError("local survivor/routing identity failed")
                    if not torch.equal(ablation.window_mixture[:q], baseline.window_mixture[:q]):
                        raise ValueError("untargeted local token rows changed")
                    torch.testing.assert_close(ablation.reconstruct(expert), ablation.mixture, **tolerances)
            try:
                with RoutingCapture(model, layer, q):
                    raise RuntimeError("injected cleanup check")
            except RuntimeError:
                pass
            clean_nll, clean_logits = forward(model, inputs)
            if clean_nll != nll or not torch.equal(clean_logits, logits):
                raise ValueError("cleanup after exceptions failed")
            checks.append({"input_id": inputs["input_id"], "layer": layer, "status": "pass",
                           "baseline_nll": nll, "expert_ids": native["expert_ids"]})
    if versions != {name: p._version for name, p in model.named_parameters()}:
        raise ValueError("model parameters changed")
    if not checks:
        raise ValueError("empty mechanics plan")
    receipt = {"status": "pass", "kind": "native-mechanics", "executor": {
        "model_catalog_sha256": digest(model_catalog), "runtime": runtime_identity(),
        "implementation_sha256": implementation_identity()},
        "research_scope": scope_identity(settings),
        "source_review_sha256": manifest["metadata"]["source_review_sha256"],
        "context_lengths": [manifest["metadata"]["window"]],
        "technical_plan_sha256": __import__("src.artifacts", fromlist=["file_digest"]).file_digest(Path(plan_dir) / "manifest.json"),
        "checks": checks, "tolerances": tolerances}
    atomic(output, canonical(receipt) + b"\n")
    return receipt
