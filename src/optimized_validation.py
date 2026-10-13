"""Independent comparison of suffix replay with native full-forward ablations.

CPU fixtures can exercise the comparison machinery, but only receipts from the
pinned native GPU runtime establish pretrained model fidelity.
"""

from __future__ import annotations

import json
import math
import struct
import sys
import time
from pathlib import Path

from .artifacts import (atomic, canonical, digest, file_digest, read_json,
                        read_rows, verify_bundle)
from .scope import (is_path_b, scope_identity, require_scope, stage_layers,
                    validate_plan)


ATOL = 1e-6
RTOL = 1e-5
NLL_ATOL = 1e-6
EXPECTED_LAYERS = [0, 7, 15]
EXPECTED_CONTEXTS = {2048, 4096}


class _TimeLimit(Exception):
    pass


def _progress(message):
    """Write a short, immediately visible validation status line."""
    print(f"[optimized-validation] {message}", file=sys.stdout, flush=True)


def _record(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        __import__("os").fsync(stream.fileno())


def _memory_metrics(torch):
    import resource
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes. The native execution target is Linux.
    rss_bytes = int(peak_rss * 1024) if peak_rss < 10**10 else int(peak_rss)
    gpu = {}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gpu = {"gpu_allocated_bytes": int(torch.cuda.memory_allocated()),
               "gpu_reserved_bytes": int(torch.cuda.memory_reserved()),
               "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
               "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved())}
    return {"host_peak_rss_bytes": rss_bytes, **gpu}


def _close_tensor(torch, name, reference, observed, *, atol=ATOL, rtol=RTOL):
    try:
        torch.testing.assert_close(reference, observed, atol=atol, rtol=rtol)
    except AssertionError as exc:
        import torch
        difference = (reference - observed).abs()
        raise AssertionError(
            f"{name}: max_abs={float(difference.max()) if difference.numel() else 0.0:.9g}; {exc}") from exc


def _nll_error(reference, observed):
    error = abs(float(reference) - float(observed))
    if not math.isfinite(error) or error > NLL_ATOL:
        raise AssertionError(f"NLL mismatch: reference={reference:.12g}, observed={observed:.12g}, abs={error:.9g}")
    return error


def _routes(ids, weights):
    values = weights.detach().to("cpu").contiguous()
    return {"expert_ids": ids.detach().cpu().tolist(),
            "weights_fp32_le_hex": struct.pack("<" + "f" * values.numel(), *values.flatten().tolist()).hex()}


def _validate_full_technical_pair(plan_data, source_review_sha256):
    """Reject screened/truncated plans and verify matched, frozen technical windows."""
    from .preparation import eligible_positions, window_for

    by_context = {manifest["metadata"]["window"]: (plan, manifest, settings, inputs)
                  for plan, manifest, settings, inputs in plan_data}
    settings = plan_data[0][2]
    path_b = is_path_b(settings)
    primary = by_context[4096]
    if not path_b:
        matched = by_context[2048]
    primary_plan, primary_manifest, settings, primary_inputs = primary
    if not path_b:
        matched_plan, matched_manifest, _, matched_inputs = matched
        if matched_manifest["metadata"].get("parent_plan_sha256") != file_digest(primary_plan / "manifest.json"):
            raise ValueError("full 2048 technical plan must link to the exact 4096 technical plan")
    expected_layers = set(stage_layers(settings, "technical"))
    for plan, manifest, _, inputs in plan_data:
        metadata = manifest["metadata"]
        if metadata.get("benchmark_screen") or metadata.get("units") != {"c4": 12, "scientific": 12, "code": 12}:
            raise ValueError("full technical validation rejects benchmark screens and requires 12 groups/domain")
        validate_plan(settings, metadata, inputs)
        if any(set(row["layers"]) != expected_layers for row in inputs):
            raise ValueError("full technical plans must retain every scoped layer per input")
        review = read_json(plan / "source-review.json")
        if (review.get("status") != "approved" or
                set(review.get("checks", {}).values()) != {"pass"} or
                review.get("prepared_manifest_sha256") != metadata.get("prepared_manifest_sha256")):
            raise ValueError("full technical plan source review is not approved and complete")
        docs = list(read_rows(plan / "documents.jsonl"))
        groups = {domain: {d["source_group"] for d in docs if d["domain"] == domain}
                  for domain in ("c4", "scientific", "code")}
        if any(len(value) != 12 for value in groups.values()):
            raise ValueError("full technical plans must contain 12 actual source groups in every domain")
        if (file_digest(plan / "source-review.json") != source_review_sha256 or
                review.get("prepared_manifest_sha256") != metadata.get("prepared_manifest_sha256")):
            raise ValueError("full technical source review identity differs")

        review_docs = review.get("documents", {})
        selected_groups = {(domain, group) for domain, values in groups.items() for group in values}
        expected_documents = {doc_id for doc_id, doc in review_docs.items()
                              if doc.get("stage") == "technical" and
                              (doc.get("domain"), doc.get("source_group")) in selected_groups}
        document_ids = {doc["document_id"] for doc in docs}
        if expected_documents != document_ids:
            raise ValueError("technical plan omits or adds documents from its approved source groups")

        rows_by_document = {}
        for row in inputs:
            rows_by_document.setdefault(row["document_id"], []).append(row)
        excluded = settings["context"].get("excluded_predictor_target_ids", [])
        for doc in docs:
            doc_rows = rows_by_document.get(doc["document_id"], [])
            eligible = eligible_positions(doc["token_ids"], 4096, excluded)
            if not eligible:
                raise ValueError("full technical document lacks a full-context predictor/target")
            expected_positions = list(dict.fromkeys(eligible[i] for i in (
                0, len(eligible) - 1, (len(eligible) - 1) // 3, 2 * (len(eligible) - 1) // 3)))
            observed = {row["predictor_position"]: row for row in doc_rows}
            if set(observed) != set(expected_positions):
                raise ValueError("technical document does not have the frozen four-position sample")
            for q, row in observed.items():
                expected = window_for(doc["token_ids"], q, manifest["metadata"]["window"])
                if any(row.get(name) != value for name, value in expected.items()):
                    raise ValueError("technical input bytes or predictor/target alignment differs from its document")

    if path_b:
        return
    primary_docs = {row["document_id"]: row for row in read_rows(primary_plan / "documents.jsonl")}
    matched_docs = {row["document_id"]: row for row in read_rows(matched_plan / "documents.jsonl")}
    if primary_docs.keys() != matched_docs.keys():
        raise ValueError("matched technical contexts have different source documents")
    primary_by_slot = {(row["document_id"], row["sample_order"]): row for row in primary_inputs}
    matched_by_slot = {(row["document_id"], row["sample_order"]): row for row in matched_inputs}
    if primary_by_slot.keys() != matched_by_slot.keys():
        raise ValueError("matched technical contexts have different selected predictor positions")
    for slot, row in primary_by_slot.items():
        short = matched_by_slot[slot]
        if (short["input_ids"] != row["input_ids"][-2048:] or
                any(short[name] != row[name] for name in
                    ("document_id", "source_group", "predictor_position", "target_position", "target_id"))):
            raise ValueError("matched technical screen changed input bytes or predictor/target identity")


def _input_case(model, inputs, elapsed, max_seconds, progress=None):
    """Compare each optimized replay with a separately executed native ablation."""
    import torch
    from contextlib import ExitStack
    from .interventions import RoutingCapture
    from .run_experiment import (_capture_layer_call, _capture_layer_residual,
                                _replay_from_layer, forward)

    def check_clock():
        if elapsed() >= max_seconds:
            raise _TimeLimit("elapsed cap reached during input validation")

    q = inputs["local_predictor_position"]
    layers = list(inputs["layers"])
    invocations, residuals = {}, {}
    versions = {name: parameter._version for name, parameter in model.named_parameters()}
    device = next(model.parameters()).device
    def synchronized_clock():
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.monotonic()

    case_started = synchronized_clock()
    timings = {"native_ablation_seconds_by_layer": {},
               "optimized_suffix_seconds_by_layer": {}}

    # First capture an untouched native baseline. Contribution capture replaces
    # the expert forward with a shape-preserving arithmetic copy, so its output
    # must be compared back to this independent baseline before it can anchor replay.
    phase_started = synchronized_clock()
    with ExitStack() as stack:
        native_captures = {layer: stack.enter_context(RoutingCapture(
            model, layer, q, capture_window=True)) for layer in layers}
        native_baseline_nll, native_baseline_logits = forward(model, inputs)
        native_routes = {layer: capture.routing() for layer, capture in native_captures.items()}
        native_mixtures = {layer: capture.mixture for layer, capture in native_captures.items()}
        native_windows = {layer: capture.window_mixture for layer, capture in native_captures.items()}
    timings["native_baseline_seconds"] = synchronized_clock() - phase_started
    check_clock()

    # A second baseline captures native-shape expert intermediates for replay.
    phase_started = synchronized_clock()
    with ExitStack() as stack:
        for layer in range(len(model.model.layers)):
            stack.callback(_capture_layer_call(model, layer, invocations, set()).remove)
        for layer in layers:
            stack.callback(_capture_layer_residual(model, layer, residuals).remove)
        captures = {layer: stack.enter_context(RoutingCapture(
            model, layer, q, capture_window=True, capture_contributions=True)) for layer in layers}
        baseline_nll, baseline_logits = forward(model, inputs)
    timings["contribution_baseline_seconds"] = synchronized_clock() - phase_started
    check_clock()
    _nll_error(native_baseline_nll, baseline_nll)
    _close_tensor(torch, "baseline predictor logits", native_baseline_logits.reshape(-1), baseline_logits.reshape(-1))
    for layer in layers:
        if captures[layer].routing() != native_routes[layer]:
            raise AssertionError(f"layer {layer}: contribution capture changed routing/coefficient bytes")
        _close_tensor(torch, "baseline mixture", native_mixtures[layer], captures[layer].full_mixture[q])
        if not torch.equal(torch.cat((native_windows[layer][:q], native_windows[layer][q + 1:]), dim=0),
                           torch.cat((captures[layer].full_mixture[:q], captures[layer].full_mixture[q + 1:]), dim=0)):
            raise AssertionError(f"layer {layer}: contribution capture changed untargeted rows")
    result = {"input_id": inputs["input_id"], "context_length": len(inputs["input_ids"]),
              "cases": [], "baseline_nll": native_baseline_nll}
    check_clock()

    for layer in layers:
        check_clock()
        capture = captures[layer]
        route = native_routes[layer]
        baseline_mixture = native_windows[layer]

        for expert in route["expert_ids"]:
            check_clock()
            accelerated_mixture = capture.recompose_for_target(expert)
            untouched = torch.cat((accelerated_mixture[:q], accelerated_mixture[q + 1:]), dim=0)
            expected_untouched = torch.cat((baseline_mixture[:q], baseline_mixture[q + 1:]), dim=0)
            if not torch.equal(untouched, expected_untouched):
                raise AssertionError(f"layer {layer}, expert {expert}: accelerated replay changed untargeted rows")

            downstream = {}
            handles = []
            for downstream_layer in range(layer + 1, len(model.model.layers)):
                experts_module = model.model.layers[downstream_layer].mlp.experts

                def capture_routes(module, args, *, index=downstream_layer):
                    if index in downstream:
                        raise ValueError(f"downstream layer {index} invoked more than once")
                    downstream[index] = _routes(args[1], args[2])

                handles.append(experts_module.register_forward_pre_hook(capture_routes))
            try:
                phase_started = synchronized_clock()
                accelerated_nll, accelerated_logits = _replay_from_layer(
                    model, inputs, invocations, residuals, captures, layer, expert)
                optimized_seconds = synchronized_clock() - phase_started
            finally:
                for handle in handles:
                    handle.remove()
            check_clock()

            # Independent full-model execution with the native local zero mask.
            native_downstream = {}
            native_handles = []
            for downstream_layer in range(layer + 1, len(model.model.layers)):
                experts_module = model.model.layers[downstream_layer].mlp.experts

                def capture_native_routes(module, args, *, index=downstream_layer):
                    if index in native_downstream:
                        raise ValueError(f"native downstream layer {index} invoked more than once")
                    native_downstream[index] = _routes(args[1], args[2])

                native_handles.append(experts_module.register_forward_pre_hook(capture_native_routes))
            try:
                phase_started = synchronized_clock()
                with RoutingCapture(model, layer, q, expert, capture_window=True) as native_capture:
                    native_nll, native_logits = forward(model, inputs)
                    native_route = native_capture.routing()
                    native_mixture = native_capture.mixture
                    survivor_exact = native_capture.survivors_unchanged
                    local_mask_exact = native_capture.mask_is_local
                    native_rows = native_capture.window_mixture
                native_seconds = synchronized_clock() - phase_started
            finally:
                for handle in native_handles:
                    handle.remove()
            check_clock()

            if native_route != route or not survivor_exact or not local_mask_exact:
                raise AssertionError(f"layer {layer}, expert {expert}: native routing/survivor contract failed")
            _close_tensor(torch, "mixture", native_mixture, accelerated_mixture[q])
            _close_tensor(torch, "predictor logits", native_logits.reshape(-1), accelerated_logits.reshape(-1))
            error = _nll_error(native_nll, accelerated_nll)
            if not torch.equal(torch.cat((native_rows[:q], native_rows[q + 1:]), dim=0),
                               torch.cat((baseline_mixture[:q], baseline_mixture[q + 1:]), dim=0)):
                raise AssertionError(f"layer {layer}, expert {expert}: native ablation changed untargeted rows")
            if downstream != native_downstream:
                raise AssertionError(f"layer {layer}, expert {expert}: downstream routes differ from native ablation")
            result["cases"].append({"layer": layer, "expert_id": expert,
                                    "native_full_forward_seconds": native_seconds,
                                    "optimized_suffix_seconds": optimized_seconds,
                                    "native_nll": native_nll, "optimized_nll": accelerated_nll,
                                    "nll_abs_error": error,
                                    "max_logit_abs_error": float((native_logits - accelerated_logits).abs().max()),
                                    "max_mixture_abs_error": float((native_mixture - accelerated_mixture[q]).abs().max()),
                                    "downstream_layers_checked": len(downstream)})
            for field, seconds in (("native_ablation_seconds_by_layer", native_seconds),
                                   ("optimized_suffix_seconds_by_layer", optimized_seconds)):
                timings[field][str(layer)] = timings[field].get(str(layer), 0.0) + seconds
            check_clock()
            if progress is not None:
                progress({"input_id": inputs["input_id"], "context_length": len(inputs["input_ids"]),
                          "completed_comparisons": len(result["cases"]),
                          "latest_case": dict(result["cases"][-1]), "status": "in_progress"})

    phase_started = synchronized_clock()
    clean_nll, clean_logits = forward(model, inputs)
    timings["cleanup_seconds"] = synchronized_clock() - phase_started
    if clean_nll != native_baseline_nll or not torch.equal(clean_logits, native_baseline_logits):
        raise AssertionError("model cleanup did not recover the native baseline")
    if versions != {name: parameter._version for name, parameter in model.named_parameters()}:
        raise AssertionError("model parameters changed during optimized validation")
    for layer in range(len(model.model.layers)):
        experts_module = model.model.layers[layer].mlp.experts
        if experts_module._forward_pre_hooks or experts_module._forward_hooks or "forward" in experts_module.__dict__:
            raise AssertionError(f"layer {layer}: optimized hooks/forward override were not cleaned up")
    result["cleanup_pass"] = True
    result["parameters_unchanged"] = True
    timings["synchronized_comparison_total_seconds"] = synchronized_clock() - case_started
    result["comparison_timings"] = timings
    result["timing_scope"] = "diagnostic captures/comparisons/progress writes; not a production collector profile"
    return result


def validate_optimized(plan_dirs, model_dir, output, max_seconds=1800,
                       validation_scope="screen", model_session=None,
                       native_mechanics_paths=None):
    """Compare replay against native full forwards on the exact supplied technical plans.

    `screen` is diagnostic evidence only. `full_technical` requires all accepted
    technical plans and a separate passing native-mechanics receipt per plan.
    Loading time counts against `max_seconds`; incomplete attempts are persisted.
    """
    from .model import (VerifiedModelSession, implementation_identity,
                        load_model, runtime_identity)

    if validation_scope not in ("screen", "preflight", "full_technical"):
        raise ValueError("validation_scope must be screen, preflight or full_technical")
    if type(max_seconds) not in (int, float) or max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    plans = [Path(p) for p in plan_dirs]
    if not plans:
        raise ValueError("at least one technical plan is required")
    if validation_scope == "full_technical" and (not native_mechanics_paths or
            len(native_mechanics_paths) != len(plans)):
        raise ValueError("full technical validation requires a native mechanics receipt for every plan")

    output = Path(output)
    cases_path = output.with_suffix(output.suffix + ".cases.jsonl")
    receipt_tmp = output.with_suffix(output.suffix + ".partial.json")
    if output.exists() or cases_path.exists() or receipt_tmp.exists():
        raise FileExistsError("validation output already exists; preserve prior attempt and choose a new path")
    started = time.monotonic()
    def elapsed():
        return time.monotonic() - started

    from .artifacts import read_rows
    plan_data = []
    catalog = None
    source_reviews = set()
    contexts, layer_coverage, layers_by_context = set(), set(), {}
    _progress(f"plan preflight start plans={len(plans)} scope={validation_scope} cap={max_seconds:.1f}s")
    try:
        for plan_index, plan in enumerate(plans, start=1):
            _progress(f"plan preflight {plan_index}/{len(plans)} start path={plan}")
            manifest = verify_bundle(plan, "experiment-plan")
            metadata = manifest["metadata"]
            if metadata.get("stage") != "technical":
                raise ValueError("optimized validation accepts technical plans only")
            review_path = plan / "source-review.json"
            if not review_path.exists() or file_digest(review_path) != metadata.get("source_review_sha256"):
                raise ValueError("technical plan source review does not match its manifest identity")
            settings = read_json(plan / "research.json")
            if plan_data and settings != plan_data[0][2]:
                raise ValueError("technical plans have different research scope/settings")
            expected_layers = set(stage_layers(settings, "technical"))
            if settings.get("mechanics", {}).get("acceptance_tolerances") != {"atol": ATOL, "rtol": RTOL} or settings.get("mechanics", {}).get("nll_atol") != NLL_ATOL:
                raise ValueError("research numerical tolerances differ from the frozen acceptance contract")
            plan_catalog = read_json(plan / "model-catalog.json")
            if catalog is not None and digest(plan_catalog) != digest(catalog):
                raise ValueError("technical plans have different pinned model catalogs")
            catalog = plan_catalog
            source_reviews.add(metadata.get("source_review_sha256"))
            context = metadata.get("window")
            contexts.add(context)
            inputs = list(read_rows(plan / "inputs.jsonl"))
            validate_plan(settings, metadata, inputs)
            for row in inputs:
                if (len(row.get("input_ids", [])) != context or
                        row.get("local_predictor_position") != context - 1 or
                        set(row.get("layers", [])) != expected_layers):
                    raise ValueError("technical input context, predictor alignment, or layer coverage is invalid")
                layer_coverage.update(row["layers"])
                layers_by_context.setdefault(context, set()).update(row["layers"])
            plan_data.append((plan, manifest, settings, inputs))
            _progress(f"plan preflight {plan_index}/{len(plans)} pass context={context} inputs={len(inputs)}")
    except BaseException as exc:
        _progress(f"plan preflight fail elapsed={elapsed():.1f}s error={type(exc).__name__}: {exc}")
        raise

    if len(source_reviews) != 1 or None in source_reviews:
        raise ValueError("all technical plans must share one reviewed source identity")
    path_b = is_path_b(plan_data[0][2])
    expected_contexts = {4096} if path_b else EXPECTED_CONTEXTS
    if contexts != expected_contexts or len(plans) != len(expected_contexts) or layer_coverage != expected_layers:
        raise ValueError("validation plans must cover all scoped contexts and layers")
    if any(layers_by_context.get(context) != expected_layers for context in expected_contexts):
        raise ValueError("each context must independently cover the scoped layers")
    if validation_scope == "preflight":
        if not path_b or len(plan_data[0][3]) != 3:
            raise ValueError("Path B preflight requires exactly three sealed inputs")
        _, m, _, rows = plan_data[0]
        if (m["metadata"].get("preflight_diagnostic") is not True or
                m["metadata"].get("benchmark_screen") is not True or
                any(sum(r["domain"] == d for r in rows) != 1 for d in ("c4", "scientific", "code"))):
            raise ValueError("preflight requires an explicit diagnostic plan with one input/domain")
    if validation_scope == "screen":
        _progress("screen preflight start: selecting the first three approved source groups per domain")
        selected_groups = None
        screened = []
        for plan, manifest, plan_settings, inputs in plan_data:
            by_domain = {}
            for row in inputs:
                groups = by_domain.setdefault(row["domain"], [])
                if row["source_group"] not in groups:
                    groups.append(row["source_group"])
            chosen = {domain: groups[:3] for domain, groups in by_domain.items()}
            if set(chosen) != {"c4", "scientific", "code"} or any(len(groups) != 3 for groups in chosen.values()):
                raise ValueError("screen plan must contain at least three technical groups per domain")
            if selected_groups is not None and chosen != selected_groups:
                raise ValueError("both context screens must select the same three source groups per domain")
            selected_groups = chosen
            screened.append((plan, manifest, plan_settings,
                             [row for row in inputs
                              if row["source_group"] in chosen.get(row["domain"], [])]))
        plan_data = screened
        _progress(f"screen preflight pass inputs={sum(len(rows) for _, _, _, rows in plan_data)}")
    if validation_scope == "full_technical":
        _progress("full technical preflight start: verifying source groups, paired windows, and native mechanics receipts")
        for (plan, manifest, _, inputs), mechanics_path in zip(plan_data, native_mechanics_paths, strict=True):
            if (manifest["metadata"].get("benchmark_screen") or
                    manifest["metadata"].get("units") != {"c4": 12, "scientific": 12, "code": 12}):
                raise ValueError("full technical plans must retain 12 source groups per domain")
            mechanics = read_json(mechanics_path)
            require_scope(settings, mechanics)
            if (mechanics.get("status") != "pass" or mechanics.get("kind") != "native-mechanics" or
                    mechanics.get("source_review_sha256") != next(iter(source_reviews)) or
                    mechanics.get("executor", {}).get("model_catalog_sha256") != digest(catalog) or
                    mechanics.get("tolerances") != {"atol": ATOL, "rtol": RTOL}):
                raise ValueError("full technical validation requires matching native mechanics passes")
            if (mechanics.get("technical_plan_sha256") != file_digest(plan / "manifest.json") or
                    mechanics.get("context_lengths") != [manifest["metadata"]["window"]]):
                raise ValueError("native mechanics receipt does not cover this exact technical plan/context")
            expected_checks = {(row["input_id"], layer) for row in inputs for layer in row["layers"]}
            observed_checks = {(row.get("input_id"), row.get("layer")) for row in mechanics.get("checks", [])}
            if observed_checks != expected_checks or any(row.get("status") != "pass" for row in mechanics.get("checks", [])):
                raise ValueError("native mechanics receipt has incomplete technical coverage")
        _validate_full_technical_pair(plan_data, next(iter(source_reviews)))
        _progress("full technical preflight pass")

    expected_inputs = sum(len(rows) for _, _, _, rows in plan_data)
    expected_comparisons = sum(len(row["layers"])*8 for _, _, _, rows in plan_data for row in rows)
    _progress(f"plan preflight pass selected_inputs={expected_inputs} expected_expert_comparisons={expected_comparisons} elapsed={elapsed():.1f}s")

    settings = plan_data[0][2]
    session = model_session or VerifiedModelSession()
    # Count catalog and model load time. Session identity includes the complete
    # runtime and implementation hash and rejects post-load parameter mutation.
    _progress(f"model load start elapsed={elapsed():.1f}/{max_seconds:.1f}s")
    try:
        model = session.acquire(model_dir, catalog, settings["sampling"]["master_seed"])
        executor = {"model_catalog_sha256": digest(catalog), "runtime": runtime_identity(),
                    "implementation_sha256": implementation_identity()}
        _progress(f"model load complete elapsed={elapsed():.1f}/{max_seconds:.1f}s")
    except Exception as exc:
        atomic(output, canonical({"status": "fail", "kind": "optimized-validation",
                                 "validation_scope": validation_scope,
                                 "technical_plan_sha256": sorted(file_digest(p / "manifest.json") for p in plans),
                                 "elapsed_seconds": elapsed(), "max_seconds": max_seconds,
                                 "stop_reason": f"{type(exc).__name__}: {exc}",
                                 "full_coverage": False}) + b"\n")
        _progress(f"model load fail elapsed={elapsed():.1f}/{max_seconds:.1f}s error={type(exc).__name__}: {exc}")
        raise
    if validation_scope == "full_technical":
        for mechanics_path in native_mechanics_paths:
            if read_json(mechanics_path).get("executor") != executor:
                raise ValueError("native mechanics and optimized validation executor identities differ")
    if elapsed() > max_seconds:
        receipt = {"status": "incomplete", "kind": "optimized-validation", "executor": executor,
                   "execution_mode": "native_shape_suffix_cleanup_reference_v2", "validation_scope": validation_scope,
                   "source_review_sha256": next(iter(source_reviews)), "context_lengths": sorted(contexts),
                   "layers": sorted(layer_coverage),
                   "technical_plan_sha256": sorted(file_digest(p / "manifest.json") for p in plans),
                   "approval": "diagnostic_only", "elapsed_seconds": elapsed(),
                   "max_seconds": max_seconds, "stop_reason": "model load exceeded validation elapsed cap",
                   "expected_inputs": sum(len(rows) for _, _, _, rows in plan_data),
                   "completed_inputs": 0, "full_coverage": False}
        atomic(output, canonical(receipt) + b"\n")
        _progress(f"validation incomplete inputs=0/{expected_inputs} comparisons=0/{expected_comparisons} elapsed={receipt['elapsed_seconds']:.1f}/{max_seconds:.1f}s full_coverage=False stop_reason={receipt['stop_reason']} receipt={output}")
        return receipt

    completed_inputs, completed_comparisons = 0, 0
    status, stop_reason = "incomplete", None
    try:
        input_number = 0
        for plan, manifest, _, rows in plan_data:
            plan_hash = file_digest(plan / "manifest.json")
            for index, inputs in enumerate(rows):
                input_number += 1
                if elapsed() >= max_seconds:
                    raise _TimeLimit("elapsed cap reached before next input")
                case_started = time.monotonic()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                input_comparison_count = 0
                _progress(f"input {input_number}/{expected_inputs} start domain={inputs.get('domain')} context={len(inputs['input_ids'])} id={inputs.get('input_id')} elapsed={elapsed():.1f}/{max_seconds:.1f}s")
                try:
                    def progress(row):
                        nonlocal completed_comparisons, input_comparison_count
                        _record(cases_path, {"record_type": "input_progress",
                                             "plan_manifest_sha256": plan_hash,
                                             "plan_input_index": index, **row})
                        completed_comparisons += row["completed_comparisons"] - input_comparison_count
                        input_comparison_count = row["completed_comparisons"]
                        latest = row["latest_case"]
                        percent = 100.0 * completed_comparisons / expected_comparisons if expected_comparisons else 100.0
                        _progress(f"comparison {completed_comparisons}/{expected_comparisons} ({percent:.1f}%) input={input_number}/{expected_inputs} domain={inputs.get('domain')} context={len(inputs['input_ids'])} layer={latest['layer']} expert={latest['expert_id']} error_nll={latest['nll_abs_error']:.3g} error_logits={latest['max_logit_abs_error']:.3g} error_mixture={latest['max_mixture_abs_error']:.3g} elapsed={elapsed():.1f}/{max_seconds:.1f}s")

                    case = _input_case(model, inputs, elapsed, max_seconds, progress=progress)
                    case_counts = {layer:len({c["expert_id"] for c in case["cases"] if c["layer"] == layer})
                                   for layer in inputs["layers"]}
                    if (set(c["layer"] for c in case["cases"]) != set(inputs["layers"]) or
                            any(n != 8 for n in case_counts.values()) or
                            len(case["cases"]) != 8*len(inputs["layers"]) or
                            input_comparison_count != len(case["cases"]) or not case.get("cleanup_pass")):
                        raise ValueError("incomplete independent expert comparisons or cleanup")
                    case.update(plan_manifest_sha256=plan_hash, plan_input_index=index,
                                status="pass", elapsed_seconds=time.monotonic() - case_started,
                                **_memory_metrics(torch))
                    persist_started = time.monotonic()
                    _record(cases_path, case)
                    _record(cases_path, {"record_type": "persistence_timing",
                                         "input_id": inputs["input_id"],
                                         "persistence_write_seconds": time.monotonic() - persist_started})
                    completed_inputs += 1
                    _progress(f"input {input_number}/{expected_inputs} durable pass comparisons={completed_comparisons}/{expected_comparisons} elapsed={elapsed():.1f}/{max_seconds:.1f}s")
                except _TimeLimit as exc:
                    stop_reason = str(exc)
                    partial = {"input_id": inputs.get("input_id"), "plan_manifest_sha256": plan_hash,
                               "plan_input_index": index, "status": "incomplete",
                               "elapsed_seconds": time.monotonic() - case_started,
                               "reason": stop_reason, **_memory_metrics(torch)}
                    _record(cases_path, partial)
                    _progress(f"input {input_number}/{expected_inputs} incomplete comparisons={completed_comparisons}/{expected_comparisons} elapsed={elapsed():.1f}/{max_seconds:.1f}s reason={stop_reason}")
                    raise
                except BaseException as exc:
                    failed = {"input_id": inputs.get("input_id"), "plan_manifest_sha256": plan_hash,
                              "plan_input_index": index, "status": "fail",
                              "elapsed_seconds": time.monotonic() - case_started,
                              "exception": type(exc).__name__, "message": str(exc), **_memory_metrics(torch)}
                    _record(cases_path, failed)
                    _progress(f"input {input_number}/{expected_inputs} fail comparisons={completed_comparisons}/{expected_comparisons} elapsed={elapsed():.1f}/{max_seconds:.1f}s error={type(exc).__name__}: {exc}")
                    raise
        status = "pass"
    except _TimeLimit:
        status = "incomplete"
    except BaseException as exc:
        status = "fail"
        stop_reason = f"{type(exc).__name__}: {exc}"

    full_coverage = (completed_inputs == expected_inputs and
                     completed_comparisons == expected_comparisons and status == "pass")
    native_checks = ({"baseline_routes_weights_mixture_logits": "pass",
                      "all_eight_native_ablation_logits_mixture_nll": "pass",
                      "survivors_and_untargeted_rows": "pass",
                      "downstream_routes_against_native_ablation": "pass",
                      "parameters_and_cleanup": "pass"} if full_coverage else {})
    receipt = {"status": status, "kind": "optimized-validation", "executor": executor,
               "research_scope": scope_identity(settings),
               "execution_mode": "native_shape_suffix_cleanup_reference_v2",
               "source_review_sha256": next(iter(source_reviews)),
               "context_lengths": sorted(contexts), "layers": sorted(layer_coverage),
               "technical_plan_sha256": sorted(file_digest(p / "manifest.json") for p in plans),
               "validation_scope": validation_scope,
               "approval": "collection_candidate" if validation_scope == "full_technical" and full_coverage else "diagnostic_only",
               "checks": native_checks,
               "tolerances": {"atol": ATOL, "rtol": RTOL, "nll_atol": NLL_ATOL},
               "expected_inputs": expected_inputs, "completed_inputs": completed_inputs,
               "completed_expert_comparisons": completed_comparisons,
               "case_report_sha256": file_digest(cases_path) if cases_path.exists() else None,
               "case_report_durability": "fsync_per_progress_and_case",
               "case_report_path": cases_path.name, "elapsed_seconds": elapsed(),
               "max_seconds": max_seconds, "stop_reason": stop_reason,
               "full_coverage": full_coverage}
    atomic(output, canonical(receipt) + b"\n")
    _progress(f"validation {status} inputs={completed_inputs}/{expected_inputs} comparisons={completed_comparisons}/{expected_comparisons} elapsed={receipt['elapsed_seconds']:.1f}/{max_seconds:.1f}s full_coverage={full_coverage} stop_reason={stop_reason or 'none'} receipt={output}")
    if status == "fail":
        raise ValueError(f"optimized validation failed: {stop_reason}; receipt: {output}")
    return receipt
