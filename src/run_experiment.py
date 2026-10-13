"""Native routing and complete eight-expert token-local effects, with raw NLLs."""

from pathlib import Path
import json
import math
import shutil
import time
from contextlib import ExitStack

from .artifacts import (Journal, atomic, canonical, digest, file_digest, read_json,
                        read_rows, require, seal, validate_measurement, verify_bundle)
from .interventions import RoutingCapture
from .model import implementation_identity, load_model, runtime_identity
from .reuse import SealedMeasurementReuse

OPTIMIZED_EXECUTION_MODE = "native_shape_suffix_cleanup_reference_v2"


def write_jsonl(path: str, record: dict):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_timing_row(path, record):
    """Durably append one physical-input timing row after its events are complete."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("ab") as stream:
        stream.write(canonical(record) + b"\n")
        stream.flush()
        __import__("os").fsync(stream.fileno())


def _shared_timing(events, name):
    values = [row[name] for row in events if name in row]
    if not values:
        return None
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"repeated shared timing differs: {name}")
    return values[0]


def causal_discrimination(delta_high: float, delta_low: float) -> float:
    return delta_high - delta_low


def forward(model, inputs):
    import torch
    device = next(model.parameters()).device
    ids = torch.tensor([inputs["input_ids"]], device=device, dtype=torch.long)
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        output = model(input_ids=ids, attention_mask=mask, use_cache=False,
                       logits_to_keep=1, return_dict=True)
        logits = output.logits[0, -1].double()
        nll = float(-torch.log_softmax(logits, dim=-1)[inputs["target_id"]])
    if not math.isfinite(nll):
        raise ValueError("nonfinite actual next-token NLL")
    return nll, logits.detach()


def event_id(inputs, layer):
    return digest([inputs["input_id"], layer])


def _capture_layer_call(model, layer, invocations, replay_layers):
    def capture(module, args, kwargs):
        kwargs = dict(kwargs)
        if args:
            hidden = args[0] if layer in replay_layers else None
            invocations[layer] = (args[1:], kwargs, hidden)
        else:
            hidden = kwargs.pop("hidden_states", None)
            invocations[layer] = ((), kwargs, hidden if layer in replay_layers else None)
    return model.model.layers[layer].register_forward_pre_hook(capture, with_kwargs=True)


def _capture_layer_residual(model, layer, residuals):
    def capture(module, args):
        if len(args) != 1:
            raise ValueError("unexpected post-attention normalization inputs")
        residuals[layer] = args[0]
    return model.model.layers[layer].post_attention_layernorm.register_forward_pre_hook(capture)


def _call_layer(layer_module, invocation, hidden_states):
    args, kwargs, _ = invocation
    if args:
        return layer_module(hidden_states, *args, **kwargs)
    kwargs = dict(kwargs)
    kwargs["hidden_states"] = hidden_states
    return layer_module(**kwargs)


def _replay_from_layer(model, inputs, invocations, residuals, captures, layer, expert):
    """Replay the changed MoE row and full-shape decoder suffix from a baseline prefix."""
    import torch
    if layer not in residuals:
        raise ValueError("missing immutable post-attention residual")
    residual = residuals[layer]
    mixture = captures[layer].recompose_for_target(expert)
    hidden_states = residual + mixture.reshape_as(residual)
    with torch.inference_mode():
        for index in range(layer + 1, len(model.model.layers)):
            hidden_states = _call_layer(model.model.layers[index], invocations[index], hidden_states)
        logits = model.lm_head(model.model.norm(hidden_states)[:, -1]).double()
        nll = float(-torch.log_softmax(logits[0], dim=-1)[inputs["target_id"]])
    if not math.isfinite(nll):
        raise ValueError("nonfinite actual next-token NLL")
    return nll, logits.detach()


def measure_input(model, inputs, calibration=False, routing_only=False, optimized=False):
    """One baseline shared across layers; eight independent interventions per layer."""
    import torch
    if optimized and (calibration or routing_only):
        raise ValueError("optimized suffix replay is currently for effect events only")
    if next(model.parameters()).device.type == "cuda":
        torch.cuda.synchronize()
    start = time.monotonic()
    if next(model.parameters()).device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    invocations = {}
    residuals = {}
    with ExitStack() as stack:
        if optimized:
            replay_layers = set()
            for l in range(len(model.model.layers)):
                handle = _capture_layer_call(model, l, invocations, replay_layers)
                stack.callback(handle.remove)
            for l in inputs["layers"]:
                handle = _capture_layer_residual(model, l, residuals)
                stack.callback(handle.remove)
        captures = {l: stack.enter_context(RoutingCapture(model, l, inputs["local_predictor_position"],
                                                         capture_window=calibration,
                                                         capture_contributions=optimized)) for l in inputs["layers"]}
        baseline, logits = forward(model, inputs)
        if next(model.parameters()).device.type == "cuda":
            torch.cuda.synchronize()
        baseline_seconds = time.monotonic() - start
        native = {l: h.routing() for l, h in captures.items()}
        windows = {l: h.window_routing() for l, h in captures.items()} if calibration else {}
    result = []
    for layer in inputs["layers"]:
        event = {k: v for k, v in inputs.items() if k not in ("input_ids", "layers")}
        event.update(event_id=event_id(inputs, layer), layer=layer,
                     kind="calibration_window" if calibration else ("probe_routing" if routing_only else "effects"),
                     routing=native[layer], baseline_nll=baseline)
        if calibration:
            event["window_routing"] = windows[layer]
        elif not routing_only:
            intervention_start = time.monotonic()
            effects = []
            for expert in native[layer]["expert_ids"]:
                if optimized:
                    ablated, _ = _replay_from_layer(model, inputs, invocations, residuals,
                                                    captures, layer, expert)
                else:
                    with RoutingCapture(model, layer, inputs["local_predictor_position"], expert) as hook:
                        ablated, _ = forward(model, inputs)
                        if (hook.routing() != native[layer] or not hook.mask_is_local or
                                not hook.survivors_unchanged):
                            raise ValueError("operator A changed native routing or survivors")
                effects.append({"expert_id": expert, "ablated_nll": ablated,
                                "delta_nll": ablated - baseline, "operator": "token_local_zero_contribution"})
                if next(model.parameters()).device.type == "cuda":
                    torch.cuda.synchronize()
            event["effects"] = effects
            event["eight_interventions_seconds"] = time.monotonic() - intervention_start
            event["effect_seconds"] = event["eight_interventions_seconds"]
        validate_measurement(event)
        result.append(event)
    # This uninstrumented full forward also supplies the independent native
    # reference for optimized contribution capture. Exact recovery is required
    # before any complete event is returned/persisted, including after all eight
    # interventions; a bad captured baseline cannot become a completed effect.
    cleanup_start = time.monotonic()
    cleaned, cleaned_logits = forward(model, inputs)
    if next(model.parameters()).device.type == "cuda":
        torch.cuda.synchronize()
    if cleaned != baseline or not torch.equal(cleaned_logits, logits):
        raise ValueError("baseline did not recover after hook cleanup")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    cleanup_seconds = time.monotonic() - cleanup_start
    input_total_seconds = time.monotonic() - start
    for event in result:
        event["shared_baseline_seconds"] = baseline_seconds
        event["shared_seconds"] = baseline_seconds
        event["shared_cleanup_seconds"] = cleanup_seconds
        event["cleanup_seconds"] = event["shared_cleanup_seconds"]
        event["input_total_seconds"] = input_total_seconds
        event["synchronized_input_total_seconds"] = event["input_total_seconds"]
        if next(model.parameters()).device.type == "cuda":
            event["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            event["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        event["host_max_rss_kib"] = __import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss
    return result


def copy_bundle(source, destination):
    verify_bundle(source)
    if Path(destination).exists():
        if file_digest(Path(source) / "manifest.json") != file_digest(Path(destination) / "manifest.json"):
            raise ValueError("copied bundle differs on resume")
        verify_bundle(destination)
    else:
        shutil.copytree(source, destination)


def collect(plan_dir, model_dir, output, mechanics_path=None, calibration_dir=None,
            optimized=False, model_session=None, optimized_validation_path=None,
            reuse_runs=()):
    from .scope import (is_path_b, require_scope, main_budgets, stage_layers, validate_plan)
    manifest = verify_bundle(plan_dir, "experiment-plan")
    settings = read_json(Path(plan_dir) / "research.json")
    stage = manifest["metadata"]["stage"]
    validate_plan(settings, manifest["metadata"], read_rows(Path(plan_dir) / "inputs.jsonl"))
    if stage not in ("technical", "calibration", "probe", "pilot", "main", "sensitivity"):
        raise ValueError("unknown measurement stage")
    if optimized and stage in ("calibration", "probe"):
        raise ValueError("optimized execution is available only for effect events")
    if optimized and optimized_validation_path is None:
        raise ValueError("full technical optimized validation receipt required")
    if mechanics_path is None:
        raise ValueError("native mechanism verification receipt required")
    mechanics = read_json(mechanics_path)
    require_scope(settings, mechanics)
    if mechanics.get("status") != "pass":
        raise ValueError("native mechanism verification has not passed")
    if manifest["metadata"]["window"] not in mechanics.get("context_lengths", []):
        raise ValueError("mechanics receipt has not verified this context length")
    if mechanics.get("source_review_sha256") != manifest["metadata"]["source_review_sha256"]:
        raise ValueError("mechanics receipt belongs to another source review")
    if stage in ("pilot", "main", "sensitivity"):
        if calibration_dir is None:
            raise ValueError("effects require frozen calibration")
        freeze = verify_bundle(calibration_dir, "calibration")
        require_scope(settings, freeze["metadata"])
        tables = read_json(Path(calibration_dir) / "frozen-tables.json")
        if tables.get("status") != "frozen" or tables.get("context_length") != manifest["metadata"]["window"]:
            raise ValueError("frozen rankings must match the input context policy")
        if stage == "main" and not manifest["metadata"].get("main_freeze"):
            raise ValueError("main outcome collection requires an exact pilot-approved freeze")
        if stage in ("main", "sensitivity"):
            main_freeze = manifest["metadata"].get("main_freeze")
            if not main_freeze or main_freeze["settings_sha256"] != digest(settings):
                raise ValueError("main settings differ from the pilot-approved freeze")
            if read_json(Path(plan_dir)/"main-freeze.json") != main_freeze:
                raise ValueError("main freeze payload differs from the frozen plan")
            if stage == "main" and manifest["metadata"]["units"] != main_budgets(settings, main_freeze["selected"]):
                raise ValueError("main sample differs from the pilot-approved allocation")
            if stage == "main" and main_freeze["calibration_manifest_sha256"] != file_digest(Path(calibration_dir)/"manifest.json"):
                raise ValueError("main calibration changed after freezing")
            if is_path_b(settings) and not main_freeze.get("remaining_workflow_budget", {}).get("feasibility_pass"):
                raise ValueError("Path B main requires a remaining-workflow budget admission")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    copy_bundle(plan_dir, output / "plan")
    if calibration_dir:
        copy_bundle(calibration_dir, output / "calibration")
    catalog = read_json(Path(plan_dir) / "model-catalog.json")
    model_acquire_started = time.monotonic()
    fresh_model_load = model_session is None or getattr(model_session, "_model", None) is None
    model = (model_session.acquire(model_dir, catalog, settings["sampling"]["master_seed"])
             if model_session is not None else
             load_model(model_dir, catalog, settings["sampling"]["master_seed"]))
    model_acquire_seconds = time.monotonic() - model_acquire_started
    runtime = runtime_identity()
    executor = {"model_catalog_sha256": digest(catalog), "runtime": runtime,
                "implementation_sha256": implementation_identity()}
    if mechanics.get("executor") != executor:
        raise ValueError("mechanism receipt belongs to another model/runtime/implementation")
    if calibration_dir and freeze["metadata"]["executor"] != executor:
        raise ValueError("frozen calibration belongs to another model/runtime/implementation")
    if stage in ("main","sensitivity") and main_freeze["executor"] != executor:
        raise ValueError("main executor differs from the pilot-approved runtime")
    execution_mode = OPTIMIZED_EXECUTION_MODE if optimized else "native_full_forward_v1"
    if optimized:
        optimized_receipt = read_json(optimized_validation_path)
        require_scope(settings, optimized_receipt)
        required = (optimized_receipt.get("status") == "pass" and
                    optimized_receipt.get("kind") == "optimized-validation" and
                    optimized_receipt.get("validation_scope") == "full_technical" and
                    optimized_receipt.get("approval") == "collection_candidate" and
                    optimized_receipt.get("full_coverage") is True and
                    optimized_receipt.get("checks") == {
                        "baseline_routes_weights_mixture_logits": "pass",
                        "all_eight_native_ablation_logits_mixture_nll": "pass",
                        "survivors_and_untargeted_rows": "pass",
                        "downstream_routes_against_native_ablation": "pass",
                        "parameters_and_cleanup": "pass"} and
                    optimized_receipt.get("executor") == executor and
                    optimized_receipt.get("execution_mode") == execution_mode and
                    manifest["metadata"]["window"] in optimized_receipt.get("context_lengths", []) and
                    set(manifest["metadata"].get("layers", stage_layers(settings, stage))) <=
                    set(optimized_receipt.get("layers", [])) and
                    optimized_receipt.get("source_review_sha256") == manifest["metadata"]["source_review_sha256"] and
                    mechanics.get("technical_plan_sha256") in
                    optimized_receipt.get("technical_plan_sha256", []) and
                    optimized_receipt.get("tolerances") == {"atol": 1e-6, "rtol": 1e-5, "nll_atol": 1e-6})
        if not required:
            raise ValueError("optimized validation receipt does not authorize this executor, context, layers or technical plan")
    identity = {"executor": executor, "execution_mode": execution_mode,
                "plan_manifest_sha256": file_digest(Path(plan_dir) / "manifest.json"),
                "calibration_manifest_sha256": file_digest(Path(calibration_dir) / "manifest.json") if calibration_dir else None}
    inputs_path = Path(plan_dir) / "inputs.jsonl"
    # Keep only identifiers/layers for coverage and reusable-source indexes.
    # Token windows are streamed one at a time during collection.
    input_index = [{key: row[key] for key in ("input_id", "document_id", "layers")}
                   for row in read_rows(inputs_path)]
    plan_manifest_sha256 = file_digest(Path(plan_dir) / "manifest.json")
    # Raw source documents may retain full tokenized bodies; this collector only
    # needs the immutable content digest for reuse identity checks.
    documents = {row["document_id"]: {"document_id": row["document_id"],
                  "content_sha256": row["content_sha256"]}
                 for row in read_rows(Path(plan_dir) / "documents.jsonl")}
    expected = {event_id(i, l) for i in input_index for l in i["layers"]}
    reusable = [SealedMeasurementReuse(
        path, input_index, executor=executor, execution_mode=execution_mode,
        model_revision=catalog["revision"],
        tokenizer_revision=manifest["metadata"]["tokenizer_sha256"], plan_stage=stage)
        for path in reuse_runs]
    atomic_receipt = output / "mechanics.json"
    if not atomic_receipt.exists():
        atomic(atomic_receipt, canonical(mechanics) + b"\n")
    if optimized:
        validation_copy = output / "optimized-validation.json"
        if validation_copy.exists():
            if read_json(validation_copy) != optimized_receipt:
                raise ValueError("optimized validation receipt changed on resume")
        else:
            atomic(validation_copy, canonical(optimized_receipt) + b"\n")
    with Journal(output, identity, compact=True) as journal:
        if set(journal.records) - expected:
            raise ValueError("journal contains events outside the frozen plan")
        for index, i in enumerate(read_rows(inputs_path)):
            wanted = {event_id(i, l) for l in i["layers"]}
            if all(identifier in journal.records for identifier in wanted):
                continue
            input_started = time.monotonic()
            persistence_seconds = 0.0
            cache_hit_seconds_by_layer = {}
            source_timing_by_layer = {}
            measured_by_layer = {}
            try:
                remaining = [l for l in i["layers"] if event_id(i,l) not in journal.records]
                events = []
                missed = []
                for layer in remaining:
                    reused = None
                    for source in reusable:
                        reused = source.get(i, layer, document=documents[i["document_id"]],
                            plan_manifest_sha256=plan_manifest_sha256, plan_stage=stage,
                            sampling={**i.get("sampling", {}), "sample_order": i.get("sample_order"),
                                      "eligible_positions": i.get("eligible_positions")})
                        if reused is not None:
                            cache_hit_seconds_by_layer[str(layer)] = reused["cache_hit_seconds"]
                            source_timing_by_layer[str(layer)] = reused.get("source_timing", {})
                            break
                    if reused is None:
                        missed.append(layer)
                    else:
                        reused["execution_mode"] = execution_mode
                        events.append(reused)
                if missed:
                    measured = measure_input(model, dict(i,layers=missed), calibration=stage == "calibration",
                        routing_only=stage == "probe", optimized=optimized)
                    for event in measured:
                        event["execution_mode"] = execution_mode
                        measured_by_layer[str(event["layer"])] = {
                            "effect_seconds": event.get("effect_seconds"),
                            "shared_seconds": event.get("shared_seconds"),
                            "cleanup_seconds": event.get("cleanup_seconds"),
                            "synchronized_input_total_seconds": event.get("synchronized_input_total_seconds"),
                        }
                    events.extend(measured)
                previous_baselines = {journal.records[e]["baseline_nll"] for e in wanted if e in journal.records}
                if any(e["baseline_nll"] != previous for e in events for previous in previous_baselines):
                    raise ValueError("resume baseline differs from completed layers")
                for event in sorted(events, key=lambda row: row["layer"]):
                    if event["event_id"] in journal.records:
                        previous = journal.records[event["event_id"]]
                        if any(previous[k] != event[k] for k in ("routing", "baseline_nll")):
                            raise ValueError("resume baseline differs from preserved measurements")
                    else:
                        append_started = time.monotonic()
                        journal.append(event)
                        persistence_seconds += time.monotonic() - append_started
                # A sidecar is append-only and belongs to the same sealed bundle. It
                # records only work observed in this invocation; resumed rows with no
                # surviving sidecar remain UNKNOWN rather than inheriting old timings.
                sidecar = output / "input-timings.jsonl"
                timing = {
                    "input_id": i["input_id"], "domain": i["domain"],
                    "context_length": i["context_length"],
                    "kind": "calibration" if stage == "calibration" else
                           "probe" if stage == "probe" else "effects",
                    "execution_mode": execution_mode,
                    "evidence_status": "measured",
                    "shared_seconds": _shared_timing(
                        [row for row in events if not row.get("cache_status")], "shared_seconds"),
                    "cleanup_seconds": _shared_timing(
                        [row for row in events if not row.get("cache_status")], "cleanup_seconds"),
                    "effect_seconds_by_layer": {
                        layer: values["effect_seconds"]
                        for layer, values in measured_by_layer.items()
                        if values["effect_seconds"] is not None
                    },
                    "measured_by_layer": measured_by_layer,
                    "cache_hit_seconds_by_layer": cache_hit_seconds_by_layer,
                    "source_timing_by_layer": source_timing_by_layer,
                    "persistence_seconds": persistence_seconds,
                    "synchronized_input_total_seconds": time.monotonic() - input_started,
                    "cache_miss_layers": sorted(measured_by_layer, key=int),
                    "cache_hit_layers": sorted(cache_hit_seconds_by_layer, key=int),
                }
                append_timing_row(sidecar, timing)
            except BaseException as error:
                with (output / "failures.jsonl").open("ab") as f:
                    f.write(canonical({"input_id": i["input_id"], "exception": type(error).__name__,
                                       "message": str(error), "status": "incomplete"}) + b"\n")
                    f.flush()
                    __import__("os").fsync(f.fileno())
                raise
            print(f"Completed input {index + 1}/{len(input_index)}", flush=True)
        if set(journal.records) != expected:
            raise ValueError("frozen plan is incomplete")
    return seal(output, "measurements", {"stage": stage, "executor": executor,
                "execution_mode": execution_mode,
                "setup_load_seconds": model_acquire_seconds,
                "model_load_evidence_status": "measured" if fresh_model_load else "reused_session_acquire_measured",
                "plan_metadata": manifest["metadata"], "expected_event_ids": sorted(expected),
                "raw_measurement_scope": "all_window_positions" if stage == "calibration" else "eight_experts_per_event"})
