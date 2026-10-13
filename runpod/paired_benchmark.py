#!/usr/bin/env python3
"""Run the bounded paired diagnostic for reference/cleanup consolidation.

This command is diagnostic only. It never calls a cloud API, downloads weights,
creates scientific collection receipts, or authorizes optimized collection.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import types


EXPECTED_CONTEXTS = {2048, 4096}
EXPECTED_LAYERS = [0, 7, 15]
FROZEN_INPUT_IDS = (
    "03eaf4fa4d52e6dff5f3b24b0e8d2994e13ea36fcafba67c7bf199f09f8e80b3",
    "25fa91b5a3e5053e115ab9abbcef3068eb50471808e2a96759a7bbb47488f32d",
    "5c095561ce2c001c260435555de2c2bfc05c4d00551e3ac34e4a3060618aa696",
    "b13680b131af949a5d145ea37d0452140d41742e891e8f3efe39a4f559aaea84",
    "caa20ee78b858632dc01dd1427b3e6ef02b3d112d441232404c5174da9e5135a",
    "04599fbea87a37f411b0111ac16ba0c2db934b3d081c05d36a55fb50049d9a4e",
)
LAYER_SHAPES = ([15, 0, 7], [15])
DEPENDENCY_FILES = ("artifacts.py", "interventions.py", "model.py", "reuse.py",
                    "observation_contract.py", "verify.py")
IMPLEMENTATION_FILES = (*DEPENDENCY_FILES, "run_experiment.py", "optimized_validation.py")
CANDIDATE_MODE = "native_shape_suffix_cleanup_reference_v2"
BASELINE_MODE = "native_shape_suffix_v1"
ATOL, RTOL, NLL_ATOL = 1e-6, 1e-5, 1e-6
PHASE_CAPS = {"model_hash_load_and_identity": 90, "paired_warmup": 60,
              "independent_native_checks": 240, "actual_paired_collector_and_durable_persistence": 240,
              "receipt_finalization": 30}
FINAL_RESERVE = 30.0
RECEIPT_WRITE_RESERVE_SECONDS = 2.0


def emit(message):
    print(f"[paired-benchmark {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {message}", flush=True)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def write_json(path, value):
    from src.artifacts import atomic
    atomic(path, canonical(value) + b"\n")


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def implementation_hash(src_root):
    from src.artifacts import digest
    src_root = Path(src_root)
    return digest({name: file_hash(src_root / name) for name in IMPLEMENTATION_FILES})


def load_findings(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    proposal = raw.get("benchmark_proposal", raw)
    selection = proposal.get("selection")
    if not isinstance(selection, list) or len(selection) != 6:
        raise ValueError("selection must contain the six frozen benchmark input IDs")
    if proposal.get("status") not in (None, "proposal_only_NOT_authorized_or_prepared"):
        # Slim immutable package specs use their own non-paid prepared status.
        if not str(proposal.get("status", "")).startswith("prepared_"):
            raise ValueError("unexpected paired benchmark selection status")
    shapes = proposal.get("layer_shapes", [list(x) for x in LAYER_SHAPES])
    if shapes != [list(x) for x in LAYER_SHAPES]:
        raise ValueError("layer shapes differ from the frozen [15,0,7] and [15] schedule")
    ids = [row.get("input_id") for row in selection]
    if len(set(ids)) != 6 or any(not isinstance(value, str) or len(value) != 64 for value in ids):
        raise ValueError("selection input IDs must be six unique SHA-256 identities")
    if tuple(ids) != FROZEN_INPUT_IDS:
        raise ValueError("selection input IDs/order differ from the reviewed six-window proposal")
    if {(row.get("context_length")) for row in selection} != EXPECTED_CONTEXTS:
        raise ValueError("selection must include both 2048 and 4096 contexts")
    if any(row.get("pair_order") not in ("baseline_then_candidate", "candidate_then_baseline")
           for row in selection):
        raise ValueError("every input must specify an AB/BA pair order")
    expected_order = ("baseline_then_candidate", "candidate_then_baseline",
                      "baseline_then_candidate", "candidate_then_baseline",
                      "baseline_then_candidate", "candidate_then_baseline")
    if tuple(row["pair_order"] for row in selection) != expected_order:
        raise ValueError("pair execution order differs from the reviewed alternating AB/BA schedule")
    return raw, proposal, selection


def _read_plan(plan):
    from src.artifacts import read_json, read_rows, verify_bundle
    plan = Path(plan)
    manifest = verify_bundle(plan, "experiment-plan")
    context = manifest["metadata"].get("window")
    if manifest["metadata"].get("stage") != "technical" or context not in EXPECTED_CONTEXTS:
        raise ValueError("paired benchmark accepts only sealed 2048/4096 technical plans")
    rows = {row["input_id"]: row for row in read_rows(plan / "inputs.jsonl")}
    if len(rows) != sum(1 for _ in read_rows(plan / "inputs.jsonl")):
        raise ValueError("technical plan contains duplicate input IDs")
    return plan, manifest, read_json(plan / "research.json"), read_json(plan / "model-catalog.json"), rows


def select_inputs(plan_dirs, selection):
    from src.artifacts import digest, file_digest
    plan_data = [_read_plan(path) for path in plan_dirs]
    by_context = {item[1]["metadata"]["window"]: item for item in plan_data}
    if set(by_context) != EXPECTED_CONTEXTS or len(plan_data) != 2:
        raise ValueError("exactly one sealed technical plan per context is required")
    primary = by_context[4096]
    shortened = by_context[2048]
    if digest(primary[3]) != digest(shortened[3]):
        raise ValueError("paired technical contexts use different model catalogs")
    if digest(primary[2]) != digest(shortened[2]):
        raise ValueError("paired technical contexts use different research settings")
    if primary[1]["metadata"].get("source_review_sha256") != shortened[1]["metadata"].get("source_review_sha256"):
        raise ValueError("paired technical contexts use different source review")
    result = []
    for selected in selection:
        context = selected["context_length"]
        plan = by_context[context]
        row = plan[4].get(selected["input_id"])
        if row is None:
            raise ValueError(f"selected input is absent from sealed {context} plan: {selected['input_id']}")
        for field in ("document_id", "domain", "source_group", "predictor_position", "target_id", "sample_order"):
            if row.get(field) != selected.get(field):
                raise ValueError(f"selected {context} input differs in {field}")
        if (len(row["input_ids"]) != context or row.get("local_predictor_position") != context - 1 or
                row.get("target_position") != 4096 or set(row.get("layers", ())) != set(EXPECTED_LAYERS)):
            raise ValueError("selected input context, target, predictor or layer coverage differs")
        if context == 2048:
            long_row = primary[4].get(row.get("parent_input_id"))
            if long_row is None or any(row.get(key) != long_row.get(key) for key in (
                    "document_id", "domain", "source_group", "predictor_position", "target_id", "sample_order")):
                raise ValueError("short-context selection does not match its exact 4096 parent input")
            if row["input_ids"] != long_row["input_ids"][-2048:]:
                raise ValueError("short-context input bytes differ from the parent suffix")
        result.append({"row": row, "plan": plan[0], "manifest_sha256": file_digest(plan[0] / "manifest.json"),
                       "source_review_sha256": plan[1]["metadata"].get("source_review_sha256"),
                       "settings": plan[2], "catalog": plan[3], "selected": selected})
    if {row["selected"]["domain"] for row in result} != {"c4", "scientific", "code"}:
        raise ValueError("selection must cover C4, scientific and code inputs")
    if any(sum(item["selected"]["domain"] == domain for item in result) != 2
           for domain in ("c4", "scientific", "code")):
        raise ValueError("selection must contain one input/domain/context")
    return result


def load_baseline(baseline_source, candidate_src):
    """Load the preserved source under a private package after strict dependency parity."""
    from src.artifacts import digest
    baseline_root = Path(baseline_source).resolve()
    baseline_src = baseline_root / "src" if (baseline_root / "src/run_experiment.py").is_file() else baseline_root
    if not (baseline_src / "run_experiment.py").is_file():
        raise FileNotFoundError("baseline source must contain src/run_experiment.py")
    candidate_src = Path(candidate_src).resolve()
    dependency_hashes = {}
    for name in DEPENDENCY_FILES:
        baseline_path, candidate_path = baseline_src / name, candidate_src / name
        if file_hash(baseline_path) != file_hash(candidate_path):
            raise ValueError(f"preserved baseline dependency differs from candidate: {name}")
        dependency_hashes[name] = file_hash(baseline_path)
    baseline_impl = implementation_hash(baseline_src)
    candidate_impl = implementation_hash(candidate_src)
    alias = "_dsic3007_pair_baseline_" + baseline_impl[:12]
    package = types.ModuleType(alias)
    package.__path__ = [str(baseline_src)]
    package.__package__ = alias
    sys.modules[alias] = package
    module_name = alias + ".run_experiment"
    spec = importlib.util.spec_from_file_location(module_name, baseline_src / "run_experiment.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load preserved baseline collector")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, {"baseline_implementation_sha256": baseline_impl,
                    "candidate_implementation_sha256": candidate_impl,
                    "shared_dependency_sha256": dependency_hashes}


def _finite_close(left, right, tolerance, label):
    left, right = float(left), float(right)
    if not math.isfinite(left) or not math.isfinite(right) or abs(left - right) > tolerance:
        raise AssertionError(f"{label} mismatch: baseline={left:.12g}, reference={right:.12g}")


def compare_measurement_events(baseline_events, candidate_events, reference_routes, native_cases,
                               native_baseline_nll=None):
    """Require both measured arms to match independent native routes and effects."""
    if [event["layer"] for event in baseline_events] != [event["layer"] for event in candidate_events]:
        raise AssertionError("baseline/candidate layer order differs")
    if len(baseline_events) != len(candidate_events):
        raise AssertionError("baseline/candidate event counts differ")
    by_case = {(case["layer"], case["expert_id"]): case for case in native_cases}
    for baseline, candidate in zip(baseline_events, candidate_events, strict=True):
        layer = baseline["layer"]
        if baseline.get("event_id") != candidate.get("event_id") or baseline.get("input_id") != candidate.get("input_id"):
            raise AssertionError(f"layer {layer}: event/input identity differs between paired arms")
        route = reference_routes[layer]
        if baseline["routing"] != route or candidate["routing"] != route:
            raise AssertionError(f"layer {layer}: event routes differ from independent native baseline")
        _finite_close(baseline["baseline_nll"], candidate["baseline_nll"], NLL_ATOL, f"layer {layer} baseline NLL")
        if native_baseline_nll is not None:
            _finite_close(baseline["baseline_nll"], native_baseline_nll, NLL_ATOL,
                          f"baseline layer {layer} native reference NLL")
            _finite_close(candidate["baseline_nll"], native_baseline_nll, NLL_ATOL,
                          f"candidate layer {layer} native reference NLL")
        baseline_effects = {row["expert_id"]: row for row in baseline["effects"]}
        candidate_effects = {row["expert_id"]: row for row in candidate["effects"]}
        if ([row["expert_id"] for row in baseline["effects"]] != route["expert_ids"] or
                [row["expert_id"] for row in candidate["effects"]] != route["expert_ids"]):
            raise AssertionError(f"layer {layer}: raw expert order differs from native routing order")
        if set(baseline_effects) != set(route["expert_ids"]) or set(candidate_effects) != set(route["expert_ids"]):
            raise AssertionError(f"layer {layer}: event expert set differs from native route")
        for expert in route["expert_ids"]:
            native = by_case.get((layer, expert))
            if native is None:
                raise AssertionError(f"missing independent native case for layer {layer}, expert {expert}")
            for label, event, field in (("baseline", baseline_effects[expert], "ablated_nll"),
                                        ("candidate", candidate_effects[expert], "ablated_nll")):
                _finite_close(event[field], native["native_nll"], NLL_ATOL,
                              f"{label} layer {layer} expert {expert} ablated NLL")
                _finite_close(event["delta_nll"], event[field] - baseline["baseline_nll"],
                              NLL_ATOL, f"{label} layer {layer} expert {expert} delta NLL")
    return True


def _synchronize(model):
    import torch
    if next(model.parameters()).device.type == "cuda":
        torch.cuda.synchronize()


def _assert_clean(model):
    for index, layer in enumerate(model.model.layers):
        if layer._forward_pre_hooks or layer._forward_hooks:
            raise RuntimeError(f"layer {index}: decoder hooks remain after diagnostic operation")
        experts = layer.mlp.experts
        if experts._forward_pre_hooks or experts._forward_hooks or "forward" in experts.__dict__:
            raise RuntimeError(f"layer {index}: hooks/forward override remain after diagnostic operation")
        norm = getattr(layer, "post_attention_layernorm", None)
        if norm is not None and (norm._forward_pre_hooks or norm._forward_hooks):
            raise RuntimeError(f"layer {index}: post-attention norm hooks remain after diagnostic operation")


def _parameter_versions(model):
    return {name: parameter._version for name, parameter in model.named_parameters()}


def _assert_parameter_versions(model, expected):
    current = _parameter_versions(model)
    if current != expected:
        changed = sorted(set(current) | set(expected))
        changed = [name for name in changed if current.get(name) != expected.get(name)]
        raise RuntimeError(f"model parameter versions changed during paired diagnostic: {changed[:8]}")


def _is_incomplete_timeout(exc, validator_time_limit):
    return isinstance(exc, TimeoutError) or isinstance(exc, validator_time_limit)


def _finalization_within_budget(started_monotonic, deadline_epoch, now_monotonic=None,
                                now_epoch=None, cap_seconds=30.0):
    now_monotonic = time.monotonic() if now_monotonic is None else now_monotonic
    now_epoch = time.time() if now_epoch is None else now_epoch
    elapsed = now_monotonic - started_monotonic
    return (elapsed + RECEIPT_WRITE_RESERVE_SECONDS < cap_seconds and
            now_epoch + RECEIPT_WRITE_RESERVE_SECONDS < deadline_epoch)


def _pair_measurement(item, layers, pair_order, model, baseline_module, candidate_module,
                      events_path, timings_path, progress_path, deadline, phase_deadline,
                      run_number, kind, parameter_versions=None):
    from src.run_experiment import OPTIMIZED_EXECUTION_MODE

    row = item["row"]
    inputs = dict(row, layers=list(layers))
    functions = {"baseline": baseline_module.measure_input, "candidate": candidate_module.measure_input}
    modes = {"baseline": BASELINE_MODE, "candidate": OPTIMIZED_EXECUTION_MODE}
    order = pair_order.split("_then_")
    if set(order) != {"baseline", "candidate"}:
        raise ValueError("invalid paired execution order")
    results = {}
    parameter_versions = _parameter_versions(model) if parameter_versions is None else parameter_versions
    for arm in order:
        check_deadline(deadline, phase_deadline)
        arm_started = time.time()
        append_jsonl(progress_path, {"record_type": "paired_arm_started", "pair_number": run_number,
                                     "benchmark_kind": kind, "input_id": row["input_id"],
                                     "context_length": row["context_length"], "layer_shape": list(layers),
                                     "benchmark_arm": arm, "pair_order": pair_order,
                                     "started_epoch": arm_started})
        emit(f"ARM START pair={run_number} kind={kind} input={row['input_id'][:12]} "
             f"context={row['context_length']} layers={layers} arm={arm} "
             f"remaining={max(0.0, deadline-time.time()):.1f}s")
        arm_wall_started = time.monotonic()
        _synchronize(model)
        started = time.monotonic()
        events = functions[arm](model, inputs, optimized=True)
        _synchronize(model)
        measured_seconds = time.monotonic() - started
        _assert_parameter_versions(model, parameter_versions)
        _assert_clean(model)
        event_persistence_started = time.monotonic()
        for event in events:
            event.update(execution_mode=modes[arm], benchmark_arm=arm,
                         benchmark_kind=kind, layer_shape=list(layers), run_number=run_number,
                         plan_manifest_sha256=item["manifest_sha256"])
            append_jsonl(events_path, event)
        event_persistence_seconds = time.monotonic() - event_persistence_started
        timing = {"record_type": "paired_measurement_timing", "input_id": row["input_id"],
                  "context_length": row["context_length"], "domain": row["domain"],
                  "benchmark_arm": arm, "benchmark_kind": kind,
                  "execution_mode": modes[arm], "layer_shape": list(layers),
                  "measurement_seconds": measured_seconds,
                  "shared_seconds": events[0].get("shared_seconds") if events else None,
                  "cleanup_seconds": events[0].get("cleanup_seconds") if events else None,
                  "effect_seconds_by_layer": {str(event["layer"]): event.get("effect_seconds") for event in events},
                  "event_journal_write_seconds": event_persistence_seconds,
                  "end_to_end_scope": "from pre-arm synchronization through event journal fsync; excludes timing/progress sidecars",
                  "end_to_end_seconds_before_timing_sidecar": time.monotonic() - arm_wall_started,
                  "event_count": len(events), "evidence_status": "measured_diagnostic"}
        timing_sidecar_started = time.monotonic()
        candidate_module.append_timing_row(timings_path, timing)
        timing_sidecar_seconds = time.monotonic() - timing_sidecar_started
        end_to_end_seconds = time.monotonic() - arm_wall_started
        timing["timing_sidecar_fsync_seconds"] = timing_sidecar_seconds
        timing["end_to_end_seconds"] = end_to_end_seconds
        results[arm] = {"events": events, "timing": timing}
        check_deadline(deadline, phase_deadline)
        _assert_clean(model)
        _assert_parameter_versions(model, parameter_versions)
        completed_epoch = time.time()
        append_jsonl(progress_path, {"record_type": "paired_arm_complete", "pair_number": run_number,
                                     "benchmark_kind": kind, "input_id": row["input_id"],
                                     "context_length": row["context_length"], "layer_shape": list(layers),
                                     "benchmark_arm": arm, "event_count": len(events),
                                     "measurement_seconds": measured_seconds,
                                     "event_journal_write_seconds": event_persistence_seconds,
                                     "timing_sidecar_fsync_seconds": timing_sidecar_seconds,
                                     "end_to_end_scope": "from pre-arm synchronization through timing-row fsync; progress fsync excluded",
                                     "end_to_end_seconds": end_to_end_seconds,
                                     "elapsed_seconds": completed_epoch-arm_started,
                                     "finished_epoch": completed_epoch})
        emit(f"ARM COMPLETE pair={run_number} arm={arm} events={len(events)} "
             f"measure={measured_seconds:.3f}s events_fsync={event_persistence_seconds:.3f}s "
             f"timing_fsync={timing_sidecar_seconds:.3f}s end_to_end={end_to_end_seconds:.3f}s "
             f"elapsed={completed_epoch-arm_started:.1f}s remaining={max(0.0, deadline-completed_epoch):.1f}s")
    return results


def check_deadline(deadline, phase_deadline=None):
    now = time.time()
    if now >= deadline - FINAL_RESERVE:
        raise TimeoutError("absolute attempt deadline reserve reached")
    if phase_deadline is not None and now >= phase_deadline:
        raise TimeoutError("diagnostic phase allowance reached")


def _capture_native_routes(model, inputs, validator, deadline, phase_deadline, progress_path):
    import src.interventions as interventions

    captured = {}
    original = interventions.RoutingCapture

    class TapRoutingCapture(original):
        def __init__(self, *args, **kwargs):
            self._pair_tap = (kwargs.get("expert_id", args[3] if len(args) > 3 else None) is None and
                              kwargs.get("capture_window", args[4] if len(args) > 4 else False) and
                              not kwargs.get("capture_contributions", False))
            self._pair_layer = kwargs.get("layer", args[1] if len(args) > 1 else None)
            super().__init__(*args, **kwargs)

        def routing(self):
            result = super().routing()
            if self._pair_tap:
                captured[self._pair_layer] = result
            return result

    start = time.monotonic()
    case_deadline = min(deadline - FINAL_RESERVE, phase_deadline)
    max_seconds = max(0.01, case_deadline - time.time())

    def progress(row):
        check_deadline(deadline, phase_deadline)
        append_jsonl(progress_path, {"record_type": "native_expert_progress", **row,
                                     "elapsed_seconds": time.monotonic() - start})

    interventions.RoutingCapture = TapRoutingCapture
    try:
        elapsed = lambda: time.monotonic() - start
        result = validator._input_case(model, dict(inputs, layers=list(EXPECTED_LAYERS)),
                                       elapsed, max_seconds, progress=progress)
    finally:
        interventions.RoutingCapture = original
    check_deadline(deadline, phase_deadline)
    if not result.get("cleanup_pass") or not result.get("parameters_unchanged"):
        raise AssertionError("independent native validation cleanup/parameter check failed")
    if set(captured) != set(EXPECTED_LAYERS):
        raise AssertionError("independent native baseline route capture is incomplete")
    summary = {"input_id": inputs["input_id"], "context_length": len(inputs["input_ids"]),
               "domain": inputs["domain"], "baseline_nll": result["baseline_nll"],
               "route_by_layer": captured, "cases": result["cases"],
               "comparison_timings": result["comparison_timings"],
               "cleanup_pass": True, "parameters_unchanged": True,
               "elapsed_seconds": time.monotonic() - start}
    append_jsonl(progress_path, {"record_type": "native_input_complete", **summary})
    return summary


def run_benchmark(plan_dirs, baseline_source, selection_path, model_dir, deadline_epoch, output):
    from src.artifacts import atomic, digest, file_digest, read_json
    from src.model import VerifiedModelSession, implementation_identity, runtime_identity
    import src.run_experiment as candidate_module
    from src.optimized_validation import _TimeLimit, _input_case

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    events_path = output.with_suffix(output.suffix + ".events.jsonl")
    timings_path = output.with_suffix(output.suffix + ".timings.jsonl")
    progress_path = output.with_suffix(output.suffix + ".progress.jsonl")
    for path in (output, events_path, timings_path, progress_path):
        if path.exists():
            raise FileExistsError(f"preserve existing benchmark attempt and choose a new path: {path}")
    if (not math.isfinite(deadline_epoch) or deadline_epoch <= time.time() or
            deadline_epoch - time.time() > 1200):
        raise ValueError("deadline must be finite, in the future, and within the 20-minute benchmark cap")

    state = {"status": "incomplete", "kind": "paired-collector-diagnostic",
             "diagnostic_only": True, "approval": "diagnostic_only",
             "started_epoch": time.time(), "deadline_epoch": deadline_epoch,
             "phases": {}, "completed_pairs": 0, "completed_native_inputs": 0,
             "raw_event_count": 0, "measured_event_count": 0,
             "warmup_event_count": 0, "full_coverage": False}
    append_jsonl(progress_path, {"record_type": "benchmark_started", **state})
    emit(f"START paired diagnostic; output={output}; absolute_deadline={deadline_epoch:.3f}")

    def phase_window(name, previous_end=None):
        start = time.time()
        stop = min(deadline_epoch - FINAL_RESERVE, start + PHASE_CAPS[name])
        if previous_end is not None:
            stop = min(stop, previous_end)
        return start, stop

    def check(name, stop):
        check_deadline(deadline_epoch, stop)
        state["current_phase"] = name
        state["updated_epoch"] = time.time()
        write_json(Path(str(output) + ".status.json"), state)

    try:
        raw_findings, proposal, selection = load_findings(selection_path)
        check_deadline(deadline_epoch)
        selected = select_inputs(plan_dirs, selection)
        candidate_src = Path(candidate_module.__file__).resolve().parent
        baseline_module, identities = load_baseline(baseline_source, candidate_src)
        expected_baseline = proposal.get("baseline_implementation_sha256")
        if expected_baseline and expected_baseline != identities["baseline_implementation_sha256"]:
            raise ValueError("baseline implementation hash differs from immutable benchmark specification")
        expected_candidate = proposal.get("candidate_implementation_sha256")
        if expected_candidate and expected_candidate != identities["candidate_implementation_sha256"]:
            raise ValueError("candidate implementation hash differs from immutable benchmark specification")
        state.update(selection_sha256=file_digest(selection_path),
                     plan_manifest_sha256={str(item["row"]["context_length"]): item["manifest_sha256"] for item in selected},
                     source_review_sha256=selected[0]["source_review_sha256"], **identities)
        model_catalog = selected[0]["catalog"]
        settings = selected[0]["settings"]

        started, stop = phase_window("model_hash_load_and_identity")
        check("model_hash_load_and_identity", stop)
        session = VerifiedModelSession()
        model = session.acquire(model_dir, model_catalog, settings["sampling"]["master_seed"])
        initial_parameter_versions = _parameter_versions(model)
        runtime = runtime_identity()
        executor = {"model_catalog_sha256": digest(model_catalog), "runtime": runtime,
                    "implementation_sha256": implementation_identity()}
        if time.time() >= stop:
            raise TimeoutError("model hash/load/identity phase exceeded its 120 second allowance")
        state["runtime"] = runtime
        state["executor"] = executor
        state["phases"]["model_hash_load_and_identity"] = {"status": "pass", "started_epoch": started,
            "finished_epoch": time.time(), "elapsed_seconds": time.time()-started}
        write_json(Path(str(output) + ".status.json"), state)
        emit(f"PHASE model_hash_load_and_identity PASS elapsed={time.time()-started:.1f}s mode={CANDIDATE_MODE}")

        started, stop = phase_window("paired_warmup")
        for context in (4096, 2048):
            item = next(entry for entry in selected if entry["row"]["context_length"] == context)
            check("paired_warmup", stop)
            order = item["selected"]["pair_order"]
            pair = _pair_measurement(item, LAYER_SHAPES[0], order, model, baseline_module,
                    candidate_module, events_path, timings_path, progress_path, deadline_epoch,
                    stop, 0, "warmup", initial_parameter_versions)
            compare_measurement_events(pair["baseline"]["events"], pair["candidate"]["events"],
                                       {event["layer"]: event["routing"] for event in pair["baseline"]["events"]},
                                       [{"layer": event["layer"], "expert_id": effect["expert_id"],
                                         "native_nll": effect["ablated_nll"]}
                                        for event in pair["baseline"]["events"] for effect in event["effects"]])
            warmup_events = len(pair["baseline"]["events"]) + len(pair["candidate"]["events"])
            state["raw_event_count"] += warmup_events
            state["warmup_event_count"] += warmup_events
            write_json(Path(str(output) + ".status.json"), state)
            append_jsonl(progress_path, {"record_type": "paired_warmup_complete", "context_length": context,
                                         "input_id": item["row"]["input_id"], "pair_order": order,
                                         "event_count": warmup_events,
                                         "baseline_seconds": pair["baseline"]["timing"]["measurement_seconds"],
                                         "candidate_seconds": pair["candidate"]["timing"]["measurement_seconds"]})
        state["phases"]["paired_warmup"] = {"status": "pass", "started_epoch": started,
            "finished_epoch": time.time(), "elapsed_seconds": time.time()-started, "input_count": 2}
        write_json(Path(str(output) + ".status.json"), state)
        emit(f"PHASE paired_warmup PASS elapsed={time.time()-started:.1f}s inputs=2")

        started, stop = phase_window("independent_native_checks")
        native = {}
        for number, item in enumerate(selected, start=1):
            check("independent_native_checks", stop)
            summary = _capture_native_routes(model, item["row"],
                    types.SimpleNamespace(_input_case=_input_case), deadline_epoch, stop, progress_path)
            native[item["row"]["input_id"]] = summary
            state["completed_native_inputs"] = number
            write_json(Path(str(output) + ".status.json"), state)
            emit(f"NATIVE {number}/6 domain={item['row']['domain']} context={item['row']['context_length']} comparisons={len(summary['cases'])}/24 elapsed={time.time()-started:.1f}s/{min(PHASE_CAPS['independent_native_checks'], deadline_epoch-started):.1f}s")
        state["phases"]["independent_native_checks"] = {"status": "pass", "started_epoch": started,
            "finished_epoch": time.time(), "elapsed_seconds": time.time()-started,
            "input_count": len(native), "expert_comparisons": sum(len(row["cases"]) for row in native.values())}
        write_json(Path(str(output) + ".status.json"), state)
        emit(f"PHASE independent_native_checks PASS elapsed={time.time()-started:.1f}s inputs=6 comparisons=144")

        started, stop = phase_window("actual_paired_collector_and_durable_persistence")
        comparison_summaries = []
        run_number = 0
        for item in selected:
            for shape in LAYER_SHAPES:
                run_number += 1
                check("actual_paired_collector_and_durable_persistence", stop)
                pair = _pair_measurement(item, shape, item["selected"]["pair_order"], model,
                        baseline_module, candidate_module, events_path, timings_path,
                        progress_path, deadline_epoch, stop, run_number, "measured",
                        initial_parameter_versions)
                independent = native[item["row"]["input_id"]]
                compare_measurement_events(pair["baseline"]["events"], pair["candidate"]["events"],
                    independent["route_by_layer"], independent["cases"],
                    native_baseline_nll=independent["baseline_nll"])
                ratio = (pair["baseline"]["timing"]["measurement_seconds"] /
                         pair["candidate"]["timing"]["measurement_seconds"]
                         if pair["candidate"]["timing"]["measurement_seconds"] > 0 else None)
                comparison = {"input_id": item["row"]["input_id"], "domain": item["row"]["domain"],
                    "context_length": item["row"]["context_length"], "layer_shape": list(shape),
                    "pair_order": item["selected"]["pair_order"], "pair_number": run_number,
                    "baseline_seconds": pair["baseline"]["timing"]["measurement_seconds"],
                    "candidate_seconds": pair["candidate"]["timing"]["measurement_seconds"],
                    "baseline_end_to_end_seconds": pair["baseline"]["timing"]["end_to_end_seconds"],
                    "candidate_end_to_end_seconds": pair["candidate"]["timing"]["end_to_end_seconds"],
                    "conditional_ratio_baseline_over_candidate": ratio,
                    "baseline_events": len(pair["baseline"]["events"]),
                    "candidate_events": len(pair["candidate"]["events"]),
                    "native_checks": "pass"}
                append_jsonl(progress_path, {"record_type": "paired_measurement_complete", **comparison})
                comparison_summaries.append(comparison)
                state["completed_pairs"] = run_number
                measured_events = len(pair["baseline"]["events"]) + len(pair["candidate"]["events"])
                state["raw_event_count"] += measured_events
                state["measured_event_count"] += measured_events
                write_json(Path(str(output) + ".status.json"), state)
                emit(f"PAIR {run_number}/12 domain={comparison['domain']} context={comparison['context_length']} layers={shape} arms={comparison['pair_order']} comparison=PASS baseline={comparison['baseline_seconds']:.3f}s candidate={comparison['candidate_seconds']:.3f}s elapsed={time.time()-started:.1f}s")
        state["phases"]["actual_paired_collector_and_durable_persistence"] = {
            "status": "pass", "started_epoch": started, "finished_epoch": time.time(),
            "elapsed_seconds": time.time()-started, "pair_count": len(comparison_summaries),
            "event_count": state["raw_event_count"]}
        state["paired_comparisons"] = comparison_summaries
        _assert_clean(model)
        _assert_parameter_versions(model, initial_parameter_versions)
        state["parameter_versions_unchanged"] = True
        state["status"] = "pass" if (len(comparison_summaries) == 12 and len(native) == 6 and
                                        state["measured_event_count"] == 48 and
                                        state["warmup_event_count"] == 12 and
                                        state["raw_event_count"] == 60) else "incomplete"
        state["full_coverage"] = state["status"] == "pass"
    except BaseException as exc:
        if _is_incomplete_timeout(exc, _TimeLimit):
            state.update(status="incomplete", stop_reason=f"{type(exc).__name__}: {exc}")
            emit(f"STOP INCOMPLETE {state['stop_reason']}")
        else:
            state.update(status="fail", stop_reason=f"{type(exc).__name__}: {exc}")
            emit(f"STOP FAIL {state['stop_reason']}")
    finalization_started = time.monotonic()
    finalization_started_epoch = time.time()
    if events_path.exists():
        persisted = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line]
        state["raw_event_count"] = len(persisted)
        state["measured_event_count"] = sum(row.get("benchmark_kind") == "measured" for row in persisted)
        state["warmup_event_count"] = sum(row.get("benchmark_kind") == "warmup" for row in persisted)
    state.update(finished_epoch=time.time(), elapsed_seconds=time.time()-state["started_epoch"],
                 candidate_mode=CANDIDATE_MODE, baseline_mode=BASELINE_MODE,
                 diagnostics_scope="six selected inputs; one paired observation per cohort/shape; diagnostic only",
                 events_path=events_path.name, timings_path=timings_path.name,
                 progress_path=progress_path.name,
                 event_journal_sha256=file_hash(events_path) if events_path.exists() else None,
                 timing_rows_sha256=file_hash(timings_path) if timings_path.exists() else None,
                 progress_sha256=file_hash(progress_path) if progress_path.exists() else None,
                 scientific_collection_authorized=False)
    state.setdefault("phases", {})["receipt_finalization"] = {
        "status": "pass", "started_epoch": finalization_started_epoch,
        "finished_epoch": time.time(), "elapsed_seconds": time.monotonic()-finalization_started,
        "scope": "journal recount, hashes and receipt metadata assembly; excludes receipt file writes",
        "cap_seconds": PHASE_CAPS["receipt_finalization"],
        "receipt_write_reserve_seconds": RECEIPT_WRITE_RESERVE_SECONDS}
    if not _finalization_within_budget(finalization_started, deadline_epoch,
                                       cap_seconds=PHASE_CAPS["receipt_finalization"]):
        finalization = state["phases"]["receipt_finalization"]
        finalization["status"] = "incomplete"
        state["full_coverage"] = False
        if state["status"] == "pass":
            state["status"] = "incomplete"
            state["stop_reason"] = "receipt finalization cap or absolute-deadline write reserve reached"
        emit(f"STOP INCOMPLETE {state.get('stop_reason', 'receipt finalization cap reached')}")
    write_json(output, state)
    write_json(Path(str(output) + ".status.json"), state)
    emit(f"FINAL {state['status']} pairs={state['completed_pairs']}/12 native_inputs={state['completed_native_inputs']}/6 "
         f"measured_events={state['measured_event_count']}/48 warmup_events={state['warmup_event_count']}/12 "
         f"journal_events={state['raw_event_count']}/60 receipt={output}")
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, nargs=2, required=True,
                        help="sealed 4096 and matched 2048 technical plan directories")
    parser.add_argument("--baseline-source", type=Path, required=True,
                        help="immutable extracted baseline source tree containing src/run_experiment.py")
    parser.add_argument("--selection", type=Path, required=True,
                        help="fresh findings JSON or immutable slim six-input benchmark specification")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (not math.isfinite(args.deadline_epoch) or args.deadline_epoch <= time.time() or
            args.deadline_epoch - time.time() > 1200):
        parser.error("--deadline-epoch must be finite, in the future, and within the 20-minute cap")
    return 0 if run_benchmark(args.plans, args.baseline_source, args.selection,
                              args.model_dir, args.deadline_epoch, args.output).get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
