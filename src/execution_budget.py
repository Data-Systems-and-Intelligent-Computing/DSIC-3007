"""Fail-closed complete-workflow time and cost accounting.

The planner consumes exact planned input IDs, synchronized timing rows and
explicit non-inference allowances. It keeps the main projection separate from
the complete remaining workflow and never treats missing evidence as zero.
"""

from collections import defaultdict
import math


_COMPONENTS = ("verification", "benchmark", "setup_load", "transfer_write",
               "host_wait", "storage_usd")
_SHARED_TIMINGS = ("synchronized_input_total_seconds", "shared_seconds",
                   "cleanup_seconds")
_REQUIRED_STAGES = {
    "verification", "benchmark", "calibration_4096", "calibration_2048",
    "probe_4096", "probe_2048", "pilot", "main", "sensitivity_2048",
    "finalization",
}
_HASH_FIELDS = ("measurement_manifest_sha256", "source_plan_manifest_sha256",
                "executor_sha256")


def _finite_nonnegative(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0



def _valid_sha256(value):
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return any(char != "0" for char in value.lower())


def _unique_input_timings(rows):
    """Collapse layer-repeated shared timing fields by physical input ID."""
    by_input = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict) or not row.get("input_id"):
            raise ValueError("every timing row needs a unique input_id")
        if type(row.get("layer")) is not int:
            raise ValueError("every timing row needs an integer layer")
        if row.get("evidence_status") != "measured" or not row.get("event_id"):
            raise ValueError("timing rows need measured evidence and event identity")
        if any(not _valid_sha256(row.get(name)) for name in _HASH_FIELDS):
            raise ValueError("timing row provenance hashes are incomplete or invalid")
        by_input[row["input_id"]].append(row)
    result = {}
    for input_id, events in by_input.items():
        layers = [event["layer"] for event in events]
        if len(layers) != len(set(layers)):
            raise ValueError(f"duplicate timing event for {input_id}")
        shared = {}
        for name in _SHARED_TIMINGS:
            values = [event.get(name) for event in events if name in event]
            if values and any(value != values[0] for value in values[1:]):
                raise ValueError(f"conflicting duplicated {name} for {input_id}")
            shared[name] = values[0] if values else None
        for name in (*_HASH_FIELDS, "event_id"):
            values = [event.get(name) for event in events]
            if len(set(values)) != len(values) and name == "event_id":
                # Event IDs differ by layer. This field is retained as an
                # immutable reference set, while hashes must stay identical.
                continue
            if name != "event_id" and len(set(values)) != 1:
                raise ValueError(f"conflicting {name} for {input_id}")
        effects = {}
        for event in events:
            value = event.get("effect_seconds")
            if value is None:
                if event.get("kind") in ("calibration_window", "probe_routing"):
                    value = 0.0
                else:
                    continue
            if not _finite_nonnegative(value):
                raise ValueError(f"invalid effect_seconds for {input_id}")
            effects[str(event["layer"])] = value
        result[input_id] = {**shared, "effect_seconds_by_layer": effects,
                            "layers_measured": sorted(layers)}
    return result


def _input_seconds(profile, layers):
    required = ("synchronized_input_total_seconds", "shared_seconds", "cleanup_seconds")
    missing = [name for name in required if profile.get(name) is None]
    if missing:
        return None, ["missing_timing:" + name for name in missing]
    if any(not _finite_nonnegative(profile[name]) for name in required):
        return None, ["invalid_shared_timing"]
    effects = profile["effect_seconds_by_layer"]
    absent = [str(layer) for layer in layers if str(layer) not in effects]
    if absent:
        return None, ["missing_effect_timing:layer_" + layer for layer in absent]
    seconds = sum(profile[name] for name in required[1:]) + sum(effects[str(layer)] for layer in layers)
    # Preserve unclassified synchronized elapsed time instead of pretending the
    # measured component breakdown explains the whole physical execution.
    seconds = max(seconds, profile["synchronized_input_total_seconds"])
    return seconds, []


def document_position_counts(eligible_positions, primary_positions=64,
                             secondary_positions=16):
    """Return frozen primary and each secondary-layer position counts."""
    if any(type(v) is not int or v < 0 for v in
           (eligible_positions, primary_positions, secondary_positions)):
        raise ValueError("position counts must be nonnegative integers")
    return {"primary_layer_15": min(eligible_positions, primary_positions),
            "secondary_each_layer": min(eligible_positions, secondary_positions)}


def plan_execution_budget(*, stages, timing_rows, hourly_rate_usd,
                          remaining_budget_usd, remaining_gpu_hours=None,
                          remaining_time_hours,
                          reserve_fraction=0.20, non_gpu_cost_usd=0.0,
                          pilot_sufficient=True, calibration_escalation=False,
                          pilot_escalation=False):
    """Assess measured workload against remaining dollars and GPU-hours.

    `stages` maps stage names to `{input_ids, required, components}`. Each
    component value is a mapping with `gpu_seconds`, `wall_seconds`, and `usd`;
    every value must be explicit (zero is valid only when evidenced). Timing
    rows may repeat shared fields for layer events; these are counted once per
    unique `input_id`, while layer effect timings are counted per layer.
    """
    if not isinstance(stages, dict) or not stages:
        raise ValueError("workflow stages are required")
    timings = _unique_input_timings(timing_rows)
    limits = {"remaining_budget_usd": remaining_budget_usd,
              "remaining_time_hours": remaining_time_hours,
              "hourly_rate_usd": hourly_rate_usd,
              "non_gpu_cost_usd": non_gpu_cost_usd,
              "reserve_fraction": reserve_fraction}
    unknown_limits = [key for key, value in limits.items() if value is None]
    invalid_limits = [key for key, value in limits.items()
                      if value is not None and not _finite_nonnegative(value)]
    if reserve_fraction >= 1:
        invalid_limits.append("reserve_fraction_must_be_lt_1")
    if invalid_limits:
        raise ValueError("invalid budget limits: " + ", ".join(invalid_limits))
    if unknown_limits:
        return {
            "status": "unknown", "feasibility_pass": False,
            "main_gpu_hours": None, "total_gpu_hours": None,
            "main_cost_usd": None, "total_cost_usd": None,
            "remaining_budget_usd": remaining_budget_usd,
            "remaining_gpu_hours": remaining_gpu_hours,
            "remaining_time_hours": remaining_time_hours,
            "reserve_fraction": reserve_fraction,
            "deduplicated_physical_inputs": 0,
            "blockers": ["unknown_limit:" + name for name in unknown_limits],
        }

    reports, unknown = {}, []
    absent_stages = sorted(_REQUIRED_STAGES - set(stages))
    if absent_stages:
        unknown.extend("missing_required_stage:" + name for name in absent_stages)
    if "main" in stages:
        main_spec = stages["main"]
        if set(main_spec.get("domains", ())) != {"scientific", "code", "c4"}:
            unknown.append("main_domain_set_must_include_scientific_code_and_c4")
        if main_spec.get("context_length") != 4096:
            unknown.append("main_context_must_be_4096")
        measured_layers = {layer for values in main_spec.get("layers_by_input", {}).values()
                           for layer in values}
        if measured_layers != {0, 7, 15}:
            unknown.append("main_must_cover_layers_0_7_15")
    if "sensitivity_2048" in stages:
        sensitivity_spec = stages["sensitivity_2048"]
        if sensitivity_spec.get("context_length") != 2048:
            unknown.append("sensitivity_context_must_be_2048")
        if set(sensitivity_spec.get("domains", ())) != {"scientific", "code", "c4"}:
            unknown.append("sensitivity_domain_set_must_include_scientific_code_and_c4")
    for name in ("verification", "benchmark"):
        if name in stages and set(stages[name].get("contexts", ())) != {2048, 4096}:
            unknown.append(name + "_must_cover_both_contexts")
    for name, context in (("calibration_4096", 4096), ("calibration_2048", 2048),
                          ("probe_4096", 4096), ("probe_2048", 2048), ("pilot", 4096)):
        if name in stages and stages[name].get("context_length") != context:
            unknown.append(f"{name}_context_must_be_{context}")
    globally_counted = set()
    union_layers = defaultdict(set)
    for stage in stages.values():
        for input_id in stage.get("input_ids", []):
            union_layers[input_id].update(stage.get("layers_by_input", {}).get(input_id, ()))
    main_seconds = total_seconds = 0.0
    main_wall_seconds = total_wall_seconds = 0.0
    main_non_gpu_usd, total_non_gpu_usd = 0.0, non_gpu_cost_usd
    main_known = not any(item.startswith("main_") for item in unknown)
    for stage_name, stage in stages.items():
        if not isinstance(stage, dict) or not isinstance(stage.get("input_ids"), list):
            raise ValueError(f"stage {stage_name} needs an explicit input_ids list")
        stage_inputs = set(stage["input_ids"])
        if len(stage_inputs) != len(stage["input_ids"]):
            raise ValueError(f"duplicate input ID in stage {stage_name}")
        main = stage_name == "main"
        stage_seconds = 0.0
        stage_unknown = []
        if stage_name in _REQUIRED_STAGES and not stage_inputs:
            stage_unknown.append("empty_required_stage")
        stage_seen = set()
        for input_id in sorted(stage_inputs):
            stage_seen.add(input_id)
            profile = timings.get(input_id)
            if profile is None:
                stage_unknown.append(f"missing_timing:{input_id}")
                continue
            seconds, issues = _input_seconds(profile, stage.get("layers_by_input", {}).get(input_id, ()))
            if issues:
                stage_unknown.extend(f"{input_id}:{issue}" for issue in issues)
            else:
                stage_seconds += seconds
        components = stage.get("components")
        if not isinstance(components, dict):
            stage_unknown.append("missing_stage_components")
            components = {}
        component_totals = {"gpu_seconds": 0.0, "wall_seconds": 0.0, "usd": 0.0}
        for component in _COMPONENTS:
            values = components.get(component)
            if not isinstance(values, dict):
                stage_unknown.append("missing_component:" + component)
                continue
            for field in component_totals:
                value = values.get(field)
                if not _finite_nonnegative(value):
                    stage_unknown.append(f"unknown_component:{component}.{field}")
                else:
                    component_totals[field] += value
        all_known = not stage_unknown
        if all_known:
            stage_seconds += component_totals["gpu_seconds"]
            stage_wall_seconds = stage_seconds + component_totals["wall_seconds"]
            total_seconds += component_totals["gpu_seconds"]
            total_wall_seconds += component_totals["gpu_seconds"] + component_totals["wall_seconds"]
            total_non_gpu_usd += component_totals["usd"]
            if main:
                main_seconds += stage_seconds
                main_wall_seconds += stage_wall_seconds
                main_non_gpu_usd += component_totals["usd"]
        elif main:
            main_known = False
        unknown.extend(f"{stage_name}:{item}" for item in stage_unknown)
        reports[stage_name] = {
            "input_count": len(stage_inputs), "gpu_seconds": stage_seconds if all_known else None,
            "wall_seconds_components": component_totals["wall_seconds"] if all_known else None,
            "component_seconds": component_totals["gpu_seconds"] if all_known else None,
            "known": all_known, "unknowns": stage_unknown,
        }

    # Physical input work is the union of requested layers, independent of stage
    # ordering. A later nested budget may add a layer but never repeat shared work.
    for input_id, layers in union_layers.items():
        globally_counted.add(input_id)
        profile = timings.get(input_id)
        if profile is None:
            continue  # the stage above already records the missing evidence
        seconds, issues = _input_seconds(profile, layers)
        if not issues:
            total_seconds += seconds
            total_wall_seconds += seconds

    main_gpu_hours = main_seconds / 3600 if main_known else None
    total_gpu_hours = total_seconds / 3600 if not unknown else None
    main_elapsed_hours = main_wall_seconds / 3600 if main_known else None
    total_elapsed_hours = total_wall_seconds / 3600 if not unknown else None
    main_cost = (main_gpu_hours * hourly_rate_usd + main_non_gpu_usd) if main_gpu_hours is not None else None
    total_cost = (total_gpu_hours * hourly_rate_usd + total_non_gpu_usd) if total_gpu_hours is not None else None
    feasible = (not unknown and pilot_sufficient and main_cost is not None and total_cost is not None and
                total_cost * (1 + reserve_fraction) <= remaining_budget_usd and
                total_elapsed_hours * (1 + reserve_fraction) <= remaining_time_hours)
    if remaining_gpu_hours is not None:
        feasible = (feasible and total_gpu_hours is not None and
                    total_gpu_hours * (1 + reserve_fraction) <= remaining_gpu_hours)
    blockers = list(unknown)
    if not pilot_sufficient:
        blockers.append("pilot_extension_or_insufficiency_unresolved")
    if calibration_escalation or pilot_escalation:
        blockers.append("required_escalation_not_included_in_workload")
        feasible = False
    elif total_cost is not None and total_cost * (1 + reserve_fraction) > remaining_budget_usd:
        blockers.append("remaining_dollar_budget_exceeded_with_reserve")
    if (remaining_gpu_hours is not None and total_gpu_hours is not None and
            total_gpu_hours * (1 + reserve_fraction) > remaining_gpu_hours):
        blockers.append("remaining_gpu_hour_limit_exceeded_with_reserve")
    if total_elapsed_hours is not None and total_elapsed_hours * (1 + reserve_fraction) > remaining_time_hours:
        blockers.append("remaining_elapsed_time_limit_exceeded_with_reserve")
    return {
        "status": "feasible" if feasible else ("unknown" if unknown else "infeasible"),
        "feasibility_pass": bool(feasible), "main_gpu_hours": main_gpu_hours,
        "total_gpu_hours": total_gpu_hours, "main_elapsed_hours": main_elapsed_hours,
        "total_elapsed_hours": total_elapsed_hours, "main_cost_usd": main_cost,
        "total_cost_usd": total_cost, "remaining_budget_usd": remaining_budget_usd,
        "remaining_gpu_hours": remaining_gpu_hours, "hourly_rate_usd": hourly_rate_usd,
        "remaining_time_hours": remaining_time_hours,
        "reserve_fraction": reserve_fraction, "main_includes_c4": "c4" in stages.get("main", {}).get("domains", ()),
        "main_components": reports, "deduplicated_physical_inputs": len(globally_counted),
        "pilot_sufficient": bool(pilot_sufficient), "calibration_escalation_required": bool(calibration_escalation),
        "pilot_escalation_required": bool(pilot_escalation), "blockers": blockers,
    }


def forecast_execution_budget(*, workload, profiles, overhead, accounting,
                              calibration_budget=500, pilot_budget=80,
                              pilot_sufficient=False, native_fidelity_pass=False,
                              reserve_fraction=0.20):
    """Forecast the unchanged workload from measured domain/context profiles.

    Profiles are individual synchronized measurements, not advertised speeds.
    Each cohort uses its largest measured component and unexplained elapsed
    overhead. This remains an extrapolation; a forecast is never a peak bound or
    evidence of achieved pilot/main precision.
    """
    if calibration_budget not in (250, 500) or pilot_budget not in (20, 40, 80):
        raise ValueError("forecast must preserve the accepted calibration/pilot ladders")
    if not _finite_nonnegative(reserve_fraction) or reserve_fraction >= 1:
        raise ValueError("invalid reserve fraction")
    buckets = defaultdict(list)
    executor_hashes = set()
    seen = set()
    for row in profiles:
        if row.get("evidence_status") != "measured":
            raise ValueError("forecast profiles require measured evidence")
        if any(not _valid_sha256(row.get(name)) for name in _HASH_FIELDS):
            raise ValueError("forecast profile provenance is incomplete")
        if not row.get("input_id") or not row.get("event_ids"):
            raise ValueError("forecast profiles require physical input/event identity")
        identity = (row["measurement_manifest_sha256"], row["input_id"])
        if identity in seen:
            raise ValueError("duplicate physical forecast profile")
        seen.add(identity)
        executor_hashes.add(row["executor_sha256"])
        kind = row.get("kind")
        if kind not in ("effects", "calibration", "probe"):
            raise ValueError("unknown forecast profile kind")
        if kind in ("calibration", "probe") and set(row.get("layers_measured", [])) != {0, 7, 15}:
            raise ValueError("routing profiles must measure all three accepted layers together")
        required = ("shared_seconds", "cleanup_seconds", "persistence_seconds", "synchronized_input_total_seconds")
        if any(not _finite_nonnegative(row.get(key)) for key in required):
            raise ValueError("forecast profile has unmeasured or invalid timings")
        effects = row.get("effect_seconds_by_layer", {})
        layers = tuple(int(layer) for layer in effects) if kind == "effects" else ()
        if kind == "effects" and (15 not in layers or not set(layers) <= {0, 7, 15}):
            raise ValueError("effects profile must retain primary layer 15 and only accepted layers")
        if any(not _finite_nonnegative(effects.get(str(layer))) for layer in layers):
            raise ValueError("effects forecast has invalid measured layer timings")
        parts = sum(row[key] for key in required[:3]) + sum(effects[str(layer)] for layer in layers)
        item = dict(row)
        item["unclassified_seconds"] = max(0, row["synchronized_input_total_seconds"] - parts)
        buckets[row.get("domain"), row.get("context_length"), kind].append(item)
    if len(executor_hashes) > 1:
        raise ValueError("forecast cannot pool different GPU/runtime/executor identities")
    blockers, stages = [], {}

    def project(name, context, kind, counts):
        seconds, missing = 0.0, []
        references = []
        for domain, count in counts.items():
            cohort = buckets.get((domain, context, kind), [])
            if kind == "effects":
                needed = set(count["layer_events"])
                cohort = [row for row in cohort if needed <= set(row["effect_seconds_by_layer"])]
            if not cohort:
                missing.append(f"missing_profile:{domain}/{context}/{kind}")
                continue
            def largest(field):
                return max(row[field] for row in cohort)
            shared = sum(largest(field) for field in ("shared_seconds", "cleanup_seconds", "persistence_seconds", "unclassified_seconds"))
            seconds += count["windows"] * shared
            if kind == "effects":
                for layer, n in count["layer_events"].items():
                    seconds += n * max(row["effect_seconds_by_layer"][str(layer)] for row in cohort)
            references.extend({"measurement_manifest_sha256": row["measurement_manifest_sha256"],
                               "source_plan_manifest_sha256": row["source_plan_manifest_sha256"],
                               "input_id": row["input_id"], "event_ids": row["event_ids"]} for row in cohort)
        blockers.extend(f"{name}:{issue}" for issue in missing)
        stages[name] = {"gpu_seconds": None if missing else seconds,
                        "counts": counts, "source_measurements": references}

    for name, source, context in (("main", "minimum_main_candidate", 4096),
                                   ("sensitivity_2048", "matched_sensitivity_for_minimum_candidate", 2048)):
        counts = {}
        for domain in ("scientific", "code", "c4"):
            row = workload[source][domain]
            counts[domain] = {"windows": row["layer15_positions"], "layer_events": {
                "15": row["layer15_positions"], "0": row["layer0_positions"], "7": row["layer7_positions"]}}
        project(name, context, "effects", counts)
    other = workload["other_frozen_workloads"]
    pilot_windows = other.get("pilot_windows_by_domain_and_budget", {}).get(str(pilot_budget))
    if pilot_windows is None:
        blockers.append("pilot:missing_whole_group_domain_census")
    else:
        project("pilot", 4096, "effects", {d: {"windows": n, "layer_events": {"15": n}} for d, n in pilot_windows.items()})
    for context in (4096, 2048):
        project(f"calibration_{context}", context, "calibration", {
            domain: {"windows": calibration_budget * 3, "layer_events": {}}
            for domain in ("scientific", "code", "c4")})
        # Native fixed probes preserve four positions on 32 documents/domain.
        project(f"probe_{context}", context, "probe", {
            domain: {"windows": 128, "layer_events": {}}
            for domain in ("scientific", "code", "c4")})
    overhead_gpu = overhead_wall = overhead_usd = 0.0
    # Nested budgets reuse inference but still read/validate/copy/persist every
    # consumer event. Require a separate measured/bounded allowance for that work.
    for name in (*_COMPONENTS, "nested_reuse", "final_cpu_analysis"):
        values = overhead.get(name, {})
        if values.get("evidence_status") not in ("measured", "bounded"):
            blockers.append(f"overhead:{name}:unknown_evidence")
        for key in ("gpu_seconds", "wall_seconds", "usd"):
            if not _finite_nonnegative(values.get(key)):
                blockers.append(f"overhead:{name}:unknown_{key}")
        if all(_finite_nonnegative(values.get(key)) for key in ("gpu_seconds", "wall_seconds", "usd")):
            overhead_gpu += values["gpu_seconds"]
            overhead_wall += max(values["gpu_seconds"], values["wall_seconds"])
            overhead_usd += values["usd"]
    rate, dollars, hours = (accounting.get(key) for key in (
        "hourly_rate_usd", "remaining_budget_usd", "remaining_time_hours"))
    if not all(_finite_nonnegative(value) for value in (rate, dollars, hours)):
        blockers.append("accounting:unknown_rate_or_remaining_limit")
    if not accounting.get("cash_boundary_confirmed") or not accounting.get("time_boundary_confirmed"):
        blockers.append("accounting:unreconciled_remaining_boundary")
    if accounting.get("price_evidence") != "live_quote":
        blockers.append("accounting:missing_live_quote")
    main_seconds = stages["main"]["gpu_seconds"]
    runtime_unknown = any(stage["gpu_seconds"] is None for stage in stages.values()) or "pilot" not in stages
    overhead_unknown = any(issue.startswith("overhead:") for issue in blockers)
    total_seconds = None if runtime_unknown or overhead_unknown else sum(stage["gpu_seconds"] for stage in stages.values()) + overhead_gpu
    wall_seconds = None if total_seconds is None else total_seconds - overhead_gpu + overhead_wall
    main_cost = main_seconds * rate / 3600 if main_seconds is not None and _finite_nonnegative(rate) else None
    total_cost = total_seconds * rate / 3600 + overhead_usd if total_seconds is not None and _finite_nonnegative(rate) else None
    if not pilot_sufficient:
        blockers.append("scientific:sufficient_pilot_and_actual_allocation_pending")
    if not native_fidelity_pass:
        blockers.append("validation:candidate_native_full_acceptance_pending")
    exceeded = []
    if total_cost is not None and _finite_nonnegative(dollars) and total_cost * (1 + reserve_fraction) > dollars:
        exceeded.append("remaining_dollar_budget_exceeded_with_reserve")
    if wall_seconds is not None and _finite_nonnegative(hours) and wall_seconds * (1 + reserve_fraction) > hours * 3600:
        exceeded.append("remaining_elapsed_time_limit_exceeded_with_reserve")
    blockers.extend(exceeded)
    return {"status": "infeasible" if exceeded else ("unknown" if blockers else "projected_feasible"),
            "feasibility_pass": not blockers, "evidence_scope": "measured_profile_extrapolation",
            "calibration_budget": calibration_budget, "pilot_budget": pilot_budget,
            "main_gpu_hours": main_seconds / 3600 if main_seconds is not None else None,
            "total_gpu_hours": total_seconds / 3600 if total_seconds is not None else None,
            "total_elapsed_hours": wall_seconds / 3600 if wall_seconds is not None else None,
            "main_cost_usd": main_cost, "total_cost_usd": total_cost,
            "reserve_fraction": reserve_fraction, "stages": stages, "blockers": blockers,
            "executor_sha256": next(iter(executor_hashes), None),
            "nested_reuse_workload": {
                "calibration_cached_windows_both_contexts":
                    2 * 3 * 3 * (100 if calibration_budget == 250 else 100 + 250),
                "pilot_cached_windows": sum(
                    other["pilot_windows_by_nested_budget"][str(budget)]
                    for budget in (20, 40) if budget < pilot_budget),
                "cost_evidence": overhead.get("nested_reuse", {}),
            },
            "assumptions": ["cohort maxima extrapolate to unseen windows; they are not a proven upper bound",
                            "minimum main candidate must be replaced by sufficient-pilot allocation",
                            "calibration/pilot escalation is explicitly included at selected ladder ceilings"]}


def profiles_from_measurements(runs):
    """Read durable measured timing sidecars from verified sealed runs, without GPU work."""
    from pathlib import Path
    from .artifacts import digest, file_digest, read_rows, verify_bundle
    profiles, excluded = [], []
    for run in runs:
        run = Path(run)
        manifest = verify_bundle(run, "measurements")
        plan = verify_bundle(run / "plan", "experiment-plan")
        sidecar = run / "input-timings.jsonl"
        if not sidecar.exists():
            excluded.append({"run": str(run), "reason": "no measured persistence timing sidecar"})
            continue
        inputs = {row["input_id"]: {"layers": row["layers"], "domain": row["domain"],
                  "context_length": row["context_length"]} for row in read_rows(run / "plan" / "inputs.jsonl")}
        for timing in read_rows(sidecar):
            planned = inputs.get(timing.get("input_id"))
            if planned is None or any(timing.get(key) != planned[key] for key in ("domain", "context_length")):
                raise ValueError("timing sidecar does not match the frozen input plan")
            # Hits and partial resume windows cannot stand in for a complete
            # physical measurement profile. Their actual costs remain in the run.
            if (timing.get("cache_hit_layers") or
                    set(map(int, timing.get("cache_miss_layers", []))) != set(planned["layers"])):
                excluded.append({"run": str(run), "input_id": timing["input_id"],
                                 "reason": "cached or partial measurement; not a complete profile"})
                continue
            profiles.append({**timing, "layers_measured": planned["layers"],
                "event_ids": [digest([timing["input_id"], layer]) for layer in planned["layers"]],
                "measurement_manifest_sha256": file_digest(run / "manifest.json"),
                "source_plan_manifest_sha256": file_digest(run / "plan" / "manifest.json"),
                "executor_sha256": digest(manifest["metadata"]["executor"]),
                "measurement_stage": plan["metadata"]["stage"]})
    return {"profiles": profiles, "excluded": excluded,
            "scope": "measured sidecars; omitted observations cannot be assumed free"}
