"""Offline replay and inference from complete scientific measurement bundles."""

from collections import defaultdict
import json
import math
from pathlib import Path
import struct

import numpy as np

from .artifacts import (canonical, digest, file_digest, read_json, read_rows, require,
                        priority, validate_measurement, verify_bundle, write_rows)
from .routing_stats import (average_seed_tables, calibration_document_tables,
                            causal_discrimination, decode_window_weights,
                            m1_tables, select_event_metrics, stable_tie_order)


def paired_difference(cd_domain, cd_global):
    a = np.asarray(cd_domain, dtype=float)
    b = np.asarray(cd_global, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Paired arrays must have identical shape")
    return a - b


def bootstrap_mean_ci(values, n_boot=5000, seed=42, ci=0.95):
    """Compatibility document bootstrap helper."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(values, size=len(values), replace=True)))
             for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return float(np.mean(values)), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def _expected_plan_events(plan_dir):
    inputs = list(read_rows(Path(plan_dir) / "inputs.jsonl"))
    expected = {}
    for row in inputs:
        for layer in row["layers"]:
            event_id = digest([row["input_id"], int(layer)])
            if event_id in expected:
                raise ValueError("duplicate planned event identity")
            expected[event_id] = dict(row, layer=int(layer))
    return expected


def _validate_observation_identity(row, planned):
    fields = ("input_id", "layer", "document_id", "dataset", "domain", "revision",
              "source_group", "predictor_position", "window_start", "window_end",
              "target_id", "context_length", "token_ids_sha256", "sampling")
    if any(field not in planned or field not in row or row[field] != planned[field]
           for field in fields):
        raise ValueError("measurement observation identity differs from frozen plan")


def _load_frozen_tables(calibration_dir):
    root = Path(calibration_dir)
    candidates = [root / "frozen-tables.json", root / "tables.json"]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise ValueError("frozen calibration bundle lacks frozen-tables.json")
    payload = read_json(path)
    if payload.get("status") != "frozen":
        raise ValueError("calibration tables are not marked frozen")
    tables = payload["tables"]
    m1 = {int(layer): np.asarray(values, dtype=np.float64)
          for layer, values in tables["M1"].items()}
    domains = {domain: {int(layer): np.asarray(values, dtype=np.float64)
                        for layer, values in by_layer.items()}
               for domain, by_layer in tables.items() if domain != "M1"}
    if set(domains) != {"c4", "scientific", "code"}:
        raise ValueError("frozen calibration requires M2 tables for C4, scientific, and code")
    if any(set(values) != set(m1) for values in domains.values()):
        raise ValueError("frozen M1 and M2 tables must cover the same layers")
    for table in [*m1.values(), *(v for d in domains.values() for v in d.values())]:
        if table.shape != (64,) or not np.isfinite(table).all() or (table < 0).any():
            raise ValueError("frozen routing tables must contain 64 finite nonnegative values")
        if not math.isclose(float(table.sum()), 8.0, rel_tol=0, abs_tol=1e-8):
            raise ValueError("utilization frequency tables must sum to eight")
    for layer, global_table in m1.items():
        recomputed = np.mean([domains[domain][layer] for domain in ("c4", "scientific", "code")], axis=0)
        if not np.allclose(global_table, recomputed, rtol=0, atol=1e-10):
            raise ValueError(f"frozen M1 does not equal the equal-domain M2 mixture at layer {layer}")
    return m1, domains, payload


def _group_variance(documents):
    """User-specified cluster robust variance for a document-weighted mean."""
    if not documents:
        return float("nan"), float("nan"), 0, 0
    groups = defaultdict(list)
    for row in documents:
        group = row["source_group"]
        if not group:
            raise ValueError("source group is required for source-cluster variance")
        groups[group].append(float(row["delta"]))
    n, g = len(documents), len(groups)
    mean = float(np.mean([row["delta"] for row in documents]))
    if g < 2:
        return mean, float("nan"), g, n
    sumsq = sum((sum(values) - len(values) * mean) ** 2 for values in groups.values())
    variance = (g / (g - 1)) * sumsq / (n * n)
    return mean, float(variance), g, n


def _bootstrap_domain(documents, replicates, seed, confidence):
    """Whole-group bootstrap-t, retaining all documents and draw multiplicities."""
    groups = defaultdict(list)
    for row in documents:
        groups[row["source_group"]].append(float(row["delta"]))
    names = sorted(groups)
    mean, variance, g, n = _group_variance(documents)
    if g < 2 or not math.isfinite(variance):
        return {"mean": mean if math.isfinite(mean) else None, "variance": None, "se": None,
                "groups": g, "documents": n, "valid_replicates": 0,
                "degenerate_replicates": replicates, "ci": None, "bootstrap_t": [],
                "interval_status": "insufficient_independent_groups", "bootstrap_group_slots": [], "bootstrap_valid_flags": []}
    rng = np.random.Generator(np.random.PCG64(seed))
    t_values, means, group_slots, valid_flags = [], [], [], []
    for _ in range(replicates):
        draws = rng.integers(0, g, size=g)
        group_slots.append([names[int(draw)] for draw in draws])
        valid_flags.append(False)
        # Each selected group occurrence is a separate bootstrap slot. Thus a
        # repeatedly selected group contributes with its full multiplicity.
        slot_sums, slot_ns = [], []
        total, total_n = 0.0, 0
        for draw in draws:
            values = groups[names[int(draw)]]
            slot_sums.append(sum(values))
            slot_ns.append(len(values))
            total += sum(values)
            total_n += len(values)
        if total_n == 0 or len(slot_sums) < 2:
            continue
        estimate = total / total_n
        residual_sums = [s - count * estimate for s, count in zip(slot_sums, slot_ns)]
        boot_variance = (g / (g - 1)) * sum(v * v for v in residual_sums) / (total_n * total_n)
        if not math.isfinite(boot_variance) or boot_variance <= 0:
            continue
        t_values.append((estimate - mean) / math.sqrt(boot_variance))
        means.append(estimate)
        valid_flags[-1] = True
    valid = len(t_values)
    result = {"mean": mean, "variance": variance, "se": math.sqrt(variance),
              "groups": g, "documents": n, "valid_replicates": valid,
              "degenerate_replicates": replicates - valid, "ci": None,
              "bootstrap_t": t_values, "bootstrap_means": means,
              "bootstrap_group_slots": group_slots, "bootstrap_valid_flags": valid_flags}
    if valid / replicates >= 0.95 and variance > 0:
        alpha = (1 - confidence) / 2
        q_low, q_high = np.quantile(t_values, [alpha, 1 - alpha])
        se = math.sqrt(variance)
        result["ci"] = [float(mean - q_high * se), float(mean - q_low * se)]
    return result


def _rng_seed(seed, purpose):
    return int(priority(int(seed), purpose, "bootstrap")[:32], 16)


def _paired_primary(documents, replicates, seed, confidence):
    by_domain = defaultdict(list)
    for row in documents:
        if row["domain"] in ("scientific", "code"):
            by_domain[row["domain"]].append(row)
    estimates = {d: _group_variance(by_domain[d]) for d in ("scientific", "code")}
    if any(not by_domain[d] for d in ("scientific", "code")):
        return {"estimate": None, "variance": None, "se": None, "ci": None,
                "valid_replicates": 0, "degenerate_replicates": replicates,
                "interval_status": "missing_primary_domain"}
    theta = 0.5 * estimates["scientific"][0] + 0.5 * estimates["code"][0]
    variance = 0.25 * (estimates["scientific"][1] + estimates["code"][1])
    gmin = min(estimates[d][2] for d in estimates)
    if gmin < 2 or not math.isfinite(variance):
        return {"estimate": theta, "variance": None, "se": None, "ci": None,
                "valid_replicates": 0, "degenerate_replicates": replicates,
                "interval_status": "insufficient_independent_groups_or_variance"}
    group_values = {}
    group_names = {}
    for domain in ("scientific", "code"):
        grouped = defaultdict(list)
        for row in by_domain[domain]:
            grouped[row["source_group"]].append(float(row["delta"]))
        group_names[domain] = sorted(grouped)
        group_values[domain] = [grouped[key] for key in group_names[domain]]
    rng = np.random.Generator(np.random.PCG64(seed))
    ts, group_slots = [], {"scientific": [], "code": []}
    components = []
    valid_flags = []
    for _ in range(replicates):
        estimates_b, variances_b = {}, {}
        drawn_groups = {}
        valid_draw = True
        for domain in ("scientific", "code"):
            groups = group_values[domain]
            g = len(groups)
            draws = rng.integers(0, g, size=g)
            drawn_groups[domain] = [group_names[domain][int(i)] for i in draws]
            sums = [sum(groups[int(i)]) for i in draws]
            counts = [len(groups[int(i)]) for i in draws]
            n_b = sum(counts)
            estimate_b = sum(sums) / n_b if n_b else float("nan")
            if g < 2 or n_b == 0:
                valid_draw = False
                break
            residuals = [s - count * estimate_b for s, count in zip(sums, counts)]
            variance_b = (g / (g - 1)) * sum(x * x for x in residuals) / (n_b * n_b)
            estimates_b[domain], variances_b[domain] = estimate_b, variance_b
        variance_b = 0.25 * (variances_b.get("scientific", 0.0) + variances_b.get("code", 0.0))
        theta_b = 0.5 * (estimates_b.get("scientific", float("nan")) + estimates_b.get("code", float("nan")))
        for domain in ("scientific", "code"):
            group_slots[domain].append(drawn_groups.get(domain, []))
        components.append({d: {"mean": estimates_b.get(d), "variance": variances_b.get(d)}
                           for d in ("scientific", "code")})
        valid_flags.append(False)
        if not valid_draw or not math.isfinite(variance_b) or variance_b <= 0:
            continue
        ts.append((theta_b - theta) / math.sqrt(variance_b))
        valid_flags[-1] = True
    count = len(ts)
    if count / replicates < 0.95 or variance <= 0:
        return {"estimate": theta, "variance": variance, "se": math.sqrt(variance), "ci": None,
                "valid_replicates": count, "degenerate_replicates": replicates - count,
                "interval_status": "fewer_than_95_percent_valid_replicates",
                "bootstrap_components": components,
                "bootstrap_group_slots": group_slots, "bootstrap_valid_flags": valid_flags}
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(ts, [alpha, 1 - alpha])
    se = math.sqrt(variance)
    return {"estimate": theta, "variance": variance, "se": se,
            "ci": [float(theta - hi * se), float(theta - lo * se)],
            "valid_replicates": count, "degenerate_replicates": replicates - count,
            "interval_status": "computed_bootstrap_t", "bootstrap_t": ts,
            "bootstrap_components": components,
            "bootstrap_group_slots": group_slots, "bootstrap_valid_flags": valid_flags}


def _holm(pvalues):
    ordered = sorted(pvalues, key=lambda key: pvalues[key])
    adjusted, running = {}, 0.0
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def analyze_events(events, m1, m2, seed=3007, m0_draws=64):
    """Replay each complete event and return event, document, and group rows."""
    by_doc = defaultdict(list)
    output = []
    for event in events:
        validate_measurement(event)
        effects = {int(e["expert_id"]): float(e["delta_nll"]) for e in event["effects"]}
        selections = select_event_metrics(event, m1, m2, seed, m0_draws)
        cd = {metric: causal_discrimination(selections[metric], effects)
              for metric in ("M1", "M2", "M4")}
        m0_values = [causal_discrimination(pair, effects) for pair in selections["M0"]]
        from .routing_stats import _select_extremes
        mismatch = {domain:list(_select_extremes(by_layer[event["layer"]],event["routing"]["expert_ids"],
                                                 selections["tie_order"]))
                    for domain,by_layer in m2.items() if domain != event["domain"]}
        row = {"event_id": event["event_id"], "input_id": event["input_id"],
               "dataset": event["dataset"], "domain": event["domain"],
               "source_group": event["source_group"], "document_id": event["document_id"],
               "token_position": event.get("predictor_position", event.get("token_position")),
               "target_id": event.get("target_id"),
               "context_length": event["context_length"], "layer": event["layer"],
               "expert_ids": event["routing"]["expert_ids"], "routing_weights": event["routing"]["weights"],
               "routing_weights_fp32_le_hex": event["routing"]["weights_fp32_le_hex"],
               "sampling": event["sampling"], "baseline_nll": event["baseline_nll"],
               "ablated_nll_by_expert": {str(e["expert_id"]): e["ablated_nll"] for e in event["effects"]},
               "delta_nll_by_expert": {str(e["expert_id"]): e["delta_nll"] for e in event["effects"]},
               "selections": {m: selections[m] for m in ("M1", "M2", "M4")},
               "shared_tie_order": selections["tie_order"],
               "mismatched_m2_selections": mismatch,
               "mismatched_m2_cd": {d:causal_discrimination(pair,effects) for d,pair in mismatch.items()},
               "m0_pairs": selections["M0"], "cd": cd,
               "delta_m2_minus_m1": cd["M2"] - cd["M1"],
               "m0_mean": float(np.mean(m0_values)),
               "m0_randomization_variance": float(np.var(m0_values, ddof=1)) if len(m0_values) > 1 else 0.0,
               "m0_randomization_draws": m0_values}
        output.append(row)
        by_doc[event["document_id"], int(event["layer"])].append(row)
    doc_rows = []
    for (doc_id, layer), rows in sorted(by_doc.items()):
        first = rows[0]
        doc_rows.append({"document_id": doc_id, "layer": layer, "dataset": first["dataset"],
                         "domain": first["domain"], "source_group": first["source_group"],
                         "n_events": len(rows), "mean_cd_m1": float(np.mean([r["cd"]["M1"] for r in rows])),
                         "mean_cd_m2": float(np.mean([r["cd"]["M2"] for r in rows])),
                         "mean_cd_m4": float(np.mean([r["cd"]["M4"] for r in rows])),
                         "mean_delta": float(np.mean([r["delta_m2_minus_m1"] for r in rows])),
                         "within_document_position_variance": float(np.var([r["delta_m2_minus_m1"] for r in rows],ddof=1)) if len(rows)>1 else None,
                         "routing_membership_counts": np.bincount(np.asarray([r["expert_ids"] for r in rows]).ravel(),minlength=64).tolist(),
                         "routing_membership_frequencies": (np.bincount(np.asarray([r["expert_ids"] for r in rows]).ravel(),minlength=64)/len(rows)).tolist(),
                         "mean_mismatched_m2_cd": {d:float(np.mean([r["mismatched_m2_cd"][d] for r in rows]))
                                                   for d in first["mismatched_m2_cd"]},
                         "mean_m0": float(np.mean([r["m0_mean"] for r in rows])),
                         "mean_m0_randomization_variance": float(np.mean([r["m0_randomization_variance"] for r in rows]))})
    groups = defaultdict(list)
    for row in doc_rows:
        groups[(row["domain"], row["source_group"], row["layer"])].append(row)
    group_rows = []
    for (domain, group, layer), rows in sorted(groups.items()):
        group_rows.append({"domain": domain, "source_group": group, "layer": layer, "n_documents": len(rows),
                           "mean_delta": float(np.mean([r["mean_delta"] for r in rows])),
                           "document_delta_sum": float(sum(r["mean_delta"] for r in rows))})
    return output, doc_rows, group_rows


def analyze_bundle(bundle_dir, output_dir=None, calibration_dir=None, settings=None,
                   replicates=None):
    """Verify and analyze a raw measurement bundle entirely offline.

    A sealed bundle must contain events.jsonl and a copied plan/inputs.jsonl.
    The explicit expected_event_ids manifest field is checked as well as the
    plan-derived identities; missing events are errors, never complete-case data.
    """
    bundle = Path(bundle_dir)
    manifest = verify_bundle(bundle, "measurements")
    metadata = manifest["metadata"]
    if metadata.get("stage") == "probe":
        raise ValueError("route-only probe measurements must be analyzed through calibration_freeze")
    if metadata.get("stage") not in ("pilot", "main", "sensitivity", "technical"):
        raise ValueError("unsupported measurement bundle stage")
    plan = bundle / "plan"
    if not (plan / "manifest.json").exists():
        raise ValueError("measurement bundle must preserve its sealed plan")
    verify_bundle(plan, "experiment-plan")
    plan_events = _expected_plan_events(plan)
    if "expected_event_ids" not in metadata:
        raise ValueError("measurement manifest lacks expected_event_ids")
    expected = set(metadata["expected_event_ids"])
    if expected != set(plan_events):
        raise ValueError("manifest expected_event_ids disagree with the frozen plan")
    events = list(read_rows(bundle / "events.jsonl"))
    actual = {row["event_id"] for row in events}
    if len(actual) != len(events) or actual != set(plan_events):
        missing, extra = sorted(set(plan_events) - actual), sorted(actual - set(plan_events))
        raise ValueError(f"measurement event coverage differs from plan; missing={len(missing)} extra={len(extra)}")
    for row in events:
        planned = plan_events[row["event_id"]]
        _validate_observation_identity(row, planned)
        if row.get("kind") != "effects" or len(row.get("effects", [])) != 8:
            raise ValueError("analysis requires all eight individual expert effects per event")
    calibration_dir = Path(calibration_dir) if calibration_dir else bundle / "calibration"
    calibration_manifest = verify_bundle(calibration_dir, "calibration")
    m1, m2, frozen = _load_frozen_tables(calibration_dir)
    context_lengths = {int(row["context_length"]) for row in events}
    if context_lengths != {int(frozen["context_length"])}:
        raise ValueError("measurement and frozen calibration context lengths differ")
    effect_plan_meta = metadata.get("plan_metadata", {})
    calibration_meta = calibration_manifest["metadata"]
    calibration_identity = calibration_meta.get("identities", {})
    if metadata.get("executor") != calibration_meta.get("executor"):
        raise ValueError("measurement executor differs from frozen calibration")
    checks = {"source_pool": effect_plan_meta.get("prepared_manifest_sha256"),
              "source_review": effect_plan_meta.get("exact_source_review_sha256",
                                                     effect_plan_meta.get("source_review_sha256")),
              "tokenizer": effect_plan_meta.get("tokenizer_sha256")}
    if any(value is None or calibration_identity.get(key) != value for key, value in checks.items()):
        raise ValueError("measurement and frozen calibration source identities differ")
    if settings is None:
        settings = read_json(bundle / "plan" / "research.json")
    elif settings != read_json(bundle / "plan" / "research.json"):
        raise ValueError("analysis settings differ from the frozen measurement plan")
    analysis_settings = settings["analysis"]
    from .scope import is_path_b, require_scope, scope_identity, validate_plan
    validate_plan(settings, effect_plan_meta, read_rows(bundle / "plan" / "inputs.jsonl"))
    require_scope(settings, calibration_meta)
    require(settings, "analysis.confidence", "analysis.initial_replicates",
            "analysis.minimum_valid_replicate_fraction")
    default_reps = (settings["planning"]["planning_replicates"] if metadata["stage"] == "pilot"
                    else analysis_settings["initial_replicates"])
    reps = int(replicates if replicates is not None else default_reps)
    seed = int(settings["sampling"]["master_seed"])
    event_rows, document_rows, group_rows = analyze_events(
        events, m1, m2, seed, int(settings["controls"]["m0_draws_per_event"]))
    layer = int(settings["sampling"]["primary_layer"])
    # Average repeated positions within document at the primary layer before
    # estimating uncertainty over independent source groups.
    by_document_layer = defaultdict(list)
    for row in event_rows:
        if row["layer"] == layer:
            by_document_layer[row["document_id"]].append(row)
    infer_docs = []
    for doc_id, rows in by_document_layer.items():
        first = rows[0]
        infer_docs.append({"document_id": doc_id, "domain": first["domain"],
                           "source_group": first["source_group"],
                           "delta": float(np.mean([r["delta_m2_minus_m1"] for r in rows])),
                           **{f"cd_{metric.lower()}": float(np.mean([r["cd"][metric] for r in rows]))
                              for metric in ("M1", "M2", "M4")}})
    domain_results = {}
    absolute_results = {}
    bootstrap_draw_rows = []
    for domain in ("scientific", "code"):
        rows = [r for r in infer_docs if r["domain"] == domain]
        if rows:
            dseed = _rng_seed(seed, f"{metadata['stage']}:inference:{domain}")
            domain_results[domain] = _bootstrap_domain(rows, reps, dseed, analysis_settings["confidence"])
            domain_results[domain]["bootstrap_seed"] = dseed
            delta_boot = domain_results[domain]
            observed_t = (delta_boot["mean"] / delta_boot["se"]
                          if delta_boot.get("se") and math.isfinite(delta_boot["se"]) else None)
            if observed_t is not None and delta_boot["valid_replicates"] / reps >= settings["analysis"]["minimum_valid_replicate_fraction"]:
                delta_boot["two_sided_p"] = (1 + sum(abs(value) >= abs(observed_t)
                                                      for value in delta_boot["bootstrap_t"])) / (delta_boot["valid_replicates"] + 1)
            else:
                delta_boot["two_sided_p"] = None
            absolute_results[domain] = {}
            for metric in ("M1", "M2", "M4"):
                metric_rows = [{"source_group": r["source_group"],
                                "delta": r[f"cd_{metric.lower()}"]} for r in rows]
                metric_seed = dseed
                absolute_results[domain][metric] = _bootstrap_domain(
                    metric_rows, reps, metric_seed, analysis_settings["confidence"])
                absolute_results[domain][metric].pop("bootstrap_t", None)
                absolute_results[domain][metric].pop("bootstrap_means", None)
                absolute_results[domain][metric].pop("bootstrap_group_slots", None)
                absolute_results[domain][metric].pop("bootstrap_valid_flags", None)
            for index, (slots, valid) in enumerate(zip(delta_boot.get("bootstrap_group_slots", []),
                                                       delta_boot.get("bootstrap_valid_flags", []))):
                bootstrap_draw_rows.append({"purpose": f"domain:{domain}", "replicate": index,
                                            "group_slots": slots, "valid": valid})
    primary = _paired_primary(infer_docs, reps, _rng_seed(seed, f"{metadata['stage']}:inference:primary"),
                              analysis_settings["confidence"])
    for domain in ("scientific", "code"):
        slots = primary.get("bootstrap_group_slots", {}).get(domain, [])
        valid = primary.get("bootstrap_valid_flags", [])
        for index, groups in enumerate(slots):
            bootstrap_draw_rows.append({"purpose": "primary", "domain": domain,
                                        "replicate": index, "group_slots": groups,
                                        "valid": valid[index] if index < len(valid) else False})
    domain_pvalues = {d: values["two_sided_p"] for d, values in domain_results.items()
                      if d in ("scientific", "code") and values.get("two_sided_p") is not None}
    holm = _holm(domain_pvalues) if len(domain_pvalues) == 2 else None
    c4_rows = [r for r in infer_docs if r["domain"] == "c4"]
    c4_groups = len({r["source_group"] for r in c4_rows})
    c4_diagnostic = {"status":"fewer_than_10_hostnames","documents":len(c4_rows),
                     "source_groups":c4_groups,"confirmatory_test":False}
    if is_path_b(settings) and not c4_rows:
        c4_diagnostic["status"] = "not_collected_deferred_by_path_b"
    if c4_groups >= 10:
        c4_boot = _bootstrap_domain(c4_rows,reps,_rng_seed(seed,f"{metadata['stage']}:inference:c4"),
                                   analysis_settings["confidence"])
        for index,(slots,valid) in enumerate(zip(c4_boot.get("bootstrap_group_slots",[]),
                                                c4_boot.get("bootstrap_valid_flags",[]))):
            bootstrap_draw_rows.append({"purpose":"diagnostic:c4","replicate":index,"group_slots":slots,"valid":valid})
        for key in ("bootstrap_t","bootstrap_means","bootstrap_group_slots","bootstrap_valid_flags"):
            c4_boot.pop(key,None)
        c4_diagnostic.update(c4_boot,status="computed_bootstrap_t" if c4_boot["ci"] else "insufficient_valid_replicates")
    for values in domain_results.values():
        values.pop("bootstrap_t", None)
        values.pop("bootstrap_means", None)
        values.pop("bootstrap_group_slots", None)
        values.pop("bootstrap_valid_flags", None)
    primary.pop("bootstrap_t", None)
    primary.pop("bootstrap_components", None)
    primary.pop("bootstrap_group_slots", None)
    primary.pop("bootstrap_valid_flags", None)
    result = {"schema_version": 1, "bundle": str(bundle), "stage": metadata["stage"],
              "events": len(event_rows), "documents": len({r["document_id"] for r in event_rows}),
              "expected_events": len(plan_events), "calibration_manifest_sha256": file_digest(Path(calibration_dir)/"manifest.json"),
              "frozen_table_evidence": frozen.get("evidence", {}),
              "absolute_discrimination": {
                  domain: {metric: float(np.mean([r[f"mean_cd_{metric.lower()}"] for r in document_rows
                                                  if r["domain"] == domain and r["layer"] == layer]))
                           for metric in ("M1", "M2", "M4")}
                  for domain in sorted({r["domain"] for r in event_rows if r["layer"] == layer})},
              "absolute_discrimination_inference": absolute_results,
              "domain_paired_delta": domain_results,
              "c4_diagnostic_uncertainty": c4_diagnostic,
              "primary_equal_domain": primary,
              "secondary_holm": {"status": "computed" if holm is not None else "insufficient_valid_replicates",
                                 "domains": ["scientific", "code"], "raw_two_sided_p": domain_pvalues,
                                 "adjusted_p_values": holm,
                                 "monte_carlo_rule": "(extreme_valid_draws + 1) / (valid_draws + 1)"},
              "m0_randomization": {"draws_per_event": int(settings["controls"]["m0_draws_per_event"]),
                                   "expected_cd": 0.0,
                                   "mean_cd_by_domain": {d: float(np.mean([r["mean_m0"] for r in document_rows if r["domain"] == d and r["layer"] == layer]))
                                                         for d in sorted({r["domain"] for r in event_rows if r["layer"] == layer})},
                                   "mean_within_event_randomization_variance_by_domain": {
                                       d: float(np.mean([r["mean_m0_randomization_variance"] for r in document_rows if r["domain"] == d and r["layer"] == layer]))
                                       for d in sorted({r["domain"] for r in event_rows if r["layer"] == layer})}},
              "bootstrap": {"replicates": reps, "seed_by_domain": {d: _rng_seed(seed, f"{metadata['stage']}:inference:{d}")
                                                                       for d in domain_results},
                            "primary_seed": _rng_seed(seed, f"{metadata['stage']}:inference:primary"),
                            "interval_rule": "studentized bootstrap quantiles; report requested, valid, and degenerate draws"},
              "unresolved_decisions": (["analysis.c4_hostname_safeguards"] if settings["analysis"].get("c4_hostname_safeguards") is None else [])}
    if metadata["stage"] == "pilot":
        ladder = settings["sampling"].get("pilot_document_ladder", [])
        plan_metadata = metadata.get("plan_metadata", {})
        units = plan_metadata.get("units", {})
        pilot_rows = {}
        from .scope import stage_domains
        for domain in stage_domains(settings, "pilot"):
            rows = [r for r in infer_docs if r["domain"] == domain]
            pilot_inference = domain_results.get(domain)
            if domain == "c4" and rows:
                pilot_inference = _bootstrap_domain(rows, reps, _rng_seed(seed, "pilot:c4"),
                                                    analysis_settings["confidence"])
            grouped = defaultdict(list)
            for row in rows:
                grouped[row["source_group"]].append(row["delta"])
            group_means = [float(np.mean(values)) for values in grouped.values()]
            se = (pilot_inference or {}).get("se")
            ci = (pilot_inference or {}).get("ci")
            half_width = ((ci[1] - ci[0]) / 2) if ci else None
            group_sd = float(np.std(group_means, ddof=1)) if len(group_means) > 1 else None
            trigger = (ci is None or len(grouped) < 10 or
                       (half_width is not None and group_sd is not None and half_width > 0.25 * group_sd))
            current_unit = units.get(domain)
            at_ceiling = bool(ladder and current_unit is not None and current_unit >= max(ladder))
            pilot_rows[domain] = {"documents": len(rows), "source_groups": len(grouped),
                                  "source_group_mean_sd": group_sd, "ci_half_width": half_width,
                                  "unit_budget": current_unit,
                                  "extend": bool(trigger and not at_ceiling),
                                  "insufficient_at_maximum": bool(trigger and at_ceiling)}
        primary_se = primary.get("se")
        primary_half_width = (primary["ci"][1]-primary["ci"][0])/2 if primary.get("ci") else None
        target_half_width = float(settings["planning"]["target_ci_half_width_nats"])
        projection = None
        if primary_half_width is not None and primary_half_width > 0:
            factor = (primary_half_width / target_half_width) ** 2
            projection = {d: int(math.ceil(len({r["source_group"] for r in infer_docs if r["domain"] == d}) * factor))
                          for d in ("scientific", "code")}
        result["pilot_planning"] = {"domains": pilot_rows,
                                    "primary_95_percent_bootstrap_half_width": primary_half_width,
                                    "target_half_width": target_half_width,
                                    "projected_source_groups_per_domain": projection,
                                    "projection_rule": "current group count scaled by squared half-width ratio; planning approximation only",
                                    "main_allocation": "run plan-main with reviewed source pool and measured technical timings"}
        if settings["planning"].get("main_allocation_rule") is None:
            result["unresolved_decisions"].append("planning.main_allocation_rule")
    result["main_minimums"] = {
        "scientific_documents": sum(r["domain"] == "scientific" for r in infer_docs),
        "scientific_documents_required": int(settings["sampling"]["main_min_scientific_documents"]),
        "code_source_groups": len({r["source_group"] for r in infer_docs if r["domain"] == "code"}),
        "code_source_groups_required": int(settings["sampling"]["main_min_code_groups"]),
        "minimums_met": (sum(r["domain"] == "scientific" for r in infer_docs) >= int(settings["sampling"]["main_min_scientific_documents"]) and
                          len({r["source_group"] for r in infer_docs if r["domain"] == "code"}) >= int(settings["sampling"]["main_min_code_groups"]))}
    result["inference_role"] = ("confirmatory_main" if metadata["stage"] == "main" and
        result["main_minimums"]["minimums_met"] and effect_plan_meta.get("main_freeze")
        else "pilot_planning" if metadata["stage"] == "pilot" else "descriptive")
    descriptive = []
    for domain in settings["domains"]:
        for depth in (0,7,15):
            rows = [r for r in document_rows if r["domain"]==domain and r["layer"]==depth]
            if not rows:
                continue
            descriptive.append({"domain":domain,"layer":depth,"documents":len(rows),
                "source_groups":len({r["source_group"] for r in rows}),
                "positions":sum(r["n_events"] for r in rows),
                "document_weighted_mean_delta":float(np.mean([r["mean_delta"] for r in rows])),
                "document_weighted_cd":{m:float(np.mean([r[f"mean_cd_{m.lower()}"] for r in rows]))
                                         for m in ("M1","M2","M4")},
                "document_weighted_mismatched_m2_cd":{d:float(np.mean([r["mean_mismatched_m2_cd"][d] for r in rows]))
                                                       for d in rows[0]["mean_mismatched_m2_cd"]},
                "document_weighted_routing_utilization":np.mean([r["routing_membership_frequencies"] for r in rows],axis=0).tolist(),
                "mean_within_document_position_variance":float(np.mean([r["within_document_position_variance"] for r in rows
                    if r["within_document_position_variance"] is not None])) if any(r["within_document_position_variance"] is not None for r in rows) else None})
    result["descriptive_by_domain_and_layer"]=descriptive
    if is_path_b(settings):
        result["research_scope"] = scope_identity(settings)
        result["deferred_outcomes"] = ["c4_pilot_main", "layers_0_7", "matched_context_2048"]
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=False)
        write_rows(out / "event-analysis.jsonl", event_rows)
        write_rows(out / "document-summary.jsonl", document_rows)
        write_rows(out / "source-group-summary.jsonl", group_rows)
        write_rows(out / "bootstrap-draws.jsonl", bootstrap_draw_rows)
        (out / "analysis.json").write_bytes(canonical(result) + b"\n")
        from .reporting import write_results
        write_results(out,result)
    return {"report": result, "events": event_rows, "documents": document_rows,
            "groups": group_rows}


def analyze_sensitivity(primary_bundle, sensitivity_bundle, primary_calibration,
                        sensitivity_calibration, settings=None, output_dir=None,
                        replicates=None):
    """Compare frozen main observations with matched 2048-token reruns."""
    primary = analyze_bundle(primary_bundle, calibration_dir=primary_calibration,
                             settings=settings, replicates=replicates)
    sensitivity = analyze_bundle(sensitivity_bundle, calibration_dir=sensitivity_calibration,
                                 settings=settings, replicates=replicates)
    p_events, s_events = primary["events"], sensitivity["events"]
    if {int(r["context_length"]) for r in p_events} != {4096} or {int(r["context_length"]) for r in s_events} != {2048}:
        raise ValueError("sensitivity comparison requires matched 4096 and 2048 contexts")
    def key(row):
        return row["document_id"], row["token_position"], row["layer"]
    pmap, smap = {key(row): row for row in p_events}, {key(row): row for row in s_events}
    if len(pmap) != len(p_events) or len(smap) != len(s_events):
        raise ValueError("duplicate document/position/layer in sensitivity inputs")
    if not set(smap) <= set(pmap):
        raise ValueError("sensitivity rows are not a subset of the primary observations")
    matched = []
    for ident, srow in smap.items():
        prow = pmap[ident]
        if (srow["domain"], srow["source_group"], srow["target_id"]) != (
                prow["domain"], prow["source_group"], prow["target_id"]):
            raise ValueError("sensitivity observation does not preserve source identity and target")
        matched.append({"document_id": ident[0], "token_position": ident[1], "layer": ident[2],
                        "domain": srow["domain"], "source_group": srow["source_group"],
                        "primary_event_id": prow["event_id"], "sensitivity_event_id": srow["event_id"],
                        "target_id": srow["target_id"], "primary_cd_m1": prow["cd"]["M1"],
                        "primary_cd_m2": prow["cd"]["M2"],
                        "primary_delta": prow["delta_m2_minus_m1"],
                        "sensitivity_cd_m1": srow["cd"]["M1"],
                        "sensitivity_cd_m2": srow["cd"]["M2"],
                        "sensitivity_delta": srow["delta_m2_minus_m1"],
                        "delta_change": srow["delta_m2_minus_m1"] - prow["delta_m2_minus_m1"]})
    primary_docs = {(r["domain"], r["document_id"]) for r in p_events}
    sensitivity_docs = {(r["domain"], r["document_id"]) for r in s_events}
    primary_code_groups = defaultdict(set)
    sensitivity_code_groups = defaultdict(set)
    for row in p_events:
        primary_code_groups[row["domain"],row["source_group"]].add(row["document_id"])
    for row in s_events:
        sensitivity_code_groups[row["domain"],row["source_group"]].add(row["document_id"])
    partial_code_groups = [g for g in sensitivity_code_groups
                           if sensitivity_code_groups[g] != primary_code_groups.get(g, set())]
    if partial_code_groups:
        raise ValueError("sensitivity subset splits a whole source group")
    selected_docs = {r["document_id"] for r in s_events}
    if set(smap) != {k for k,r in pmap.items() if r["document_id"] in selected_docs}:
        raise ValueError("sensitivity must preserve all evaluated positions and layers of selected documents")
    p_inputs = {r["input_id"]:r for r in read_rows(Path(primary_bundle)/"plan"/"inputs.jsonl")}
    for row in read_rows(Path(sensitivity_bundle)/"plan"/"inputs.jsonl"):
        parent = p_inputs.get(row.get("parent_input_id"))
        if parent is None or row["input_ids"] != parent["input_ids"][-2048:]:
            raise ValueError("sensitivity input is not the exact shorter original context")
    if settings is None:
        settings = read_json(Path(primary_bundle)/"plan"/"research.json")
    by_document = defaultdict(list)
    for row in matched:
        by_document[row["document_id"],row["layer"]].append(row)
    matched_documents = [{"document_id":doc,"layer":layer,"domain":rows[0]["domain"],
        "source_group":rows[0]["source_group"],"delta":float(np.mean([r["delta_change"] for r in rows]))}
        for (doc,layer),rows in sorted(by_document.items())]
    inference_rows = [r for r in matched_documents if r["layer"] == 15]
    reps = replicates or settings["analysis"]["initial_replicates"]
    seed = settings["sampling"]["master_seed"]
    uncertainty = {}
    for domain in ("scientific","code","c4"):
        rows = [r for r in inference_rows if r["domain"]==domain]
        if not rows or (domain=="c4" and len({r["source_group"] for r in rows})<10):
            continue
        result = _bootstrap_domain(rows,reps,_rng_seed(seed,f"sensitivity:change:{domain}"),.95)
        for key_ in ("bootstrap_t","bootstrap_means","bootstrap_group_slots","bootstrap_valid_flags"):
            result.pop(key_,None)
        uncertainty[domain]=result
    paired_primary = _paired_primary(inference_rows,reps,_rng_seed(seed,"sensitivity:change:primary"),.95)
    for key_ in ("bootstrap_t","bootstrap_group_slots","bootstrap_valid_flags","bootstrap_components"):
        paired_primary.pop(key_,None)
    summary = {"matched_events": len(matched), "primary_events": len(p_events),
               "sensitivity_fraction": len(smap) / len(pmap) if pmap else None,
               "code_groups": sum(d=="code" for d,g in sensitivity_code_groups), "partial_source_groups": 0,
               "role":"exploratory_matched_context","domain_change_inference":uncertainty,
               "primary_equal_domain_change":paired_primary,
               "selected_documents_by_domain":{d:sum(r["domain"]==d for r in inference_rows) for d in settings["domains"]},
               "mean_delta_change_by_domain": {
                   d: float(np.mean([r["delta"] for r in inference_rows if r["domain"] == d]))
                   for d in sorted({r["domain"] for r in inference_rows})}}
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=False)
        write_rows(out / "sensitivity-event-comparison.jsonl", matched)
        (out / "sensitivity-analysis.json").write_bytes(canonical(summary) + b"\n")
    return {"summary": summary, "events": matched}


def _choose_calibration_budget(comparisons, budgets, pass_fraction_min=.9):
    frozen_budget, qualifying = None, {}
    for lower,higher in zip(budgets,budgets[1:]):
        relevant = [r for r in comparisons if r["comparison"] == f"budget:{lower}:{higher}" or
                    (r["comparison"].startswith("seed:") and
                     r["comparison"].rsplit(":budget:",1)[-1] == str(higher))]
        by_test = defaultdict(list)
        for r in relevant:
            by_test[r["comparison"]].append(r)
        fractions = {label:sum(r["pass"] for r in items)/len(items) for label,items in by_test.items()}
        qualifying[higher] = {"previous_budget":lower,"comparison_count":len(relevant),
            "pass_fraction":sum(r["pass"] for r in relevant)/len(relevant) if relevant else 0.,
            "passes":sum(r["pass"] for r in relevant),"pass_fraction_by_comparison":fractions}
        if len(fractions)==4 and all(f>=pass_fraction_min for f in fractions.values()) and frozen_budget is None:
            frozen_budget=higher
    return frozen_budget,qualifying


def calibration_freeze(seed_budget_bundles, output=None, domains=("c4", "scientific", "code"),
                       l1_max=0.05, jaccard_min=0.75, pass_fraction_min=0.90,
                       probe_bundle=None, output_dir=None):
    """Build calibration tables from raw seed/budget bundles and report stability.

    `seed_budget_bundles` maps integer document budgets to three sealed
    measurements bundle paths. Rows must preserve one all-position window for
    each document/seed/layer. This function does not select a budget when no
    explicit, consecutive passing pair exists.
    """
    from .artifacts import verify_bundle
    seed_tables, provenance, document_sets = {}, {}, {}
    identity_facts = {}
    document_statistics, probe_selections = [], []
    contexts = set()
    scope = None
    research_scope = None
    expected_layers = None
    budgets = sorted(seed_budget_bundles)
    if budgets not in ([100, 250], [100, 250, 500]):
        raise ValueError("calibration freeze requires consecutive 100/250 or 100/250/500 budgets")
    for budget, paths in sorted(seed_budget_bundles.items()):
        if len(paths) != 3:
            raise ValueError("calibration requires exactly three seed streams per budget")
        budget_seeds = set()
        for path in paths:
            root = Path(path)
            manifest = verify_bundle(root, "measurements")
            settings = read_json(root/"plan"/"research.json")
            from .scope import is_path_b, scope_identity, require_scope, stage_layers, validate_plan
            run_research_scope = scope_identity(settings)
            if research_scope is not None and run_research_scope != research_scope:
                raise ValueError("calibration runs have different research scope/settings")
            research_scope = run_research_scope
            expected_layers = stage_layers(settings, "calibration")
            if is_path_b(settings) and (list(domains) != settings["domains"] or
                    (l1_max, jaccard_min, pass_fraction_min) != (.05, .75, .90)):
                raise ValueError("Path B retains all three calibration domains and frozen stability tolerances")
            require(settings,"calibration.stability_gate_scope")
            run_scope = settings["calibration"]["stability_gate_scope"]
            if scope is not None and scope != run_scope:
                raise ValueError("calibration runs have different stability gate scopes")
            if run_scope != "candidate_seed_pairs_and_consecutive_average":
                raise ValueError("unknown calibration gate scope")
            scope=run_scope
            if manifest["metadata"].get("stage") != "calibration":
                raise ValueError("expected calibration measurement bundle")
            rows = list(read_rows(root / "events.jsonl"))
            planned = _expected_plan_events(root / "plan")
            if "expected_event_ids" not in manifest["metadata"]:
                raise ValueError("calibration manifest lacks expected_event_ids")
            expected_ids = set(manifest["metadata"]["expected_event_ids"])
            if expected_ids != set(planned):
                raise ValueError("calibration expected IDs differ from plan")
            if {row["event_id"] for row in rows} != set(planned) or len(rows) != len(planned):
                raise ValueError("calibration rows do not cover the frozen plan")
            if any(row.get("kind") != "calibration_window" for row in rows):
                raise ValueError("calibration bundles must contain calibration_window rows")
            for row in rows:
                _validate_observation_identity(row, planned[row["event_id"]])
                validate_measurement(row)
                counts = np.bincount(np.asarray(row["window_routing"]["expert_ids"]).ravel(), minlength=64)
                document_statistics.append({"budget": int(budget), "seed": row["sampling"]["seed"],
                    "dataset": row["dataset"], "domain": row["domain"], "source_group": row["source_group"],
                    "document_id": row["document_id"], "layer": row["layer"], "input_id": row["input_id"],
                    "predictor_position": row["predictor_position"], "context_length": row["context_length"],
                    "expert_selection_counts": counts.tolist(), "input_positions": row["context_length"],
                    "expert_utilization": (counts/row["context_length"]).tolist()})
            tables = calibration_document_tables(rows)
            plan_meta = manifest["metadata"].get("plan_metadata", {})
            validate_plan(settings, plan_meta, read_rows(root / "plan" / "inputs.jsonl"))
            seed_value = plan_meta.get("seed", manifest["metadata"].get("seed"))
            seed = int(seed_value if seed_value is not None else rows[0]["sampling"]["seed"])
            if seed not in (3008, 3009, 3010):
                raise ValueError("calibration position stream seed must be 3008, 3009, or 3010")
            if seed in budget_seeds:
                raise ValueError("duplicate calibration seed stream for budget")
            budget_seeds.add(seed)
            docs = {row["document_id"] for row in rows}
            document_sets[int(budget), seed] = docs
            plan_context = int(plan_meta.get("window", read_json(root / "plan" / "research.json")["context"]["primary_window"]))
            contexts.add(plan_context)
            executor = manifest["metadata"].get("executor")
            stable = {"executor": executor,
                      "runtime": executor.get("runtime") if isinstance(executor, dict) else None,
                      "source_pool": plan_meta.get("prepared_manifest_sha256"),
                      "source_review": plan_meta.get("source_review_sha256"),
                      "tokenizer": plan_meta.get("tokenizer_sha256")}
            if any(stable[name] is None for name in stable):
                raise ValueError("calibration bundle lacks required executor or source identity")
            for name, value in stable.items():
                if name in identity_facts and identity_facts[name] != value:
                    raise ValueError(f"calibration bundles differ in {name} identity")
                identity_facts[name] = value
            budget_units = plan_meta.get("units", {})
            for domain in domains:
                actual_documents = len({row["document_id"] for row in rows if row["domain"] == domain})
                if actual_documents != int(budget) or budget_units.get(domain) != int(budget):
                    raise ValueError(f"calibration budget document count mismatch for {domain}")
            if set(tables) != {(seed, domain, layer) for domain in domains for layer in expected_layers}:
                raise ValueError("calibration must cover every domain and scoped layer")
            for (row_seed, domain, layer), table in tables.items():
                if row_seed != seed:
                    raise ValueError("calibration rows mix seed streams")
                seed_tables[int(budget), seed, domain, layer] = table
            provenance[int(budget), seed] = file_digest(root/"manifest.json")
        if budget_seeds != {3008, 3009, 3010}:
            raise ValueError("each calibration budget must contain streams 3008, 3009, and 3010")
        expected_docs = document_sets[int(budget), 3008]
        if any(document_sets[int(budget), seed] != expected_docs for seed in (3009, 3010)):
            raise ValueError("calibration seed streams must use the same documents")
    for lower, higher in zip(sorted(seed_budget_bundles), sorted(seed_budget_bundles)[1:]):
        for seed in (3008, 3009, 3010):
            if not document_sets[int(lower), seed] <= document_sets[int(higher), seed]:
                raise ValueError("calibration document budgets must be nested")
    if len(contexts) != 1:
        raise ValueError("calibration bundles mix context lengths")
    if not contexts <= {2048,4096}:
        raise ValueError("calibration context must be 4096 or matched 2048")
    if probe_bundle is None:
        raise ValueError("fixed probe measurements are required to evaluate calibration")
    comparisons = []
    budgets = sorted(seed_budget_bundles)
    seeds = sorted({seed for _, seed, _, _ in seed_tables})
    table_keys = sorted({(domain, layer) for _, _, domain, layer in seed_tables})
    def add_comparison(label, left, right, domain, layer):
        a, b = seed_tables[left], seed_tables[right]
        order_a = np.argsort(-a, kind="stable")
        order_b = np.argsort(-b, kind="stable")
        top_a, top_b = set(order_a[:8]), set(order_b[:8])
        bottom_a, bottom_b = set(order_a[-8:]), set(order_b[-8:])
        jac_top = len(top_a & top_b) / len(top_a | top_b)
        jac_bottom = len(bottom_a & bottom_b) / len(bottom_a | bottom_b)
        drift = float(np.abs(a - b).sum())
        comparisons.append({"comparison": label, "domain": domain, "layer": layer,
                            "l1_drift": drift, "top8_jaccard": jac_top,
                            "bottom8_jaccard": jac_bottom,
                            "pass": drift <= l1_max and jac_top >= jaccard_min and jac_bottom >= jaccard_min})
    for budget in budgets:
        for i, seed_a in enumerate(seeds):
            for seed_b in seeds[i + 1:]:
                for domain, layer in table_keys:
                    akey, bkey = (budget, seed_a, domain, layer), (budget, seed_b, domain, layer)
                    if akey in seed_tables and bkey in seed_tables:
                        add_comparison(f"seed:{seed_a}:{seed_b}:budget:{budget}", akey, bkey, domain, layer)
    for lower, higher in zip(budgets, budgets[1:]):
        if higher not in (2 * lower, 2.5 * lower):
            continue
        for domain, layer in table_keys:
            left_vals, right_vals = [], []
            for seed in seeds:
                akey, bkey = (lower, seed, domain, layer), (higher, seed, domain, layer)
                if akey in seed_tables and bkey in seed_tables:
                    left_vals.append(seed_tables[akey]); right_vals.append(seed_tables[bkey])
            if len(left_vals) == 3:
                left = np.mean(left_vals, axis=0); right = np.mean(right_vals, axis=0)
                # Store temporary keys so the same comparison helper applies.
                akey, bkey = (lower, -1, domain, layer), (higher, -1, domain, layer)
                seed_tables[akey], seed_tables[bkey] = left, right
                add_comparison(f"budget:{lower}:{higher}", akey, bkey, domain, layer)
    frozen_budget,qualifying = _choose_calibration_budget(comparisons,budgets,pass_fraction_min)
    pass_fraction = (sum(row["pass"] for row in comparisons) / len(comparisons)) if comparisons else 0.0
    averaged = average_seed_tables({(seed, domain, layer): table
                                    for (budget, seed, domain, layer), table in seed_tables.items()
                                    if seed != -1 and budget == frozen_budget}) if frozen_budget else {}
    m1 = m1_tables(averaged, domains) if averaged else {}
    tables = {"M1": {str(layer): values.tolist() for layer, values in m1.items()}}
    for (domain, layer), values in averaged.items():
        tables.setdefault(domain, {})[str(layer)] = values.tolist()
    rankings = {}
    for metric, by_layer in tables.items():
        rankings[metric] = {}
        for layer_s, values in by_layer.items():
            order = sorted(range(64), key=lambda e: (-values[e], e))
            tied, current = [], []
            for expert in order:
                if current and values[current[-1]] != values[expert]:
                    if len(current) > 1:
                        tied.append({"score": values[current[0]], "expert_ids": current})
                    current = []
                current.append(expert)
            if len(current) > 1:
                tied.append({"score": values[current[0]], "expert_ids": current})
            rankings[metric][layer_s] = {"ordered_expert_ids": order, "tie_blocks": tied}
    probe_report = None
    if probe_bundle is not None:
        probe_root = Path(probe_bundle)
        probe_manifest = read_json(probe_root / "manifest.json")
        probe_kind = probe_manifest.get("kind")
        if probe_kind not in ("probe", "measurements"):
            raise ValueError("fixed probe bundle must have kind probe or measurements")
        probe_manifest = verify_bundle(probe_root, probe_kind)
        if probe_manifest["metadata"].get("stage") != "probe":
            raise ValueError("fixed probe measurement bundle must have stage probe")
        probe_rows = list(read_rows(probe_root / "events.jsonl"))
        if {r["document_id"] for r in probe_rows} & set.union(*document_sets.values()):
            raise ValueError("probe documents overlap calibration")
        probe_units = defaultdict(set)
        probe_positions = defaultdict(set)
        probe_layers = defaultdict(set)
        calibration_groups = {(r["domain"],r["source_group"]) for r in document_statistics}
        for row in probe_rows:
            validate_measurement(row)
            probe_units[row["domain"]].add(row["document_id"])
            probe_positions[row["document_id"]].add(row["predictor_position"])
            probe_layers[row["document_id"],row["predictor_position"]].add(row["layer"])
            if (row["domain"],row["source_group"]) in calibration_groups:
                raise ValueError("probe source groups overlap calibration")
        if any(len(probe_units[d]) != 32 for d in domains) or any(len(v) != 4 for v in probe_positions.values()):
            raise ValueError("fixed probe requires 32 documents per domain and four positions per document")
        if any(v != set(expected_layers) for v in probe_layers.values()):
            raise ValueError("fixed probe requires every scoped layer at every position")
        planned = _expected_plan_events(probe_root / "plan")
        if "expected_event_ids" not in probe_manifest["metadata"] or set(probe_manifest["metadata"]["expected_event_ids"]) != set(planned):
            raise ValueError("fixed probe expected IDs differ from plan")
        ids = {row["event_id"] for row in probe_rows}
        if ids != set(planned) or len(ids) != len(probe_rows):
            raise ValueError("fixed probe rows do not cover the frozen plan")
        for row in probe_rows:
            _validate_observation_identity(row, planned[row["event_id"]])
        probe_meta = probe_manifest["metadata"].get("plan_metadata", {})
        if read_json(probe_root / "plan" / "research.json") != settings:
            raise ValueError("fixed probe differs from calibration research settings")
        validate_plan(settings, probe_meta, read_rows(probe_root / "plan" / "inputs.jsonl"))
        probe_context = int(probe_meta.get(
            "window", read_json(probe_root / "plan" / "research.json")["context"]["primary_window"]))
        if probe_context != next(iter(contexts)):
            raise ValueError("fixed probe and calibration context lengths differ")
        probe_executor = probe_manifest["metadata"].get("executor")
        stable_probe = {"executor": probe_executor,
                        "runtime": probe_executor.get("runtime") if isinstance(probe_executor, dict) else None,
                        "source_pool": probe_meta.get("prepared_manifest_sha256"),
                        "source_review": probe_meta.get("source_review_sha256"),
                        "tokenizer": probe_meta.get("tokenizer_sha256")}
        if any(stable_probe.get(name) is None for name in identity_facts):
            raise ValueError("fixed probe bundle lacks required matching executor or source identity")
        for name, value in stable_probe.items():
            if name in identity_facts and value != identity_facts[name]:
                raise ValueError(f"fixed probe differs from calibration in {name} identity")
        for row in probe_rows:
            if row.get("kind") != "probe_routing":
                raise ValueError("fixed probe must contain route-only probe_routing rows")
        versions = []
        for budget in budgets:
            for seed in [*seeds,-1]:
                version_domain = {(d, layer): seed_tables[budget, seed, d, layer]
                                  for d, layer in table_keys if (budget, seed, d, layer) in seed_tables}
                if version_domain:
                    versions.append((budget, seed, version_domain,
                                     m1_tables(version_domain, domains)))
        probe_counts = defaultdict(lambda: {"comparisons": 0, "agreements": 0,
                                             "tied_comparisons": 0, "tied_agreements": 0,
                                             "untied_comparisons": 0, "untied_agreements": 0})
        for row in probe_rows:
            active = row["routing"]["expert_ids"]
            weights = row["routing"]["weights"]
            if len(active) != 8 or len(set(active)) != 8 or len(weights) != 8:
                raise ValueError("invalid fixed probe routing")
            if struct.pack("<8f", *weights).hex() != row["routing"]["weights_fp32_le_hex"]:
                raise ValueError("fixed probe routing bytes mismatch")
            identity = row["event_id"]
            tie_order = stable_tie_order(3007, "metric-selection", identity)
            by_version = []
            for budget, seed, domain_tables, global_tables in versions:
                layer = int(row["layer"])
                if layer not in global_tables or (row["domain"], layer) not in domain_tables:
                    continue
                metrics = {"M1": global_tables[layer], "M2:matched": domain_tables[row["domain"], layer]}
                for domain in domains:
                    if domain != row["domain"] and (domain, layer) in domain_tables:
                        metrics[f"M2:mismatched:{domain}"] = domain_tables[domain, layer]
                for metric, scores in metrics.items():
                    from .routing_stats import _select_extremes
                    pair = _select_extremes(scores, active, tie_order)
                    tied = bool(sum(scores[e] == scores[pair[0]] for e in active) > 1 or
                                sum(scores[e] == scores[pair[1]] for e in active) > 1)
                    by_version.append((budget, seed, metric, pair, tied))
                    probe_selections.append({"event_id": identity, "domain": row["domain"],
                        "document_id": row["document_id"], "source_group": row["source_group"],
                        "predictor_position": row["predictor_position"], "context_length": row["context_length"],
                        "layer": layer, "budget": budget, "seed": seed, "metric": metric,
                        "selected_pair": list(pair), "exact_extreme_tie": tied,
                        "frozen_average_ranking": budget == frozen_budget and seed == -1})
            for i, left in enumerate(by_version):
                for right in by_version[i + 1:]:
                    if left[2] != right[2] or left[:2] == right[:2]:
                        continue
                    # Only compare seed pairs at the same budget and averaged
                    # calibration rankings across adjacent budgets.
                    if left[0] == right[0] and left[1] != right[1]:
                        label = f"seed:{left[0]}"
                    elif abs(budgets.index(left[0]) - budgets.index(right[0])) == 1 and left[1] == right[1]:
                        label = f"budget:{min(left[0], right[0])}:{max(left[0], right[0])}"
                    else:
                        continue
                    key = (label, left[2],row["domain"],row["layer"])
                    counts = probe_counts[key]
                    is_tied = left[4] or right[4]
                    counts["comparisons"] += 1
                    counts["agreements"] += int(left[3] == right[3])
                    if is_tied:
                        counts["tied_comparisons"] += 1
                        counts["tied_agreements"] += int(left[3] == right[3])
                    else:
                        counts["untied_comparisons"] += 1
                        counts["untied_agreements"] += int(left[3] == right[3])
        probe_report = [{"comparison": label, "metric": metric, "domain":domain,"layer":layer, **counts,
                         "agreement_fraction": counts["agreements"] / counts["comparisons"] if counts["comparisons"] else None,
                         "tied_agreement_fraction": counts["tied_agreements"] / counts["tied_comparisons"] if counts["tied_comparisons"] else None,
                         "untied_agreement_fraction": counts["untied_agreements"] / counts["untied_comparisons"] if counts["untied_comparisons"] else None}
                        for (label, metric,domain,layer), counts in sorted(probe_counts.items())]
    payload = {"status": "frozen" if frozen_budget is not None else "unstable",
               "research_scope": research_scope,
               "layers": expected_layers,
               "context_length": next(iter(contexts)), "selected_document_budget": frozen_budget,
               "tables": tables,
               "rankings": rankings,
               "token_weighted_calibration_note":"One equal-length window per document per seed makes token-weighted and equal-document utilization identical.",
               "identities": identity_facts,
               "evidence": {"source_bundle_sha256": {f"{b}/{s}": h for (b, s), h in provenance.items()},
                            "fixed_probe_bundle_sha256":file_digest(Path(probe_bundle)/"manifest.json"),
                            "comparisons": comparisons, "pass_fraction": pass_fraction,
                            "consecutive_budget_checks": qualifying,
                            "calibration_document_counts": {str(b):{d:len({r["document_id"] for r in document_statistics
                                if r["budget"]==b and r["domain"]==d}) for d in domains} for b in budgets},
                            "fixed_probe_agreement": probe_report,
                            "criteria": {"normalized_l1_max": l1_max,
                                         "stability_gate_scope":scope,
                                         "extreme_set_jaccard_min": jaccard_min,
                                         "comparison_pass_fraction_min": pass_fraction_min},
                            "ranking_ties": "stable expert ID order; no outcomes used"}}
    if output_dir is not None:
        from .artifacts import seal
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=False)
        (outdir / "frozen-tables.json").write_bytes(canonical(payload) + b"\n")
        report = {"status": payload["status"], "selected_document_budget": frozen_budget,
                  "evidence": payload["evidence"]}
        (outdir / "stability-report.json").write_bytes(canonical(report) + b"\n")
        write_rows(outdir / "routing-document-statistics.jsonl", document_statistics)
        write_rows(outdir / "routing-domain-statistics.jsonl", ({"budget": b, "seed": s,
            "domain": d, "layer": l, "expert_utilization": table.tolist()}
            for (b,s,d,l),table in sorted(seed_tables.items()) if s != -1))
        write_rows(outdir / "probe-selections.jsonl", probe_selections)
        payload["manifest"] = seal(outdir, "calibration", {
            "research_scope": research_scope,
            "context_length": next(iter(contexts)),
            "selected_document_budget": frozen_budget,
            "executor": identity_facts["executor"],
            "identities": {k: identity_facts[k] for k in ("source_pool", "source_review", "tokenizer")}})
    elif output is not None:
        Path(output).write_bytes(canonical(payload) + b"\n")
    return payload
