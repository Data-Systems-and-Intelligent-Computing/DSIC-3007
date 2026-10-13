"""Local pilot planning and an exact, outcome-independent main input freeze."""

from bisect import bisect_left
from collections import defaultdict
import math
from pathlib import Path

import numpy as np

from .artifacts import (atomic, canonical, digest, file_digest, priority, read_json,
                        read_rows, seal, verify_bundle, write_rows)
from .analyze import (analyze_bundle, _paired_primary, _rng_seed,
                      _expected_plan_events, _validate_observation_identity)
from .preparation import make_plan, reviewed_documents
from .scope import is_path_b, main_budgets, scope_identity, stage_layers, validate_plan


def plan_main(pilot, technical, calibration, prepared, review, settings, output,
              remaining_workflow=None):
    """Project pilot cluster variance; minimize time over available nested inputs."""
    pilot_manifest = verify_bundle(pilot, "measurements")
    technical_manifest = verify_bundle(technical, "measurements")
    if pilot_manifest["metadata"]["stage"] != "pilot" or technical_manifest["metadata"]["stage"] != "technical":
        raise ValueError("planning requires a scientific pilot and measured technical benchmark")
    if pilot_manifest["metadata"]["executor"] != technical_manifest["metadata"]["executor"]:
        raise ValueError("pilot and technical timings must use the same model/runtime")
    _, groups = reviewed_documents(prepared, review)
    for manifest in (pilot_manifest, technical_manifest):
        meta = manifest["metadata"]["plan_metadata"]
        if (meta["prepared_manifest_sha256"] != file_digest(Path(prepared)/"manifest.json") or
                meta["source_review_sha256"] != file_digest(review) or meta["window"] != 4096):
            raise ValueError("planning inputs do not belong to this reviewed 4096-token study")
    frozen_settings = read_json(Path(pilot)/"plan"/"research.json")
    if frozen_settings != settings:
        raise ValueError("settings changed after pilot inputs were frozen")
    path_b = is_path_b(settings)
    if read_json(Path(technical)/"plan"/"research.json") != settings:
        raise ValueError("technical timing settings differ from pilot")
    for root, manifest in ((pilot,pilot_manifest),(technical,technical_manifest)):
        validate_plan(settings, manifest["metadata"]["plan_metadata"],
                      read_rows(Path(root)/"plan"/"inputs.jsonl"))
    if path_b:
        technical_meta = technical_manifest["metadata"]["plan_metadata"]
        if technical_meta.get("benchmark_screen") or technical_meta.get("preflight_diagnostic"):
            raise ValueError("main planning cannot use diagnostic technical timings as full acceptance")
        from .optimized_validation import _validate_full_technical_pair
        technical_root = Path(technical)/"plan"
        _validate_full_technical_pair([(technical_root,verify_bundle(technical_root,"experiment-plan"),settings,
                                       list(read_rows(technical_root/"inputs.jsonl")))],
                                      file_digest(technical_root/"source-review.json"))
        # Pilot-to-main group variance transfer needs the same position budget.
        by_doc = defaultdict(set)
        for row in read_rows(Path(pilot)/"plan"/"inputs.jsonl"):
            by_doc[row["document_id"]].add(row["predictor_position"])
            n = min(8, row["eligible_positions"])
            if not 1 <= n <= 8:
                raise ValueError("invalid eight-position pilot eligibility")
        for row in read_rows(Path(pilot)/"plan"/"inputs.jsonl"):
            if len(by_doc[row["document_id"]]) != min(8, row["eligible_positions"]):
                raise ValueError("Path B planning requires the complete eight-position pilot")
    analyzed = analyze_bundle(pilot, calibration_dir=calibration)
    pilot_report = analyzed["report"]
    domains = pilot_report["pilot_planning"]["domains"]
    if any(domains[d]["extend"] or domains[d]["insufficient_at_maximum"]
           for d in ("scientific", "code")):
        raise ValueError("pilot extension or insufficiency must be resolved before main planning")
    docs = [{"domain": d["domain"], "document_id": d["document_id"],
             "source_group": d["source_group"], "delta": d["mean_delta"]}
            for d in analyzed["documents"] if d["layer"] == 15]
    seed = _rng_seed(settings["sampling"]["master_seed"], "pilot:main-planning")
    boot = _paired_primary(docs, 2000, seed, .95)
    if boot["ci"] is None:
        raise ValueError("pilot planning bootstrap has fewer than 95% valid draws or degenerate variance")
    low, high = np.quantile(boot["bootstrap_t"], [.025, .975], method="linear")
    critical_half_span = float((high-low)/2)
    sample = settings["sampling"]
    master = sample["master_seed"]
    scientific = [d for (domain, stage, _), members in groups.items()
                  if domain == "scientific" and stage == "main" for d in members]
    scientific.sort(key=lambda d: (priority(master, "source-documents",
                             ["scientific", "main", d["document_id"]]), d["document_id"]))
    code = [(g, members) for (domain, stage, g), members in groups.items()
            if domain == "code" and stage == "main"]
    code.sort(key=lambda p: (priority(master, "source-allocation", ["code","main",p[0]]),p[0]))
    c4 = [d for (domain, stage, _), members in groups.items()
          if domain == "c4" and stage == "main" for d in members]
    if len(c4) < sample["main_c4_documents"]:
        raise ValueError("insufficient reviewed C4 main diagnostic documents")
    c4.sort(key=lambda d: (priority(master, "source-documents",
                     ["c4", "main", d["document_id"]]), d["document_id"]))
    technical_inputs = defaultdict(dict)
    expected = _expected_plan_events(Path(technical)/"plan")
    seen_events = set()
    for row in read_rows(Path(technical)/"events.jsonl"):
        from .artifacts import validate_measurement
        validate_measurement(row)
        if row["event_id"] in seen_events or row["event_id"] not in expected:
            raise ValueError("invalid technical benchmark event coverage")
        seen_events.add(row["event_id"])
        _validate_observation_identity(row,expected[row["event_id"]])
        technical_inputs[row["domain"],row["input_id"]][row["layer"]] = row
    if seen_events != set(expected):
        raise ValueError("technical timing benchmark is incomplete")
    timings = {}
    layers = stage_layers(settings, "technical")
    for domain in settings["domains"]:
        observations = [r for (d,_),r in technical_inputs.items() if d == domain]
        if not observations or any(set(r) != set(layers) for r in observations):
            raise ValueError("timing benchmark must include all technical domains and layers")
        values = [[r[15].get("shared_baseline_seconds"), r[15].get("shared_cleanup_seconds"),
                   *(r[l].get("eight_interventions_seconds") for l in layers)]
                  for r in observations]
        if any(v is None or not math.isfinite(v) or v <= 0 for vs in values for v in vs):
            raise ValueError("technical benchmark lacks positive measured forward/intervention timings")
        mean = np.mean(values, axis=0)
        timings[domain] = {"shared_forward_seconds": float(mean[0]+mean[1]),
                          "eight_interventions_seconds": {str(l):float(mean[2+i]) for i,l in enumerate(layers)}}
        if path_b:
            total = [r[15].get("input_total_seconds") for r in observations]
            if any(v is None or not math.isfinite(v) or v <= 0 for v in total):
                raise ValueError("Path B technical timings require complete synchronized input elapsed time")
            timings[domain]["input_total_seconds"] = float(np.mean(total))

    def document_time(doc, domain):
        # Keep all available positions below the frozen stage budget.
        n = min(sample["main_positions_per_document"], doc["eligible_positions"])
        t = timings[domain]
        if path_b:
            return n*max(t["input_total_seconds"], t["shared_forward_seconds"]+t["eight_interventions_seconds"]["15"])
        return (n*(t["shared_forward_seconds"]+t["eight_interventions_seconds"]["15"]) +
                min(n,16)*(t["eight_interventions_seconds"]["0"]+t["eight_interventions_seconds"]["7"]))

    c4_time = sum(document_time(doc, "c4") for doc in c4[:sample["main_c4_documents"]])

    scientific_groups, scientific_times = [], []
    seen, cumulative = set(), 0.0
    for doc in scientific:
        seen.add(doc["source_group"])
        cumulative += document_time(doc, "scientific")
        scientific_groups.append(len(seen))
        scientific_times.append(cumulative)
    vs = pilot_report["domain_paired_delta"]["scientific"]["variance"]
    vc = pilot_report["domain_paired_delta"]["code"]["variance"]
    gs, gc = domains["scientific"]["source_groups"], domains["code"]["source_groups"]
    target = settings["planning"]["target_ci_half_width_nats"]
    candidate_rows = []
    code_time, code_documents = 0.0, 0
    for n_code, (_, members) in enumerate(code, 1):
        code_time += sum(document_time(d,"code") for d in members)
        code_documents += len(members)
        if n_code < sample["main_min_code_groups"]:
            continue
        allowance = 4*(target/critical_half_span)**2 - vc*gc/n_code
        if allowance <= 0:
            continue
        required_scientific_groups = max(2, math.ceil(vs*gs/allowance))
        n_scientific = max(sample["main_min_scientific_documents"],
                           bisect_left(scientific_groups,required_scientific_groups)+1)
        if n_scientific > len(scientific):
            continue
        variance = .25*(vs*gs/scientific_groups[n_scientific-1]+vc*gc/n_code)
        half_width = critical_half_span*math.sqrt(variance)
        if half_width > target:
            continue
        candidate_rows.append({"scientific_documents":n_scientific,
            "scientific_source_groups":scientific_groups[n_scientific-1],
            "code_source_groups":n_code,"code_documents":code_documents,
            "c4_documents":sample["main_c4_documents"],
            "c4_measured_runtime_projection_seconds":c4_time,
            "projected_variance":variance,"projected_ci_half_width":half_width,
            "measured_runtime_projection_seconds":scientific_times[n_scientific-1]+code_time+c4_time})
    selected = min(candidate_rows, key=lambda r:(r["measured_runtime_projection_seconds"],
                   r["scientific_documents"],r["code_source_groups"])) if candidate_rows else None
    report = {"status":"qualifying_allocation" if selected else "no_feasible_allocation",
        "selected":selected,"candidate_allocations":candidate_rows,"timings":timings,
        "planning_replicates":2000,"valid_replicates":boot["valid_replicates"],
        "degenerate_replicates":boot["degenerate_replicates"],"planning_seed":seed,
        "bootstrap_t_quantiles":[float(low),float(high)],
        "variance_projection_rule":settings["planning"]["variance_projection_rule"],
        "precision_is_projection":True,"target_ci_half_width_nats":target,
        "c4_documents":sample["main_c4_documents"],"settings_sha256":digest(settings),
        "pilot_manifest_sha256":file_digest(Path(pilot)/"manifest.json"),
        "technical_manifest_sha256":file_digest(Path(technical)/"manifest.json"),
        "calibration_manifest_sha256":file_digest(Path(calibration)/"manifest.json"),
        "prepared_manifest_sha256":file_digest(Path(prepared)/"manifest.json"),
        "source_review_sha256":file_digest(review),
        "executor":pilot_manifest["metadata"]["executor"]}
    report["c4_pilot_diagnostic"] = domains.get("c4", {"status":"not_collected_deferred_by_path_b"})
    if path_b:
        from .admission import admit_remaining
        report["research_scope"] = scope_identity(settings)
        report["position_budget_transfer"] = "pilot_and_main_eight_uniform_nested_positions; group_variance_projection_only"
        identities = {k:report[k] for k in ("pilot_manifest_sha256", "technical_manifest_sha256", "calibration_manifest_sha256")}
        report["remaining_workflow_budget"] = (admit_remaining(settings,
            selected["measured_runtime_projection_seconds"], identities, remaining_workflow)
            if selected else {"status":"no_precision_qualifying_allocation", "feasibility_pass":False})
        if selected and not report["remaining_workflow_budget"]["feasibility_pass"]:
            report["status"] = "precision_qualifies_budget_unresolved"
        if remaining_workflow is not None:
            report["remaining_workflow_ledger"] = remaining_workflow
    out = Path(output)
    out.mkdir(parents=True,exist_ok=False)
    atomic(out/"main-planning.json",canonical(report)+b"\n")
    write_rows(out/"bootstrap-draws.jsonl", ({"replicate":i,"valid":valid,
        "scientific_group_slots":boot["bootstrap_group_slots"]["scientific"][i],
        "code_group_slots":boot["bootstrap_group_slots"]["code"][i]}
        for i,valid in enumerate(boot["bootstrap_valid_flags"])))
    seal(out,"local-analysis",{"purpose":"main-planning"})
    return report


def freeze_main(planning, prepared, review, settings, output):
    verify_bundle(planning,"local-analysis")
    report = read_json(Path(planning)/"main-planning.json")
    if report["status"] != "qualifying_allocation" or report["settings_sha256"] != digest(settings):
        raise ValueError("main planning has no qualifying allocation or settings changed")
    if (report["prepared_manifest_sha256"] != file_digest(Path(prepared)/"manifest.json") or
            report["source_review_sha256"] != file_digest(review)):
        raise ValueError("reviewed source pool changed after sample-size planning")
    selected = report["selected"]
    budgets = main_budgets(settings, selected)
    if is_path_b(settings):
        from .admission import admit_remaining
        identities = {k:report[k] for k in ("pilot_manifest_sha256", "technical_manifest_sha256", "calibration_manifest_sha256")}
        admitted = admit_remaining(settings, selected["measured_runtime_projection_seconds"],
                                   identities, report.get("remaining_workflow_ledger"))
        if not admitted["feasibility_pass"] or admitted != report.get("remaining_workflow_budget"):
            raise ValueError("Path B main freeze requires the complete remaining workflow budget admission")
    manifest = make_plan(prepared,review,output,settings,"main",budgets,
                         positions=settings["sampling"]["main_positions_per_document"])
    # Re-seal once, locally, before the plan is exposed or any outcomes exist.
    (Path(output)/"manifest.json").unlink()
    freeze = {"planning_manifest_sha256":file_digest(Path(planning)/"manifest.json"),
              **{k:report[k] for k in ("pilot_manifest_sha256","technical_manifest_sha256",
                                      "calibration_manifest_sha256","executor","settings_sha256")},
              "selected":selected,"analysis":settings["analysis"],
              "positions_per_document":settings["sampling"]["main_positions_per_document"]}
    if is_path_b(settings):
        freeze["remaining_workflow_budget"] = admitted
    atomic(Path(output)/"main-freeze.json",canonical(freeze)+b"\n")
    return seal(output,"experiment-plan",dict(manifest["metadata"],main_freeze=freeze))
