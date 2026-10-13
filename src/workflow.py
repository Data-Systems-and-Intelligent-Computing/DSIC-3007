"""The explicit research workflow: prepare, collect, freeze and analyze."""

import argparse
from collections import defaultdict
from pathlib import Path

from .artifacts import (atomic, canonical, digest, file_digest, priority, read_json,
                        read_rows, seal, unresolved, verify_bundle, write_rows)
from .scope import is_path_b, scope_identity, stage_domains


def review_draft(prepared, output):
    verify_bundle(prepared, "prepared-inputs")
    decisions, counts = {}, defaultdict(int)
    for d in read_rows(Path(prepared) / "documents.jsonl"):
        excluded = not d["eligible_positions"] or not d["source_group"]
        reason = ("no full 4096 input and subsequent target" if not d["eligible_positions"]
                  else "missing native source group" if not d["source_group"] else None)
        decisions[d["document_id"]] = {"source_group": d["source_group"],
            "stage": "excluded" if excluded else "pending", "reason": reason,
            "domain": d["domain"], "retained_split": d["retained_split"],
            "duplicate_alias_review_required": bool(d["source_group_aliases"])}
        counts[d["domain"] + "/" + ("excluded" if excluded else "eligible_candidate")] += 1
    result = {"status": "pending", "prepared_manifest_sha256": file_digest(Path(prepared) / "manifest.json"),
              "checks": {"duplicate_content": "pending", "article_variants": "pending",
                         "repository_aliases_and_known_clones": "pending", "source_group_disjointness": "pending"},
              "documents": decisions, "counts": dict(counts)}
    atomic(output, canonical(result) + b"\n")
    return result


def shorten_plan(plan_dir, output, window=2048, subset=False):
    """Exact matched predictors/targets; deterministic whole-group subset for main."""
    manifest = verify_bundle(plan_dir, "experiment-plan")
    settings = read_json(Path(plan_dir) / "research.json")
    if is_path_b(settings):
        raise ValueError("matched 2048 is deferred in Path B; a new explicit scope amendment is required")
    if manifest["metadata"]["window"] != 4096 or window != 2048:
        raise ValueError("the selected context sensitivity is matched 4096 -> 2048")
    docs = list(read_rows(Path(plan_dir) / "documents.jsonl"))
    inputs = list(read_rows(Path(plan_dir) / "inputs.jsonl"))
    selected = {d["document_id"] for d in docs}
    achieved = {}
    if subset:
        if manifest["metadata"]["stage"] != "main":
            raise ValueError("sensitivity subset must come from frozen main inputs")
        selected = set()
        for domain in settings["domains"]:
            grouped = defaultdict(list)
            for d in docs:
                if d["domain"] == domain:
                    grouped[d["source_group"]].append(d["document_id"])
            ordered = sorted(grouped, key=lambda g: (priority(settings["sampling"]["master_seed"],
                                      "context-sensitivity-groups", [domain, g]), g))
            total = sum(map(len, grouped.values()))
            # Dynamic programming chooses the closest attainable document count.
            # First attainable hash-ordered subset wins ties, without outcomes.
            choices = {0: ()}
            for group in ordered:
                for count, members in sorted(list(choices.items()), reverse=True):
                    new = count + len(grouped[group])
                    choices.setdefault(new, members + (group,))
            best = min((n for n in choices if n > 0),
                       key=lambda n: (abs(n - total * settings["sensitivity"]["subset_fraction"]), n))
            for g in choices[best]:
                selected.update(grouped[g])
            achieved[domain] = {"main_documents": total, "target_fraction": 0.25,
                                "selected_documents": best, "selected_groups": list(choices[best])}
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    shortened = []
    for i in inputs:
        if i["document_id"] not in selected:
            continue
        row = dict(i, input_ids=i["input_ids"][-window:], context_length=window,
                   window_start=i["window_end"] - window, local_predictor_position=window - 1,
                   parent_input_id=i["input_id"])
        row["input_id"] = digest({"document_id": row["document_id"],
            "token_ids_sha256": row["token_ids_sha256"], "window_start": row["window_start"],
            "window_end": row["window_end"], "target_id": row["target_id"], "context_length": window})
        shortened.append(row)
    write_rows(out / "inputs.jsonl", shortened)
    write_rows(out / "documents.jsonl", (d for d in docs if d["document_id"] in selected))
    for name in ("research.json", "source-review.json", "model-catalog.json"):
        atomic(out / name, (Path(plan_dir) / name).read_bytes())
    if (Path(plan_dir)/"main-freeze.json").exists():
        atomic(out/"main-freeze.json",(Path(plan_dir)/"main-freeze.json").read_bytes())
    metadata = dict(manifest["metadata"], window=window,
        stage="sensitivity" if subset else manifest["metadata"]["stage"],
        parent_plan_sha256=file_digest(Path(plan_dir) / "manifest.json"),
        matched_predictors_and_targets=True, subset=achieved, input_count=len(shortened))
    return seal(out, "experiment-plan", metadata)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=Path("configs/research-path-b-v1.json"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="report scientific decisions still pending")
    p = sub.add_parser("inventory", help="verify retained bytes and list source identities")
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--catalog", type=Path, default=Path("data/manifests/collection-10k-v1.json"))
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("tokenizer", help="fetch only the small pinned tokenizer files")
    p.add_argument("--catalog", type=Path, default=Path("data/manifests/model-fp32.json"))
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("weights", help="explicitly acquire and verify original pinned FP32 model files")
    p.add_argument("--catalog",type=Path,default=Path("data/manifests/model-fp32.json"))
    p.add_argument("--output",type=Path,required=True)
    p = sub.add_parser("prepare", help="tokenize unchanged retained text and count exact eligibility")
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--catalog", type=Path, default=Path("data/manifests/collection-10k-v1.json"))
    p.add_argument("--model-catalog", type=Path, default=Path("data/manifests/model-fp32.json"))
    p.add_argument("--tokenizer-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("review-draft", help="create an explicitly pending source audit/allocation file")
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("rebind-prepared", help="create a Path B successor from unchanged verified token bytes")
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("audit-sources", help="audit source metadata and duplicates without approving a sample")
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("combine-inputs", help="reference verified retained and supplemental bytes in one catalog")
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--retained-catalog", type=Path, default=Path("data/manifests/collection-10k-v1.json"))
    p.add_argument("--supplements", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("allocate", help="allocate audited source groups deterministically before outcomes")
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--calibration-ceiling", type=int, choices=(100, 250, 500), default=500)
    p.add_argument("--exclude", nargs="*", default=[])
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("plan", help="freeze inputs from reviewed independent source partitions")
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--stage", choices=("technical", "calibration", "probe", "pilot"), required=True)
    p.add_argument("--units", type=int, required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("shorten-plan", help="matched 2048 calibration or whole-group main sensitivity")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--subset", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("verify", help="explicit native GPU mechanism checks on technical inputs")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("prepare-benchmark", help="prepare a bounded technical screen locally; launches no GPU")
    p.add_argument("--primary", type=Path, required=True)
    p.add_argument("--shortened", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("execution-budget", help="assess complete measured workflow cost and time locally")
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("timing-profiles", help="extract measured persistence/runtime profiles from sealed runs locally")
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("validate-optimized", help="explicit bounded GPU comparison with native full forwards")
    p.add_argument("--plans", type=Path, nargs="+", required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-seconds", type=float, default=1800)
    p.add_argument("--scope", choices=("screen", "preflight", "full_technical"), default="screen")
    p.add_argument("--native-mechanics", type=Path, nargs="+")
    p = sub.add_parser("collect-sequence", help="collect explicit requests using one verified loaded model")
    p.add_argument("--requests", type=Path, required=True)
    p = sub.add_parser("collect", help="explicit native GPU routing/effect collection")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--mechanics", type=Path, required=True)
    p.add_argument("--calibration", type=Path)
    p.add_argument("--optimized", action="store_true")
    p.add_argument("--optimized-validation", type=Path)
    p.add_argument("--reuse-runs", type=Path, nargs="*", default=[])
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("freeze-calibration", help="assess seed/budget stability and freeze averaged tables locally")
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("analyze", help="replay all eight raw effects and analyze source-group uncertainty locally")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--calibration", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("plan-main", help="choose the fastest reviewed allocation from pilot precision and measured technical timings")
    for name in ("pilot","technical","calibration","prepared","review","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--remaining-workflow", type=Path, help="complete reconciled remaining stage ledger; otherwise main freeze is blocked")
    p = sub.add_parser("freeze-main", help="freeze exact main IDs, positions, calibration and analysis before outcomes")
    for name in ("planning","prepared","review","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p = sub.add_parser("analyze-sensitivity", help="compare matched full/short contexts locally")
    p.add_argument("--main", type=Path, required=True)
    p.add_argument("--sensitivity", type=Path, required=True)
    p.add_argument("--main-calibration", type=Path, required=True)
    p.add_argument("--sensitivity-calibration", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    settings = read_json(args.settings)
    if args.command == "status":
        result = {"authority": settings["authority"], "research_scope": scope_identity(settings),
                  "pending_decisions": [p for p in unresolved(settings)
                      if not (is_path_b(settings) and p == "context.sensitivity_window")],
                  "compute_policy": "operator-controlled Terraform; disabled by default"}
        collection_proposal = args.settings.parent / "collection-proposal.json"
        if collection_proposal.exists():
            proposal = read_json(collection_proposal)
            result["data_preparation"] = {
                "collection_decisions": proposal["status"],
                "scientific_variant_decision": proposal["scientific_variant_decision"],
                "supplements": {s["domain"]: s["download_status"] for s in proposal["sources"]}}
    elif args.command == "inventory":
        from .preparation import inventory
        result = inventory(args.root, args.catalog, args.output)["metadata"]
    elif args.command == "tokenizer":
        from .preparation import acquire_tokenizer
        acquire_tokenizer(args.catalog, args.output)
        result = {"status": "verified", "weights_downloaded": False}
    elif args.command == "weights":
        from .model import acquire_model
        acquire_model(args.catalog,args.output)
        result = {"status":"original_fp32_bytes_verified"}
    elif args.command == "prepare":
        from .preparation import prepare
        result = prepare(args.root, args.catalog, args.model_catalog, args.tokenizer_dir, args.output, settings)["metadata"]
    elif args.command == "review-draft":
        result = review_draft(args.prepared, args.output)
        result = {"status": result["status"], "counts": result["counts"]}
    elif args.command == "audit-sources":
        from .source_audit import audit_sources
        result = audit_sources(args.prepared, args.output)["metadata"]
    elif args.command == "combine-inputs":
        from .preparation import combine_collections
        result = combine_collections(args.root, args.retained_catalog, args.supplements, args.output)
        result = {"snapshot_identity_sha256": result["snapshot_identity_sha256"],
                  "source_entries": len(result["sources"]), "assets": len(result["files"])}
    elif args.command == "rebind-prepared":
        from .preparation import rebind_prepared
        result = rebind_prepared(args.prepared, args.output, settings)["metadata"]
    elif args.command == "allocate":
        from .allocation import allocate
        review = allocate(args.prepared, settings, args.output, args.exclude,
                          args.calibration_ceiling)
        result = {"status": review["status"], "capacity": review["capacity"],
                  "shortfalls": review["shortfalls"], "stage_counts": review["stage_counts"]}
    elif args.command == "plan":
        from .preparation import make_plan
        if args.stage == "calibration" and (args.units not in (100,250,500) or args.seed not in (3008,3009,3010)):
            raise ValueError("calibration uses the chosen 100/250/500 document ladder and three explicit seeds")
        if args.stage == "pilot" and args.units not in (20,40,80):
            raise ValueError("pilot uses nested 20/40/80 source-unit budgets")
        if args.stage == "probe" and args.units != 32:
            raise ValueError("fixed probe uses 32 documents per domain")
        if args.stage == "technical" and args.units != 12:
            raise ValueError("technical testing reserves 12 source groups per domain")
        result = make_plan(args.prepared, args.review, args.output, settings, args.stage,
                           {d: args.units for d in stage_domains(settings, args.stage)}, args.seed)["metadata"]
    elif args.command == "shorten-plan":
        result = shorten_plan(args.plan, args.output, subset=args.subset)["metadata"]
    elif args.command == "verify":
        from .verify import verify_native
        receipt = verify_native(args.plan, args.model_dir, args.output)
        result = {"status": receipt["status"], "checks": len(receipt["checks"])}
    elif args.command == "prepare-benchmark":
        from .benchmark import prepare_benchmark
        result = prepare_benchmark(args.primary, args.shortened, args.output)
    elif args.command == "execution-budget":
        from .execution_budget import forecast_execution_budget, plan_execution_budget
        parameters = read_json(args.inputs)
        budget = forecast_execution_budget if "workload" in parameters else plan_execution_budget
        result = budget(**parameters)
        atomic(args.output, canonical(result) + b"\n")
    elif args.command == "validate-optimized":
        from .optimized_validation import validate_optimized
        receipt = validate_optimized(args.plans, args.model_dir, args.output,
            max_seconds=args.max_seconds, validation_scope=args.scope,
            native_mechanics_paths=args.native_mechanics)
        result = {"status": receipt["status"], "validation_scope": receipt["validation_scope"],
                  "receipt": str(args.output)}
    elif args.command == "timing-profiles":
        from .execution_budget import profiles_from_measurements
        result = profiles_from_measurements(args.runs)
        atomic(args.output, canonical(result) + b"\n")
    elif args.command == "collect-sequence":
        from .model import VerifiedModelSession
        from .run_experiment import collect
        requests = read_json(args.requests)
        if not isinstance(requests, list) or not requests:
            raise ValueError("collection sequence needs explicit request objects")
        session = VerifiedModelSession()
        result = []
        for request in requests:
            if not isinstance(request, dict):
                raise ValueError("invalid collection sequence request")
            metadata = collect(**request, model_session=session)["metadata"]
            result.append({"stage": metadata["stage"], "events": len(metadata["expected_event_ids"])})
    elif args.command == "collect":
        from .run_experiment import collect
        result = collect(args.plan, args.model_dir, args.output, args.mechanics, args.calibration,
            optimized=args.optimized, optimized_validation_path=args.optimized_validation,
            reuse_runs=args.reuse_runs)["metadata"]
        result = {"stage": result["stage"], "events": len(result["expected_event_ids"])}
    elif args.command == "freeze-calibration":
        from .analyze import calibration_freeze
        grouped = defaultdict(list)
        for run in args.runs:
            metadata = verify_bundle(run, "measurements")["metadata"]["plan_metadata"]
            units = set(metadata["units"].values())
            if len(units) != 1:
                raise ValueError("calibration budgets must match across domains")
            grouped[next(iter(units))].append(run)
        result = calibration_freeze(grouped, probe_bundle=args.probe, output_dir=args.output)
        result = {"status": result["status"], "selected_document_budget": result["selected_document_budget"]}
    elif args.command == "analyze":
        from .analyze import analyze_bundle
        analyzed = analyze_bundle(args.run, args.output, args.calibration)
        seal(args.output, "local-analysis", {"run_manifest_sha256": file_digest(args.run / "manifest.json")})
        result = analyzed["report"]
    elif args.command == "plan-main":
        from .planning import plan_main
        planned = plan_main(args.pilot,args.technical,args.calibration,args.prepared,args.review,settings,args.output,
                            remaining_workflow=read_json(args.remaining_workflow) if args.remaining_workflow else None)
        result = {"status":planned["status"],"selected":planned["selected"]}
    elif args.command == "freeze-main":
        from .planning import freeze_main
        result = freeze_main(args.planning,args.prepared,args.review,settings,args.output)["metadata"]
    else:
        from .analyze import analyze_sensitivity
        result = analyze_sensitivity(args.main, args.sensitivity, args.main_calibration,
                                     args.sensitivity_calibration, None, args.output)
        seal(args.output, "local-analysis", {"main_manifest_sha256": file_digest(args.main / "manifest.json"),
             "sensitivity_manifest_sha256": file_digest(args.sensitivity / "manifest.json")})
        result = result["summary"]
    print(canonical(result).decode())


if __name__ == "__main__":
    main()
