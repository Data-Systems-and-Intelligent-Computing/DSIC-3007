#!/usr/bin/env python3
"""User-run bounded Path B setup/diagnostic; separate explicit model-resume grant."""

import argparse
import json
import math
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runpod import screen
from src.artifacts import (atomic, canonical, digest, file_digest, priority,
                           read_json, read_rows, seal, verify_bundle, write_rows)
from src.model import implementation_identity
from src.scope import scope_identity, validate_plan

PACKAGE = ROOT / "data/optimization/path-b-global-preflight-2026-10-13-v11"
TECHNICAL = ROOT / "data/plans/path-b-v1-2026-10-12/technical"


def prepare(package=PACKAGE, technical=TECHNICAL):
    """Freeze explicit source/window hashes, without any network or paid action."""
    manifest = verify_bundle(technical, "experiment-plan")
    settings = read_json(technical/"research.json")
    inputs = list(read_rows(technical/"inputs.jsonl"))
    validate_plan(settings, manifest["metadata"], inputs)
    chosen = []
    for domain in settings["domains"]:
        rows = [r for r in inputs if r["domain"] == domain]
        if not rows:
            raise ValueError("missing technical domain")
        chosen.append(min(rows, key=lambda r: (
            priority(3007,"source-allocation",[domain,"technical",r["source_group"]]),
            r["sample_order"], r["document_id"], r["input_id"])))
    package.mkdir(parents=True, exist_ok=False)
    plan = package/"plans/path-b-preflight-v1/4096"
    plan.mkdir(parents=True)
    ids = {r["document_id"] for r in chosen}
    write_rows(plan/"inputs.jsonl", chosen)
    write_rows(plan/"documents.jsonl", (d for d in read_rows(technical/"documents.jsonl") if d["document_id"] in ids))
    for name in ("research.json", "source-review.json", "model-catalog.json"):
        atomic(plan/name, (technical/name).read_bytes())
    seal(plan,"experiment-plan",dict(manifest["metadata"],input_count=3, benchmark_screen=True,
        preflight_diagnostic=True, units={d:1 for d in settings["domains"]},
        screen_parent_sha256=file_digest(technical/"manifest.json")))
    source_paths = [ROOT/"README.md", ROOT/"WORKFLOW.md", ROOT/"pyproject.toml", ROOT/"uv.lock",
        ROOT/"docs/validation/PATH-B-WEIGHTS-REVERIFICATION-REMOVAL-2026-10-13.md",
        *sorted((ROOT/"src").glob("*.py")), *sorted((ROOT/"runpod").glob("*.py")),
        *sorted((ROOT/"configs").glob("*.json"))]
    files = {p.relative_to(ROOT).as_posix():file_digest(p) for p in source_paths}
    screen_files = {p.relative_to(package/"plans").as_posix():file_digest(p)
                    for p in plan.rglob("*") if p.is_file()}
    with tarfile.open(package/"source.tar.gz", "w:gz") as tar:
        for path in source_paths:
            tar.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
    with tarfile.open(package/"screen.tar.gz", "w:gz") as tar:
        for name in sorted(screen_files):
            tar.add(package/"plans"/name, arcname=name, recursive=False)
    previous = read_json(ROOT/"data/optimization/paired-collector-2026-10-11-v6/identity.json")
    proposal = {"kind":"path-b-preflight", "status":"prepared_user_authorized_not_launched",
        "settings_sha256":digest(settings), "research_scope":scope_identity(settings),
        "plans":[{"context":4096,"manifest_sha256":file_digest(plan/"manifest.json")}],
        "selected_cases":[{k:r[k] for k in ("input_id","domain","document_id","source_group","predictor_position","target_id")} for r in chosen],
        "startup_seconds":600, "supported_maximum_wall_seconds":1800,
        "paid_authorization":"one user-run $.30/900s benchmark; explicit authorization ID required",
        "maximum_all_in_cost_usd":.30, "maximum_all_phase_wall_seconds":900,
        "phase_caps_seconds":{"startup":600,"upload_preflight_after_ssh":90,
            "verified_load_native_warmup_collection":"shared remaining work deadline; no independent phase caps",
            "remote_finalization":30,"retrieval_teardown":90},
        "counts":{"windows":3,"native_comparisons":24,"native_validation_full_forwards":33,
            "warmup_inputs":1,"measured_inputs":3,"collector_full_forwards_including_warmup":8,
            "suffix_replays_including_validation_warmup_measurement":56,"measured_effects":24},
        "acceptance":"diagnostic only; full technical validation still required",
        "weights":"completed preparation receipt required; per-run SHA256/F32 scan removed by user; no copy/download/repair",
        "fixed_cost_bound_usd":.03,
        "cash_deadline":"min(operator_wall_cap,3600*(operator_cash_cap-fixed_cost_bound)/(max(live_rate,1.09)+.02)); reserves90s cleanup plus30s remote finalization",
        "completion":"phase caps and historical transfers are conditional; incomplete is a valid stop"}
    identity = {"base_image":screen.IMAGE,"expected_runtime":previous["expected_runtime"],
        "implementation_sha256":implementation_identity(), "proposal":proposal,
        "files":files,"local_files":files,"screen_files":screen_files,
        "archives":{n:{"sha256":file_digest(package/n),"size_bytes":(package/n).stat().st_size}
                    for n in ("source.tar.gz","screen.tar.gz")}}
    atomic(package/"identity.json",canonical(identity)+b"\n")
    return identity


def configure(package=PACKAGE):
    screen.PACKAGE = package
    screen.CASH_CAP = .20
    screen.MAX_SECONDS = 1800


def benchmark_authorization(config, authorization_id):
    """Only a separately authorized grant can release paid work."""
    auth = config["paid_authorization"]["benchmark_retry_proposal"]
    if auth.get("status") != "authorized":
        raise ValueError("paid benchmark disabled: original diagnostic consumed and v9 retry proposal withdrawn; no active grant")
    if not authorization_id or authorization_id != auth["authorization_id"]:
        raise ValueError("the $.20 diagnostic grant is consumed; a new budget decision is required; "
                         "an explicit active authorization ID is required")
    if (type(auth.get("maximum_usd")) not in (int, float) or not math.isfinite(auth["maximum_usd"]) or
            not 0 < auth["maximum_usd"] <= 12 or type(auth.get("maximum_wall_seconds")) not in (int, float) or
            not math.isfinite(auth["maximum_wall_seconds"]) or not 90 < auth["maximum_wall_seconds"] <= 1800 or
            auth.get("maximum_attempts") != 1):
        raise ValueError("invalid separately authorized one-attempt budget")
    if read_json(ROOT/auth["parent_run"]/"cleanup.json").get("status") != "confirmed":
        raise ValueError("preceding benchmark cleanup is not confirmed")
    for candidate in (ROOT/"runpod/runs").glob("*/candidate.json"):
        if read_json(candidate).get("proposal", {}).get("benchmark_authorization_id") == authorization_id:
            raise ValueError("benchmark grant already used; reconcile budget before another attempt")
    return auth


def resume_authorization(config):
    auth = config["paid_authorization"]["model_resume"]
    if (auth.get("maximum_usd"), auth.get("maximum_wall_seconds"), auth.get("maximum_attempts")) != (.30, 900, 1):
        raise ValueError("expected the one-attempt $.30/900s resume grant")
    parent = ROOT/auth["parent_run"]
    if auth.get("preceding_resume_run") and read_json(ROOT/auth["preceding_resume_run"]/"cleanup.json").get("status") != "confirmed":
        raise ValueError("preceding resume cleanup is not confirmed")
    previous = read_json(parent/"candidate.json")
    if (previous.get("hardware_profile", {}).get("global_volume_id") != config["global_volume_id"] or
            previous.get("proposal", {}).get("global_operation") != "prepare_model" or
            read_json(parent/"cleanup.json").get("status") != "confirmed" or
            read_json(parent/"results/model-preparation.json").get("status") != "incomplete"):
        raise ValueError("resume requires the matching incomplete setup and confirmed cleanup")
    for candidate in (ROOT/"runpod/runs").glob("*/candidate.json"):
        if read_json(candidate).get("proposal", {}).get("resume_authorization_id") == auth["authorization_id"]:
            raise ValueError("resume grant already used; reconcile budget before another attempt")
    return auth


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--check-live", action="store_true", help="read-only quote/stock/pod/registry checks")
    group.add_argument("--setup-model", action="store_true", help="USER: one bounded paid fresh-model setup attempt")
    group.add_argument("--resume-model", action="store_true", help="USER: one authorized $.30/900s partial-cache resume")
    parser.add_argument("--fixed-cost-bound-usd", type=float, default=.03)
    parser.add_argument("--benchmark-authorization-id",
                        help="USER: release only a separately authorized active grant; the v9 proposal is withdrawn")
    parser.add_argument("--legacy-regional", action="store_true",
                        help="explicit historical L40S/network-volume profile; not Global Volume")
    args = parser.parse_args(argv)
    if args.benchmark_authorization_id and (not args.run or args.legacy_regional):
        parser.error("benchmark authorization applies only to --run with the Global Volume")
    if args.run and args.legacy_regional:
        raise ValueError("historical diagnostic grant is consumed; use the proposed Global Volume retry")
    if not args.legacy_regional:
        from runpod.global_preflight import CONFIG, status, live_profile, cache_ready
        config = read_json(CONFIG)
        if args.prepare:
            identity = prepare()
            identity["hardware_profile"] = {k:config[k] for k in ("storage_type","gpu","global_volume_id",
                "global_volume_mount_path","global_volume_model_path")}
            identity["global_config_sha256"] = digest(config)
            atomic(PACKAGE/"identity.json",canonical(identity)+b"\n")
            print(canonical({"status":"global_package_prepared_no_paid_resources", "package":str(PACKAGE)}).decode())
            return 0
        configure()
        screen.MAX_RATE = config["maximum_gpu_rate_usd_per_hour"]
        identity = screen.verify_package(PACKAGE)
        if identity.get("global_config_sha256") != digest(config):
            raise ValueError("global profile changed after freeze; prepare a new successor")
        if not args.run and not args.setup_model and not args.resume_model:
            result = status(read_live=args.check_live)
            result.update(package_verified=True, global_attachment_backend="official SDK GraphQL volumeMounts")
            print(canonical(result).decode())
            return 0
        setup = args.setup_model or args.resume_model
        if setup and config.get("model_preparation_status") == "complete":
            raise ValueError("model preparation is complete; setup/resume reruns are disabled")
        operation = "prepare_model" if setup else "benchmark"
        cash_cap, wall_cap, resume, benchmark = .20, 900, None, None
        if args.resume_model:
            resume = resume_authorization(config)
            cash_cap = resume["maximum_usd"]
        if args.setup_model:
            auth = config["paid_authorization"]
            if auth.get("download_setup_cash_and_time_limit") != {"maximum_usd":.20,"maximum_wall_seconds":900}:
                raise ValueError("bounded model preparation has not been authorized")
        elif not args.resume_model:
            benchmark = benchmark_authorization(config, args.benchmark_authorization_id)
            cash_cap, wall_cap = benchmark["maximum_usd"], benchmark["maximum_wall_seconds"]
            cache_ready(config)
        screen.CASH_CAP = cash_cap
        profile, rate, key = live_profile(operation=operation)
        static = {k:profile[k] for k in identity["hardware_profile"]}
        if static != identity["hardware_profile"]:
            raise ValueError("live volume/GPU differs from frozen package")
        fixed = .08 if setup else max(.03,args.fixed_cost_bound_usd)
        if not math.isfinite(fixed) or not .03 <= fixed < cash_cap:
            raise ValueError("fixed cost allowance must be finite and below cash cap")
        proposal = dict(identity["proposal"],global_operation=operation)
        if resume:
            proposal.update(resume_authorization_id=resume["authorization_id"], resume_parent_run=resume["parent_run"])
        if benchmark:
            proposal.update(benchmark_authorization_id=benchmark["authorization_id"],
                            benchmark_parent_run=benchmark["parent_run"],
                            status="explicit_operator_release_one_attempt",
                            maximum_all_in_cost_usd=cash_cap, maximum_all_phase_wall_seconds=wall_cap,
                            fixed_cost_bound_usd=fixed)
        if setup:
            proposal.update(weights="explicit user-authorized pinned download; setup only",
                            startup_seconds=600,
                            phase_caps_seconds={"startup":600,"upload_preflight_after_ssh":90,
                                                "remote_finalization":30,"retrieval_teardown":90},
                            maximum_all_in_cost_usd=cash_cap, fixed_cost_bound_usd=fixed,
                            cash_deadline="min(900,3600*(operator cash cap-fixed allowance)/(live GPU guard rate+.02)); reserves90s cleanup",
                            counts={"native_comparisons":0,"measured_inputs":0,"measured_effects":0},
                            acceptance="model byte/dtype preparation only; no numerical/scientific acceptance")
        identity = dict(identity, hardware_profile=profile,
            proposal=proposal,
            operator_limits={"remaining_cash_usd":cash_cap,"remaining_wall_seconds":wall_cap,"fixed_cost_bound_usd":fixed})
        screen.emit(f"GLOBAL {operation}; eligible regions={profile['eligible_data_centers']}; "
                    f"quote=${rate:.2f}/h; ${cash_cap:.2f}/{wall_cap}s planning guard; fixed allowance=${fixed:.2f}; no retry")
        return screen.run_attempt(identity,max(rate,screen.MAX_RATE),key,max_seconds=wall_cap,profile=profile)
    if args.resume_model:
        raise ValueError("resume grant applies only to the selected Global Volume")
    if args.prepare:
        identity = prepare()
        print(canonical(identity["proposal"]).decode())
        return 0
    configure()
    # verify_package's historical default argument binds the old package; pass explicitly.
    identity = screen.verify_package(PACKAGE)
    if identity["proposal"]["kind"] != "path-b-preflight":
        raise ValueError("Path B diagnostic package required")
    if not args.run and not args.check_live:
        print(canonical({"status":"offline_package_verified_no_gpu_launched", "proposal":identity["proposal"]}).decode())
        return 0
    # The inherited preflight calls verify_package without args. Bind this exact successor.
    original = screen.verify_package
    screen.verify_package = lambda: original(PACKAGE)
    identity, rate, stock, key = screen.preflight()
    fixed = args.fixed_cost_bound_usd
    if not math.isfinite(fixed) or not .03 <= fixed < .20:
        parser.error("fixed cost bound must be finite, >=$.03 and <$.20")
    lease = min(900,3600*(.20-fixed)/(max(rate,screen.MAX_RATE)+screen.STORAGE_ALLOWANCE_PER_HOUR))
    if lease <= screen.SHUTDOWN_ALLOWANCE:
        raise ValueError("no remaining interval after cleanup reserve")
    screen.emit(f"PATH B PREFLIGHT live stock={stock}, quote=${rate:.2f}/h; guarded lease={lease:.1f}s, cash cap=$.20 including ${fixed:.2f} fixed allowance; cleanup90s")
    if not args.run:
        return 0
    identity = dict(identity,operator_limits={"remaining_cash_usd":.20,
        "remaining_wall_seconds":900,"fixed_cost_bound_usd":fixed})
    return screen.run_attempt(identity, max(rate,screen.MAX_RATE),key,max_seconds=900)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        screen.emit(f"ERROR {type(error).__name__}: {error}")
        raise SystemExit(1)
