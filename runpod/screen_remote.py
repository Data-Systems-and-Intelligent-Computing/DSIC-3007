#!/usr/bin/env python3
"""Bounded, single-attempt remote worker for the prepared diagnostic screen.

This helper does not create or stop cloud resources. The caller owns the billed
lease deadline. It only stages the immutable archives, acquires the pinned FP32
weights, and runs the diagnostic validator inside the supplied deadline.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import select
import shutil
import signal
import subprocess
import sys
import tarfile
import time


IMPLEMENTATION_FILES = (
    "model.py", "interventions.py", "run_experiment.py", "verify.py",
    "artifacts.py", "reuse.py", "observation_contract.py", "optimized_validation.py",
)
EXPECTED_TOPDIR = "optimization-screen-2026-10-11-v2"
PYTHON = Path("/opt/dsic3007/.venv/bin/python")
MODEL_DIR = Path("/workspace/models/OLMoE-fp32")
FINAL_RESERVE_SECONDS = 30.0
HEARTBEAT_SECONDS = 12.0


def emit(message):
    print(f"[screen-remote {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {message}", flush=True)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_file(path, deadline=None):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError("deadline reached while hashing candidate input")
            block = stream.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def safe_member(member):
    name = PurePosixPath(member.name)
    if (name.is_absolute() or not name.parts or ".." in name.parts or
            any(part in ("", ".") for part in name.parts)):
        raise ValueError(f"unsafe archive path: {member.name!r}")
    if not (member.isdir() or member.isfile()):
        raise ValueError(f"archive contains a link or special file: {member.name!r}")


def extract_checked(archive, destination, expected_files, deadline):
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        if not members:
            raise ValueError(f"empty archive: {archive.name}")
        for member in members:
            if time.time() >= deadline - FINAL_RESERVE_SECONDS:
                raise TimeoutError("deadline reached while inspecting archive")
            safe_member(member)
        payload = {member.name: member for member in members if member.isfile()}
        if set(payload) != set(expected_files):
            raise ValueError(f"archive member set differs from identity: {archive.name}")
        for member in members:
            target = destination.joinpath(*PurePosixPath(member.name).parts)
            if target.exists() and not member.isdir():
                raise FileExistsError(f"refusing to overwrite extracted candidate file: {member.name}")
        tar.extractall(destination, members=members, filter="data")
    for name, expected in expected_files.items():
        path = destination.joinpath(*PurePosixPath(name).parts)
        if not path.is_file() or sha256_file(path, deadline - FINAL_RESERVE_SECONDS) != expected:
            raise ValueError(f"extracted file hash differs from identity: {name}")


def implementation_hash(candidate):
    from_hashes = {}
    names = (*IMPLEMENTATION_FILES, "scope.py") if (candidate / "src/scope.py").exists() else IMPLEMENTATION_FILES
    for name in names:
        path = candidate / "src" / name
        from_hashes[name] = sha256_file(path)
    encoded = json.dumps(from_hashes, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def bind_image_identity(identity):
    # SSH need not inherit the container environment. The controller binds this
    # declared identity to its pod read-back; this is not container attestation.
    image = identity.get("base_image", "")
    if not isinstance(image, str):
        raise ValueError("candidate requires a digest-pinned base image")
    prefix, separator, digest = image.rpartition("@sha256:")
    if (not prefix or not separator or len(digest) != 64 or
            any(char not in "0123456789abcdef" for char in digest)):
        raise ValueError("candidate requires a digest-pinned base image")
    inherited = os.environ.get("RESEARCH_IMAGE_DIGEST")
    if inherited and inherited != image:
        raise ValueError("inherited image identity differs from frozen candidate")
    os.environ["RESEARCH_IMAGE_DIGEST"] = image
    return image


def validate_runtime(expected, cpu_only_setup=False):
    import torch
    import transformers
    actual = {"python": ".".join(map(str, sys.version_info[:3])),
              "torch": importlib.metadata.version("torch").split("+", 1)[0],
              "torch_cuda": torch.version.cuda,
              "transformers": transformers.__version__}
    for key in actual:
        if str(actual[key]) != str(expected[key]):
            raise ValueError(f"runtime identity mismatch for {key}: expected {expected[key]}, got {actual[key]}")
    if cpu_only_setup:
        # Download/hash and F32 metadata checks use CPU only. Do not initialize
        # the pinned CUDA13 runtime on an older host driver during preparation.
        names = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, timeout=10).strip().splitlines()
        if len(names) != 1 or "L40S" not in names[0]:
            raise ValueError("expected exactly one visible L40S device")
        actual.update(gpu_name=names[0], cuda_compute_tested=False,
                      gpu_interpretation="CPU-only model preparation; CUDA compatibility untested")
        return actual
    if not torch.cuda.is_available():
        raise ValueError("pinned runtime does not see a CUDA device")
    if torch.cuda.device_count() != 1 or "L40S" not in torch.cuda.get_device_name(0):
        raise ValueError("expected exactly one visible L40S device")
    properties = torch.cuda.get_device_properties(0)
    actual.update(gpu_name=properties.name, gpu_memory_total_bytes=int(properties.total_memory),
                  gpu_interpretation="observed device metadata; not a fit or peak-bound proof")
    return actual


def validate_plans(identity, candidate):
    from src.artifacts import digest, file_digest, verify_bundle

    proposal = identity["proposal"]
    path_b = proposal.get("kind") == "path-b-preflight"
    statuses = {"prepared_user_authorized_not_launched", "prepared_local_not_authorized"}
    if path_b:
        # Plan integrity is checked offline too; paid release is enforced by
        # the controller before live calls and recorded in its attempt identity.
        statuses.update({"prepared_pending_new_benchmark_authorization", "explicit_operator_release_one_attempt"})
    if proposal.get("status") not in statuses:
        raise ValueError("screen proposal status is not a prepared diagnostic state")
    settings_path = candidate / ("configs/research-path-b-v1.json" if path_b else "configs/research.json")
    if proposal.get("settings_sha256") != digest(read_json(settings_path)):
        raise ValueError("candidate research settings differ from the prepared screen")
    if (read_json(candidate / "configs/research.json").get("mechanics", {}).get("acceptance_tolerances") !=
            {"atol": 1e-6, "rtol": 1e-5} or
            read_json(candidate / "configs/research.json").get("mechanics", {}).get("nll_atol") != 1e-6):
        raise ValueError("frozen validation tolerances differ")
    plans = []
    for expected in proposal["plans"]:
        context = expected["context"]
        path = candidate / "data/plans" / ("path-b-preflight-v1" if path_b else EXPECTED_TOPDIR) / str(context)
        manifest = verify_bundle(path, "experiment-plan")
        if file_digest(path / "manifest.json") != expected["manifest_sha256"]:
            raise ValueError(f"sealed plan manifest changed for context {context}")
        metadata = manifest["metadata"]
        if metadata.get("stage") != "technical" or metadata.get("window") != context:
            raise ValueError(f"screen plan stage/context mismatch for {context}")
        plans.append(path)
    if path_b:
        from src.scope import is_path_b, validate_plan
        from src.artifacts import read_rows
        if len(plans) != 1 or not is_path_b(read_json(settings_path)):
            raise ValueError("Path B preflight uses one 4096/layer15 diagnostic plan")
        if read_json(plans[0] / "research.json") != read_json(settings_path):
            raise ValueError("Path B preflight plan settings differ from candidate")
        validate_plan(read_json(settings_path), verify_bundle(plans[0])["metadata"],
                      read_rows(plans[0] / "inputs.jsonl"))
        return plans, plans[0] / "model-catalog.json"
    primary_catalog = read_json(plans[0] / "model-catalog.json")
    matched_catalog = read_json(plans[1] / "model-catalog.json")
    if digest(primary_catalog) != digest(matched_catalog):
        raise ValueError("paired technical plans have different model catalogs")
    return plans, plans[0] / "model-catalog.json"


def model_bytes(catalog_path):
    catalog = read_json(catalog_path)
    return sum(item["size_bytes"] for item in catalog["files"])


def present_model_bytes(catalog_path):
    catalog = read_json(catalog_path)
    total = 0
    for item in catalog["files"]:
        path = MODEL_DIR / item["path"]
        if path.is_file():
            total += min(path.stat().st_size, item["size_bytes"])
    partial = 0
    incomplete_root = MODEL_DIR / ".cache/huggingface/download"
    if incomplete_root.is_dir():
        for path in incomplete_root.rglob("*.incomplete"):
            try:
                partial += path.stat().st_size
            except OSError:
                pass
    return total + min(partial, max(0, model_bytes(catalog_path) - total))


def gpu_sample(results):
    row = {"time_epoch": time.time(), "status": "unavailable"}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True)
        fields = [value.strip() for value in result.stdout.strip().split(",")]
        if len(fields) == 5:
            row = {"time_epoch": time.time(), "status": "observed", "name": fields[0],
                   "memory_total_mib": int(fields[1]), "memory_used_mib": int(fields[2]),
                   "memory_free_mib": int(fields[3]), "utilization_percent": int(fields[4]),
                   "interpretation": "observed telemetry only; not a fit or peak-bound proof"}
        else:
            row["reason"] = "unrecognized nvidia-smi output"
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        row["reason"] = f"{type(exc).__name__}: {exc}"
    with (results / "gpu-samples.jsonl").open("ab") as stream:
        stream.write(canonical(row) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    return row


def run_stage(command, candidate, results, status_path, deadline, stage, catalog_path=None):
    if time.time() >= deadline - FINAL_RESERVE_SECONDS:
        raise TimeoutError(f"insufficient bounded lease remaining to start {stage}")
    stage_started = time.time()
    state = read_json(status_path) if status_path.exists() else {}
    state.update(status="running", phase=stage, phase_started_epoch=stage_started,
                 deadline_epoch=deadline, elapsed_seconds=0,
                 remaining_seconds=max(0, deadline-time.time()))
    write_json(status_path, state)
    emit(f"STAGE {stage} START command={' '.join(command)}")
    gpu_sample(results)
    process = subprocess.Popen(command, cwd=candidate, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, start_new_session=True, bufsize=0,
                               env={**os.environ,
                                    "PYTHONPATH": str(candidate) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                                    "PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "0",
                                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    log_path = results / f"{stage}.log"
    last_heartbeat = time.monotonic()
    last_gpu_sample = 0.0
    with log_path.open("ab", buffering=0) as log:
        try:
            while True:
                now = time.time()
                if now >= deadline - FINAL_RESERVE_SECONDS:
                    raise TimeoutError(f"{stage} stopped at bounded work deadline")
                ready, _, _ = select.select([process.stdout], [], [], 1.0)
                if ready:
                    chunk = os.read(process.stdout.fileno(), 65536)
                    if chunk:
                        log.write(chunk)
                        sys.stdout.buffer.write(chunk)
                        sys.stdout.buffer.flush()
                    elif process.poll() is not None:
                        break
                mono = time.monotonic()
                if mono - last_gpu_sample >= HEARTBEAT_SECONDS:
                    sample = gpu_sample(results)
                    last_gpu_sample = mono
                if mono - last_heartbeat >= HEARTBEAT_SECONDS:
                    details = {}
                    if catalog_path is not None:
                        expected, present = model_bytes(catalog_path), present_model_bytes(catalog_path)
                        details = {"downloaded_bytes_estimate": present, "expected_model_bytes": expected,
                                   "downloaded_percent_estimate": round(100 * present / expected, 2) if expected else 0}
                    emit(f"HEARTBEAT phase={stage} elapsed={now-stage_started:.1f}s remaining={max(0, deadline-now):.1f}s gpu={json.dumps(sample, separators=(',', ':'))} {json.dumps(details, separators=(',', ':'))}")
                    state = read_json(status_path) if status_path.exists() else {}
                    state.update(status="running", phase=stage, phase_started_epoch=stage_started,
                                 updated_epoch=now, elapsed_seconds=now-stage_started,
                                 remaining_seconds=max(0, deadline-now), gpu=sample, **details)
                    write_json(status_path, state)
                    last_heartbeat = mono
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            os.fsync(log.fileno())
    phase = {"status": "pass", "started_epoch": stage_started,
             "finished_epoch": time.time(), "elapsed_seconds": time.time()-stage_started,
             "log": log_path.name, "returncode": process.returncode}
    state = read_json(status_path) if status_path.exists() else {}
    state.update(status="running", phase=stage + "_complete", updated_epoch=time.time(),
                 phases={**state.get("phases", {}), stage: phase})
    write_json(status_path, state)
    return phase


def run(directory, deadline_epoch, preflight_deadline_epoch=None):
    global MODEL_DIR
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if not math.isfinite(deadline_epoch) or deadline_epoch <= time.time() or deadline_epoch - time.time() > 1800:
        raise ValueError("deadline must be finite, in the future, and within the 30-minute screen limit")
    directory = Path(directory).resolve(strict=True)
    results = directory / "results"
    results.mkdir(exist_ok=True)
    status_path = results / "runner-status.json"
    receipt_path = results / "runner-receipt.json"
    state = {"status": "incomplete", "phase": "preflight", "started_epoch": time.time(),
             "deadline_epoch": deadline_epoch, "phases": {}, "validation_status": None}
    write_json(status_path, state)
    final_status = "incomplete"
    try:
        if not PYTHON.is_file():
            raise FileNotFoundError(f"pinned environment Python missing: {PYTHON}")
        if preflight_deadline_epoch is not None:
            if not math.isfinite(preflight_deadline_epoch):
                raise ValueError("preflight deadline must be finite")
            if time.time() >= preflight_deadline_epoch:
                raise TimeoutError("startup/upload/preflight allowance reached")
        preflight_cutoff = min(deadline_epoch-FINAL_RESERVE_SECONDS,
                               preflight_deadline_epoch if preflight_deadline_epoch is not None else deadline_epoch)
        if time.time() >= deadline_epoch - FINAL_RESERVE_SECONDS:
            raise TimeoutError("deadline leaves no bounded work interval")
        identity_path = directory / "identity.json"
        identity = read_json(identity_path)
        profile = identity.get("hardware_profile")
        if profile:
            mount = Path(profile["global_volume_mount_path"])
            if profile.get("storage_type") != "global" or not os.path.ismount(mount):
                raise ValueError("expected mounted Global Volume; no regional/local cache fallback")
            if str(directory).startswith(str(mount)+"/"):
                raise ValueError("attempt/journals must use local POSIX disk")
            MODEL_DIR = mount / "models/OLMoE-fp32"
        if not all(key in identity for key in ("archives", "local_files", "files", "implementation_sha256", "proposal", "expected_runtime")):
            raise ValueError("candidate identity lacks required hash and runtime fields")
        paired = identity["proposal"].get("kind") == "paired-collector-benchmark"
        path_b = identity["proposal"].get("kind") == "path-b-preflight"
        archives = ("source.tar.gz", "screen.tar.gz", "baseline.tar.gz") if paired else ("source.tar.gz", "screen.tar.gz")
        for archive_name in archives:
            spec = identity["archives"].get(archive_name)
            archive = directory / archive_name
            if not spec or archive.stat().st_size != spec["size_bytes"] or sha256_file(archive, preflight_cutoff) != spec["sha256"]:
                raise ValueError(f"candidate archive size/hash mismatch: {archive_name}")
        local_helper_hash = identity["local_files"].get("runpod/screen_remote.py")
        if not local_helper_hash or sha256_file(Path(__file__)) != local_helper_hash:
            raise ValueError("uploaded remote runner differs from its candidate identity")
        candidate = directory / "candidate"
        if candidate.exists():
            raise FileExistsError("candidate directory already exists; do not overwrite prior extraction")
        source_destination = candidate
        extract_checked(directory / "source.tar.gz", source_destination,
                        identity["files"], preflight_cutoff+FINAL_RESERVE_SECONDS)
        sys.path.insert(0, str(source_destination))
        plans_root = source_destination / "data/plans"
        extract_checked(directory / "screen.tar.gz", plans_root,
                        identity.get("screen_files", _archive_member_hashes(directory / "screen.tar.gz", preflight_cutoff)),
                        preflight_cutoff+FINAL_RESERVE_SECONDS)
        if paired:
            baseline_source = directory / "baseline"
            if baseline_source.exists():
                raise FileExistsError("baseline source already exists; preserve immutable previous extraction")
            extract_checked(directory / "baseline.tar.gz", baseline_source,
                            identity["baseline_files"], preflight_cutoff+FINAL_RESERVE_SECONDS)
            if implementation_hash(baseline_source) != identity["baseline_implementation_sha256"]:
                raise ValueError("preserved baseline implementation identity mismatch")
        local_files = identity["local_files"]
        for name, expected in local_files.items():
            if name in identity["files"] and identity["files"][name] != expected:
                raise ValueError(f"local/source archive identity hash differs: {name}")
        actual_impl = implementation_hash(source_destination)
        if actual_impl != identity["implementation_sha256"]:
            raise ValueError("optimized implementation identity mismatch")
        declared_image = bind_image_identity(identity)
        cpu_only_setup = bool(profile and identity["proposal"].get("global_operation") == "prepare_model")
        runtime = (validate_runtime(identity["expected_runtime"], cpu_only_setup=True) if cpu_only_setup
                   else validate_runtime(identity["expected_runtime"]))
        plans, catalog_path = validate_plans(identity, source_destination)
        if preflight_deadline_epoch is not None and time.time() >= preflight_deadline_epoch:
            raise TimeoutError("startup/upload/preflight allowance reached")
        model_parent = MODEL_DIR.parent
        model_parent.mkdir(parents=True, exist_ok=True)
        needed_bytes = max(0, model_bytes(catalog_path) - present_model_bytes(catalog_path))
        available_bytes = shutil.disk_usage(model_parent).free
        if available_bytes < needed_bytes + 1024**3:
            raise OSError(f"insufficient model volume space: need about {needed_bytes + 1024**3} bytes including 1 GiB margin; free={available_bytes}")
        state.update(phase="preflight", candidate_implementation_sha256=actual_impl,
                     declared_image=declared_image,
                     image_identity_source="frozen candidate bound to controller pod read-back; not in-container attestation",
                     runtime=runtime, plan_contexts=[4096] if path_b else [4096, 2048], status="running")
        write_json(status_path, state)
        state["phases"]["preflight"] = {"status": "pass", "finished_epoch": time.time()}
        write_json(status_path, state)

        if identity["proposal"].get("global_operation") == "prepare_model":
            if not profile:
                raise ValueError("model preparation requires an explicit global profile")
            setup_output = results / "model-preparation.json"
            command = [str(PYTHON), "-u", "-m", "runpod.global_model_setup",
                       "--catalog", str(catalog_path), "--mount", str(mount),
                       "--output", str(setup_output), "--deadline-epoch", str(deadline_epoch-FINAL_RESERVE_SECONDS)]
            state["phases"]["model_preparation"] = run_stage(command, source_destination, results,
                status_path, deadline_epoch, "model_preparation", catalog_path)
            setup = read_json(setup_output)
            if setup.get("status") != "pass":
                raise ValueError("model preparation did not complete; partial global cache retained")
            state.update(status="pass", phase="model_preparation_complete", finished_epoch=time.time(),
                         elapsed_seconds=time.time()-state["started_epoch"],
                         global_volume_id=profile["global_volume_id"], validation_status="not_run_setup_only",
                         validation_is_scientific_approval=False, model_cache_ready=True)
            write_json(status_path, state)
            write_json(receipt_path, state)
            return 0

        weights_command = [str(PYTHON), "-u", "-m", "src.workflow", "weights",
                           "--catalog", str(catalog_path), "--output", str(MODEL_DIR)]
        weights_deadline = deadline_epoch
        if paired or path_b:
            # Never acquire or repair weights during this bounded paired screen.
            for asset in read_json(catalog_path)["files"]:
                path = MODEL_DIR / asset["path"]
                if not path.is_file() or path.stat().st_size != asset["size_bytes"]:
                    raise ValueError("paired benchmark requires complete cached weights; no download permitted")
            weights_command = [str(PYTHON), "-u", "-c",
                               "from src.model import verify_model; from src.artifacts import read_json; import sys; verify_model(sys.argv[1], read_json(sys.argv[2]))",
                               str(MODEL_DIR), str(catalog_path)]
            weights_deadline = min(deadline_epoch, time.time() + 180 + FINAL_RESERVE_SECONDS)
        if time.time() >= preflight_cutoff:
            raise TimeoutError("startup/upload/preflight allowance reached before weights")
        # User removed repeated weights verification for Path B. The controller
        # requires the completed setup receipt; only cheap presence/size checks
        # remain here. Native comparisons still gate timings.
        try:
            if path_b:
                phase = {"status": "not_run_user_removed_reverification",
                         "verification": "completed preparation receipt only; no fresh byte/F32 check",
                         "standalone_hash_scan": False}
            else:
                phase = run_stage(weights_command, source_destination, results, status_path,
                                  weights_deadline, "weights", catalog_path)
            state["phases"]["weights"] = phase
            state.update(phase="cache_presence_complete" if path_b else "weights_complete",
                         updated_epoch=time.time(), status="running")
            write_json(status_path, state)
        except BaseException as exc:
            state["phases"]["weights"] = {"status": "incomplete" if isinstance(exc, TimeoutError) else "fail",
                                             "finished_epoch": time.time(),
                                             "error": f"{type(exc).__name__}: {exc}"}
            state.update(phase="weights_incomplete", updated_epoch=time.time(), status="running")
            write_json(status_path, state)
            raise

        remaining_for_validation = deadline_epoch - time.time() - FINAL_RESERVE_SECONDS
        if remaining_for_validation <= 0:
            raise TimeoutError("weight acquisition left no validation time before the work deadline")
        validation_cap = min(1800.0, remaining_for_validation - 15.0)
        if validation_cap <= 0:
            raise TimeoutError("weight acquisition left less than 15 seconds to persist a validation receipt")
        validation_path = results / "validation.json"
        command = [str(PYTHON), "-u", "-m", "src.workflow", "validate-optimized",
                   "--plans", *map(str, plans), "--model-dir", str(MODEL_DIR),
                   "--scope", "screen", "--max-seconds", str(validation_cap),
                   "--output", str(validation_path)]
        if paired:
            command = [str(PYTHON), "-u", "-m", "runpod.paired_benchmark",
                       "--plans", str(plans[0]), str(plans[1]),
                       "--baseline-source", str(baseline_source),
                       "--selection", str(source_destination / "runpod/paired-benchmark.json"),
                       "--model-dir", str(MODEL_DIR),
                       "--deadline-epoch", str(deadline_epoch-FINAL_RESERVE_SECONDS),
                       "--output", str(validation_path)]
        if path_b:
            command = [str(PYTHON), "-u", "-m", "runpod.path_b_benchmark",
                       "--plan", str(plans[0]), "--model-dir", str(MODEL_DIR),
                       "--deadline-epoch", str(deadline_epoch-FINAL_RESERVE_SECONDS),
                       "--output", str(validation_path)]
        try:
            phase = run_stage(command, source_destination, results, status_path,
                              deadline_epoch, "validation", catalog_path)
            state["phases"]["validation_command"] = phase
            state.update(phase="validation_command_complete", updated_epoch=time.time(), status="running")
            write_json(status_path, state)
        except BaseException as exc:
            state["phases"]["validation_command"] = {"status": "incomplete" if isinstance(exc, TimeoutError) else "fail",
                                                        "finished_epoch": time.time(),
                                                        "error": f"{type(exc).__name__}: {exc}"}
            state.update(phase="validation_command_incomplete", updated_epoch=time.time(), status="running")
            if validation_path.is_file():
                try:
                    validation = read_json(validation_path)
                    state["validation_status"] = validation.get("status")
                    state["phases"]["validation_receipt"] = {"status": validation.get("status"),
                                                                  "receipt_sha256": sha256_file(validation_path)}
                except Exception:
                    pass
            write_json(status_path, state)
            raise
        if not validation_path.is_file():
            raise ValueError("validator process returned success without a validation receipt")
        validation = read_json(validation_path)
        state["validation_status"] = validation.get("status")
        state["phases"]["validation_receipt"] = {"status": validation.get("status"),
                                                     "receipt_sha256": sha256_file(validation_path)}
        if validation.get("status") != "pass":
            raise ValueError(f"validation receipt status is {validation.get('status')!r}")
        final_status = "pass"
    except BaseException as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            final_status = "interrupted"
        elif isinstance(exc, TimeoutError) or state.get("validation_status") == "incomplete":
            final_status = "incomplete"
        else:
            final_status = "fail"
        emit(f"STOP {final_status}: {state['error']}")
    state.update(status=final_status, finished_epoch=time.time(), elapsed_seconds=time.time()-state["started_epoch"],
                 validation_is_scientific_approval=False,
                 note="screen results are diagnostic only; runner does not approve collection")
    write_json(status_path, state)
    write_json(receipt_path, state)
    emit(f"FINAL {final_status}; validation_status={state.get('validation_status')} status={status_path} receipt={receipt_path}")
    return 0 if final_status == "pass" else 1


def _archive_member_hashes(archive, deadline):
    hashes = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            safe_member(member)
            if not member.isfile():
                continue
            if time.time() >= deadline:
                raise TimeoutError("deadline reached while verifying archive members")
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError(f"unreadable archive member: {member.name}")
            h = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                if time.time() >= deadline:
                    raise TimeoutError("deadline reached while hashing archive members")
                h.update(block)
            hashes[member.name] = h.hexdigest()
    return hashes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--preflight-deadline-epoch", type=float)
    args = parser.parse_args(argv)
    if (not math.isfinite(args.deadline_epoch) or args.deadline_epoch <= time.time() or
            args.deadline_epoch - time.time() > 1800):
        parser.error("--deadline-epoch must be finite, in the future, and within the 30-minute screen limit")
    return run(args.directory, args.deadline_epoch, args.preflight_deadline_epoch)


if __name__ == "__main__":
    raise SystemExit(main())
