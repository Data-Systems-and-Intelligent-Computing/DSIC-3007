#!/usr/bin/env python3
"""User-invoked, single-attempt L40S screen. Default is a read-only dry run."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import select
import shlex
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.runpod.io/v2"
PACKAGE = ROOT / "data/optimization/paired-collector-2026-10-11-v6"
VOLUME = "k6u810gze3"
REGION = "US-MO-1"
GPU = "NVIDIA L40S"
IMAGE = "docker.io/aryamudas/dsic-3007@sha256:ae58005a9187a2e69f3d63bd3802eea063ae206801239abdc20e9286f77612d3"
REGISTRY = "cmv2myqnw003lvauwsz43uqnt"
REGISTRY_NAME = "dsic3007-dockerhub-aryamudas"
MAX_RATE = 1.09
STORAGE_ALLOWANCE_PER_HOUR = .02  # conservative allowance: 35 GB disk + 50 GB STANDARD volume (also covers HIGH_PERFORMANCE)
CASH_CAP = .40
MAX_SECONDS = 1200
SHUTDOWN_ALLOWANCE = 90
BOOT_MAX_SECONDS = 600
PAIRED_STARTUP_SECONDS = 240
LOG_PATH = None


class HTTPFailure(RuntimeError):
    def __init__(self, method, path, status):
        self.status = status
        super().__init__(f"{method} {path}: HTTP {status}; no retry")


def emit(message):
    line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}"
    if LOG_PATH is not None:
        with LOG_PATH.open("a") as stream:
            stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    print(line, flush=True)


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def api_key():
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        for line in (ROOT / ".env").read_text().splitlines():
            if line.strip().startswith("RUNPOD_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key or any(c in key for c in '\r\n"\\'):
        raise ValueError("valid RUNPOD_API_KEY required; credential is never logged")
    return key


def api(method, path, payload=None, *, allow_missing=False):
    if method == "POST" and path == "/pods" and payload and "volumeMounts" in payload:
        # Global Volume attachment is exposed by GraphQL, not REST v2.
        sys.path.insert(0, str(ROOT))
        from runpod.global_api import deploy
        return deploy(payload)
    # Curl's config arrives on stdin: the key is absent from process arguments,
    # persisted request files, remote env, and logs. No mutation is retried here.
    config = f'header = "Authorization: Bearer {api_key()}"\n'
    command = ["curl", "--config", "-", "--silent", "--show-error", "--max-time", "25",
               "--request", method, "--write-out", "\n%{http_code}", API + path]
    if payload is not None:
        encoded = json.dumps(payload, separators=(",", ":"))
        config += 'header = "Content-Type: application/json"\n'
        config += 'data = "' + encoded.replace("\\", "\\\\").replace('"', '\\"') + '"\n'
    result = subprocess.run(command, input=config, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise ConnectionError(f"{method} {path}: HTTP client failed ({result.returncode})")
    body, code = result.stdout.rsplit("\n", 1)
    status = int(code)
    if allow_missing and status in (404, 410):
        return None
    if status not in (200, 201, 204):
        raise HTTPFailure(method, path, status)
    return json.loads(body) if body.strip() else {}


def verify_package(package=PACKAGE):
    identity = json.loads((package / "identity.json").read_text())
    for name, value in identity["archives"].items():
        if file_hash(package / name) != value["sha256"]:
            raise ValueError(f"candidate archive changed: {name}")
        if (package / name).stat().st_size != value["size_bytes"]:
            raise ValueError(f"candidate archive size changed: {name}")
    required_archives = {"source.tar.gz", "screen.tar.gz"}
    if identity.get("proposal", {}).get("kind") == "paired-collector-benchmark":
        required_archives.add("baseline.tar.gz")
    if set(identity["archives"]) != required_archives:
        raise ValueError("both frozen candidate and screen archives are required")
    if identity.get("base_image") != IMAGE:
        raise ValueError("candidate base image differs from the pinned pod request")
    required = {"README.md", "configs/research.json", "src/verify.py",
                "runpod/screen.py", "runpod/screen_remote.py"}
    if identity.get("proposal", {}).get("kind") == "paired-collector-benchmark":
        required |= {"runpod/paired_benchmark.py", "runpod/paired-benchmark.json"}
        if not identity.get("baseline_files") or not identity.get("baseline_implementation_sha256"):
            raise ValueError("paired package lacks immutable baseline bindings")
    if not required.issubset(identity["local_files"]) or not identity.get("implementation_sha256"):
        raise ValueError("frozen package lacks protected source/runner/implementation bindings")
    for name, expected in identity["local_files"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"source changed after freeze: {name}; repackage before spending")
    return identity


def preflight():
    identity = verify_package()
    registry = api("GET", f"/registries/{REGISTRY}")
    if registry.get("id") != REGISTRY or registry.get("name") != REGISTRY_NAME:
        raise ValueError("existing Docker Hub registry credential identity differs")
    volume = api("GET", f"/network-volumes/{VOLUME}")
    if volume["dataCenter"] != REGION or volume["size"] != 50 or volume["type"] not in ("STANDARD", "HIGH_PERFORMANCE"):
        raise ValueError("replacement volume identity, region or size differs")
    pods = account_pods()
    if pods:
        raise ValueError("account has existing pods; preserve them and resolve verification status first")
    rows = api("GET", "/catalog/gpus?include=AVAILABILITY&product=POD&cloud=SECURE")["gpus"]
    card = next(row for row in rows if row["id"] == GPU)
    centers = {row["id"]: row["availability"] for row in card.get("dataCenters", [])}
    if centers.get(REGION) not in ("LOW", "MEDIUM", "HIGH"):
        raise ValueError("no current L40S stock in the replacement volume's region")
    rate = card["price"]["secure"]
    if type(rate) not in (int, float) or not math.isfinite(rate) or not 0 < rate <= MAX_RATE:
        raise ValueError("secure L40S catalog rate exceeds the frozen screen allowance")
    if not any(row["available"] and row["version"] == "13.0" for row in card.get("cudaVersions", [])):
        raise ValueError("CUDA 13.0 capacity is unavailable")
    private_key = Path.home() / ".ssh/id_ed25519"
    public_key = private_key.with_suffix(".pub")
    if not private_key.is_file() or not public_key.is_file():
        raise ValueError("existing ~/.ssh/id_ed25519 key pair required")
    return identity, rate, centers[REGION], public_key.read_text().strip()


def request_body(name, public_key, remote, profile=None, deadline_epoch=None):
    # The custom runtime image has no sshd. Install only OS access tooling;
    # Torch/Transformers and accepted numerical settings remain untouched.
    boot = f"""set -eu
mkdir -p {shlex.quote(remote + '/results')} /run/sshd /root/.ssh
exec > >(tee -a {shlex.quote(remote + '/results/bootstrap.log')}) 2>&1
echo '[BOOT] installing SSH access; staging and model work start only from user command'
export DEBIAN_FRONTEND=noninteractive
printf '#!/bin/sh\\nexit 101\\n' > /usr/sbin/policy-rc.d
chmod 755 /usr/sbin/policy-rc.d
apt-get update -qq
apt-get install -y --no-install-recommends openssh-server
printf '%s\\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
chmod 700 /root/.ssh
chmod 600 /root/.ssh/authorized_keys
ssh-keygen -A
echo '[BOOT] SSH ready'
exec /usr/sbin/sshd -D -e -o PermitRootLogin=prohibit-password -o PasswordAuthentication=no
"""
    if profile is not None:
        if profile.get("storage_type") != "global" or not profile.get("eligible_data_centers"):
            raise ValueError("verified global placement profile required")
        return {"name": name, "imageName": IMAGE, "containerRegistryAuthId": REGISTRY,
                "cloudType": "SECURE", "dataCenterIds": profile["eligible_data_centers"],
                "gpuTypeId": profile["gpu"], "gpuCount": 1, "minMemoryInGb": 64,
                "allowedCudaVersions": profile.get("allowed_cuda_versions", ["13.0"]), "containerDiskInGb": 35,
                "volumeMounts": [{"volumeId": profile["global_volume_id"],
                                  "volumeType": "OBJECT_STORE_VOLUME",
                                  "mountPath": profile["global_volume_mount_path"]}],
                "dockerArgs": shlex.join(["/bin/bash", "-lc", boot]),
                "ports": "22/tcp", "supportPublicIp": True, "startSsh": True,
                "terminateAfter": datetime.fromtimestamp(deadline_epoch, timezone.utc).isoformat(),
                "env": [{"key": k, "value": v} for k, v in {
                    "RESEARCH_IMAGE_DIGEST": IMAGE, "PUBLIC_KEY": public_key,
                    "DSIC_GLOBAL_VOLUME_ID": profile["global_volume_id"]}.items()]}
    return {"name": name, "image": IMAGE, "registry": REGISTRY, "cloud": "SECURE", "dataCenterIds": [REGION],
            "gpu": {"id": GPU, "count": 1, "minRamPerGpu": 64, "allowedCudaVersions": ["13.0"]},
            "disk": 35, "mounts": {"network": [{"volumeId": VOLUME, "path": "/workspace"}]},
            "entrypoint": [], "cmd": ["/bin/bash", "-lc", boot], "ports": ["22/tcp"],
            "startSsh": True, "env": {"RESEARCH_IMAGE_DIGEST": IMAGE, "PUBLIC_KEY": public_key}}


def matches_attempt(pod, state):
    profile = state.get("hardware_profile")
    if profile:
        env = pod.get("env", {})
        if isinstance(env, list):
            env = {r["key"]: r["value"] for r in env if isinstance(r, dict) and "key" in r and "value" in r}
        return (pod.get("name") == state["name"] and pod.get("image") == IMAGE and
                pod.get("registry") == state.get("registry_id") and
                pod.get("gpu", {}).get("id") == profile["gpu"] and
                env.get("DSIC_GLOBAL_VOLUME_ID") == profile["global_volume_id"])
    return (pod.get("name") == state["name"] and pod.get("image") == IMAGE and
            pod.get("registry") == state.get("registry_id") and
            pod.get("gpu", {}).get("id") == GPU and
            pod.get("mounts", {}).get("network") == [{"volumeId": VOLUME, "path": "/workspace"}])


def boot_failure(identifier):
    """Read a bounded SSE tail and return only a decisive container-start error."""
    response = subprocess.run(
        ["curl", "--config", "-", "--silent", "--show-error", "--max-time", "4",
         "--write-out", "\n%{http_code}", API + f"/pods/{identifier}/logs?source=system&tail=30"],
        input=f'header = "Authorization: Bearer {api_key()}"\n',
        capture_output=True, text=True, timeout=8)
    body, status = response.stdout.rsplit("\n", 1)
    if status != "200" or response.returncode not in (0, 28):
        raise ConnectionError("startup-log read failed; do not assume the image is loading")
    for event in body.splitlines():
        if event.startswith("data:"):
            row = json.loads(event[5:])
            line = row.get("line", "")
            if any(marker in line.lower() for marker in ("unauthorized: repository", "error creating container", "manifest unknown", "pull access denied")):
                return line
    return None


def account_pods():
    response = api("GET", "/pods?includeClusterPods=true&limit=100")
    if not isinstance(response, dict) or not isinstance(response.get("pods"), list):
        raise ValueError("unrecognized pod-list response; do not assume the account is empty")
    pagination = response.get("pagination")
    if (not isinstance(pagination, dict) or type(pagination.get("hasNextPage")) is not bool or
            "nextCursor" not in pagination):
        raise ValueError("pod-list pagination evidence missing; do not assume the account is empty")
    if response.get("nextCursor") or pagination.get("nextCursor") or pagination.get("hasNextPage"):
        raise ValueError("pod list paginated; ownership/readiness cannot be established")
    return response["pods"]


def validate_ready_pod(pod, profile=None):
    rate = pod.get("cost")
    if type(rate) not in (int, float) or not math.isfinite(rate) or not 0 < rate <= MAX_RATE:
        raise ValueError("actual running pod cost is missing, nonfinite or above the allowance")
    gpu = pod.get("gpu", {})
    memory = gpu.get("memory")
    if (gpu.get("count") != 1 or type(memory) not in (int, float) or not math.isfinite(memory) or memory < 64 or
            pod.get("cloud") != "SECURE" or
            pod.get("dataCenterId") not in (profile["eligible_data_centers"] if profile else [REGION]) or
            pod.get("cudaVersion") not in (profile.get("allowed_cuda_versions", ["13.0"]) if profile else ["13.0"])):
        raise ValueError("actual GPU count/RAM/cloud/region/CUDA differs from bounded request")


def terminate(attempt):
    state = json.loads((attempt / "attempt.json").read_text())
    identifier = state.get("pod_id")
    if not identifier:
        # Resolve only the unique name/request of this attempt after an ambiguous
        # create response. Never repeat POST or touch an unrelated listed pod.
        pods = account_pods()
        matching = [pod for pod in pods if matches_attempt(pod, state)]
        if len(matching) > 1:
            raise ValueError("ambiguous own-attempt pod; no deletion performed")
        identifier = matching[0]["id"] if matching else None
        if identifier is None and state.get("create_uncertain", True):
            emit("TEARDOWN no matching pod visible yet; uncertain create remains guarded")
            return False
    if identifier:
        pod = api("GET", f"/pods/{identifier}", allow_missing=True)
        if pod is not None:
            if not matches_attempt(pod, state):
                raise ValueError("pod is not the exact screen attempt; no deletion performed")
            api("DELETE", f"/pods/{identifier}", allow_missing=True)
            if api("GET", f"/pods/{identifier}", allow_missing=True) is not None:
                raise RuntimeError("pod deletion not confirmed; inspect Runpod console immediately")
        volume = state.get("hardware_profile", {}).get("global_volume_id", VOLUME)
        emit(f"TEARDOWN pod {identifier} deleted; user volume {volume} retained")
    write_json(attempt / "cleanup.json", {"status": "confirmed", "pod_id": identifier,
               "volume_deleted": False, "confirmed_at": time.time()})
    return True


def guard(attempt):
    state = json.loads((attempt / "attempt.json").read_text())
    emit(f"GUARD armed until epoch {state['deadline_epoch']:.0f}")
    while time.time() < state["deadline_epoch"] - 60:
        if (attempt / "cleanup.json").exists():
            return
        time.sleep(1)
    # Read-only waits may retry; deletion failure is retried only for this exact
    # own disposable pod. Every failed guard attempt remains visible in guard.log.
    for _ in range(3):
        try:
            if terminate(attempt):
                return
        except Exception as error:
            emit(f"GUARD ERROR {type(error).__name__}: {error}")
            time.sleep(2)
    emit("URGENT: guard could not confirm termination; delete this attempt's pod in the console")


def ssh_args(endpoint, attempt, *, scp=False):
    return ["scp" if scp else "ssh", "-i", str(Path.home() / ".ssh/id_ed25519"),
            "-P" if scp else "-p", str(endpoint["port"]), "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={attempt / 'known_hosts'}",
            "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2"]


def pipe_command(command, attempt, deadline, label):
    emit(f"STAGE {label}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               start_new_session=True)
    with (attempt / "console.log").open("ab") as log:
        last_heartbeat = time.monotonic()
        try:
            while True:
                if time.time() >= deadline:
                    raise TimeoutError(f"{label} reached its stage deadline")
                ready, _, _ = select.select([process.stdout], [], [], .5)
                if ready:
                    chunk = os.read(process.stdout.fileno(), 65536)
                    if chunk:
                        log.write(chunk)
                        log.flush()
                        sys.stdout.buffer.write(chunk)
                        sys.stdout.buffer.flush()
                    elif process.poll() is not None:
                        break
                if time.monotonic() - last_heartbeat >= 15:
                    emit(f"WAIT {label}; remaining stage budget {max(0, deadline-time.time()):.0f}s; log {attempt/'console.log'}")
                    last_heartbeat = time.monotonic()
            code = process.wait()
            if code:
                raise RuntimeError(f"{label} exited {code}; no automatic retry")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            os.fsync(log.fileno())


def run_attempt(identity, rate, public_key, max_seconds=MAX_SECONDS, profile=None):
    global LOG_PATH
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not SHUTDOWN_ALLOWANCE < max_seconds <= MAX_SECONDS:
        raise ValueError(f"attempt lease must be finite, greater than 90 and at most {MAX_SECONDS} seconds")
    limits = identity.get("operator_limits", {})
    if identity.get("proposal", {}).get("kind") == "paired-collector-benchmark":
        if not all(key in limits for key in ("remaining_cash_usd", "remaining_wall_seconds", "fixed_cost_bound_usd")):
            raise ValueError("paired benchmark requires explicit reconciled remaining cash/time and fixed cost bound")
    for key, value in limits.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid operator limit: {key}")
    if limits.get("remaining_cash_usd", 0) > 12 or limits.get("remaining_wall_seconds", 0) > 36000:
        raise ValueError("operator remaining limits exceed the user $12/ten-hour ceiling")
    cash_cap = min(CASH_CAP, limits.get("remaining_cash_usd", CASH_CAP))
    fixed_cost_bound = limits.get("fixed_cost_bound_usd", 0)
    wall_cap = min(max_seconds, limits.get("remaining_wall_seconds", max_seconds))
    lease = min(wall_cap, (cash_cap-fixed_cost_bound) / (rate + STORAGE_ALLOWANCE_PER_HOUR) * 3600)
    if not math.isfinite(lease) or lease <= SHUTDOWN_ALLOWANCE:
        raise ValueError("reconciled cash/time leaves no bounded work interval")
    name = "dsic-screen-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    attempt = ROOT / "runpod/runs" / name
    attempt.mkdir(parents=True)
    LOG_PATH = attempt / "console.log"
    latest = attempt.parent / "latest"
    if latest.is_symlink():
        latest.unlink()
    if latest.exists():
        raise FileExistsError("runpod/runs/latest is not a managed symlink")
    latest.symlink_to(name, target_is_directory=True)
    remote = ("/tmp/dsic3007-screen-attempts/" if profile else "/workspace/screen-attempts/") + name
    started = time.time()
    deadline = started + lease
    state = {"name": name, "pod_id": None, "started_epoch": started, "deadline_epoch": deadline,
             "cash_cap_usd": cash_cap, "operator_limits": limits,
             "fixed_cost_bound_usd": fixed_cost_bound, "catalog_rate_usd_per_hour": rate,
             "storage_hourly_allowance_usd": STORAGE_ALLOWANCE_PER_HOUR, "remote_directory": remote,
             "network_volume_id": VOLUME, "status": "creating"}
    state["registry_id"] = REGISTRY
    if profile:
        state["hardware_profile"] = profile
        state["operation"] = identity["proposal"].get("global_operation")
        state.pop("network_volume_id", None)
    state["create_uncertain"] = True
    write_json(attempt / "attempt.json", state)
    write_json(attempt / "candidate.json", identity)
    if profile:
        # Bind this exact operation/live profile on the remote side, rather than
        # uploading the package's operation-neutral preparation identity.
        write_json(attempt / "identity.json", identity)
    body = request_body(name, public_key, remote, profile, deadline) if profile else request_body(name, public_key, remote)
    write_json(attempt / "request.json", body)  # public key only; no API/R2/registry credential
    with (attempt / "guard.log").open("ab") as log:
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--guard", str(attempt)],
                         stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
    emit(f"ATTEMPT {name}; deadline includes create/stage/load/failure/teardown; cap ${cash_cap:.2f}")
    endpoint = None
    result = {"status": "incomplete", "scientific_collection_authorized": False}
    try:
        try:
            created = api("POST", "/pods", body)
        except HTTPFailure as error:
            if 400 <= error.status < 500:
                state["create_uncertain"] = False
                write_json(attempt / "attempt.json", state)
            raise
        state["pod_id"] = created["id"]
        state["create_uncertain"] = False
        write_json(attempt / "attempt.json", state)
        emit(f"POD {created['id']} created; waiting for runtime and SSH")
        last_wait = 0
        last_boot_log = 0
        startup_cap = identity.get("proposal", {}).get("startup_seconds",
            PAIRED_STARTUP_SECONDS if identity.get("proposal", {}).get("kind") == "paired-collector-benchmark" else BOOT_MAX_SECONDS)
        model_setup = bool(profile and state["operation"] == "prepare_model")
        if model_setup:
            # User increased setup startup to 600s. The overall cash deadline
            # and cleanup reserve still take precedence; no scientific change.
            startup_cap = 600
        boot_deadline = min(deadline - SHUTDOWN_ALLOWANCE, started + startup_cap)
        emit(f"BOOT allowance={max(0,boot_deadline-time.time()):.0f}s; "
             f"policy={'model setup with work/cleanup reserves' if model_setup else 'frozen diagnostic startup'}")
        while time.time() < boot_deadline:
            pod = api("GET", f"/pods/{created['id']}")
            if not matches_attempt(pod, state):
                raise ValueError("created pod configuration differs from exact request")
            if isinstance(pod.get("cost"), (int, float)) and pod["cost"] > MAX_RATE:
                raise ValueError("actual pod hourly cost exceeds the frozen allowance")
            if pod.get("status") == "EXITED" and time.time() - last_boot_log >= 30:
                failure = boot_failure(created["id"])
                last_boot_log = time.time()
                if failure:
                    raise RuntimeError(f"container startup failed: {failure}")
            endpoint = pod.get("ssh", {}).get("direct")
            write_json(attempt / "boot-status.json", {
                "observed_epoch": time.time(), "elapsed_seconds": time.time()-started,
                "status": pod.get("status"), "ssh_endpoint_present": bool(endpoint),
                "startup_deadline_epoch": boot_deadline,
                "pod": {k:pod.get(k) for k in ("id", "gpu", "cost", "dataCenterId", "cudaVersion")}})
            if endpoint:
                probe = subprocess.run(ssh_args(endpoint, attempt) +
                    [f"{endpoint['username']}@{endpoint['host']}", "true"], capture_output=True, timeout=15)
                if probe.returncode == 0:
                    validate_ready_pod(pod, profile) if profile else validate_ready_pod(pod)
                    write_json(attempt / "pod.json", {k: pod.get(k) for k in
                        ("id", "name", "status", "gpu", "cost", "dataCenterId", "cudaVersion", "ssh", "mounts")})
                    break
            if time.time()-last_wait >= 15:
                emit(f"BOOT status={pod.get('status')} SSH-ready={bool(endpoint)}; remaining={deadline-time.time():.0f}s")
                last_wait = time.time()
            time.sleep(3)
        else:
            raise TimeoutError("setup reached its bounded startup allowance; no benchmark started")
        target = f"{endpoint['username']}@{endpoint['host']}"
        work_deadline = deadline - SHUTDOWN_ALLOWANCE
        path_b = identity.get("proposal", {}).get("kind") == "path-b-preflight"
        upload_deadline = (min(work_deadline, time.time()+90) if model_setup or path_b
                           else min(work_deadline, started + startup_cap))
        emit(f"SETUP boot elapsed {time.time()-started:.0f}s; upload/preflight stage remaining {max(0, upload_deadline-time.time()):.0f}s; overall lease remaining {max(0, deadline-time.time()):.0f}s")
        files = [PACKAGE / name for name in sorted(identity.get("archives") or {"source.tar.gz": {}, "screen.tar.gz": {}})]
        files.append((attempt if profile else PACKAGE) / "identity.json")
        pipe_command(ssh_args(endpoint, attempt, scp=True) + [str(p) for p in files] + [target+":"+remote+"/"],
                     attempt, upload_deadline, "upload immutable candidate and screens")
        # Remote helper is copied separately and its hash is bound by identity.
        helper = ROOT / "runpod/screen_remote.py"
        if file_hash(helper) != identity["local_files"]["runpod/screen_remote.py"]:
            raise ValueError("remote runner changed after preflight; stop this attempt")
        pipe_command(ssh_args(endpoint, attempt, scp=True) + [str(helper), target+":"+remote+"/"],
                     attempt, upload_deadline, "upload bounded remote runner")
        remote_args = ["/opt/dsic3007/.venv/bin/python", "-u", remote+"/screen_remote.py",
                       "--directory", remote, "--deadline-epoch", str(work_deadline)]
        if identity.get("proposal", {}).get("kind") in ("paired-collector-benchmark", "path-b-preflight"):
            remote_args += ["--preflight-deadline-epoch", str(upload_deadline)]
        command = shlex.join(remote_args)
        label = ("prepare pinned model cache" if profile and state["operation"] == "prepare_model"
                 else "stage weights and validate optimized screen")
        pipe_command(ssh_args(endpoint, attempt) + [target, command], attempt, work_deadline, label)
        result["status"] = "remote_command_finished"
    except BaseException as error:
        result.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "incomplete_or_failed",
                      reason=f"{type(error).__name__}: {error}")
        emit(f"STOP {result['reason']}")
    finally:
        if endpoint and time.time() < deadline - 65:
            try:
                target = f"{endpoint['username']}@{endpoint['host']}"
                pipe_command(ssh_args(endpoint, attempt, scp=True) + ["-r", target+":"+remote+"/results", str(attempt)],
                             attempt, deadline - 65, "retrieve receipt, case progress and GPU/log files")
                result["local_results"] = str(attempt / "results")
            except Exception as error:
                location = (f"results on ephemeral pod disk may be lost during cleanup: {remote}" if profile
                            else f"results remain on user volume: {remote}")
                emit(f"RETRIEVAL incomplete ({type(error).__name__}); {location}")
                result["retrieval_error"] = f"{type(error).__name__}: {error}"
        try:
            if not terminate(attempt):
                result["teardown_error"] = "create response uncertain; guard remains armed, inspect console"
        except Exception as error:
            emit(f"TEARDOWN ERROR {error}; guard remains armed; inspect console")
            result["teardown_error"] = str(error)
        result.update(elapsed_seconds=time.time()-started, remote_directory=remote,
                      network_volume_retained=not bool(profile), whole_workflow_feasibility="unknown")
        if profile:
            result.update(global_volume_retained=True,
                          ephemeral_results_retrieval_required=True, operation=state["operation"])
        write_json(attempt / "result.json", result)
    retained_volume = profile["global_volume_id"] if profile else VOLUME
    emit(f"FINISHED {result['status']}; logs/results: {attempt}; volume {retained_volume} retained")
    return 0 if (result["status"] == "remote_command_finished" and "teardown_error" not in result and
                 "local_results" in result and "retrieval_error" not in result) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="USER command: one paid paired attempt, <=20min and $.40 including setup")
    parser.add_argument("--max-seconds", type=float, default=MAX_SECONDS,
                        help="reduce this attempt's overall lease; must be between 90 and 1200 seconds")
    parser.add_argument("--remaining-cash-usd", type=float,
                        help="operator-reconciled remaining portion of the $12 work allowance")
    parser.add_argument("--remaining-wall-seconds", type=float,
                        help="operator-reconciled remaining portion of the ten-hour allowance")
    parser.add_argument("--fixed-cost-bound-usd", type=float, default=.03,
                        help="planning bound for additional charges outside hourly burn; default $.03")
    parser.add_argument("--guard", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.guard:
        guard(args.guard)
        return 0
    if not math.isfinite(args.max_seconds) or not SHUTDOWN_ALLOWANCE < args.max_seconds <= MAX_SECONDS:
        parser.error("--max-seconds must be finite, greater than 90 and at most 1200")
    identity, rate, stock, public_key = preflight()
    emit(f"PREFLIGHT PASS L40S {REGION}: {stock} stock, ${rate:.2f}/h; volume {VOLUME}; one attempt, no retries")
    emit("Budget uses $.02/h disk+volume allowance; GPU fit/throughput unproven; screen remains diagnostic")
    if not args.run:
        emit("DRY RUN ONLY. User command requires reconciled limits: python runpod/screen.py --run --remaining-cash-usd CASH --remaining-wall-seconds SECONDS")
        return 0
    if identity.get("proposal", {}).get("kind") == "paired-collector-benchmark":
        if args.remaining_cash_usd is None or args.remaining_wall_seconds is None:
            parser.error("paid paired attempt requires --remaining-cash-usd and --remaining-wall-seconds; no remaining budget is inferred")
        values = (args.remaining_cash_usd, args.remaining_wall_seconds, args.fixed_cost_bound_usd)
        if any(not math.isfinite(value) or value < 0 for value in values):
            parser.error("operator budget limits must be finite and nonnegative")
        if args.remaining_cash_usd > 12 or args.remaining_wall_seconds > 36000:
            parser.error("remaining limits cannot exceed the user $12/ten-hour ceiling")
        identity = dict(identity, operator_limits={
            "remaining_cash_usd": args.remaining_cash_usd,
            "remaining_wall_seconds": args.remaining_wall_seconds,
            "fixed_cost_bound_usd": args.fixed_cost_bound_usd})
    return run_attempt(identity, rate, public_key, args.max_seconds)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        emit(f"ERROR {type(error).__name__}: {error}")
        raise SystemExit(1)
