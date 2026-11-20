"""User-run GPU diagnostic configured in execution.json, with API pod cleanup."""
from __future__ import annotations

import argparse
from collections import deque
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.runpod.io/v2"


class APIError(RuntimeError):
    def __init__(self, status, message, response_headers=None):
        super().__init__(message)
        self.status = status
        self.response_headers = response_headers or {}


def api_key():
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        for line in (ROOT / ".env").read_text().splitlines():
            match = re.match(r'^\s*(?:export\s+)?RUNPOD_API_KEY\s*=\s*(.*?)\s*$', line)
            if match:
                key = match[1].strip().strip("\"'")
                break
    if not key:
        raise ValueError("Set RUNPOD_API_KEY in the environment or ignored .env")
    return key


def api(method, path, body=None):
    key = api_key()
    request = urllib.request.Request(API + path, method=method,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        data=None if body is None else json.dumps(body).encode())
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return None if response.status == 204 else json.load(response)
    except urllib.error.HTTPError as error:
        if method == "GET" and error.code == 404:
            return None
        detail = error.read().decode(errors="replace").replace(key, "[REDACTED]")[:1500]
        headers = {name: error.headers[name].replace(key, "[REDACTED]")
                   for name in ("x-request-id", "x-correlation-id", "cf-ray", "traceparent")
                   if error.headers and error.headers.get(name)}
        raise APIError(error.code, f"RunPod {method} {path}: HTTP {error.code}: {detail}", headers) from None


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def event(env, phase, **fields):
    if env.get("PATH_B_RUN_DIR"):
        with (Path(env["PATH_B_RUN_DIR"]) / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "phase": phase, **fields}) + "\n")


def record_startup(pod_id, folder, stop, poll_seconds):
    """Keep provider system/container logs locally while the pod still exists."""
    key, cursor = api_key(), None
    with (folder / "startup.jsonl").open("a") as output:
        finishing = False
        while not finishing:
            # On cleanup, do one final bounded read before the pod is deleted.
            finishing = stop.is_set()
            headers = {"Authorization": "Bearer " + key, "Accept": "text/event-stream"}
            if cursor:
                headers["Last-Event-ID"] = cursor
            request = urllib.request.Request(API + f"/pods/{pod_id}/logs?tail=1000", headers=headers)
            opened = False
            try:
                with urllib.request.urlopen(request, timeout=3) as response:
                    opened = True
                    deadline = time.monotonic() + 3
                    pending_cursor = cursor
                    for raw in response:
                        if (stop.is_set() and not finishing) or time.monotonic() >= deadline:
                            break
                        line = raw.decode(errors="replace").strip()
                        if line.startswith("id:"):
                            pending_cursor = line[3:].strip()
                        elif line.startswith("data:"):
                            # Preserve the provider's source/timestamp/line JSON.
                            output.write(line[5:].strip().replace(key, "[REDACTED]") + "\n")
                            output.flush()
                            cursor = pending_cursor
            except Exception as error:
                if not (opened and isinstance(error, TimeoutError)):
                    output.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(),
                        "log_read_error": str(error).replace(key, "[REDACTED]")}) + "\n")
                    output.flush()
                if isinstance(error, urllib.error.HTTPError) and error.code == 404:
                    break
            if not finishing:
                stop.wait(poll_seconds)


def command(args, *, env, seconds, capture=False):
    # Kill the entire local process group on timeout/interruption, including rsync.
    process = subprocess.Popen(args, cwd=ROOT, env=env, start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE if capture else subprocess.STDOUT, text=True)
    tail = deque(maxlen=30)
    def stream_output():
        log = (Path(env["PATH_B_RUN_DIR"]) / "controller.log").open("a") if env.get("PATH_B_RUN_DIR") else None
        try:
            for line in process.stdout:
                tail.append(line)
                print(line, end="", flush=True)
                if log:
                    log.write(line)
                    log.flush()
        finally:
            if log:
                log.close()
    reader = None if capture else threading.Thread(target=stream_output, daemon=True)
    if reader:
        reader.start()
    try:
        if capture:
            stdout, stderr = process.communicate(timeout=seconds)
            if stderr and env.get("PATH_B_RUN_DIR"):
                with (Path(env["PATH_B_RUN_DIR"]) / "controller.log").open("a") as log:
                    log.write(stderr)
        else:
            process.wait(timeout=seconds)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise
    finally:
        if reader:
            reader.join(timeout=3)
    if not capture:
        stdout, stderr = "", "".join(tail)
    if process.returncode:
        raise RuntimeError(f"{args[0]} failed ({process.returncode}): {(stderr or '')[-1500:]}")
    return stdout or ""


def ssh_args(env, *, attempts=None):
    options = ["ssh", "-i", env["PATH_B_SSH_KEY"], "-p", env["PATH_B_PORT"],
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=" + env.get("PATH_B_SSH_CONNECT_SECONDS", "10"),
            "-o", "ConnectionAttempts=" + str(attempts or env.get("PATH_B_SSH_ATTEMPTS", "1")),
            "-o", "StrictHostKeyChecking=accept-new", "-o",
            "UserKnownHostsFile=" + env["PATH_B_KNOWN_HOSTS"]]
    if env.get("PATH_B_SSH_CONTROL_PATH"):
        options += ["-o", "ControlMaster=auto", "-o", "ControlPersist=60",
                    "-o", "ControlPath=" + env["PATH_B_SSH_CONTROL_PATH"]]
    return options + ["root@" + env["PATH_B_HOST"]]


def preflight(execution):
    for executable in ("ssh", "rsync", "bash"):
        if not shutil.which(executable):
            raise ValueError(f"Install {executable} locally before renting a pod")
    prepared = ROOT / execution["prepared_path"]
    if not all((prepared / name).is_file() for name in ("tokens.bin", "documents.jsonl", "prepared.json")):
        raise ValueError("Prepared input is missing; prepare it before renting a GPU")
    r = execution["runpod"]
    for name in ("ssh_connect_seconds", "ssh_attempts"):
        if type(r[name]) is not int or r[name] < 1:
            raise ValueError(f"runpod.{name} must be a positive integer")
    public_key = Path(r["ssh_public_key_path"]).expanduser()
    private_key = Path(os.environ.get("PATH_B_SSH_KEY", str(public_key).removesuffix(".pub"))).expanduser()
    if not private_key.is_file():
        raise ValueError(f"SSH private key missing: {private_key}")
    for name in ("network_volume_id", "registry_id", "data_center", "image"):
        if not r[name]:
            raise ValueError(f"Set runpod.{name}")
    volume = api("GET", "/network-volumes/" + r["network_volume_id"])
    if not volume:
        raise ValueError("Configured network volume was not found")
    if volume["dataCenter"] != r["data_center"]:
        raise ValueError(f"Network volume is in {volume['dataCenter']}; set runpod.data_center to match")
    if not api("GET", "/registries/" + r["registry_id"]):
        raise ValueError("Configured private-image credential was not found")
    keys = api("GET", "/account/ssh-keys")["keys"]
    public = public_key.read_text().strip()
    if not any(k.split()[:2] == public.split()[:2] for k in keys):
        raise ValueError("Register the configured public SSH key before renting a pod")
    query = urllib.parse.urlencode({"include": "AVAILABILITY", "product": "POD",
                                    "cloud": "SECURE", "cudaVersions": ",".join(r["allowed_cuda_versions"])})
    gpu = next(g for g in api("GET", "/catalog/gpus?" + query)["gpus"] if g["id"] == r["gpu_type"])
    stock = next((d["availability"] for d in gpu.get("dataCenters", []) if d["id"] == r["data_center"]), "NONE")
    if stock not in ("LOW", "MEDIUM", "HIGH"):
        available = [d["id"] for d in gpu.get("dataCenters", []) if d["availability"] in ("LOW", "MEDIUM", "HIGH")]
        raise ValueError(f"No {r['gpu_type']} stock in {r['data_center']} for exact CUDA versions "
                         f"{r['allowed_cuda_versions']}. Available locations for this filter: {available or 'none'}")
    print(f"API catalog: {gpu['name']} in {r['data_center']}, stock={stock}, ${gpu['price']['secure']}/h + disk; "
          f"GPU diagnostic limit {execution['diagnostic']['max_seconds']}s. No new volume. "
          "Catalog availability does not confirm allocation for the full request.", flush=True)
    return public, private_key


def wait_ssh(pod_id, execution, env):
    limits = execution["diagnostic"]
    deadline = time.monotonic() + limits["ssh_ready_seconds"]
    while time.monotonic() < deadline:
        pod = api("GET", "/pods/" + pod_id)
        if pod is None:
            raise RuntimeError("Pod disappeared while waiting for SSH")
        direct = (pod.get("ssh") or {}).get("direct")
        reason = "Direct TCP mapping not published yet"
        if direct:
            env.update(PATH_B_HOST=direct["host"], PATH_B_PORT=str(direct["port"]))
            try:
                seconds = min(float(env.get("PATH_B_SSH_CONNECT_SECONDS", "10")) + 5, max(0.1, deadline-time.monotonic()))
                command(ssh_args(env, attempts=1) + ["true"], env=env, seconds=seconds, capture=True)
                event(env, "ssh_ready", host=direct["host"], port=direct["port"])
                print(f"SSH ready: {direct['host']}:{direct['port']}; pod rate ${pod['cost']}/h", flush=True)
                return pod
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                reason = str(error)[-1500:]
        event(env, "ssh_wait", status=pod["status"], direct=direct, error=reason)
        print(f"Waiting for direct SSH: {pod['status']}; {max(0, int(deadline-time.monotonic()))}s left; {reason[-400:]}", flush=True)
        time.sleep(min(limits["poll_seconds"], max(0, deadline - time.monotonic())))
    raise TimeoutError("SSH readiness deadline reached; no GPU diagnostic was started")


def find_model(execution, env):
    expected = [(a["path"], a["size_bytes"]) for a in json.loads((ROOT / "data/manifests/model-fp32.json").read_text())["files"]]
    # Validate the configured path; acquisition is a separate CPU preparation step.
    code = "\n".join([
        "import json,pathlib",
        f"expected={expected!r}", f"configured=pathlib.Path({execution['model_path']!r})",
        "missing=[n for n,s in expected if not (configured/n).is_file() or (configured/n).stat().st_size!=s]",
        "assert not missing, 'Model missing/incomplete at '+str(configured)+': '+repr(missing)+'; finish model acquisition first'",
        "print(json.dumps(str(configured)))",
    ])
    result = command(ssh_args(env) + ["python -c " + shlex.quote(code)], env=env, seconds=60, capture=True)
    path = json.loads(result)
    print(f"Existing model: {path}; original asset sizes match the pinned catalog", flush=True)
    return path


def diagnostic_settings(execution):
    diagnostic = execution["diagnostic"]
    if diagnostic["mode"] not in ("load", "benchmark"):
        raise ValueError("Set diagnostic.mode to 'load' or 'benchmark' in execution config")
    windows = diagnostic["windows"]
    if type(windows) is not int or windows < 1:
        raise ValueError("diagnostic.windows must be a positive integer")
    return diagnostic


def worker(execution_file):
    """Load directly from the configured network-volume path, inside one timeout."""
    from scripts.common import settings, snapshot
    from scripts.measurement import load_model, memory, runtime
    import torch
    research, execution = settings(argparse.Namespace(research=ROOT / "config/research.json", execution=execution_file))
    diagnostic = diagnostic_settings(execution)
    started = time.monotonic()
    device = torch.device(execution["device"])
    if device.type != "cuda":
        raise ValueError("The GPU diagnostic requires a CUDA device")
    # The allocator reset does not initialize CUDA. Check initialization, the
    # configured device and the allocator before attempting model loading.
    torch.cuda.init()
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    print(f"CUDA initialized and allocator ready: {device}", flush=True)
    folder = Path(execution["results_path"])
    snapshot(folder, research, execution)
    df = subprocess.check_output(["df", "-hT", execution["model_path"]], text=True, timeout=10)
    print(df, flush=True)
    (folder / "filesystems.txt").write_text(df)
    timings = {}
    model = load_model(research, execution, timings=timings)
    summary = {"model_path": execution["model_path"], **timings, **memory(model), "runtime": runtime(model),
               "mode": diagnostic["mode"], "load_complete": True}
    save(folder / "model-load.json", summary)
    print("MODEL LOAD REPORT: " + json.dumps(summary), flush=True)
    print(f"from_pretrained: {timings['from_pretrained_seconds']:.2f}s | "
          f"to GPU: {timings['to_gpu_seconds']:.2f}s | peak VRAM allocated/reserved: "
          f"{summary['gpu_allocated_gb']:.3f}/{summary['gpu_reserved_gb']:.3f} GB", flush=True)
    if diagnostic["mode"] == "benchmark":
        from scripts.benchmark import benchmark
        remaining = execution["diagnostic"]["max_seconds"] - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Loading consumed the diagnostic deadline; no forward window started")
        benchmark(research, {**execution, "max_seconds": remaining}, model=model, limit=diagnostic["windows"],
                  loaded_seconds=timings["total_load_seconds"])


def terminate(pod_id):
    if api("GET", "/pods/" + pod_id) is not None:
        api("DELETE", "/pods/" + pod_id)
    for _ in range(3):
        if api("GET", "/pods/" + pod_id) is None:
            print(f"Pod {pod_id} deleted (read-back 404); network volume retained", flush=True)
            return
        time.sleep(2)
    raise RuntimeError(f"Deletion not confirmed for pod {pod_id}; inspect it in RunPod")


def finish(report, execution, env, folder):
    pod_id = report.get("pod_id")
    if report.get("deleted"):
        return
    if not pod_id:
        if report.get("create_response_uncertain"):
            raise RuntimeError(f"Cannot confirm allocation/deletion for request {report['name']}")
        return
    if report.get("job_attempted"):
        # Stop only this diagnostic's recorded timeout process group before backup.
        remote = execution["runpod"]["remote_root"]
        pidfile = remote + "/" + execution["results_path"] + "/job.pid"
        stop = (f"p=$(cat {shlex.quote(pidfile)} 2>/dev/null || true); "
                "case \"$p\" in ''|*[!0-9]*) exit 0;; esac; "
                "if kill -0 \"$p\" 2>/dev/null && grep -aqE 'scripts\\.(benchmark|diagnose_gpu)' /proc/\"$p\"/cmdline; then "
                "kill -TERM -- -\"$p\" 2>/dev/null || true; sleep 2; "
                "kill -KILL -- -\"$p\" 2>/dev/null || true; fi")
        print("Cleanup: stopping this diagnostic process, then retrieving logs/results...", flush=True)
        command(ssh_args(env) + [stop], env=env, seconds=20, capture=True)
        command(["bash", "scripts/runpod.sh", "fetch"], env=env,
                seconds=execution["diagnostic"]["retrieval_seconds"])
        report["retrieved"] = True
        save(folder / "run.json", report)
    terminate(pod_id)
    report["deleted"] = True
    save(folder / "run.json", report)


def diagnose(execution, *, plan=False):
    diagnostic = diagnostic_settings(execution)
    public, private = preflight(execution)
    if plan:
        print("Read-only plan finished. No pod created.")
        return 0
    name = "path-b-diagnostic-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = ROOT / "results/diagnostics" / name
    folder.mkdir(parents=True)
    execution = copy.deepcopy(execution)
    execution.update(results_path=str(folder.relative_to(ROOT)), max_seconds=execution["diagnostic"]["max_seconds"])
    execution["runpod"].update(ssh_host="", ssh_port=None)
    save(folder / "execution.json", execution)
    control = tempfile.TemporaryDirectory(prefix="path-b-ssh-")
    env = {**os.environ, "PATH_B_SSH_KEY": str(private), "PATH_B_EXECUTION_FILE": str(folder / "execution.json"),
           "PATH_B_KNOWN_HOSTS": str(folder / "known_hosts"), "PATH_B_RUN_DIR": str(folder),
           "PATH_B_SSH_CONNECT_SECONDS": str(execution["runpod"]["ssh_connect_seconds"]),
           "PATH_B_SSH_ATTEMPTS": str(execution["runpod"]["ssh_attempts"]),
           "PATH_B_SSH_CONTROL_PATH": str(Path(control.name) / "%C")}
    stop_logs, recorder = threading.Event(), None
    report = {"name": name, "pod_id": None, "job_attempted": False, "retrieved": False, "deleted": False}
    report["mode"] = diagnostic["mode"]
    save(folder / "run.json", report)
    r = execution["runpod"]
    payload = {"name": name, "image": r["image"], "registry": r["registry_id"], "cloud": "SECURE",
        "gpu": {"id": r["gpu_type"], "count": 1, "minRamPerGpu": r["min_ram_gb"],
                "allowedCudaVersions": r["allowed_cuda_versions"]}, "dataCenterIds": [r["data_center"]],
        "disk": r["container_disk_gb"], "ports": ["22/tcp"], "startSsh": True,
        "mounts": {"network": [{"volumeId": r["network_volume_id"], "path": "/workspace"}]},
        "env": {"PUBLIC_KEY": public}}
    save(folder / "create-request.json", {**payload, "env": {key: "[REDACTED]" for key in payload["env"]}})
    status = 0
    try:
        try:
            pod = api("POST", "/pods", payload)
        except BaseException as error:
            if isinstance(error, APIError) and error.status in (400, 401, 403, 404, 422):
                report["creation_rejected"] = True
                save(folder / "run.json", report)
                print(f"Pod creation rejected (HTTP {error.status}); no pod allocated. No cleanup needed.", flush=True)
                raise
            # No POST retry: recover a lost response by this invocation's exact unique name.
            report["create_response_uncertain"] = True
            save(folder / "run.json", report)
            owned = [p for p in api("GET", "/pods?limit=1000")["pods"] if p["name"] == name and p["image"] == r["image"]]
            if len(owned) == 1:
                report["pod_id"] = owned[0]["id"]
            else:
                report["create_response_uncertain"] = True
                print(f"No creation response/id: inspect RunPod for this exact request name: {name}. Do not repeat POST.", flush=True)
            raise
        report["pod_id"] = pod["id"]
        save(folder / "run.json", report)
        event(env, "created", pod_id=pod["id"])
        recorder = threading.Thread(target=record_startup,
            args=(pod["id"], folder, stop_logs, diagnostic["poll_seconds"]), daemon=True)
        recorder.start()
        print(f"Created {pod['id']}, ${pod['cost']}/h; automatic cleanup enabled. Results: {folder}", flush=True)
        pod = wait_ssh(pod["id"], execution, env)
        report.update(host=env["PATH_B_HOST"], port=env["PATH_B_PORT"], hourly_usd=pod["cost"])
        r.update(ssh_host=env["PATH_B_HOST"], ssh_port=int(env["PATH_B_PORT"]))
        save(folder / "execution.json", execution)
        save(folder / "run.json", report)
        event(env, "model_check")
        execution["model_path"] = find_model(execution, env)
        save(folder / "execution.json", execution)
        event(env, "push_start")
        command(["bash", "scripts/runpod.sh", "push"], env=env, seconds=execution["diagnostic"]["transfer_seconds"])
        event(env, "push_finished")
        remote = r["remote_root"]
        destination = remote + "/" + execution["results_path"]
        command(ssh_args(env) + ["mkdir -p " + shlex.quote(destination)], env=env, seconds=20)
        seconds = execution["diagnostic"]["max_seconds"]
        job = (f"echo $$ > {shlex.quote(destination+'/job.pid')}; "
               f"exec timeout --signal=TERM --kill-after=10 {seconds}s /opt/path-b/.venv/bin/python "
               f"-u -m scripts.diagnose_gpu --worker")
        shell = (f"cd {shlex.quote(remote)} && export CUBLAS_WORKSPACE_CONFIG=:4096:8 && "
                 f"bash -o pipefail -c {shlex.quote('bash -c '+shlex.quote(job)+' 2>&1 | tee '+shlex.quote(destination+'/console.log'))}")
        report["job_attempted"] = True
        event(env, "gpu_start", mode=diagnostic["mode"])
        save(folder / "run.json", report)
        task = "network-volume load timing only" if diagnostic["mode"] == "load" else f"{diagnostic['windows']} diagnostic window(s)"
        print(f"Starting {task}; "
              f"loading + GPU work capped at {seconds}s", flush=True)
        command(ssh_args(env) + [shell], env=env, seconds=seconds+30)
        report["load_passed" if diagnostic["mode"] == "load" else "diagnostic_passed"] = True
    except KeyboardInterrupt:
        status = 130
        report["error"] = "Interrupted by user"
        event(env, "interrupted")
        print("Cancellation received. Finishing log retrieval and pod cleanup...", flush=True)
    except Exception as error:
        status = 1
        report["error"] = str(error)
        if isinstance(error, APIError):
            report.update(http_status=error.status, response_headers=error.response_headers)
        event(env, "failed", error=str(error))
        print(f"Diagnostic stopped: {error}", flush=True)
    finally:
        # Let cleanup finish even if Ctrl-C caused this path.
        handlers = {s: signal.signal(s, signal.SIG_IGN) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            stop_logs.set()
            if recorder:
                recorder.join(timeout=10)
                event(env, "startup_logs_saved", reader_stopped=not recorder.is_alive())
            event(env, "cleanup_start")
            finish(report, execution, env, folder)
        except Exception as error:
            status = status or 1
            report["cleanup_error"] = str(error)
            print(f"RECOVERY REQUIRED: {error}. Pod {report['pod_id']} may still bill. "
                  f"Run: python -m scripts.diagnose_gpu --recover {folder}", flush=True)
        finally:
            if env.get("PATH_B_HOST"):
                try:
                    options = ssh_args(env)
                    command(options[:-1] + ["-O", "exit", options[-1]], env=env, seconds=5, capture=True)
                except Exception:
                    pass  # The master may already be gone after pod deletion.
            control.cleanup()
            for signum, handler in handlers.items():
                signal.signal(signum, handler)
            save(folder / "run.json", report)
    print(f"Diagnostic exit {status}; retrieved={report['retrieved']}, pod_deleted={report['deleted']}; {folder}", flush=True)
    return status


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execution", type=Path, default=ROOT / "config/execution.json")
    p.add_argument("--plan", action="store_true", help="Read current stock/price/credentials, without creating a pod")
    p.add_argument("--recover", type=Path, help="Retry retrieval and deletion for a diagnostic's saved run directory")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        worker(args.execution)
        return
    if args.recover:
        folder = args.recover.resolve()
        report = json.loads((folder / "run.json").read_text())
        execution = json.loads((folder / "execution.json").read_text())
        if not report["pod_id"]:
            owned = [p for p in api("GET", "/pods?limit=1000")["pods"]
                     if p["name"] == report["name"] and p["image"] == execution["runpod"]["image"]]
            if len(owned) != 1:
                raise RuntimeError(f"Cannot resolve the creation response; inspect RunPod for exact name {report['name']}")
            report["pod_id"] = owned[0]["id"]
            save(folder / "run.json", report)
        pod = api("GET", "/pods/" + report["pod_id"])
        if pod and pod["name"] != report["name"]:
            raise ValueError("Saved diagnostic does not identify this pod")
        if pod is None:
            print("Pod is already gone; remote data cannot be retrieved.")
            return
        env = {**os.environ, "PATH_B_EXECUTION_FILE": str(folder / "execution.json"),
               "PATH_B_KNOWN_HOSTS": str(folder / "known_hosts"),
               "PATH_B_SSH_KEY": str(Path(os.environ.get("PATH_B_SSH_KEY", execution["runpod"]["ssh_public_key_path"].removesuffix(".pub"))).expanduser())}
        if report["job_attempted"]:
            env.update(PATH_B_HOST=report["host"], PATH_B_PORT=report["port"])
        finish(report, execution, env, folder)
        return
    sys.exit(diagnose(json.loads(args.execution.read_text()), plan=args.plan))


if __name__ == "__main__":
    main()
