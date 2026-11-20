import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import zipfile

import pytest

from scripts.common import snapshot


@pytest.mark.parametrize("run_status", [0, 17])
@pytest.mark.parametrize("fetch_status", [0, 9])
@pytest.mark.parametrize("managed", [False, True])
def test_run_retrieves_before_releasing_owned_compute(tmp_path, run_status, fetch_status, managed):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "scripts").mkdir()
    (tmp_path / "config").mkdir()
    (tmp_path / "bin").mkdir()
    for name in ["runpod.sh", "__init__.py", "common.py", "export_results.py"]:
        shutil.copy2(root / "scripts" / name, tmp_path / "scripts" / name)
    execution = json.loads((root / "config/execution.json").read_text())
    execution["runpod"].update(remote_root="/root/path-b", terraform_managed=managed)
    (tmp_path / "config/execution.json").write_text(json.dumps(execution))
    ssh = tmp_path / "bin/ssh"
    ssh.write_text('#!/bin/sh\nprintf "ssh %s\\n" "$*" >> "$MOCK_LOG"\ncase "$*" in\n *scripts.run_gpu*) exit "$MOCK_RUN_STATUS" ;;\n *scripts.export_results*) exit "$MOCK_FETCH_STATUS" ;;\nesac\n')
    rsync = tmp_path / "bin/rsync"
    rsync.write_text('#!/bin/sh\nprintf "rsync %s\\n" "$*" >> "$MOCK_LOG"\nfor destination; do :; done\nprintf "[]" > "$destination/transfer.json"\n')
    terraform = tmp_path / "bin/terraform"
    terraform.write_text('#!/bin/sh\nprintf "terraform %s\\n" "$*" >> "$MOCK_LOG"\n')
    for file in [ssh, rsync, terraform]:
        file.chmod(0o755)
    env = {**os.environ, "PATH": str(tmp_path / "bin") + ":" + os.environ["PATH"],
           "PATH_B_HOST": "mock.invalid", "PATH_B_PORT": "22", "PATH_B_SSH_KEY": "mock-key",
           "MOCK_LOG": str(tmp_path / "operations.log"), "MOCK_RUN_STATUS": str(run_status),
           "MOCK_FETCH_STATUS": str(fetch_status), "PATH_B_MODEL_PATH": "/workspace/model path/OLMoE",
           "PATH_B_SSH_CONTROL_PATH": str(tmp_path / "control-%C")}
    result = subprocess.run(["bash", str(tmp_path / "scripts/runpod.sh"), "run", "pilot", "--units", "20"], env=env, capture_output=True, text=True)
    assert result.returncode == (run_status or fetch_status), result.stderr
    operations = (tmp_path / "operations.log").read_text()
    assert ("terraform" in operations) == (managed and fetch_status == 0)
    assert "cd /root/path-b" in operations
    assert "PATH_B_MODEL_PATH=/workspace/model\\ path/OLMoE" in operations
    assert "ConnectionAttempts=3" in operations
    assert "ControlMaster=auto" in operations and "ControlPath=" in operations
    if managed and fetch_status == 0:
        assert operations.index("scripts.export_results") < operations.index("terraform")
    if fetch_status:
        assert "Keep this pod" in result.stderr


def test_snapshot_keeps_lock_and_excludes_state_and_variable_secrets(tmp_path, monkeypatch):
    import scripts.common as common
    for directory in ["scripts", "config", "terraform"]:
        (tmp_path / directory).mkdir()
    (tmp_path / "scripts/measure.py").write_text("# actual source\n")
    (tmp_path / "terraform/main.tf").write_text("# source\n")
    (tmp_path / "terraform/terraform.tfstate.backup").write_text("private state")
    (tmp_path / "terraform/private.auto.tfvars.json").write_text("private variables")
    (tmp_path / "uv.lock").write_text("pinned lock")
    monkeypatch.setattr(common, "ROOT", tmp_path)
    snapshot(tmp_path / "results", {}, {})
    with zipfile.ZipFile(tmp_path / "results/source.zip") as archive:
        names = set(archive.namelist())
    assert "uv.lock" in names and "scripts/measure.py" in names
    assert not any("tfstate" in name or "tfvars" in name for name in names)


def test_restore_verified_sqlite_and_refuse_existing_observations(tmp_path):
    root = Path(__file__).resolve().parents[1]
    local, remote, bin_dir = tmp_path / "local", tmp_path / "remote", tmp_path / "bin"
    for base in [local, remote]:
        (base / "scripts").mkdir(parents=True)
        (base / "config").mkdir()
        for name in ["runpod.sh", "__init__.py", "common.py", "export_results.py"]:
            shutil.copy2(root / "scripts" / name, base / "scripts" / name)
        shutil.copy2(root / "config/research.json", base / "config/research.json")
        execution = json.loads((root / "config/execution.json").read_text())
        execution["runpod"].update(remote_root=str(remote), terraform_managed=False)
        (base / "config/execution.json").write_text(json.dumps(execution))
    source = local / "results/path-b"
    source.mkdir(parents=True)
    with sqlite3.connect(source / "pilot.sqlite") as db:
        db.execute("CREATE TABLE observations (value INTEGER)")
        db.execute("INSERT INTO observations VALUES (17)")
    (source / "calibration.json").write_text('{"id":"frozen"}')
    bin_dir.mkdir()
    # Execute the actual remote validation/move locally; emulate only SSH and rsync.
    (bin_dir / "ssh").write_text('#!/bin/bash\ncommand=${!#}\ncommand=${command//\\/opt\\/path-b\\/.venv\\/bin\\/python/python}\nexec bash -c "$command"\n')
    (bin_dir / "rsync").write_text('#!/bin/bash\nsource=${@: -2:1}\ndestination=${!#}\ndestination=${destination#root@mock.invalid:}\nmkdir -p "$destination"\ncp -a "$source/." "$destination/"\n')
    for executable in bin_dir.iterdir():
        executable.chmod(0o755)
    env = {**os.environ, "PATH": str(bin_dir) + ":" + os.environ["PATH"],
           "PATH_B_HOST": "mock.invalid", "PATH_B_PORT": "22", "PATH_B_SSH_KEY": "mock-key"}
    command = ["bash", str(local / "scripts/runpod.sh"), "restore"]
    first = subprocess.run(command, env=env, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    restored = remote / "results/path-b"
    with sqlite3.connect(restored / "pilot.sqlite") as db:
        assert db.execute("SELECT value FROM observations").fetchall() == [(17,)]
        db.execute("INSERT INTO observations VALUES (23)")
    second = subprocess.run(command, env=env, capture_output=True, text=True)
    assert second.returncode != 0 and "Restore refused" in second.stderr
    with sqlite3.connect(restored / "pilot.sqlite") as db:
        assert db.execute("SELECT value FROM observations").fetchall() == [(17,), (23,)]


@pytest.mark.parametrize("failure,expected_status,deleted", [
    (None, 0, True), ("ready", 1, True), ("model", 1, True), ("push", 1, True),
    ("job", 1, True), ("interrupt", 130, True), ("fetch", 1, False),
    ("stop", 1, False), ("lost-create-response", 1, True),
    ("load", 0, True), ("rejected", 1, False), ("server-error", 1, True),
])
def test_user_gpu_diagnostic_lifecycle(tmp_path, monkeypatch, capsys, failure, expected_status, deleted):
    import scripts.diagnose_gpu as diagnostic
    execution = json.loads((Path(__file__).resolve().parents[1] / "config/execution.json").read_text())
    execution["diagnostic"]["mode"] = "load" if failure == "load" else "benchmark"
    execution["runpod"]["remote_root"] = "/root/configured-research"
    operations = []
    pod = {"id": "this-invocation-only", "cost": 1.09, "image": execution["runpod"]["image"]}
    gone = False

    def fake_api(method, path, body=None):
        nonlocal gone
        operations.append((method, path))
        if method == "POST":
            assert body["mounts"] == {"network": [{"volumeId": execution["runpod"]["network_volume_id"], "path": "/workspace"}]}
            assert body["gpu"]["id"] == execution["runpod"]["gpu_type"]
            assert body["dataCenterIds"] == [execution["runpod"]["data_center"]]
            assert body["ports"] == ["22/tcp"]
            assert "args" not in body and "entrypoint" not in body  # preserve prebuilt bootstrap
            pod["name"] = body["name"]
            if failure == "rejected":
                raise diagnostic.APIError(400, "There are no longer any instances available with the requested specifications")
            if failure == "server-error":
                raise diagnostic.APIError(500, "response failed after allocation")
            if failure == "lost-create-response":
                raise TimeoutError("response lost after allocation")
            return pod.copy()
        if path.startswith("/pods?"):
            return {"pods": [pod.copy()]}
        if method == "DELETE":
            if failure not in ("lost-create-response", "server-error"):
                assert ("logs_saved", None) in operations
            gone = True
            return None
        return None if gone else pod.copy()

    def ready(pod_id, execution, env):
        if failure == "ready":
            raise TimeoutError("SSH not ready")
        env.update(PATH_B_HOST="mock.invalid", PATH_B_PORT="1234")
        return pod.copy()

    def find_model(execution, env):
        if failure == "model":
            raise ValueError("original FP32 catalog mismatch")
        return "/workspace/original model"

    def run(args, *, env, seconds, capture=False):
        if args[0] == "bash":
            action = args[-1]
            operations.append((action, None))
            if failure == action:
                raise RuntimeError(action + " failed")
            resolved = json.loads(Path(env["PATH_B_EXECUTION_FILE"]).read_text())
            assert resolved["max_seconds"] == execution["diagnostic"]["max_seconds"]
            assert resolved["model_path"] == "/workspace/original model"
            assert resolved["results_path"].startswith("results/diagnostics/")
            assert resolved["diagnostic"] == execution["diagnostic"]
            assert resolved["runpod"]["remote_root"] == "/root/configured-research"
            assert resolved["runpod"]["ssh_host"] == "mock.invalid"
            assert resolved["runpod"]["ssh_port"] == 1234
            assert "%C" in env["PATH_B_SSH_CONTROL_PATH"]
        elif args[-1].startswith("p="):
            operations.append(("stop", None))
            if failure == "stop":
                raise RuntimeError("cannot stop remote diagnostic before backup")
        elif "scripts.diagnose_gpu --worker" in args[-1]:
            operations.append(("job", None))
            assert f"{execution['diagnostic']['max_seconds']}s" in args[-1] and "--kill-after=10" in args[-1]
            assert "--load-only" not in args[-1]
            assert "cd /root/configured-research" in args[-1]
            if failure == "job":
                raise RuntimeError("GPU OOM")
            if failure == "interrupt":
                raise KeyboardInterrupt
        return ""

    monkeypatch.setattr(diagnostic, "ROOT", tmp_path)
    def logs(pod_id, folder, stop, poll_seconds):
        (folder / "startup.jsonl").write_text('{"source":"system","line":"starting pod"}\n')
        stop.wait()
        operations.append(("logs_saved", None))
    monkeypatch.setattr(diagnostic, "record_startup", logs)
    monkeypatch.setattr(diagnostic, "preflight", lambda x: ("ssh-ed25519 public-test-key", tmp_path / "key"))
    monkeypatch.setattr(diagnostic, "api", fake_api)
    monkeypatch.setattr(diagnostic, "wait_ssh", ready)
    monkeypatch.setattr(diagnostic, "find_model", find_model)
    monkeypatch.setattr(diagnostic, "command", run)
    assert diagnostic.diagnose(execution) == expected_status
    report = json.loads(next((tmp_path / "results/diagnostics").glob("*/run.json")).read_text())
    assert report["deleted"] is deleted and gone is deleted
    assert report["mode"] == execution["diagnostic"]["mode"]
    if failure == "rejected":
        assert report["creation_rejected"] and report["pod_id"] is None
        assert "create_response_uncertain" not in report and "cleanup_error" not in report
        assert operations == [("POST", "/pods")]
        output = capsys.readouterr().out
        assert "No cleanup needed" in output and "RECOVERY REQUIRED" not in output
    if failure == "load":
        assert report["load_passed"] and "diagnostic_passed" not in report
    if report["job_attempted"] and deleted:
        names = [name for name, _ in operations]
        assert names.index("stop") < names.index("fetch") < names.index("DELETE")
    if failure in ("ready", "model", "push", "lost-create-response", "server-error", "rejected"):
        assert not report["job_attempted"]
    assert all(path.startswith("/pods/") for method, path in operations if method == "DELETE")


def test_diagnostic_mode_does_not_change_create_request(tmp_path, monkeypatch):
    import copy
    import scripts.diagnose_gpu as diagnostic
    execution = json.loads((Path(__file__).resolve().parents[1] / "config/execution.json").read_text())
    requests = []
    monkeypatch.setattr(diagnostic, "ROOT", tmp_path)
    monkeypatch.setattr(diagnostic, "preflight", lambda x: ("public-key-not-for-logs", tmp_path / "key"))

    def reject(method, path, body=None):
        assert (method, path) == ("POST", "/pods")
        requests.append(copy.deepcopy(body))
        raise diagnostic.APIError(400, "allocator rejection", {"x-request-id": "provider-request-123"})

    monkeypatch.setattr(diagnostic, "api", reject)
    monkeypatch.setattr(diagnostic, "command", lambda *a, **kw: pytest.fail("No worker or SSH on creation rejection"))
    for mode in ("load", "benchmark"):
        execution["diagnostic"]["mode"] = mode
        assert diagnostic.diagnose(execution) == 1
    assert len(requests) == 2
    assert {k: v for k, v in requests[0].items() if k != "name"} == {
        k: v for k, v in requests[1].items() if k != "name"}
    for request in requests:
        folder = tmp_path / "results/diagnostics" / request["name"]
        persisted = json.loads((folder / "create-request.json").read_text())
        assert persisted == {**request, "env": {"PUBLIC_KEY": "[REDACTED]"}}
        assert "public-key-not-for-logs" not in (folder / "create-request.json").read_text()
        report = json.loads((folder / "run.json").read_text())
        assert report["http_status"] == 400
        assert report["response_headers"] == {"x-request-id": "provider-request-123"}
        assert report["creation_rejected"] and not report["job_attempted"]


def test_startup_log_stream_keeps_both_sources_and_resumes_without_exposing_key(tmp_path, monkeypatch):
    import io
    import threading
    import scripts.diagnose_gpu as diagnostic
    key = "test-api-key"
    monkeypatch.setattr(diagnostic, "api_key", lambda: key)
    stop = threading.Event()
    requests = []
    data = [b'id: event-one\n',  # interrupted before the data: line was received
            b'id: event-one\ndata: {"source":"system","line":"image pulled","ts":"one"}\n\n',
            b'id: event-two\ndata: {"source":"container","line":"sshd ready test-api-key","ts":"two"}\n\n',
            b'id: event-three\ndata: {"source":"system","line":"last event before cleanup","ts":"three"}\n\n']
    def wait(seconds):
        if len(requests) == 3:
            stop.set()
        return stop.is_set()
    monkeypatch.setattr(stop, "wait", wait)
    def stream(request, timeout):
        requests.append(request)
        return io.BytesIO(data[len(requests)-1])
    monkeypatch.setattr(diagnostic.urllib.request, "urlopen", stream)
    diagnostic.record_startup("owned-pod", tmp_path, stop, 10)
    content = (tmp_path / "startup.jsonl").read_text()
    rows = [json.loads(line) for line in content.splitlines()]
    assert [row["source"] for row in rows] == ["system", "container", "system"]
    assert rows[-1]["line"] == "last event before cleanup"
    assert key not in content and "[REDACTED]" in content
    assert requests[1].get_header("Last-event-id") is None
    assert requests[2].get_header("Last-event-id") == "event-one"
    assert all("source=" not in request.full_url for request in requests)


def test_ssh_probe_records_exact_failure_before_eventual_readiness(tmp_path, monkeypatch):
    from unittest.mock import Mock
    import scripts.diagnose_gpu as diagnostic
    pod = {"status": "RUNNING", "cost": 0.59, "ssh": {"direct": {"host": "new.invalid", "port": 22001}}}
    monkeypatch.setattr(diagnostic, "api", lambda *args: pod)
    monkeypatch.setattr(diagnostic.time, "sleep", lambda seconds: None)
    attempts = []
    def probe(args, **kwargs):
        attempts.append(args)
        assert "ConnectionAttempts=1" in args
        if len(attempts) == 1:
            raise RuntimeError("ssh failed (255): Connection timed out")
        return ""
    monkeypatch.setattr(diagnostic, "command", probe)
    env = {"PATH_B_RUN_DIR": str(tmp_path), "PATH_B_SSH_KEY": "key", "PATH_B_KNOWN_HOSTS": "hosts",
           "PATH_B_HOST": "old.invalid", "PATH_B_PORT": "22", "PATH_B_SSH_ATTEMPTS": "3",
           "PATH_B_SSH_CONTROL_PATH": "/tmp/unique/%C"}
    diagnostic.wait_ssh("owned-pod", {"diagnostic": {"ssh_ready_seconds": 60, "poll_seconds": 10}}, env)
    rows = [json.loads(line) for line in (tmp_path / "progress.jsonl").read_text().splitlines()]
    assert rows[0]["phase"] == "ssh_wait" and "Connection timed out" in rows[0]["error"]
    assert rows[1]["phase"] == "ssh_ready"
    assert env["PATH_B_HOST"] == "new.invalid" and env["PATH_B_PORT"] == "22001"
    options = diagnostic.ssh_args(env)
    assert "ConnectionAttempts=3" in options and "ControlPath=/tmp/unique/%C" in options


def test_command_preserves_transfer_error_in_terminal_log_and_exception(tmp_path):
    import sys
    import scripts.diagnose_gpu as diagnostic
    code = "import sys; print('upload started', flush=True); print('ssh: Connection timed out', file=sys.stderr); sys.exit(255)"
    with pytest.raises(RuntimeError, match="Connection timed out"):
        diagnostic.command([sys.executable, "-c", code], env={**os.environ, "PATH_B_RUN_DIR": str(tmp_path)}, seconds=5)
    content = (tmp_path / "controller.log").read_text()
    assert "upload started" in content and "ssh: Connection timed out" in content


def test_read_only_gpu_plan_never_creates_pod(tmp_path, monkeypatch):
    import scripts.diagnose_gpu as diagnostic
    monkeypatch.setattr(diagnostic, "preflight", lambda x: ("public", tmp_path / "key"))
    monkeypatch.setattr(diagnostic, "api", lambda *a: pytest.fail("Plan must not create/delete a pod"))
    assert diagnostic.diagnose({"diagnostic": {"mode": "load", "windows": 1}}, plan=True) == 0


@pytest.mark.parametrize("mode,windows", [("unknown", 1), ("load", 0), ("benchmark", True)])
def test_invalid_diagnostic_settings_fail_before_cloud_access(monkeypatch, mode, windows):
    import scripts.diagnose_gpu as diagnostic
    monkeypatch.setattr(diagnostic, "preflight", lambda *a: pytest.fail("Invalid config must fail before cloud access"))
    with pytest.raises(ValueError):
        diagnostic.diagnose({"diagnostic": {"mode": mode, "windows": windows}})


@pytest.mark.parametrize("mode", ["load", "benchmark"])
def test_worker_uses_config_for_mode_model_path_and_window_count(tmp_path, monkeypatch, mode):
    import torch
    import scripts.common as common
    import scripts.measurement as measurement
    import scripts.benchmark as benchmark_module
    import scripts.diagnose_gpu as diagnostic
    from unittest.mock import Mock

    execution = {
        "results_path": str(tmp_path / "results"), "model_path": "/workspace/original",
        "device": "cuda:0", "diagnostic": {"mode": mode, "windows": 3,
            "max_seconds": 600},
    }
    monkeypatch.setattr(common, "settings", lambda args: ({}, execution))
    monkeypatch.setattr(common, "snapshot", lambda folder, *args: folder.mkdir(exist_ok=True))
    calls = []
    monkeypatch.setattr(torch.cuda, "init", lambda: calls.append("cuda.init"))
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: calls.append("cuda.device"))

    def reset(device):
        # Reproduce the real allocator's failure when called before CUDA init.
        if calls != ["cuda.init", "cuda.device"]:
            raise RuntimeError("Invalid device argument")
        assert str(device) == "cuda:0"
        calls.append("cuda.reset")

    def disk_check(args, **kwargs):
        assert calls == ["cuda.init", "cuda.device", "cuda.reset"]
        assert args == ["df", "-hT", "/workspace/original"]
        calls.append("disk-check")
        return "Filesystem Type\nnetwork nfs\n"

    monkeypatch.setattr(diagnostic.subprocess, "check_output", disk_check)
    monkeypatch.setattr(diagnostic.subprocess, "run", lambda *a, **kw: pytest.fail("Worker must not copy/download weights"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset)
    model = object()

    def load(research, resolved, *, timings):
        assert resolved["model_path"] == "/workspace/original"
        assert calls == ["cuda.init", "cuda.device", "cuda.reset", "disk-check"]
        timings.update(from_pretrained_seconds=3, to_gpu_seconds=1, total_load_seconds=4)
        return model

    monkeypatch.setattr(measurement, "load_model", load)
    monkeypatch.setattr(measurement, "memory", lambda model: {"gpu_allocated_gb": 2, "gpu_reserved_gb": 3})
    monkeypatch.setattr(measurement, "runtime", lambda model: {})
    benchmark = Mock()
    monkeypatch.setattr(benchmark_module, "benchmark", benchmark)
    diagnostic.worker(tmp_path / "execution.json")
    report = json.loads((tmp_path / "results/model-load.json").read_text())
    assert report["mode"] == mode and report["load_complete"]
    assert report["model_path"] == "/workspace/original" and "copy_seconds" not in report
    assert (tmp_path / "results/filesystems.txt").is_file()
    if mode == "load":
        benchmark.assert_not_called()
    else:
        benchmark.assert_called_once()
        assert benchmark.call_args.kwargs["limit"] == 3
        assert benchmark.call_args.kwargs["model"] is model
        assert 0 < benchmark.call_args.args[1]["max_seconds"] <= 600


@pytest.mark.parametrize("failure", ["init", "set_device", "reset_peak_memory_stats"])
def test_cuda_failure_does_not_start_model_loading(tmp_path, monkeypatch, failure):
    import torch
    import scripts.common as common
    import scripts.diagnose_gpu as diagnostic
    from unittest.mock import Mock
    execution = {"device": "cuda:0", "diagnostic": {"mode": "load", "windows": 1}}
    monkeypatch.setattr(common, "settings", lambda args: ({}, execution))
    for name in ("init", "set_device", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, Mock(side_effect=RuntimeError("CUDA unavailable") if name == failure else None))
    import scripts.measurement as measurement
    load = Mock()
    monkeypatch.setattr(measurement, "load_model", load)
    with pytest.raises(RuntimeError, match="CUDA unavailable"):
        diagnostic.worker(tmp_path / "execution.json")
    load.assert_not_called()


def test_gpu_api_error_does_not_print_key(monkeypatch):
    import io
    import urllib.error
    import scripts.diagnose_gpu as diagnostic
    monkeypatch.setenv("RUNPOD_API_KEY", "test-secret-that-must-not-appear")
    def fail(*args, **kwargs):
        raise urllib.error.HTTPError("https://api.runpod.io", 403, "Forbidden",
            {"x-request-id": "trace-123", "cf-ray": "test-secret-that-must-not-appear", "Set-Cookie": "private-cookie"},
            io.BytesIO(b"forbidden test-secret-that-must-not-appear"))
    monkeypatch.setattr(diagnostic.urllib.request, "urlopen", fail)
    with pytest.raises(RuntimeError) as error:
        diagnostic.api("GET", "/pods")
    assert "HTTP 403" in str(error.value) and "[REDACTED]" in str(error.value)
    assert isinstance(error.value, diagnostic.APIError) and error.value.status == 403
    assert error.value.response_headers == {"x-request-id": "trace-123", "cf-ray": "[REDACTED]"}
    assert "test-secret-that-must-not-appear" not in str(error.value)


@pytest.mark.parametrize("location", ["CA-MTL-1", "WRONG-DC", None])
def test_preflight_checks_existing_network_volume_location(tmp_path, monkeypatch, location):
    import scripts.diagnose_gpu as diagnostic
    execution = json.loads((Path(__file__).resolve().parents[1] / "config/execution.json").read_text())
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    for name in ("tokens.bin", "documents.jsonl", "prepared.json"):
        (prepared / name).touch()
    execution["prepared_path"] = str(prepared)
    execution["runpod"]["ssh_public_key_path"] = str(tmp_path / "key.pub")
    (tmp_path / "key.pub").write_text("ssh-ed25519 test-key")
    (tmp_path / "key").touch()
    monkeypatch.delenv("PATH_B_SSH_KEY", raising=False)
    monkeypatch.setattr(diagnostic.shutil, "which", lambda name: name)
    reads = []

    def api(method, path, body=None):
        assert method == "GET"
        reads.append(path)
        if path == "/network-volumes/2fkb3b9n7h":
            return {"dataCenter": location} if location else None
        if path.startswith("/registries/"):
            return {"id": "existing"}
        if path == "/account/ssh-keys":
            return {"keys": ["ssh-ed25519 test-key"]}
        if path.startswith("/catalog/gpus?"):
            return {"gpus": [{"id": "NVIDIA A40", "name": "A40", "price": {"secure": 0.59},
                "dataCenters": [{"id": "CA-MTL-1", "availability": "LOW"}]}]}
        pytest.fail("Unexpected API request: " + path)

    monkeypatch.setattr(diagnostic, "api", api)
    if location == "CA-MTL-1":
        assert diagnostic.preflight(execution) == ("ssh-ed25519 test-key", tmp_path / "key")
        assert any(path.startswith("/catalog/gpus?") for path in reads)
    else:
        with pytest.raises(ValueError, match="data_center to match" if location else "not found"):
            diagnostic.preflight(execution)
        assert reads == ["/network-volumes/2fkb3b9n7h"]


@pytest.mark.parametrize("complete", [True, False])
def test_model_check_uses_only_the_configured_network_path(tmp_path, monkeypatch, complete):
    import io
    import shlex
    from contextlib import redirect_stdout
    import scripts.diagnose_gpu as diagnostic
    manifests = tmp_path / "data/manifests"
    manifests.mkdir(parents=True)
    (manifests / "model-fp32.json").write_text(json.dumps({"files": [{"path": "weights", "size_bytes": 4}]}))
    configured = tmp_path / "network/model"
    configured.mkdir(parents=True)
    (configured / "weights").write_bytes(b"1234" if complete else b"1")
    monkeypatch.setattr(diagnostic, "ROOT", tmp_path)
    monkeypatch.setattr(diagnostic, "ssh_args", lambda env: [])

    def remote_read(args, **kwargs):
        code = shlex.split(args[-1])
        assert code[:2] == ["python", "-c"] and "rglob" not in code[2]
        output = io.StringIO()
        with redirect_stdout(output):
            exec(code[2], {})
        return output.getvalue()

    monkeypatch.setattr(diagnostic, "command", remote_read)
    if complete:
        assert diagnostic.find_model({"model_path": str(configured)}, {}) == str(configured)
    else:
        with pytest.raises(AssertionError, match="finish model acquisition first"):
            diagnostic.find_model({"model_path": str(configured)}, {})


@pytest.mark.parametrize("ssh_status", [0, 17])
@pytest.mark.parametrize("connection", ["config", "environment"])
def test_acquire_wrapper_uses_config_and_propagates_download_failure(tmp_path, ssh_status, connection):
    root = Path(__file__).resolve().parents[1]
    for directory in ("scripts", "config", "bin"):
        (tmp_path / directory).mkdir()
    shutil.copy2(root / "scripts/runpod.sh", tmp_path / "scripts/runpod.sh")
    shutil.copy2(root / "uv.lock", tmp_path / "uv.lock")
    execution = json.loads((root / "config/execution.json").read_text())
    execution["model_path"] = "/workspace/configured model"
    execution["runpod"].update(ssh_host="configured.invalid", ssh_port=31415,
        ssh_public_key_path=str(tmp_path / "configured-key.pub"))
    (tmp_path / "config/execution.json").write_text(json.dumps(execution))
    ssh = tmp_path / "bin/ssh"
    ssh.write_text('#!/bin/sh\nprintf "%s\\n" "$*" > "$MOCK_LOG"\nexit "$MOCK_SSH_STATUS"\n')
    ssh.chmod(0o755)
    env = {**os.environ, "PATH": str(tmp_path / "bin") + ":" + os.environ["PATH"],
        "PATH_B_HOST": "mock.invalid", "PATH_B_PORT": "22", "PATH_B_SSH_KEY": "mock-key",
        "MOCK_LOG": str(tmp_path / "ssh.log"), "MOCK_SSH_STATUS": str(ssh_status)}
    env.pop("PATH_B_MODEL_PATH", None)
    env.pop("PATH_B_EXECUTION_FILE", None)
    if connection == "config":
        for name in ("PATH_B_HOST", "PATH_B_PORT", "PATH_B_SSH_KEY"):
            env.pop(name, None)
    result = subprocess.run(["bash", str(tmp_path / "scripts/runpod.sh"), "acquire"], env=env, capture_output=True, text=True)
    assert result.returncode == ssh_status
    command = (tmp_path / "ssh.log").read_text()
    assert "PATH_B_MODEL_PATH=/workspace/configured\\ model" in command
    assert "scripts.acquire_model" in command and "scripts.run_gpu" not in command
    assert "/usr/bin/python3 -m venv .acquire-venv" in command
    assert ".acquire-venv/bin/python -u -m scripts.acquire_model" in command
    assert "huggingface_hub==" in command and "/opt/path-b/.venv" not in command
    syntax = subprocess.run(["bash", "-n"], input=command.split("root@", 1)[1].split(" ", 1)[1], text=True, capture_output=True)
    assert syntax.returncode == 0, syntax.stderr
    assert "StrictHostKeyChecking=accept-new" in command
    if connection == "config":
        assert "root@configured.invalid" in command and "-p 31415" in command
        assert "-i " + str(tmp_path / "configured-key") in command
    else:
        assert "root@mock.invalid" in command and "-p 22" in command and "-i mock-key" in command
