"""Local, mocked lifecycle checks for the user-invoked Runpod screen."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import pytest

import runpod.screen as screen
import runpod.screen_remote as screen_remote


@pytest.fixture(autouse=True)
def _reset_screen_log_path(monkeypatch):
    monkeypatch.setattr(screen, "LOG_PATH", None)


def _identity(package: Path):
    package.mkdir(parents=True, exist_ok=True)
    value = {"archives": {}, "local_files": {}}
    (package / "identity.json").write_text(json.dumps(value))
    return value


def _catalog(rate=screen.MAX_RATE):
    return {"gpus": [{
        "id": screen.GPU,
        "dataCenters": [{"id": screen.REGION, "availability": "LOW"}],
        "price": {"secure": rate},
        "cudaVersions": [{"version": "13.0", "available": True}],
    }]}


def _valid_preflight(monkeypatch, tmp_path, *, rate=screen.MAX_RATE, pods=None):
    package = tmp_path / "candidate"
    identity = _identity(package)
    monkeypatch.setattr(screen, "PACKAGE", package)
    monkeypatch.setattr(screen, "verify_package", lambda: identity)
    key = tmp_path / ".ssh" / "id_ed25519"
    key.parent.mkdir()
    key.write_text("PRIVATE-KEY-MUST-NOT-BE-READ-OR-SENT")
    key.with_suffix(".pub").write_text("ssh-ed25519 public-demo user@host\n")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    calls = []

    def api(method, path, payload=None, **kwargs):
        calls.append((method, path, payload))
        if path == f"/registries/{screen.REGISTRY}":
            return {"id": screen.REGISTRY, "name": screen.REGISTRY_NAME}
        if path == f"/network-volumes/{screen.VOLUME}":
            return {"id": screen.VOLUME, "dataCenter": screen.REGION, "size": 50, "type": "HIGH_PERFORMANCE"}
        if path.startswith("/pods?"):
            return {"pods": pods or [], "pagination": {"hasNextPage": False, "nextCursor": None}}
        if path.startswith("/catalog/gpus"):
            return _catalog(rate)
        raise AssertionError((method, path))

    monkeypatch.setattr(screen, "api", api)
    return identity, calls


def test_preflight_checks_replacement_volume_and_live_l40s_quote(monkeypatch, tmp_path):
    identity, calls = _valid_preflight(monkeypatch, tmp_path)

    result_identity, rate, stock, public_key = screen.preflight()

    assert result_identity == identity
    assert (rate, stock) == (screen.MAX_RATE, "LOW")
    assert public_key == "ssh-ed25519 public-demo user@host"
    assert [path for _, path, _ in calls] == [
        f"/registries/{screen.REGISTRY}",
        f"/network-volumes/{screen.VOLUME}", "/pods?includeClusterPods=true&limit=100",
        "/catalog/gpus?include=AVAILABILITY&product=POD&cloud=SECURE",
    ]


@pytest.mark.parametrize("bad_rate", [screen.MAX_RATE + 0.01, float("nan"), 0])
def test_preflight_fails_closed_on_invalid_or_over_cap_rate(monkeypatch, tmp_path, bad_rate):
    _valid_preflight(monkeypatch, tmp_path, rate=bad_rate)
    with pytest.raises(ValueError, match="catalog rate"):
        screen.preflight()


def test_preflight_refuses_to_touch_any_existing_pod(monkeypatch, tmp_path):
    _, calls = _valid_preflight(monkeypatch, tmp_path, pods=[{"id": "user-verification"}])

    with pytest.raises(ValueError, match="existing pods"):
        screen.preflight()

    assert not any(method == "POST" or method == "DELETE" for method, _, _ in calls)


def test_missing_or_wrong_existing_registry_credential_fails_before_pod_create(monkeypatch, tmp_path):
    identity, calls = _valid_preflight(monkeypatch, tmp_path)
    original_api = screen.api

    def api(method, path, payload=None, **kwargs):
        if path == f"/registries/{screen.REGISTRY}":
            calls.append((method, path, None))
            return {"id": "wrong-registry", "name": screen.REGISTRY_NAME}
        return original_api(method, path, payload, **kwargs)

    monkeypatch.setattr(screen, "api", api)
    monkeypatch.setattr(screen, "verify_package", lambda: identity)

    with pytest.raises(ValueError, match="registry credential identity"):
        screen.preflight()

    assert calls == [("GET", f"/registries/{screen.REGISTRY}", None)]


@pytest.mark.parametrize("envelope", [
    {}, {"pods": []},
    {"pods": [], "pagination": {"hasNextPage": True, "nextCursor": "page-2"}},
    {"pods": [], "pagination": {"hasNextPage": False}},
])
def test_preflight_fails_closed_on_unknown_or_paginated_pod_list(monkeypatch, tmp_path, envelope):
    _valid_preflight(monkeypatch, tmp_path)
    # Keep fixture's deterministic API but replace only the pod-list response.
    calls = []
    def fixture_api(method, path, payload=None, **kwargs):
        calls.append((method, path))
        if path == f"/registries/{screen.REGISTRY}":
            return {"id": screen.REGISTRY, "name": screen.REGISTRY_NAME}
        if path == f"/network-volumes/{screen.VOLUME}":
            return {"id": screen.VOLUME, "dataCenter": screen.REGION, "size": 50, "type": "HIGH_PERFORMANCE"}
        if path.startswith("/pods?"):
            return envelope
        raise AssertionError((method, path))
    monkeypatch.setattr(screen, "api", fixture_api)

    with pytest.raises(ValueError, match="pod-list response|pagination evidence|paginated"):
        screen.preflight()
    assert not any(method == "POST" for method, _ in calls)


def test_package_drift_is_rejected_before_preflight_api_calls(monkeypatch, tmp_path):
    package = tmp_path / "candidate"
    identity = _identity(package)
    (package / "source.tar.gz").write_bytes(b"frozen source")
    identity["archives"] = {"source.tar.gz": {"sha256": "0" * 64}}
    (package / "identity.json").write_text(json.dumps(identity))
    monkeypatch.setattr(screen, "PACKAGE", package)
    called = []
    monkeypatch.setattr(screen, "api", lambda *a, **k: called.append(a))

    with pytest.raises((KeyError, ValueError), match="candidate archive changed|archives|source|screen"):
        screen.verify_package(package)

    assert called == []


def test_request_preserves_only_existing_network_volume_and_exposes_no_private_secret():
    body = screen.request_body("attempt-one", "ssh-ed25519 public-demo", "/workspace/screen-attempts/attempt-one")
    encoded = json.dumps(body)

    assert body["mounts"]["network"] == [{"volumeId": screen.VOLUME, "path": "/workspace"}]
    assert body["gpu"]["id"] == screen.GPU
    assert body["image"] == screen.IMAGE
    assert body["registry"] == screen.REGISTRY
    assert body["env"]["PUBLIC_KEY"] == "ssh-ed25519 public-demo"
    assert "PRIVATE-KEY" not in encoded and "RUNPOD_API_KEY" not in encoded
    assert "AWS_SECRET_ACCESS_KEY" not in encoded and "R2_SECRET" not in encoded
    assert "volumeId" in body["mounts"]["network"][0]


@pytest.mark.parametrize("cost", [None, float("nan"), screen.MAX_RATE + 0.01])
def test_actual_running_pod_cost_must_be_finite_and_inside_cap(cost):
    pod = {"cost": cost, "gpu": {"count": 1, "memory": 64}, "cloud": "SECURE",
           "dataCenterId": screen.REGION, "cudaVersion": "13.0"}
    with pytest.raises(ValueError, match="actual running pod cost"):
        screen.validate_ready_pod(pod)


@pytest.mark.parametrize("change", [
    {"gpu": {"count": 1, "memory": 63}},
    {"cudaVersion": "12.8"},
    {"dataCenterId": "CA-MTL-1"},
    {"cloud": "COMMUNITY"},
    {"gpu": {"count": 2, "memory": 96}},
])
def test_actual_running_pod_must_match_ram_region_cuda_cloud_and_gpu_count(change):
    pod = {"cost": 1.09, "gpu": {"count": 1, "memory": 64}, "cloud": "SECURE",
           "dataCenterId": screen.REGION, "cudaVersion": "13.0"}
    pod.update(change)
    with pytest.raises(ValueError, match="actual GPU count/RAM/cloud/region/CUDA"):
        screen.validate_ready_pod(pod)


def test_dry_run_is_default_and_paid_attempt_requires_user_flag(monkeypatch, capsys):
    identity = {"candidate": "frozen"}
    monkeypatch.setattr(screen, "preflight", lambda: (identity, 1.09, "LOW", "ssh-ed25519 public"))
    attempts = []
    monkeypatch.setattr(screen, "run_attempt", lambda *args: attempts.append(args) or 0)

    monkeypatch.setattr(sys, "argv", ["runpod/screen.py"])
    assert screen.main() == 0
    output = capsys.readouterr().out
    assert "DRY RUN ONLY" in output
    assert "python runpod/screen.py --run" in output
    assert attempts == []

    monkeypatch.setattr(sys, "argv", ["runpod/screen.py", "--run"])
    assert screen.main() == 0
    assert attempts == [(identity, 1.09, "ssh-ed25519 public", screen.MAX_SECONDS)]


@pytest.mark.parametrize("cap", ["nan", "inf", "90", "1801"])
def test_cli_rejects_invalid_lease_cap_before_preflight(monkeypatch, cap):
    monkeypatch.setattr(screen, "preflight", lambda: (_ for _ in ()).throw(AssertionError("preflight must not run")))
    monkeypatch.setattr(sys, "argv", ["runpod/screen.py", "--max-seconds", cap])

    with pytest.raises(SystemExit):
        screen.main()


def test_ambiguous_create_is_never_retried_and_attempt_is_logged(monkeypatch, tmp_path, capsys):
    package = tmp_path / "candidate"
    identity = _identity(package)
    monkeypatch.setattr(screen, "PACKAGE", package)
    monkeypatch.setattr(screen, "ROOT", tmp_path)
    calls = []

    def api(method, path, payload=None, **kwargs):
        calls.append((method, path))
        if method == "POST":
            raise TimeoutError("create response timed out")  # ambiguous: do not repeat
        if path.startswith("/pods?"):
            return {"pods": [], "pagination": {"hasNextPage": False, "nextCursor": None}}
        raise AssertionError((method, path))

    monkeypatch.setattr(screen, "api", api)
    # Do not launch the independent local guard in this deterministic test.
    monkeypatch.setattr(screen.subprocess, "Popen", lambda *a, **k: object())

    assert screen.run_attempt(identity, 1.09, "ssh-ed25519 public") == 1

    assert sum(method == "POST" for method, _ in calls) == 1
    attempt = next((tmp_path / "runpod/runs").glob("dsic-screen-*"))
    result = json.loads((attempt / "result.json").read_text())
    state = json.loads((attempt / "attempt.json").read_text())
    assert result["status"] == "incomplete_or_failed"
    assert result["network_volume_retained"] is True
    assert "TimeoutError" in result["reason"]
    assert 0 < state["deadline_epoch"] - state["started_epoch"] <= screen.MAX_SECONDS
    request = (attempt / "request.json").read_text()
    assert "ssh-ed25519 public" in request
    assert "RUNPOD_API_KEY" not in request and "PRIVATE-KEY" not in request
    logs = capsys.readouterr().out
    assert "ATTEMPT" in logs and "STOP TimeoutError" in logs and "FINISHED" in logs


def test_known_create_rejection_is_not_retried_or_left_as_an_ambiguous_pod(monkeypatch, tmp_path):
    package = tmp_path / "candidate"
    identity = _identity(package)
    monkeypatch.setattr(screen, "PACKAGE", package)
    monkeypatch.setattr(screen, "ROOT", tmp_path)
    monkeypatch.setattr(screen.subprocess, "Popen", lambda *a, **k: object())
    calls = []

    def api(method, path, payload=None, **kwargs):
        calls.append((method, path))
        if method == "POST":
            raise screen.HTTPFailure(method, path, 400)
        if method == "GET" and path.startswith("/pods?"):
            return {"pods": [], "pagination": {"hasNextPage": False, "nextCursor": None}}
        raise AssertionError((method, path))

    monkeypatch.setattr(screen, "api", api)

    assert screen.run_attempt(identity, 1.09, "ssh-ed25519 public") == 1

    assert calls == [("POST", "/pods"), ("GET", "/pods?includeClusterPods=true&limit=100")]
    attempt = next((tmp_path / "runpod/runs").glob("dsic-screen-*"))
    state = json.loads((attempt / "attempt.json").read_text())
    assert state["create_uncertain"] is False
    assert (attempt / "cleanup.json").is_file()


def test_private_image_pull_fatal_tears_down_without_upload_or_retry(monkeypatch, tmp_path, capsys):
    package = tmp_path / "candidate"
    identity = _identity(package)
    monkeypatch.setattr(screen, "PACKAGE", package)
    monkeypatch.setattr(screen, "ROOT", tmp_path)
    monkeypatch.setattr(screen.subprocess, "Popen", lambda *a, **k: object())
    monkeypatch.setattr(screen, "boot_failure", lambda identifier: "unauthorized: repository access denied")
    monkeypatch.setattr(screen, "pipe_command", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not upload")))
    pod = {
        "id": "own-screen-pod", "name": None, "image": screen.IMAGE, "registry": screen.REGISTRY,
        "gpu": {"id": screen.GPU, "count": 1, "memory": 125},
        "mounts": {"network": [{"volumeId": screen.VOLUME, "path": "/workspace"}]},
        "cost": 1.09, "status": "EXITED", "cloud": "SECURE", "dataCenterId": screen.REGION,
        "cudaVersion": "13.0",
    }
    calls = []
    deleted = False

    def api(method, path, payload=None, **kwargs):
        nonlocal deleted
        calls.append((method, path))
        if method == "POST":
            pod["name"] = payload["name"]
            assert payload["registry"] == screen.REGISTRY
            return {"id": pod["id"]}
        if method == "GET" and path == f"/pods/{pod['id']}":
            return None if deleted else pod
        if method == "DELETE" and path == f"/pods/{pod['id']}":
            deleted = True
            return {}
        raise AssertionError((method, path))

    monkeypatch.setattr(screen, "api", api)
    monkeypatch.setattr(screen.time, "sleep", lambda _: None)

    assert screen.run_attempt(identity, 1.09, "ssh-ed25519 public", 120) == 1

    assert sum(method == "POST" for method, _ in calls) == 1
    assert sum(method == "DELETE" for method, _ in calls) == 1
    assert deleted
    assert all(path == "/pods" for method, path in calls if method == "POST")
    attempt = next((tmp_path / "runpod/runs").glob("dsic-screen-*"))
    result = json.loads((attempt / "result.json").read_text())
    assert "container startup failed" in result["reason"]
    assert "unauthorized" in result["reason"]
    assert (attempt / "cleanup.json").is_file()
    assert "STOP RuntimeError" in capsys.readouterr().out


def test_boot_failure_reads_bounded_sse_when_curl_times_out_after_http_200(monkeypatch):
    observed = {}

    def run(command, **kwargs):
        observed.update(command=command, kwargs=kwargs)
        return type("Response", (), {
            "returncode": 28,
            "stdout": 'data: {"line":"failed to pull image: unauthorized: repository denied"}\n200',
        })()

    monkeypatch.setattr(screen.subprocess, "run", run)
    monkeypatch.setattr(screen, "api_key", lambda: "test-token")

    assert "unauthorized: repository denied" in screen.boot_failure("pod-1")
    assert observed["command"][observed["command"].index("--max-time") + 1] == "4"
    assert observed["kwargs"]["timeout"] == 8
    assert "tail=30" in observed["command"][-1]
    assert "test-token" in observed["kwargs"]["input"]


def test_ambiguous_create_with_no_listed_pod_does_not_claim_cleanup(monkeypatch, tmp_path):
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    screen.write_json(attempt / "attempt.json", {"name": "our-screen", "pod_id": None, "create_uncertain": True})
    monkeypatch.setattr(screen, "api", lambda method, path, *a, **k: {
        "pods": [], "pagination": {"hasNextPage": False, "nextCursor": None}})

    assert screen.terminate(attempt) is False
    assert not (attempt / "cleanup.json").exists()


def test_teardown_refuses_an_unrelated_pod_without_deleting_or_confirming(monkeypatch, tmp_path):
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    screen.write_json(attempt / "attempt.json", {"name": "our-screen", "pod_id": "other-pod"})
    other = {"id": "other-pod", "name": "user-pod", "image": screen.IMAGE,
             "gpu": {"id": screen.GPU},
             "mounts": {"network": [{"volumeId": screen.VOLUME, "path": "/workspace"}]}}
    calls = []
    def api(method, path, *a, **k):
        calls.append((method, path))
        return other
    monkeypatch.setattr(screen, "api", api)

    with pytest.raises(ValueError, match="not the exact screen attempt"):
        screen.terminate(attempt)

    assert calls == [("GET", "/pods/other-pod")]
    assert not (attempt / "cleanup.json").exists()


def test_remote_failure_tears_down_only_its_pod_and_retains_volume(monkeypatch, tmp_path):
    package = tmp_path / "candidate"
    identity = _identity(package)
    for name in ("source.tar.gz", "screen.tar.gz"):
        (package / name).write_bytes(name.encode())
    (package / "identity.json").write_text(json.dumps(identity))
    monkeypatch.setattr(screen, "PACKAGE", package)
    monkeypatch.setattr(screen, "ROOT", tmp_path)
    monkeypatch.setattr(screen.subprocess, "Popen", lambda *a, **k: object())
    monkeypatch.setattr(screen.subprocess, "run", lambda *a, **k: type("Result", (), {"returncode": 0})())
    pod = {
        "id": "own-screen-pod", "name": None, "image": screen.IMAGE,
        "registry": screen.REGISTRY,
        "gpu": {"id": screen.GPU},
        "mounts": {"network": [{"volumeId": screen.VOLUME, "path": "/workspace"}]},
        "cost": 1.09, "status": "RUNNING", "ssh": {"direct": {"host": "host", "port": 22, "username": "root"}},
        "cloud": "SECURE", "dataCenterId": screen.REGION, "cudaVersion": "13.0",
    }
    pod["gpu"].update(count=1, memory=64)
    calls = []
    deleted = False

    def api(method, path, payload=None, **kwargs):
        nonlocal deleted
        calls.append((method, path))
        if method == "POST":
            pod["name"] = payload["name"]
            return {"id": pod["id"]}
        if method == "GET" and path == f"/pods/{pod['id']}":
            return None if deleted else pod
        if method == "DELETE" and path == f"/pods/{pod['id']}":
            deleted = True
            return {}
        raise AssertionError((method, path))

    monkeypatch.setattr(screen, "api", api)
    labels = []

    def fake_pipe(command, attempt, deadline, label):
        labels.append(label)
        if label == "upload bounded remote runner":
            raise RuntimeError("injected stage failure")

    monkeypatch.setattr(screen, "pipe_command", fake_pipe)

    assert screen.run_attempt(identity, 1.09, "ssh-ed25519 public") == 1

    assert sum(method == "POST" for method, _ in calls) == 1
    assert sum(method == "DELETE" for method, _ in calls) == 1
    assert deleted
    assert "upload immutable candidate and screens" in labels
    attempt = next((tmp_path / "runpod/runs").glob("dsic-screen-*"))
    result = json.loads((attempt / "result.json").read_text())
    cleanup = json.loads((attempt / "cleanup.json").read_text())
    assert result["network_volume_retained"] is True
    assert result["whole_workflow_feasibility"] == "unknown"
    assert cleanup["volume_deleted"] is False
    assert cleanup["status"] == "confirmed"


def test_pipe_command_records_progress_and_kills_local_command_at_deadline(tmp_path, capsys):
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    deadline = time.time() + 0.4

    with pytest.raises(TimeoutError, match="deadline"):
        screen.pipe_command(
            [sys.executable, "-c", "import time; print('stage started', flush=True); time.sleep(5)"],
            attempt, deadline, "test bounded stage",
        )

    logged = (attempt / "console.log").read_text()
    output = capsys.readouterr().out
    assert logged.count("stage started") == 1 and "stage started" in output
    assert "STAGE test bounded stage" in output
    assert "WAIT test bounded stage" not in output  # deadline test remains sub-15 seconds


@pytest.mark.parametrize("deadline", ["nan", "inf", "-inf"])
def test_remote_cli_rejects_nonfinite_lease_deadlines(monkeypatch, tmp_path, deadline):
    with pytest.raises((SystemExit, ValueError)):
        screen_remote.main(["--directory", str(tmp_path), "--deadline-epoch", deadline])

    assert not (tmp_path / "results").exists()


def _write_tar(path, members):
    with tarfile.open(path, "w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, __import__("io").BytesIO(data))


@pytest.mark.parametrize("helper_matches", [True, False])
@pytest.mark.parametrize("paired", [False, True, "path_b"])
def test_remote_runner_verifies_its_own_identity_before_staging(monkeypatch, tmp_path, helper_matches, paired):
    import hashlib

    directory = tmp_path / "attempt"
    directory.mkdir()
    helper_path = Path(screen_remote.__file__)
    helper_bytes = helper_path.read_bytes()
    source_members = {
        "configs/research.json": b'{"mechanics":{"acceptance_tolerances":{"atol":1e-6,"rtol":1e-5},"nll_atol":1e-6}}',
        "src/verify.py": b"frozen verifier",
        "runpod/screen_remote.py": helper_bytes,
        "model-catalog.json": b'{"files":[]}',
    }
    screen_members = {
        f"{screen_remote.EXPECTED_TOPDIR}/4096/manifest.json": b"long-plan",
        f"{screen_remote.EXPECTED_TOPDIR}/2048/manifest.json": b"short-plan",
    }
    _write_tar(directory / "source.tar.gz", source_members)
    _write_tar(directory / "screen.tar.gz", screen_members)
    hash_bytes = lambda value: hashlib.sha256(value).hexdigest()
    archive_specs = {}
    for name in ("source.tar.gz", "screen.tar.gz"):
        payload = (directory / name).read_bytes()
        archive_specs[name] = {"size_bytes": len(payload), "sha256": hash_bytes(payload)}
    source_hashes = {name: hash_bytes(payload) for name, payload in source_members.items()}
    screen_hashes = {name: hash_bytes(payload) for name, payload in screen_members.items()}
    identity = {
        "archives": archive_specs,
        "base_image": screen.IMAGE,
        "files": source_hashes,
        "screen_files": screen_hashes,
        "local_files": {"configs/research.json": source_hashes["configs/research.json"],
                        "src/verify.py": source_hashes["src/verify.py"],
                        "runpod/screen_remote.py": source_hashes["runpod/screen_remote.py"]},
        "implementation_sha256": "impl-hash",
        "expected_runtime": {"python": "3.14.6", "torch": "2.12.1", "torch_cuda": "13.0",
                             "transformers": "5.12.1"},
        "proposal": {"status": "prepared_user_authorized_not_launched", "settings_sha256": "settings-hash",
                     "plans": [{"context": 4096}, {"context": 2048}]},
    }
    path_b = paired == "path_b"
    if paired is True:
        _write_tar(directory / "baseline.tar.gz", source_members)
        archive = directory / "baseline.tar.gz"
        identity["archives"]["baseline.tar.gz"] = {"size_bytes": archive.stat().st_size,
                                                      "sha256": hash_bytes(archive.read_bytes())}
        identity["baseline_files"] = source_hashes
        identity["baseline_implementation_sha256"] = "impl-hash"
        identity["proposal"]["kind"] = "paired-collector-benchmark"
    if path_b:
        identity["proposal"]["kind"] = "path-b-preflight"
    (directory / "identity.json").write_text(json.dumps(identity))
    monkeypatch.setattr(screen_remote, "MODEL_DIR", tmp_path / "models")
    monkeypatch.delenv("RESEARCH_IMAGE_DIGEST", raising=False)
    if not helper_matches:
        changed_helper = tmp_path / "changed-screen-remote.py"
        changed_helper.write_bytes(b"different executed helper")
        monkeypatch.setattr(screen_remote, "__file__", str(changed_helper))
    monkeypatch.setattr(screen_remote, "PYTHON", Path(sys.executable))
    monkeypatch.setattr(screen_remote, "implementation_hash", lambda candidate: "impl-hash")
    monkeypatch.setattr(screen_remote, "validate_runtime", lambda expected: expected)
    monkeypatch.setattr(screen_remote, "validate_plans", lambda identity, candidate: (
        [candidate / "data/plans" / screen_remote.EXPECTED_TOPDIR / "4096",
         candidate / "data/plans" / screen_remote.EXPECTED_TOPDIR / "2048"],
        candidate / "model-catalog.json"))
    stages = []

    def run_stage(command, candidate, results, status_path, deadline, stage, catalog_path=None):
        stages.append(stage)
        if paired is True:
            if stage == "weights":
                assert "weights" not in command and "verify_model" in command[3]
            if stage == "validation":
                assert command[3] == "runpod.paired_benchmark"
                assert "--baseline-source" in command and "--selection" in command
        if stage == "validation":
            if path_b:
                assert command[3] == "runpod.path_b_benchmark"
                assert float(command[command.index("--deadline-epoch")+1]) == work_deadline - 30
            output_path = Path(command[command.index("--output") + 1])
            output_path.write_text('{"status":"pass"}')
        return {"status": "pass", "returncode": 0, "log": stage + ".log"}

    monkeypatch.setattr(screen_remote, "run_stage", run_stage)

    work_deadline = time.time() + (1500 if path_b else 120)
    result = screen_remote.run(directory, work_deadline)

    receipt = json.loads((directory / "results/runner-receipt.json").read_text())
    if helper_matches:
        assert result == 0
        assert stages == (["validation"] if path_b else ["weights", "validation"])
        if path_b:
            assert receipt["phases"]["weights"]["status"] == "not_run_user_removed_reverification"
        assert receipt["status"] == "pass"
        assert receipt["validation_is_scientific_approval"] is False
        assert receipt["declared_image"] == screen.IMAGE
        assert "not in-container attestation" in receipt["image_identity_source"]
        if paired is True:
            assert (directory / "baseline/configs/research.json").is_file()
        assert (directory / "candidate/configs/research.json").is_file()
        assert (directory / "candidate/data/plans" / screen_remote.EXPECTED_TOPDIR / "4096/manifest.json").is_file()
    else:
        assert result == 1
        assert stages == []
        assert receipt["status"] == "fail"
        assert "uploaded remote runner differs" in receipt["error"]


def test_image_identity_survives_missing_ssh_environment(monkeypatch):
    monkeypatch.delenv("RESEARCH_IMAGE_DIGEST", raising=False)
    assert screen_remote.bind_image_identity({"base_image": screen.IMAGE}) == screen.IMAGE
    # Exercise actual subprocess inheritance, which the validator uses.
    actual = subprocess.check_output(
        [sys.executable, "-c", "import os; print(os.environ['RESEARCH_IMAGE_DIGEST'])"],
        text=True).strip()
    assert actual == screen.IMAGE


@pytest.mark.parametrize("image", [None, "latest", "repo@sha256:bad", "repo@sha256:" + "g" * 64])
def test_unpinned_image_identity_is_rejected(monkeypatch, image):
    monkeypatch.delenv("RESEARCH_IMAGE_DIGEST", raising=False)
    identity = {} if image is None else {"base_image": image}
    with pytest.raises(ValueError, match="digest-pinned"):
        screen_remote.bind_image_identity(identity)
    assert "RESEARCH_IMAGE_DIGEST" not in os.environ


def test_conflicting_image_identity_is_rejected(monkeypatch):
    monkeypatch.setenv("RESEARCH_IMAGE_DIGEST", "other@sha256:" + "0" * 64)
    with pytest.raises(ValueError, match="differs"):
        screen_remote.bind_image_identity({"base_image": screen.IMAGE})


def test_package_image_mismatch_is_rejected_with_valid_archive_hashes(tmp_path):
    package = tmp_path / "candidate"
    package.mkdir()
    archives = {}
    for name in ["source.tar.gz", "screen.tar.gz"]:
        path = package / name
        path.write_bytes(b"test archive identity")
        archives[name] = {"sha256": screen.file_hash(path), "size_bytes": path.stat().st_size}
    (package / "identity.json").write_text(json.dumps({
        "archives": archives, "base_image": "other@sha256:" + "0" * 64}))
    with pytest.raises(ValueError, match="base image differs"):
        screen.verify_package(package)


def test_paired_run_requires_reconciled_limits_before_create(monkeypatch, tmp_path):
    monkeypatch.setattr(screen, "ROOT", tmp_path)
    monkeypatch.setattr(screen, "api", lambda *a, **k: pytest.fail("no API allowed"))
    identity={"proposal": {"kind": "paired-collector-benchmark"}}
    with pytest.raises(ValueError, match="explicit reconciled"):
        screen.run_attempt(identity, 1.09, "public")
    assert not (tmp_path / "runpod/runs").exists()


@pytest.mark.parametrize("cash,wall,fixed", [(0.01,1200,0.03),(0.40,90,0.03),
    (float("nan"),1200,0.03),(13,1200,0.03),(0.40,36001,0.03)])
def test_paired_invalid_budget_stops_before_create(monkeypatch, tmp_path, cash, wall, fixed):
    monkeypatch.setattr(screen, "ROOT", tmp_path)
    monkeypatch.setattr(screen, "api", lambda *a, **k: pytest.fail("no API allowed"))
    identity={"proposal": {"kind": "paired-collector-benchmark"}, "operator_limits": {
        "remaining_cash_usd":cash,"remaining_wall_seconds":wall,"fixed_cost_bound_usd":fixed}}
    with pytest.raises(ValueError):
        screen.run_attempt(identity, 1.09, "public")
    assert not (tmp_path / "runpod/runs").exists()


def test_paired_cli_refuses_missing_reconciled_limits(monkeypatch):
    identity={"proposal": {"kind":"paired-collector-benchmark"}}
    monkeypatch.setattr(screen, "preflight", lambda: (identity,1.09,"LOW","public"))
    monkeypatch.setattr(screen, "run_attempt", lambda *a: pytest.fail("no paid attempt"))
    monkeypatch.setattr(sys,"argv",["screen.py","--run"])
    with pytest.raises(SystemExit):screen.main()


def test_extracted_hashing_stops_at_same_cutoff_as_archive_inspection(monkeypatch, tmp_path):
    payload = tmp_path / "payload.txt"
    payload.write_bytes(b"immutable source")
    expected = {payload.name: screen.file_hash(payload)}
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(payload, arcname=payload.name)
    clock = [100.0]
    monkeypatch.setattr(screen_remote.time, "time", lambda: clock[0])
    original_extract = tarfile.TarFile.extractall

    def extract_then_cross_cutoff(self, *args, **kwargs):
        original_extract(self, *args, **kwargs)
        clock[0] = 121.0

    monkeypatch.setattr(tarfile.TarFile, "extractall", extract_then_cross_cutoff)
    # Inspection is permitted before 120; hashing must not use the later 150 deadline.
    with pytest.raises(TimeoutError, match="hashing"):
        screen_remote.extract_checked(archive, tmp_path / "candidate", expected, 150.0)


@pytest.mark.parametrize("lease", [1200, 300])
def test_paired_upload_after_slow_boot_uses_setup_budget_bounded_by_overall_lease(monkeypatch, tmp_path, lease):
    package = tmp_path / "package"
    identity = _identity(package)
    identity.update(proposal={"kind": "paired-collector-benchmark"}, operator_limits={
        "remaining_cash_usd": .40, "remaining_wall_seconds": lease, "fixed_cost_bound_usd": .03})
    monkeypatch.setattr(screen, "ROOT", tmp_path)
    monkeypatch.setattr(screen, "PACKAGE", package)
    monkeypatch.setattr(screen.subprocess, "Popen", lambda *args, **kwargs: object())
    monkeypatch.setattr(screen.subprocess, "run", lambda *args, **kwargs: type("Result", (), {"returncode": 0})())
    clock = [10000.0]
    monkeypatch.setattr(screen.time, "time", lambda: clock[0])
    pod = {"id": "our-pod", "image": screen.IMAGE, "registry": screen.REGISTRY,
           "gpu": {"id": screen.GPU, "count": 1, "memory": 125},
           "mounts": {"network": [{"volumeId": screen.VOLUME, "path": "/workspace"}]},
           "cost": 1.09, "cloud": "SECURE", "dataCenterId": screen.REGION,
           "cudaVersion": "13.0", "status": "RUNNING",
           "ssh": {"direct": {"host": "host", "port": 22, "username": "root"}}}
    deleted = [False]

    def api(method, path, payload=None, **kwargs):
        if method == "POST":
            pod["name"] = payload["name"]
            return {"id": pod["id"]}
        if method == "GET":
            clock[0] = max(clock[0], 10081.0)
            return None if deleted[0] else pod
        if method == "DELETE":
            deleted[0] = True
            return {}
        raise AssertionError(method)

    observed = []

    def pipe(command, attempt, deadline, label):
        if label == "upload immutable candidate and screens":
            state = json.loads((attempt / "attempt.json").read_text())
            assert deadline == min(10240.0, state["deadline_epoch"] - screen.SHUTDOWN_ALLOWANCE)
            assert state["deadline_epoch"] - state["started_epoch"] <= lease
            clock[0] += 100.0
            assert clock[0] < deadline  # 81s boot plus 100s upload exceeded the old 120s cap.
            observed.append(label)
            raise RuntimeError("stop mock after completed upload")

    monkeypatch.setattr(screen, "api", api)
    monkeypatch.setattr(screen, "pipe_command", pipe)
    assert screen.run_attempt(identity, 1.09, "public", lease) == 1
    assert observed == ["upload immutable candidate and screens"]
    assert deleted[0]


def test_paired_phase_reallocation_keeps_total_budget_and_adapter_contract():
    from runpod import paired_benchmark
    spec = json.loads((Path(screen.__file__).parent / "paired-benchmark.json").read_text())
    phases = spec["phase_allowance_seconds"]
    assert sum(phases.values()) == spec["maximum_all_phase_wall_seconds"] == 1200
    assert phases["provision_startup_upload_preflight"] == screen.PAIRED_STARTUP_SECONDS
    assert all(phases[name] == value for name, value in paired_benchmark.PHASE_CAPS.items())
    assert screen.CASH_CAP == spec["maximum_all_in_cost_usd"] == .40
