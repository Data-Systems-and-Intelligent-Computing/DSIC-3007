import io
import json
import hashlib
import time
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from runpod import screen, global_model_setup as setup
from src.artifacts import canonical, digest


def profile():
    return {"storage_type":"global", "gpu":"NVIDIA L40S", "global_volume_id":"global-id",
            "global_volume_mount_path":"/workspace-global",
            "eligible_data_centers":["EU-A","AP-B"]}


def test_global_request_uses_official_mount_contract_and_all_live_regions():
    p=profile()
    body=screen.request_body("own-attempt","ssh-ed25519 public","/tmp/own-attempt",p,time.time()+400)
    assert body["dataCenterIds"] == ["EU-A","AP-B"]
    assert body["volumeMounts"] == [{"volumeId":"global-id","volumeType":"OBJECT_STORE_VOLUME",
                                     "mountPath":"/workspace-global"}]
    assert "networkVolumeId" not in body and "mounts" not in body
    assert body["gpuTypeId"] == p["gpu"] and body["gpuCount"] == 1
    assert body["imageName"] == screen.IMAGE and body["containerRegistryAuthId"] == screen.REGISTRY
    assert body["allowedCudaVersions"] == ["13.0"]
    assert "terminateAfter" in body and "openssh-server" in body["dockerArgs"]


def test_global_guard_rejects_wrong_volume_or_wrong_gpu():
    state={"name":"own-attempt","registry_id":screen.REGISTRY,"hardware_profile":profile()}
    pod={"name":"own-attempt","image":screen.IMAGE,"registry":screen.REGISTRY,
         "gpu":{"id":"NVIDIA L40S"},"env":{"DSIC_GLOBAL_VOLUME_ID":"global-id"}}
    assert screen.matches_attempt(pod,state)
    pod["env"]["DSIC_GLOBAL_VOLUME_ID"]="other-volume"
    assert not screen.matches_attempt(pod,state)
    pod["env"]["DSIC_GLOBAL_VOLUME_ID"]="global-id"
    pod["gpu"]["id"]="NVIDIA L40"
    assert not screen.matches_attempt(pod,state)


def test_global_ready_pod_accepts_selected_stock_region_not_old_volume_region():
    pod={"cost":1.09,"gpu":{"count":1,"memory":96},"cloud":"SECURE",
         "dataCenterId":"AP-B","cudaVersion":"13.0"}
    screen.validate_ready_pod(pod,profile())
    pod["dataCenterId"]="US-MO-1"
    with pytest.raises(ValueError,match="region"):
        screen.validate_ready_pod(pod,profile())


def test_setup_request_and_readback_use_selected_host_cuda_versions():
    p = dict(profile(), operation="prepare_model", allowed_cuda_versions=["12.8"])
    body = screen.request_body("own-attempt", "ssh-ed25519 public", "/tmp/own", p, time.time()+400)
    assert body["allowedCudaVersions"] == ["12.8"]
    pod = {"cost":1.09,"gpu":{"count":1,"memory":96},"cloud":"SECURE",
           "dataCenterId":"AP-B","cudaVersion":"12.8"}
    screen.validate_ready_pod(pod, p)
    with pytest.raises(ValueError, match="CUDA"):
        screen.validate_ready_pod(pod, profile())


def test_cpu_setup_validates_software_without_initializing_cuda(monkeypatch):
    from runpod import screen_remote as remote
    def forbidden(*args, **kwargs):
        pytest.fail("CPU-only setup must never initialize torch.cuda")
    torch = SimpleNamespace(version=SimpleNamespace(cuda="13.0"),
                            cuda=SimpleNamespace(is_available=forbidden))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(__version__="fixture"))
    monkeypatch.setattr(remote.importlib.metadata, "version", lambda name: "fixture")
    monkeypatch.setattr(remote.subprocess, "check_output", lambda *args, **kwargs: "NVIDIA L40S\n")
    expected = {"python":".".join(map(str, sys.version_info[:3])), "torch":"fixture",
                "torch_cuda":"13.0", "transformers":"fixture"}
    assert remote.validate_runtime(expected, cpu_only_setup=True)["cuda_compute_tested"] is False
    with pytest.raises(pytest.fail.Exception):
        remote.validate_runtime(expected)


def download_fixture(tmp_path,monkeypatch,prefix=0,status=206):
    data=canonical({"num_hidden_layers":16,"num_experts":64,"num_experts_per_tok":8,"norm_topk_prob":False})
    catalog={"revision":"c6f6954f182ffdd9f6ab4f4d772714b7dcd6836b","repo_id":"allenai/OLMoE-1B-7B-0924",
             "files":[{"path":"config.json","size_bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}]}
    cat=tmp_path/"catalog.json";cat.write_bytes(canonical(catalog))
    mount=tmp_path/"global";mount.mkdir();root=mount/"models/OLMoE-fp32";root.mkdir(parents=True)
    (root/".dsic-download-owner.json").write_bytes(canonical({"catalog_sha256":digest(catalog),"revision":catalog["revision"]}))
    if prefix: (root/"config.json").write_bytes(data[:prefix])
    calls=[]
    class Response(io.BytesIO):
        headers={"Content-Range":f"bytes {prefix}-{len(data)-1}/{len(data)}"}
    def open_url(request,timeout):
        calls.append(request)
        response=Response(data[prefix:]);response.status=status if prefix else 200
        return response
    monkeypatch.setattr(setup.os.path,"ismount",lambda path:Path(path)==mount)
    monkeypatch.setattr(setup.urllib.request,"urlopen",open_url)
    monkeypatch.setitem(sys.modules,"safetensors",SimpleNamespace(safe_open=lambda *args:pytest.fail("no tensor assets in transport fixture")))
    return cat,mount,root,data,calls


def test_owned_download_prefix_resumes_at_exact_offset_and_verifies_hash(tmp_path,monkeypatch):
    cat,mount,root,data,calls=download_fixture(tmp_path,monkeypatch,prefix=12)
    result=setup.prepare(cat,mount,tmp_path/"receipt.json",time.time()+300)
    assert result["status"]=="pass" and (root/"config.json").read_bytes()==data
    assert calls[0].get_header("Range")=="bytes=12-"
    assert result["verified_files"][0]["sha256"]==hashlib.sha256(data).hexdigest()


def test_wrong_resume_response_preserves_partial_file(tmp_path,monkeypatch):
    cat,mount,root,data,calls=download_fixture(tmp_path,monkeypatch,prefix=12,status=200)
    result=setup.prepare(cat,mount,tmp_path/"receipt.json",time.time()+300)
    assert result["status"]=="fail" and (root/"config.json").read_bytes()==data[:12]


def test_setup_deadline_stops_before_network_request(tmp_path,monkeypatch):
    cat,mount,root,data,calls=download_fixture(tmp_path,monkeypatch)
    result=setup.prepare(cat,mount,tmp_path/"receipt.json",time.time()+1)
    assert result["status"]=="incomplete" and not calls


def test_setup_missing_mount_never_downloads_to_local_fallback(tmp_path,monkeypatch):
    monkeypatch.setattr(setup.os.path,"ismount",lambda path:False)
    with pytest.raises(ValueError,match="not mounted"):
        setup.prepare(tmp_path/"missing-catalog",tmp_path/"not-mount",tmp_path/"receipt",time.time()+300)
    assert not (tmp_path/"not-mount").exists()


@pytest.mark.parametrize("boot_delay", [0, 180])
@pytest.mark.parametrize("operation", ["prepare_model", "benchmark"])
def test_global_controller_uploads_operation_identity_and_deletes_only_own_pod(tmp_path,monkeypatch,capsys,boot_delay,operation):
    package=tmp_path/"package";package.mkdir()
    for name in ("source.tar.gz","screen.tar.gz","identity.json"):
        (package/name).write_text("operation-neutral package")
    helper=tmp_path/"runpod/screen_remote.py";helper.parent.mkdir();helper.write_text("bound helper")
    monkeypatch.setattr(screen,"ROOT",tmp_path)
    monkeypatch.setattr(screen,"PACKAGE",package)
    cash_cap, wall_cap, fixed = (.60, 1800, .03) if operation == "benchmark" else (.30, 900, .08)
    monkeypatch.setattr(screen,"CASH_CAP",cash_cap)
    monkeypatch.setattr(screen,"MAX_SECONDS",1800)
    clock=[time.time()]; started=clock[0]
    monkeypatch.setattr(screen.time,"time",lambda:clock[0])
    monkeypatch.setattr(screen.time,"sleep",lambda seconds:clock.__setitem__(0,clock[0]+seconds))
    monkeypatch.setattr(screen.subprocess,"Popen",lambda *args,**kwargs:SimpleNamespace())
    monkeypatch.setattr(screen.subprocess,"run",lambda *args,**kwargs:SimpleNamespace(returncode=0))
    p=profile();p["global_volume_model_path"]="/workspace-global/models/OLMoE-fp32"
    identity={"archives":{"source.tar.gz":{},"screen.tar.gz":{}},
              "local_files":{"runpod/screen_remote.py":screen.file_hash(helper)},
              "proposal":{"kind":"path-b-preflight","global_operation":operation,"startup_seconds":600},
              "hardware_profile":p,
              "operator_limits":{"remaining_cash_usd":cash_cap,"remaining_wall_seconds":wall_cap,"fixed_cost_bound_usd":fixed}}
    pod=None;mutations=[];uploaded=[]
    def api(method,path,payload=None,**kwargs):
        nonlocal pod
        if method=="POST":
            assert path=="/pods" and payload["volumeMounts"][0]["volumeId"]=="global-id"
            mutations.append((method,"own-pod"))
            pod={"id":"own-pod","name":payload["name"],"image":screen.IMAGE,"registry":screen.REGISTRY,
                 "gpu":{"id":"NVIDIA L40S","count":1,"memory":96},"cloud":"SECURE",
                 "cost":1.09,"dataCenterId":"AP-B","cudaVersion":"13.0","status":"RUNNING",
                 "env":{"DSIC_GLOBAL_VOLUME_ID":"global-id"},
                 "ssh":{"direct":{"username":"root","host":"example.invalid","port":1234}}}
            return {"id":"own-pod"}
        if method=="DELETE":
            assert path=="/pods/own-pod";mutations.append((method,"own-pod"));pod=None;return {}
        assert method=="GET" and path=="/pods/own-pod"
        if pod is not None and clock[0]-started < boot_delay:
            return dict(pod,ssh={})
        return pod
    def pipe(command,attempt,deadline,label):
        if label=="upload immutable candidate and screens":
            state = json.loads((attempt/"attempt.json").read_text())
            assert deadline == min(state["deadline_epoch"]-90, clock[0]+90)
            clock[0] += 40  # boot180 + upload40 must exceed the old startup/upload150 cap.
            sent=attempt/"identity.json"
            assert str(sent) in command and str(package/"identity.json") not in command
            uploaded.append(json.loads(sent.read_text()))
        if label.startswith("retrieve"):
            (attempt/"results").mkdir()
    monkeypatch.setattr(screen,"api",api)
    monkeypatch.setattr(screen,"pipe_command",pipe)
    assert screen.run_attempt(identity,1.09,"public-key",wall_cap,profile=p)==0
    state = json.loads((tmp_path/"runpod/runs/latest/attempt.json").read_text())
    assert state["deadline_epoch"]-state["started_epoch"] <= wall_cap
    if operation == "benchmark":
        assert state["deadline_epoch"]-state["started_epoch"] == 1800
    assert uploaded[0]["proposal"]["global_operation"]==operation
    assert uploaded[0]["hardware_profile"]["eligible_data_centers"]==["EU-A","AP-B"]
    assert mutations==[("POST","own-pod"),("DELETE","own-pod")]
    latest=tmp_path/"runpod/runs/latest"
    assert json.loads((latest/"cleanup.json").read_text())["volume_deleted"] is False
    assert json.loads((latest/"result.json").read_text())["global_volume_retained"] is True
    assert json.loads((latest/"boot-status.json").read_text())["elapsed_seconds"] >= boot_delay
    final = [line for line in capsys.readouterr().out.splitlines() if "FINISHED" in line]
    assert len(final) == 1 and "volume global-id retained" in final[0]
    assert screen.VOLUME not in final[0]
