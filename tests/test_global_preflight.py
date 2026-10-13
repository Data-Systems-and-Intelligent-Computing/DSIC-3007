import pytest

from runpod import global_preflight as g
from runpod import path_b_preflight as cli
from runpod.path_b_benchmark import critical_metrics, TimedModelSession


def config():
    return {"storage_type": "global", "global_volume_id": "global-demo",
            "gpu": "NVIDIA L40", "maximum_gpu_rate_usd_per_hour": .82,
            "required_cuda": "13.0"}


def catalog(regions):
    return {"gpus": [{"id": "NVIDIA L40", "secure": True, "memory": 48,
                      "price": {"secure": .82},
                      "cudaVersions": [{"version": "13.0", "available": True}],
                      "dataCenters": [{"id": r, "availability": s} for r, s in regions]}]}


def test_global_volume_follows_stock_when_region_changes():
    # The old regional cache location is absent from both independent catalogs.
    a = g.placements(config(), catalog([("EU-A", "LOW"), ("US-B", "HIGH")]))
    b = g.placements(config(), catalog([("AP-C", "MEDIUM")]))
    assert [r["data_center_id"] for r in a] == ["US-B", "EU-A"]
    assert [r["data_center_id"] for r in b] == ["AP-C"]
    assert {r["gpu"] for r in a+b} == {"NVIDIA L40"}


def test_global_mode_cannot_inherit_old_volume_region_or_switch_gpu():
    with pytest.raises(ValueError, match="region constraint"):
        g.placements(dict(config(), dataCenterId="US-MO-1"), catalog([("US-MO-1", "LOW")]))
    with pytest.raises(ValueError, match="absent"):
        g.placements(dict(config(), gpu="NVIDIA L40S"), catalog([("EU-A", "LOW")]))


def test_global_placement_keeps_cost_and_cuda_gates():
    c = catalog([("EU-A", "LOW")])
    c["gpus"][0]["price"]["secure"] = .83
    with pytest.raises(ValueError, match="price guard"):
        g.placements(config(), c)
    c["gpus"][0]["price"]["secure"] = .82
    c["gpus"][0]["cudaVersions"][0]["available"] = False
    with pytest.raises(ValueError, match="CUDA"):
        g.placements(config(), c)


def test_cuda12_stock_is_setup_only_and_newer_driver_is_benchmark_compatible():
    cfg = dict(config(), setup_host_cuda_floor="12.8")
    c = catalog([("EU-A", "LOW")])
    c["gpus"][0]["cudaVersions"] = [{"version": "12.8", "available": True}]
    assert g.placements(cfg, c, "prepare_model")[0]["allowed_cuda_versions"] == ["12.8"]
    with pytest.raises(ValueError, match="benchmark requires host CUDA >= 13.0"):
        g.placements(cfg, c)
    c["gpus"][0]["cudaVersions"].append({"version": "13.2", "available": True})
    assert g.placements(cfg, c)[0]["allowed_cuda_versions"] == ["13.2"]


def test_read_only_status_reports_setup_and_benchmark_capacity_separately(tmp_path, monkeypatch):
    from src.artifacts import canonical
    cfg = dict(config(), setup_host_cuda_floor="12.8", global_volume_model_path="/global/model",
               local_working_directory="/tmp/work")
    path = tmp_path/"config.json"; path.write_bytes(canonical(cfg))
    monkeypatch.setattr(g, "cache_ready", lambda config: "successful-receipt")
    c = catalog([("EU-A", "LOW")])
    c["gpus"][0]["cudaVersions"] = [{"version":"12.8", "available":True}]
    result = g.status(path, read_live=True, api=lambda *args: c)
    assert result["model_setup_placements"] and not result["available_placements"]
    assert not result["ready_for_paid_run"] and result["paid_resources_created"] == 0


@pytest.mark.parametrize("args", [["--run"],
    ["--run", "--benchmark-authorization-id", "global-path-b-benchmark-2026-10-13-v9"],
    ["--setup-model"], ["--resume-model"]])
def test_global_paid_cli_never_calls_regional_fallback_or_api(monkeypatch, args):
    def forbidden(*args, **kwargs):
        pytest.fail("no API, regional preflight, or paid create allowed")
    monkeypatch.setattr(cli.screen, "api", forbidden)
    monkeypatch.setattr(cli.screen, "preflight", forbidden)
    monkeypatch.setattr(cli.screen, "run_attempt", forbidden)
    monkeypatch.setattr(cli, "configure", lambda: None)
    monkeypatch.setattr(cli.screen, "MAX_RATE", cli.screen.MAX_RATE)
    monkeypatch.setattr(cli.screen, "CASH_CAP", cli.screen.CASH_CAP)
    from src.artifacts import digest, read_json
    actual = read_json(g.CONFIG)
    monkeypatch.setattr(cli.screen,"verify_package",lambda *args:{"global_config_sha256":digest(actual)})
    with pytest.raises(ValueError, match="grant is consumed|paid benchmark disabled|preparation is complete"):
        cli.main(args)


def test_resume_grant_requires_matching_parent_and_cannot_be_reused(tmp_path, monkeypatch):
    from src.artifacts import read_json, canonical
    cfg = read_json(g.CONFIG)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    auth = cfg["paid_authorization"]["model_resume"]
    prior = tmp_path/auth["preceding_resume_run"]; prior.mkdir(parents=True)
    (prior/"cleanup.json").write_bytes(canonical({"status":"confirmed"}))
    parent = tmp_path/auth["parent_run"]
    (parent/"results").mkdir(parents=True)
    candidate = {"hardware_profile":{"global_volume_id":cfg["global_volume_id"]},
                 "proposal":{"global_operation":"prepare_model"}}
    (parent/"candidate.json").write_bytes(canonical(candidate))
    (parent/"cleanup.json").write_bytes(canonical({"status":"confirmed"}))
    (parent/"results/model-preparation.json").write_bytes(canonical({"status":"incomplete"}))
    assert cli.resume_authorization(cfg)["maximum_usd"] == .30
    (parent/"cleanup.json").write_bytes(canonical({"status":"pending"}))
    with pytest.raises(ValueError, match="confirmed cleanup"):
        cli.resume_authorization(cfg)
    (parent/"cleanup.json").write_bytes(canonical({"status":"confirmed"}))
    spent = tmp_path/"runpod/runs/resume-attempt"; spent.mkdir()
    (spent/"candidate.json").write_bytes(canonical({"proposal":{"resume_authorization_id":auth["authorization_id"]}}))
    with pytest.raises(ValueError, match="already used"):
        cli.resume_authorization(cfg)


@pytest.mark.parametrize("flag,expected_cap,operation", [("--resume-model",.30,"prepare_model"), ("--run",.30,"benchmark")])
def test_resume_cap_is_separate_from_diagnostic_cap(monkeypatch, tmp_path, flag, expected_cap, operation):
    from src.artifacts import digest, read_json
    cfg = read_json(g.CONFIG)
    cfg.pop("model_preparation_status", None)  # historical setup fixture only
    synthetic_config = tmp_path/"config.json"
    from src.artifacts import canonical
    synthetic_config.write_bytes(canonical(cfg))
    monkeypatch.setattr(g, "CONFIG", synthetic_config)
    profile = {k:cfg[k] for k in ("storage_type","gpu","global_volume_id",
                                 "global_volume_mount_path","global_volume_model_path")}
    identity = {"global_config_sha256":digest(cfg),"hardware_profile":profile,"proposal":{}}
    monkeypatch.setattr(cli, "configure", lambda: None)
    monkeypatch.setattr(cli.screen, "CASH_CAP", .20)
    monkeypatch.setattr(cli.screen, "MAX_RATE", cli.screen.MAX_RATE)
    monkeypatch.setattr(cli.screen, "verify_package", lambda *args: identity)
    monkeypatch.setattr(g, "cache_ready", lambda config: "complete-receipt")
    monkeypatch.setattr(cli, "resume_authorization", lambda config: cfg["paid_authorization"]["model_resume"])
    monkeypatch.setattr(cli, "benchmark_authorization", lambda config, authorization_id: cfg["paid_authorization"]["benchmark_retry_proposal"])
    monkeypatch.setattr(g, "live_profile", lambda operation: (dict(profile, eligible_data_centers=["EU-A"]),1.09,"public-key"))
    def simulate(identity, rate, key, max_seconds, profile):
        assert cli.screen.CASH_CAP == expected_cap
        assert identity["operator_limits"]["remaining_cash_usd"] == expected_cap
        assert identity["proposal"]["global_operation"] == operation
        assert ("resume_authorization_id" in identity["proposal"]) == (flag == "--resume-model")
        assert max_seconds == 900
        return 0
    monkeypatch.setattr(cli.screen, "run_attempt", simulate)
    assert cli.main([flag]) == 0


def test_benchmark_retry_requires_explicit_release_and_refuses_reuse(tmp_path, monkeypatch):
    from src.artifacts import canonical, read_json
    cfg = read_json(g.CONFIG)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    auth = cfg["paid_authorization"]["benchmark_retry_proposal"]
    auth["status"] = "withdrawn_not_authorized"
    with pytest.raises(ValueError, match="withdrawn"):
        cli.benchmark_authorization(cfg, auth["authorization_id"])
    auth["status"] = "authorized"
    with pytest.raises(ValueError, match="grant is consumed"):
        cli.benchmark_authorization(cfg, None)
    with pytest.raises(ValueError, match="grant is consumed"):
        cli.benchmark_authorization(cfg, "old-grant")
    parent = tmp_path/auth["parent_run"]; parent.mkdir(parents=True)
    (parent/"cleanup.json").write_bytes(canonical({"status":"pending"}))
    with pytest.raises(ValueError, match="cleanup"):
        cli.benchmark_authorization(cfg, auth["authorization_id"])
    (parent/"cleanup.json").write_bytes(canonical({"status":"confirmed"}))
    assert cli.benchmark_authorization(cfg, auth["authorization_id"])["maximum_usd"] == .30
    (parent/"candidate.json").write_bytes(canonical({"proposal":{"benchmark_authorization_id":auth["authorization_id"]}}))
    with pytest.raises(ValueError, match="already used"):
        cli.benchmark_authorization(cfg, auth["authorization_id"])


def test_metric_summary_preserves_units_and_marks_unmeasured():
    events = [{"shared_seconds": .9, "shared_cleanup_seconds": .8,
               "eight_interventions_seconds": .02, "cuda_peak_reserved_bytes": 100,
               "cuda_peak_allocated_bytes": 80, "host_max_rss_kib": 50}]
    m = critical_metrics(events, {"elapsed_seconds": 2})
    assert m["baseline_with_capture_ms_per_batch"] == [900]
    assert m["eight_zero_output_effects_ms"] == [20]
    assert m["collector_phase_windows_per_second"] == .5
    assert m["peak_cuda_reserved_bytes"] == 100
    assert m["r2_upload"] == "not performed"
    missing = critical_metrics([], {})
    assert missing["peak_cuda_allocated_bytes"] is None
    assert missing["collector_phase_windows_per_second"] is None


def test_timed_session_retains_parent_identity_and_loads_once(monkeypatch, tmp_path):
    import torch
    from src import model as loader
    model = torch.nn.Linear(2, 2).eval()
    model.config = type("Config", (), {"use_cache": False})()
    calls = []
    monkeypatch.setattr(loader, "implementation_identity", lambda: "implementation")
    monkeypatch.setattr(loader, "runtime_identity", lambda: {"fixture": "cpu"})
    monkeypatch.setattr(loader, "load_model", lambda *args, **kwargs: calls.append((args, kwargs)) or model)
    phases = {}
    session = TimedModelSession(phases)
    assert session.acquire(tmp_path, {"revision": "pinned"}, 3007) is model
    timing = dict(phases["verified_model_loading"])
    assert session.acquire(tmp_path, {"revision": "pinned"}, 3007) is model
    assert len(calls) == 1 and phases["verified_model_loading"] == timing
    with pytest.raises(ValueError, match="identity changed"):
        session.acquire(tmp_path, {"revision": "different"}, 3007)
