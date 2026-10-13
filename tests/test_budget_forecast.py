"""The future workload forecast must fail closed and count every measurement."""

from copy import deepcopy

import pytest

from src.artifacts import read_json
from src.execution_budget import forecast_execution_budget


def fixture():
    workload = read_json("docs/validation/optimization-workload-2026-10-11.json")
    profiles = []
    for domain in ("scientific", "code", "c4"):
        for context in (4096, 2048):
            for kind in ("effects", "calibration", "probe"):
                profiles.append(dict(domain=domain, context_length=context, kind=kind,
                    input_id=f"{domain}-{context}-{kind}", event_ids=["case"],
                    layers_measured=[0, 7, 15],
                    measurement_manifest_sha256="a" * 64,
                    source_plan_manifest_sha256="b" * 64, executor_sha256="c" * 64,
                    evidence_status="measured", shared_seconds=.2, cleanup_seconds=.1,
                    persistence_seconds=.1, synchronized_input_total_seconds=.55 if kind == "effects" else .4,
                    effect_seconds_by_layer={str(layer): .05 for layer in (0, 7, 15)}))
    overhead = {name: dict(gpu_seconds=0, wall_seconds=0, usd=0, evidence_status="bounded")
        for name in ("verification", "benchmark", "setup_load", "transfer_write",
                     "host_wait", "storage_usd", "nested_reuse", "final_cpu_analysis")}
    return dict(workload=workload, profiles=profiles, overhead=overhead,
        accounting=dict(hourly_rate_usd=1.09, remaining_budget_usd=12,
            remaining_time_hours=10, cash_boundary_confirmed=True,
            time_boundary_confirmed=True, price_evidence="live_quote"),
        native_fidelity_pass=True, pilot_sufficient=True)


def test_forecast_includes_c4_whole_groups_escalation_and_matched_context():
    result = forecast_execution_budget(**fixture())
    assert result["status"] == "projected_feasible"
    assert result["stages"]["main"]["counts"]["code"]["windows"] == 12957
    assert result["stages"]["main"]["counts"]["c4"]["windows"] == 12584
    assert result["stages"]["sensitivity_2048"]["counts"]["code"]["windows"] == 3234
    assert result["stages"]["pilot"]["counts"]["code"]["windows"] == 162
    assert result["stages"]["calibration_2048"]["counts"]["c4"]["windows"] == 1500
    assert result["total_gpu_hours"] > result["main_gpu_hours"]
    assert result["nested_reuse_workload"]["calibration_cached_windows_both_contexts"] == 6300
    assert result["nested_reuse_workload"]["pilot_cached_windows"] == 360


def test_missing_profile_or_boundary_and_failed_science_never_pass():
    kwargs = fixture()
    kwargs["profiles"] = kwargs["profiles"][:-1]
    kwargs["accounting"]["cash_boundary_confirmed"] = False
    kwargs["pilot_sufficient"] = False
    result = forecast_execution_budget(**kwargs)
    assert result["status"] == "unknown" and not result["feasibility_pass"]
    assert result["total_cost_usd"] is None
    assert any("missing_profile" in blocker for blocker in result["blockers"])


def test_both_cash_and_wall_time_include_overhead_and_reserve():
    kwargs = fixture()
    kwargs["overhead"]["final_cpu_analysis"]["wall_seconds"] = 8 * 3600
    kwargs["accounting"]["remaining_budget_usd"] = 1
    result = forecast_execution_budget(**kwargs)
    assert result["status"] == "infeasible"
    assert "remaining_elapsed_time_limit_exceeded_with_reserve" in result["blockers"]
    assert "remaining_dollar_budget_exceeded_with_reserve" in result["blockers"]


def test_forecast_rejects_assumed_or_mixed_executor_timing():
    kwargs = fixture()
    kwargs["profiles"][0]["evidence_status"] = "assumed"
    with pytest.raises(ValueError, match="measured evidence"):
        forecast_execution_budget(**kwargs)
    kwargs = fixture()
    kwargs["profiles"][0]["executor_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="different GPU"):
        forecast_execution_budget(**kwargs)


def test_primary_only_pilot_profile_does_not_authorize_missing_main_layers():
    kwargs = fixture()
    for row in kwargs["profiles"]:
        if row["kind"] == "effects":
            row["effect_seconds_by_layer"] = {"15": .05}
    result = forecast_execution_budget(**kwargs)
    assert result["stages"]["pilot"]["gpu_seconds"] is not None
    assert result["stages"]["main"]["gpu_seconds"] is None
    assert not result["feasibility_pass"]
