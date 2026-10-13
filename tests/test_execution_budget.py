import pytest

from src.execution_budget import document_position_counts, plan_execution_budget


def _components():
    return {name: {"gpu_seconds": 0, "wall_seconds": 0, "usd": 0}
            for name in ("verification", "benchmark", "setup_load",
                         "transfer_write", "host_wait", "storage_usd")}


def _event(input_id, layer, total=10):
    return {"input_id": input_id, "layer": layer,
            "synchronized_input_total_seconds": total, "shared_seconds": 2,
            "cleanup_seconds": 1, "effect_seconds": 3,
            "evidence_status": "measured", "event_id": f"{input_id}-{layer}",
            "measurement_manifest_sha256": "a" * 64,
            "source_plan_manifest_sha256": "b" * 64,
            "executor_sha256": "c" * 64}


def _stage(inputs=(), layers=None, *, domains=()):
    return {"input_ids": list(inputs), "layers_by_input": layers or {},
            "components": _components(), "domains": list(domains)}


def _workflow(main):
    stages = {name: _stage() for name in (
        "verification", "benchmark", "calibration_4096", "calibration_2048",
        "probe_4096", "probe_2048", "pilot", "sensitivity_2048", "finalization")}
    stages["verification"]["contexts"] = [2048, 4096]
    stages["benchmark"]["contexts"] = [2048, 4096]
    for name, context in (("calibration_4096", 4096), ("calibration_2048", 2048),
                          ("probe_4096", 4096), ("probe_2048", 2048), ("pilot", 4096)):
        stages[name]["context_length"] = context
    stages["main"] = main
    sensitivity = _stage(domains=("scientific", "code", "c4"))
    sensitivity["context_length"] = 2048
    stages["sensitivity_2048"] = sensitivity
    return stages


def _kwargs(**extra):
    return {"hourly_rate_usd": 1, "remaining_budget_usd": 12,
            "remaining_gpu_hours": None, "remaining_time_hours": 10, **extra}


def test_short_document_positions_use_available_counts():
    assert document_position_counts(9) == {"primary_layer_15": 9, "secondary_each_layer": 9}
    assert document_position_counts(100) == {"primary_layer_15": 64, "secondary_each_layer": 16}


def test_repeated_layer_timing_fields_count_once_and_require_full_workflow():
    main = _stage(["a"], {"a": [15, 0, 7]}, domains=("scientific", "code", "c4"))
    main["context_length"] = 4096
    result = plan_execution_budget(stages=_workflow(main),
        timing_rows=[_event("a", layer) for layer in (15, 0, 7)], **_kwargs())
    assert result["deduplicated_physical_inputs"] == 1
    assert result["total_gpu_hours"] is None  # empty future workflow stages fail closed
    assert any("empty_required_stage" in blocker for blocker in result["blockers"])
    assert result["main_includes_c4"] is True


def test_conflicting_shared_timing_copy_is_rejected():
    main = _stage(["a"], {"a": [15, 7]}, domains=("scientific", "code", "c4"))
    main["context_length"] = 4096
    with pytest.raises(ValueError, match="conflicting duplicated"):
        plan_execution_budget(stages=_workflow(main),
            timing_rows=[_event("a", 15), _event("a", 7, total=11)], **_kwargs())


def test_partial_stages_unknown_inputs_and_missing_components_never_pass():
    main = _stage(["a", "b"], {"a": [15, 0, 7], "b": [15]},
                  domains=("scientific", "code", "c4"))
    main["context_length"] = 4096
    del main["components"]["storage_usd"]
    stages = _workflow(main)
    del stages["probe_2048"]
    result = plan_execution_budget(stages=stages, timing_rows=[_event("a", 15)], **_kwargs())
    assert result["status"] == "unknown"
    assert result["feasibility_pass"] is False
    assert any("missing_timing:b" in blocker for blocker in result["blockers"])
    assert "missing_required_stage:probe_2048" in result["blockers"]


def test_required_c4_context_and_layer_gates_and_measurement_provenance():
    main = _stage(["a"], {"a": [15]}, domains=("scientific", "code"))
    main["context_length"] = 2048
    event = _event("a", 15)
    result = plan_execution_budget(stages=_workflow(main), timing_rows=[event], **_kwargs())
    assert result["feasibility_pass"] is False
    assert "main_domain_set_must_include_scientific_code_and_c4" in result["blockers"]
    assert "main_context_must_be_4096" in result["blockers"]
    assert any("main_must_cover_layers_0_7_15" in x for x in result["blockers"])
    event["evidence_status"] = "assumed"
    with pytest.raises(ValueError, match="measured evidence"):
        plan_execution_budget(stages=_workflow(main), timing_rows=[event], **_kwargs())


def test_escalation_or_insufficient_pilot_blocks_a_budget_pass():
    main = _stage(["a"], {"a": [15, 0, 7]}, domains=("scientific", "code", "c4"))
    main["context_length"] = 4096
    events = [_event("a", layer, total=100_000) for layer in (15, 0, 7)]
    result = plan_execution_budget(stages=_workflow(main), timing_rows=events,
        pilot_sufficient=False, pilot_escalation=True, **_kwargs())
    assert result["feasibility_pass"] is False
    assert "pilot_extension_or_insufficiency_unresolved" in result["blockers"]
    assert "required_escalation_not_included_in_workload" in result["blockers"]


def test_unknown_time_or_price_limit_never_passes():
    main = _stage(domains=("scientific", "code", "c4"))
    main["context_length"] = 4096
    result = plan_execution_budget(stages=_workflow(main), timing_rows=[],
        **_kwargs(hourly_rate_usd=None))
    assert result["status"] == "unknown"
    assert result["feasibility_pass"] is False
    assert "unknown_limit:hourly_rate_usd" in result["blockers"]
