"""Exact observation keys and fresh nested-plan provenance on cache hits."""

from copy import deepcopy

import pytest

from src.reuse import ObservationReuseIndex
from src.observation_contract import token_ids_digest
from src.artifacts import atomic, canonical, digest, file_digest, seal, write_rows
from src.reuse import SealedMeasurementReuse


def request():
    identity = {
        "model_revision": "fp32-pinned-revision",
        "tokenizer_revision": "tokenizer-pinned-revision",
        "executor": {"implementation_sha256": "a" * 64,
                     "runtime": {"torch": "pinned"}, "device": "CPU-fixture"},
        "measurement_kind": "effects",
    }
    observation = {
        "dataset": "fixture", "dataset_revision": "fixture-rev",
        "domain": "scientific", "source_group": "paper-17",
        "document_id": "doc-17", "content_sha256": "a" * 64,
        "input_ids": [4, 8, 15, 16, 23, 42],
        "token_ids_sha256": token_ids_digest([4, 8, 15, 16, 23, 42]),
        "context_length": 6, "window_start": 0, "window_end": 6,
        "absolute_predictor_position": 5, "local_predictor_position": 5,
        "target_id": 13, "stage_kind": "pilot",
        "layer": 15, "operator": "token_local_zero_contribution",
    }
    return identity, observation


def test_exact_raw_hit_gets_fresh_plan_lineage_without_replaying_execution_time():
    identity, observation = request()
    event = {"event_id": "source-event", "baseline_nll": 2.0,
             "effects": [{"expert_id": i, "ablated_nll": 2.1, "delta_nll": 0.1}
                         for i in range(8)],
             "synchronized_input_total_seconds": 9.0,
             "effect_seconds": 5.0, "cleanup_seconds": 1.0}
    index = ObservationReuseIndex()
    key = index.add(identity=identity, observation=observation, event=event,
                    source_measurement_manifest_sha256="measurements-sha",
                    source_event_id="source-event", source_plan_manifest_sha256="plan-100")

    hit = index.get(identity=identity, observation=observation,
                    plan_manifest_sha256="plan-250", plan_stage="calibration",
                    sampling={"seed": 3008, "document_budget": 250})
    assert hit is not None and hit["cache_status"] == "hit"
    assert hit["lineage"]["raw_observation_key"] == key
    assert hit["lineage"]["plan_metadata"] == {
        "plan_manifest_sha256": "plan-250", "stage": "calibration",
        "sampling": {"seed": 3008, "document_budget": 250}}
    assert hit["lineage"]["raw_source"] == {
        "source_measurement_manifest_sha256": "measurements-sha",
        "source_event_id": "source-event", "source_plan_manifest_sha256": "plan-100"}
    assert hit["source_timing"] == {"synchronized_input_total_seconds": 9.0,
                                    "effect_seconds": 5.0, "cleanup_seconds": 1.0}
    assert not ({"synchronized_input_total_seconds", "effect_seconds", "cleanup_seconds"}
                & hit["event"].keys())
    assert hit["cache_hit_seconds"] >= 0
    hit["event"]["effects"].clear()
    assert len(index) == 1


@pytest.mark.parametrize("change", [
    lambda identity, observation: identity["executor"].update(implementation_sha256="b" * 64),
    lambda identity, observation: observation.update(context_length=5),
    lambda identity, observation: observation.update(target_id=12),
    lambda identity, observation: observation.update(source_group="different-paper"),
    lambda identity, observation: observation.update(content_sha256="b" * 64),
])
def test_cache_miss_for_executor_context_target_or_source_change(change):
    identity, observation = request()
    index = ObservationReuseIndex()
    index.add(identity=identity, observation=observation, event={"baseline_nll": 1.0},
              source_measurement_manifest_sha256="measurements-sha",
              source_event_id="source-event", source_plan_manifest_sha256="plan-100")
    changed_identity, changed_observation = deepcopy(identity), deepcopy(observation)
    change(changed_identity, changed_observation)
    assert index.get(identity=changed_identity, observation=changed_observation,
                     plan_manifest_sha256="plan-250", plan_stage="calibration",
                     sampling={"seed": 3008, "document_budget": 250}) is None


def test_conflicting_duplicate_key_fails_closed():
    identity, observation = request()
    index = ObservationReuseIndex()
    refs = {"source_measurement_manifest_sha256": "measurements-sha",
            "source_event_id": "source-event", "source_plan_manifest_sha256": "plan-100"}
    index.add(identity=identity, observation=observation, event={"baseline_nll": 1.0}, **refs)
    with pytest.raises(ValueError, match="conflicting raw observation"):
        index.add(identity=identity, observation=observation, event={"baseline_nll": 2.0}, **refs)


def test_sealed_nested_measurement_reuse_keeps_fresh_sampling_and_source_refs(tmp_path):
    run = tmp_path / "source-run"
    plan = run / "plan"
    plan.mkdir(parents=True)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    input_row = {
        "input_ids": tokens, "document_id": "doc-1",
        "dataset": "fixture", "revision": "model-revision", "domain": "scientific",
        "source_group": "paper-1", "context_length": 8, "window_start": 0,
        "window_end": 8, "predictor_position": 7, "local_predictor_position": 7,
        "target_position": 8, "target_id": 9, "token_ids_sha256": token_ids_digest(tokens),
        "sampling": {"seed": 3007, "document_budget": 100}, "sample_order": 0,
        "eligible_positions": 2, "layers": [15],
    }
    input_row["input_id"] = digest({"document_id": input_row["document_id"],
        "token_ids_sha256": input_row["token_ids_sha256"], "window_start": 0,
        "window_end": 8, "target_id": 9, "context_length": 8})
    document = {"document_id": "doc-1", "content_sha256": "e" * 64}
    executor = {"implementation_sha256": "f" * 64,
                "runtime": {"python": "pinned"}, "device": "CPU-fixture"}
    mode = "native_full_forward_v1"
    atomic(plan / "model-catalog.json", canonical({"revision": "model-revision"}) + b"\n")
    atomic(plan / "research.json", canonical({"sampling": {"master_seed": 3007}}) + b"\n")
    write_rows(plan / "inputs.jsonl", [input_row])
    write_rows(plan / "documents.jsonl", [document])
    seal(plan, "experiment-plan", {"stage": "pilot", "window": 8,
                                    "tokenizer_sha256": "c" * 64})
    weights = [0.125] * 8
    event_id = digest([input_row["input_id"], 15])
    baseline = 2.0
    effects = []
    for expert in range(8):
        value = baseline + expert / 100.0
        effects.append({"expert_id": expert, "ablated_nll": value,
                        "delta_nll": value - baseline,
                        "operator": "token_local_zero_contribution"})
    import struct
    event = {"event_id": event_id, "input_id": input_row["input_id"], "layer": 15,
             "document_id": input_row["document_id"], "dataset": input_row["dataset"],
             "revision": input_row["revision"], "domain": input_row["domain"],
             "source_group": input_row["source_group"],
             "context_length": input_row["context_length"], "window_start": 0,
             "window_end": 8, "predictor_position": 7,
             "local_predictor_position": 7, "target_position": 8,
             "target_id": 9, "token_ids_sha256": input_row["token_ids_sha256"],
             "kind": "effects", "routing": {"expert_ids": list(range(8)), "weights": weights,
                  "weights_fp32_le_hex": struct.pack("<8f", *weights).hex(),
                  "coefficient_dtype": "float32"},
             "baseline_nll": baseline, "effects": effects,
             "synchronized_input_total_seconds": 3.0, "effect_seconds": 1.0,
             "sampling": input_row["sampling"]}
    run.mkdir(exist_ok=True)
    atomic(run / "identity.json", canonical({"executor": executor,
        "execution_mode": mode, "plan_manifest_sha256": file_digest(plan / "manifest.json")}) + b"\n")
    write_rows(run / "events.jsonl", [event])
    seal(run, "measurements", {"executor": executor, "stage": "pilot",
                                "execution_mode": mode})

    current = dict(input_row, sampling={"seed": 3010, "document_budget": 250}, sample_order=3)
    source = SealedMeasurementReuse(run, [current], executor=executor,
        execution_mode=mode, model_revision="model-revision",
        tokenizer_revision="c" * 64, plan_stage="pilot")
    hit = source.get(current, 15, document=document,
        plan_manifest_sha256="a" * 64, plan_stage="pilot",
        sampling={"seed": 3010, "document_budget": 250, "sample_order": 3})
    assert hit["sampling"] == {"seed": 3010, "document_budget": 250}
    assert hit["sample_order"] == 3
    assert hit["raw_source"]["source_plan_manifest_sha256"] == source.plan_manifest_sha256
    assert hit["raw_source"]["source_measurement_manifest_sha256"] == source.source_manifest_sha256
    assert hit["source_timing"] == {"synchronized_input_total_seconds": 3.0,
                                    "effect_seconds": 1.0}
    assert "effect_seconds" not in hit and len(hit["effects"]) == 8
