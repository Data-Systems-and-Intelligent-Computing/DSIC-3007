from copy import deepcopy

from src.observation_contract import (measurement_cache_compatible,
                                      plan_observation_metadata,
                                      raw_observation_key, token_ids_digest)


def _identity():
    return {"model_revision": "model-commit", "tokenizer_revision": "tokenizer-hash",
            "executor": {"implementation_sha256": "a" * 64,
                         "runtime": "torch-pinned", "device": "A40"},
            "measurement_kind": "effects"}


def _observation():
    ids = [11, 12, 13, 14]
    return {"dataset": "scientific-corpus", "dataset_revision": "dataset-commit",
            "domain": "scientific", "source_group": "paper-1",
            "document_id": "doc-1", "content_sha256": "b" * 64,
            "input_ids": ids, "token_ids_sha256": token_ids_digest(ids),
            "context_length": 4, "window_start": 20, "window_end": 24,
            "absolute_predictor_position": 23, "local_predictor_position": 3,
            "target_id": 15,
            "stage_kind": "pilot", "layer": 15, "operator": "zero-one-active-expert"}


def _cached():
    return {"identity": _identity(), "observation": _observation(),
            "source_measurement_manifest_sha256": "c" * 64,
            "source_event_id": "event-1", "source_plan_manifest_sha256": "d" * 64,
            "plan_metadata": {"seed": 3007, "budget": 20}}


def test_exact_raw_cache_match_and_fresh_plan_metadata():
    requested = {"identity": _identity(), "observation": _observation()}
    compatible, reason = measurement_cache_compatible(requested, _cached())
    assert compatible is True
    assert reason == "exact_raw_observation_match"
    metadata = plan_observation_metadata(
        plan_manifest_sha256="e" * 64, plan_stage="pilot",
        sampling={"seed": 3007, "budget": 40},
        raw_key=raw_observation_key(_identity(), _observation()),
        source_measurement_manifest_sha256="c" * 64,
        source_event_id="event-1", source_plan_manifest_sha256="d" * 64)
    assert metadata["raw_source"]["source_measurement_manifest_sha256"] == "c" * 64
    assert metadata["plan_metadata"]["sampling"] == {"seed": 3007, "budget": 40}


def test_changed_bytes_runtime_context_target_or_source_fails_closed():
    wanted = {"identity": _identity(), "observation": _observation()}
    for mutate in (
        lambda x: x["observation"].update(content_sha256="f" * 64),
        lambda x: x["identity"]["executor"].update(runtime="other-runtime"),
        lambda x: x["observation"].update(context_length=3),
        lambda x: x["observation"].update(target_id=99),
        lambda x: x["observation"].update(source_group="other-paper"),
    ):
        cached = _cached()
        mutate(cached)
        compatible, _ = measurement_cache_compatible(wanted, cached)
        assert compatible is False


def test_missing_source_measurement_reference_is_rejected():
    cached = _cached()
    del cached["source_measurement_manifest_sha256"]
    assert measurement_cache_compatible(
        {"identity": _identity(), "observation": _observation()}, cached) == (
            False, "missing_immutable_source_measurement_reference")


def test_invalid_short_window_and_missing_provenance_are_not_cacheable():
    bad = _observation()
    bad["context_length"] = 4096
    bad["local_predictor_position"] = 4095
    try:
        raw_observation_key(_identity(), bad)
    except ValueError as exc:
        assert "context" in str(exc)
    else:
        raise AssertionError("wrong context length was accepted")
    cached = _cached()
    cached["source_plan_manifest_sha256"] = ""
    assert measurement_cache_compatible(
        {"identity": _identity(), "observation": _observation()}, cached) == (
            False, "missing_source_plan_reference")


def test_plan_metadata_changes_do_not_change_raw_measurement_key():
    before = _cached()
    after = deepcopy(before)
    after["plan_metadata"] = {"seed": 3010, "budget": 100}
    request = {"identity": _identity(), "observation": _observation()}
    assert measurement_cache_compatible(request, before)[0]
    assert measurement_cache_compatible(request, after)[0]
