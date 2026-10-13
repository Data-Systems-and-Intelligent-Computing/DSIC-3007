"""Exact identity and provenance rules for reusable raw observations.

Raw measurements are keyed by the computation and immutable input identity.
Budget, seed, and sampling metadata belong to the consuming plan and are never
part of a cached measurement's identity.
"""

from collections.abc import Mapping
import hashlib
import json


IDENTITY_FIELDS = (
    "model_revision", "tokenizer_revision", "executor", "measurement_kind",
)
OBSERVATION_FIELDS = (
    "dataset", "dataset_revision", "domain", "source_group", "document_id",
    "content_sha256", "input_ids", "token_ids_sha256", "context_length",
    "window_start", "window_end", "absolute_predictor_position",
    "local_predictor_position", "target_id", "stage_kind", "layer", "operator",
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def token_ids_digest(input_ids) -> str:
    """Hash the exact canonical integer token sequence for this raw window."""
    if not isinstance(input_ids, (list, tuple)) or any(type(token) is not int for token in input_ids):
        raise ValueError("input IDs must be an integer sequence")
    return hashlib.sha256(_canonical(list(input_ids))).hexdigest()


def _valid_sha256(value):
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return any(char != "0" for char in value.lower())


def raw_observation_key(identity: Mapping, observation: Mapping) -> str:
    """Return a SHA-256 key after validating complete execution/input identity."""
    if not isinstance(identity, Mapping) or not isinstance(observation, Mapping):
        raise ValueError("identity and observation must be mappings")
    missing = [name for name in IDENTITY_FIELDS if name not in identity]
    missing.extend(name for name in OBSERVATION_FIELDS if name not in observation)
    if missing:
        raise ValueError("raw observation identity is incomplete: " + ", ".join(missing))
    if any(identity[name] in (None, "", {}) for name in IDENTITY_FIELDS):
        raise ValueError("executor identity fields must be explicit")
    if identity["measurement_kind"] not in ("routing", "effects", "calibration", "probe", "technical"):
        raise ValueError("unknown measurement kind")
    executor = identity["executor"]
    if (not isinstance(executor, Mapping) or
            not _valid_sha256(executor.get("implementation_sha256")) or
            not executor.get("runtime") or not executor.get("device")):
        raise ValueError("exact executor implementation, runtime and device identity are required")
    ids = observation["input_ids"]
    context = observation["context_length"]
    local_predictor = observation["local_predictor_position"]
    absolute_predictor = observation["absolute_predictor_position"]
    window_start = observation["window_start"]
    window_end = observation["window_end"]
    if (not isinstance(ids, (list, tuple)) or not ids or type(context) is not int or
            context != len(ids) or type(local_predictor) is not int or
            type(absolute_predictor) is not int or type(window_start) is not int or
            type(window_end) is not int or local_predictor != context - 1 or
            window_end - window_start != context or
            absolute_predictor != window_start + local_predictor):
        raise ValueError("input IDs, context, window or predictor identities are invalid")
    if observation["token_ids_sha256"] != token_ids_digest(ids):
        raise ValueError("canonical token IDs digest does not match the input window")
    if type(observation["target_id"]) is not int or type(observation["layer"]) is not int:
        raise ValueError("target and layer must be explicit integers")
    if (not observation["dataset"] or not observation["dataset_revision"] or
            not observation["domain"] or not observation["source_group"] or
            not observation["document_id"]):
        raise ValueError("dataset and source identity are required")
    if not _valid_sha256(observation["content_sha256"]):
        raise ValueError("source content SHA-256 is required")
    payload = {"identity": {key: identity[key] for key in IDENTITY_FIELDS},
               "observation": {key: observation[key] for key in OBSERVATION_FIELDS}}
    return hashlib.sha256(_canonical(payload)).hexdigest()


def measurement_cache_compatible(requested: Mapping, cached: Mapping) -> tuple[bool, str]:
    """Fail closed unless cached raw bytes can answer this exact request.

    `requested` and `cached` each contain `identity` and `observation` mappings.
    Plan metadata is intentionally excluded and must be freshly rendered by the
    consumer after a hit.
    """
    try:
        wanted = raw_observation_key(requested["identity"], requested["observation"])
        found = raw_observation_key(cached["identity"], cached["observation"])
    except (KeyError, TypeError, ValueError) as exc:
        return False, f"incomplete_identity:{exc}"
    if wanted != found:
        return False, "raw_observation_identity_mismatch"
    if cached.get("source_measurement_manifest_sha256") is None or not cached.get("source_event_id"):
        return False, "missing_immutable_source_measurement_reference"
    if not cached.get("source_plan_manifest_sha256"):
        return False, "missing_source_plan_reference"
    return True, "exact_raw_observation_match"


def plan_observation_metadata(*, plan_manifest_sha256: str, plan_stage: str,
                              sampling: Mapping, raw_key: str,
                              source_measurement_manifest_sha256: str,
                              source_event_id: str,
                              source_plan_manifest_sha256: str) -> dict:
    """Build fresh per-plan lineage around a reusable raw observation."""
    required = (plan_manifest_sha256, plan_stage, raw_key,
                source_measurement_manifest_sha256, source_event_id,
                source_plan_manifest_sha256)
    if any(not isinstance(v, str) or not v for v in required):
        raise ValueError("plan and immutable source measurement references are required")
    if not isinstance(sampling, Mapping) or not sampling:
        raise ValueError("fresh plan sampling metadata is required")
    return {
        "raw_observation_key": raw_key,
        "raw_source": {
            "source_measurement_manifest_sha256": source_measurement_manifest_sha256,
            "source_event_id": source_event_id,
            "source_plan_manifest_sha256": source_plan_manifest_sha256,
        },
        "plan_metadata": {
            "plan_manifest_sha256": plan_manifest_sha256,
            "stage": plan_stage,
            "sampling": dict(sampling),
        },
    }
