"""Fail-closed in-memory index for exact raw observation reuse."""

from copy import deepcopy
import json
import time
from collections.abc import Mapping
from pathlib import Path

from .artifacts import (file_digest, read_json, validate_measurement,
                        verify_bundle)
from .observation_contract import (measurement_cache_compatible,
                                   plan_observation_metadata,
                                   raw_observation_key, token_ids_digest)


_TIMING_FIELDS = (
    "input_total_seconds", "synchronized_input_total_seconds",
    "shared_baseline_seconds", "shared_seconds", "shared_cleanup_seconds",
    "cleanup_seconds", "eight_interventions_seconds", "effect_seconds",
)
_RESOURCE_FIELDS = ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes",
                    "host_max_rss_kib")


class ObservationReuseIndex:
    """Index complete raw rows while keeping fresh plan lineage separate."""

    def __init__(self):
        self._rows = {}

    def add(self, *, identity: Mapping, observation: Mapping, event: Mapping,
            source_measurement_manifest_sha256: str, source_event_id: str,
            source_plan_manifest_sha256: str) -> str:
        key = raw_observation_key(identity, observation)
        if not isinstance(event, Mapping) or not event:
            raise ValueError("complete raw event required")
        source = {
            "source_measurement_manifest_sha256": source_measurement_manifest_sha256,
            "source_event_id": source_event_id,
            "source_plan_manifest_sha256": source_plan_manifest_sha256,
        }
        if any(not isinstance(value, str) or not value for value in source.values()):
            raise ValueError("immutable source measurement references are required")
        record = {"identity": deepcopy(dict(identity)),
                  "observation": deepcopy(dict(observation)),
                  "event": deepcopy(dict(event)), "source": source}
        previous = self._rows.get(key)
        if previous is not None and previous != record:
            raise ValueError("conflicting raw observation for exact cache key")
        self._rows[key] = record
        return key

    def get(self, *, identity: Mapping, observation: Mapping,
            plan_manifest_sha256: str, plan_stage: str,
            sampling: Mapping) -> dict | None:
        start = time.perf_counter()
        try:
            key = raw_observation_key(identity, observation)
        except (TypeError, ValueError):
            return None
        record = self._rows.get(key)
        if record is None:
            return None
        cached = {"identity": record["identity"],
                  "observation": record["observation"], **record["source"]}
        requested = {"identity": identity, "observation": observation}
        compatible, reason = measurement_cache_compatible(requested, cached)
        if not compatible:
            return None
        lineage = plan_observation_metadata(
            plan_manifest_sha256=plan_manifest_sha256, plan_stage=plan_stage,
            sampling=sampling, raw_key=key,
            source_measurement_manifest_sha256=record["source"]["source_measurement_manifest_sha256"],
            source_event_id=record["source"]["source_event_id"],
            source_plan_manifest_sha256=record["source"]["source_plan_manifest_sha256"])
        event = deepcopy(record["event"])
        timings = {name: event.pop(name) for name in _TIMING_FIELDS if name in event}
        resources = {name: event.pop(name) for name in _RESOURCE_FIELDS if name in event}
        return {
            "event": event,
            "source_timing": timings,
            "source_resources": resources,
            "lineage": lineage,
            "cache_hit_seconds": time.perf_counter() - start,
            "cache_status": "hit",
        }

    def __len__(self):
        return len(self._rows)


def _jsonl_offsets(path, wanted, field):
    offsets = {}
    with Path(path).open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            row = json.loads(line)
            key = row.get(field)
            if key in wanted:
                if key in offsets:
                    raise ValueError(f"duplicate {field} in sealed source bundle")
                offsets[key] = offset
    return offsets


def _row_at(path, offset):
    with Path(path).open("rb") as stream:
        stream.seek(offset)
        return json.loads(stream.readline())


def _observation(input_row, document, layer, stage):
    ids = input_row["input_ids"]
    return {
        "dataset": input_row["dataset"],
        "dataset_revision": input_row["revision"],
        "domain": input_row["domain"],
        "source_group": input_row["source_group"],
        "document_id": input_row["document_id"],
        "content_sha256": document["content_sha256"],
        "input_ids": ids,
        "token_ids_sha256": token_ids_digest(ids),
        "context_length": input_row["context_length"],
        "window_start": input_row["window_start"],
        "window_end": input_row["window_end"],
        "absolute_predictor_position": input_row["predictor_position"],
        "local_predictor_position": input_row["local_predictor_position"],
        "target_id": input_row["target_id"],
        "stage_kind": stage,
        "layer": layer,
        "operator": ("token_local_zero_contribution" if stage not in ("calibration", "probe")
                     else f"native_{stage}_routing"),
    }


def _validate_event_against_plan(event, input_row, layer):
    """Reject source rows whose visible raw identity disagrees with its sealed plan."""
    expected = {
        "input_id": input_row["input_id"], "layer": layer,
        "document_id": input_row["document_id"], "dataset": input_row["dataset"],
        "revision": input_row["revision"], "domain": input_row["domain"],
        "source_group": input_row["source_group"],
        "context_length": input_row["context_length"],
        "window_start": input_row["window_start"], "window_end": input_row["window_end"],
        "predictor_position": input_row["predictor_position"],
        "local_predictor_position": input_row["local_predictor_position"],
        "target_position": input_row["target_position"], "target_id": input_row["target_id"],
        "token_ids_sha256": input_row["token_ids_sha256"],
    }
    mismatches = [name for name, value in expected.items()
                  if event.get(name) != value]
    if mismatches:
        raise ValueError("sealed source event disagrees with its plan: " + ", ".join(mismatches))


class SealedMeasurementReuse:
    """Read exact matching rows lazily from a verified immutable measurement bundle."""

    def __init__(self, directory, requested_inputs, *, executor, execution_mode,
                 model_revision, tokenizer_revision, plan_stage):
        self.directory = Path(directory).resolve(strict=True)
        self.manifest = verify_bundle(self.directory, "measurements")
        self.source_manifest_sha256 = file_digest(self.directory / "manifest.json")
        self.identity = read_json(self.directory / "identity.json")
        self.source_executor = self.identity.get("executor")
        if self.source_executor != executor or self.identity.get("execution_mode") != execution_mode:
            raise ValueError("reuse source executor or execution mode differs")
        if self.manifest["metadata"].get("executor") != executor:
            raise ValueError("reuse source manifest executor differs from journal identity")
        source_plan = self.directory / "plan"
        self.plan_manifest = verify_bundle(source_plan, "experiment-plan")
        self.plan_manifest_sha256 = file_digest(source_plan / "manifest.json")
        source_meta = self.plan_manifest["metadata"]
        if (source_meta.get("stage") != plan_stage or
                source_meta.get("tokenizer_sha256") != tokenizer_revision):
            raise ValueError("reuse source plan stage or tokenizer differs")
        if (self.identity.get("plan_manifest_sha256") != self.plan_manifest_sha256 or
                self.manifest["metadata"].get("stage") != plan_stage or
                self.manifest["metadata"].get("execution_mode") != execution_mode):
            raise ValueError("reuse source plan provenance does not match its sealed run")
        catalog = read_json(source_plan / "model-catalog.json")
        if catalog.get("revision") != model_revision:
            raise ValueError("reuse source model revision differs")
        self.stage = plan_stage
        self.inputs_path = source_plan / "inputs.jsonl"
        self.events_path = self.directory / "events.jsonl"
        requested = {row["input_id"]: row for row in requested_inputs}
        # Event IDs are digests; index only rows requested by the current plan.
        from .run_experiment import event_id
        wanted_events = {event_id(row, layer) for row in requested_inputs for layer in row["layers"]}
        self.event_offsets = _jsonl_offsets(self.events_path, wanted_events, "event_id")
        self.input_offsets = _jsonl_offsets(self.inputs_path, set(requested), "input_id")
        wanted_docs = {row["document_id"] for row in requested_inputs}
        self.document_offsets = {}
        with (source_plan / "documents.jsonl").open("rb") as stream:
            for line in stream:
                offset = stream.tell() - len(line)
                document = json.loads(line)
                if document.get("document_id") in wanted_docs:
                    self.document_offsets[document["document_id"]] = offset
        runtime = executor.get("runtime", {})
        device = runtime.get("gpu_name") or runtime.get("device") or "CPU"
        self.executor = {**executor, "execution_mode": execution_mode, "device": device}
        self.model_revision = model_revision
        self.tokenizer_revision = tokenizer_revision

    def get(self, input_row, layer, *, document, plan_manifest_sha256,
            plan_stage, sampling):
        from .run_experiment import event_id
        start = time.perf_counter()
        identifier = event_id(input_row, layer)
        event_offset = self.event_offsets.get(identifier)
        input_offset = self.input_offsets.get(input_row["input_id"])
        if event_offset is None or input_offset is None:
            return None
        source_event = _row_at(self.events_path, event_offset)
        source_input = _row_at(self.inputs_path, input_offset)
        validate_measurement(source_event)
        if (source_event.get("event_id") != identifier or source_event.get("layer") != layer or
                event_id(source_input, layer) != identifier):
            raise ValueError("reused event ID/layer does not match the requested plan")
        _validate_event_against_plan(source_event, source_input, layer)
        document_offset = self.document_offsets.get(input_row["document_id"])
        source_document = (_row_at(self.directory / "plan" / "documents.jsonl", document_offset)
                           if document_offset is not None else None)
        if source_document is None or source_document.get("content_sha256") != document.get("content_sha256"):
            raise ValueError("reused document content identity differs")
        source_identity = {
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "executor": self.executor,
            "measurement_kind": ("calibration" if source_event["kind"] == "calibration_window"
                                 else "probe" if source_event["kind"] == "probe_routing" else "effects"),
        }
        requested_identity = {**source_identity}
        source_observation = _observation(source_input, source_document, layer, self.stage)
        requested_observation = _observation(input_row, document, layer, plan_stage)
        cached = {"identity": source_identity, "observation": source_observation,
                  "source_measurement_manifest_sha256": self.source_manifest_sha256,
                  "source_event_id": identifier,
                  "source_plan_manifest_sha256": self.plan_manifest_sha256}
        compatible, reason = measurement_cache_compatible(
            {"identity": requested_identity, "observation": requested_observation}, cached)
        if not compatible:
            raise ValueError("sealed reuse source rejected: " + reason)
        key = raw_observation_key(source_identity, source_observation)
        fresh = plan_observation_metadata(
            plan_manifest_sha256=plan_manifest_sha256, plan_stage=plan_stage,
            sampling=sampling, raw_key=key,
            source_measurement_manifest_sha256=self.source_manifest_sha256,
            source_event_id=identifier,
            source_plan_manifest_sha256=self.plan_manifest_sha256)
        event = deepcopy(source_event)
        timings = {name: event.pop(name) for name in _TIMING_FIELDS if name in event}
        resources = {name: event.pop(name) for name in _RESOURCE_FIELDS if name in event}
        # Plan-owned fields are regenerated for this consumer; raw routes/effects remain source bytes.
        for name in ("document_id", "dataset", "revision", "domain", "source_group",
                     "sampling", "sample_order", "eligible_positions", "context_length",
                     "window_start", "window_end", "predictor_position",
                     "local_predictor_position", "target_position", "target_id",
                     "token_ids_sha256"):
            if name in input_row:
                event[name] = deepcopy(input_row[name])
        event.update(raw_observation_key=key, raw_source=fresh["raw_source"],
                     plan_metadata=fresh["plan_metadata"], source_timing=timings,
                     source_resources=resources,
                     cache_hit_seconds=time.perf_counter() - start, cache_status="hit")
        return event
