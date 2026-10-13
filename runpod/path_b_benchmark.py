"""Three-window Path B GPU diagnostic; never a collection acceptance receipt."""

import argparse
import math
import time
from pathlib import Path

from src.artifacts import atomic, canonical, digest, file_digest, read_json, read_rows, verify_bundle
from src.model import VerifiedModelSession, implementation_identity, runtime_identity
from src.optimized_validation import _record, validate_optimized
from src.run_experiment import measure_input
from src.scope import scope_identity, validate_plan


class TimedModelSession(VerifiedModelSession):
    """Observe acquisition timing while preserving all parent identity checks."""

    def __init__(self, phases):
        # User explicitly removed repeated weight checks from this benchmark;
        # completed model preparation is required by its controller instead.
        super().__init__(verify_weights=False)
        self.phases = phases

    def acquire(self, *args, **kwargs):
        fresh = self._model is None
        started = time.monotonic()
        succeeded = False
        try:
            model = super().acquire(*args, **kwargs)
            succeeded = True
            return model
        finally:
            if fresh:
                self.phases["verified_model_loading"] = {
                    "elapsed_seconds": time.monotonic()-started,
                    "status": "pass" if succeeded else "fail",
                    "nested_in": "independent_validation_and_load; do not add twice to workflow total",
                    "includes": "runtime identity, model load and session identity; no weight hash/F32 rescan; not isolated H2D copy"}


def critical_metrics(events, phase):
    """Summarize observed units; absent data remains absent, never zero-filled."""
    def maximum(key):
        values = [e[key] for e in events if key in e]
        return max(values) if values else None

    elapsed = phase.get("elapsed_seconds")
    return {"unit": "one 4096-token input window, batch_size=1, layer=15",
            "baseline_with_capture_ms_per_batch": [1000*e["shared_seconds"] for e in events],
            "native_cleanup_ms_per_batch": [1000*e["shared_cleanup_seconds"] for e in events],
            "eight_zero_output_effects_ms": [1000*e["eight_interventions_seconds"] for e in events],
            "peak_cuda_allocated_bytes": maximum("cuda_peak_allocated_bytes"),
            "peak_cuda_reserved_bytes": maximum("cuda_peak_reserved_bytes"),
            "peak_process_rss_kib": maximum("host_max_rss_kib"),
            "completed_measured_windows": len(events),
            "collector_phase_windows_per_second": len(events)/elapsed if elapsed and elapsed > 0 else None,
            "throughput_scope": "collector/persistence only; excludes load, validation, setup and teardown",
            "first_ablation_separate": "not measured; total all-eight recorded",
            "router_suppression": "not run; outside Path B operator",
            "cpu_utilization_percent": "not measured; process peak RSS recorded",
            "r2_upload": "not performed",
            "interpretation": "bounded diagnostic observations; not batch-size capacity or main throughput proof"}


def run(plan, model_dir, output, deadline_epoch):
    started = time.time()
    if not math.isfinite(deadline_epoch) or not started < deadline_epoch <= started+1800:
        raise ValueError("work deadline must be finite, in the future, and within 1800 seconds")
    output = Path(output)
    status_path = output.with_suffix(".status.json")
    if output.exists() or status_path.exists():
        raise FileExistsError("preserve the prior diagnostic; choose a new output")
    phases = {}
    measured_events = []
    receipt = {"status": "incomplete", "kind": "path-b-preflight",
               "diagnostic_only": True, "scientific_collection_authorized": False,
               "phases": phases, "completed_native_comparisons": 0,
               "completed_measured_inputs": 0, "completed_measured_events": 0,
               "weights_verification": "completed setup receipt only; per-run byte/F32 verification disabled by user",
               "deadline_epoch": deadline_epoch}

    def persist(phase):
        receipt.update(phase=phase, elapsed_seconds=time.time()-started,
                       remaining_seconds=max(0, deadline_epoch-time.time()))
        atomic(status_path, canonical(receipt)+b"\n")
        print(f"[path-b-preflight] {phase} elapsed={receipt['elapsed_seconds']:.1f}s "
              f"remaining={receipt['remaining_seconds']:.1f}s", flush=True)

    def check():
        if time.time() >= deadline_epoch-5:
            raise TimeoutError("Path B work deadline reached; reserve final receipt")

    try:
        manifest = verify_bundle(plan, "experiment-plan")
        settings = read_json(Path(plan)/"research.json")
        rows = list(read_rows(Path(plan)/"inputs.jsonl"))
        validate_plan(settings, manifest["metadata"], rows)
        if len(rows) != 3 or not manifest["metadata"].get("preflight_diagnostic"):
            raise ValueError("exact three-window diagnostic plan required")
        receipt.update(research_scope=scope_identity(settings),
                       plan_manifest_sha256=file_digest(Path(plan)/"manifest.json"),
                       selected_input_ids=[r["input_id"] for r in rows])
        session = TimedModelSession(phases)
        check()
        persist("independent_native_validation_including_model_load")
        phase_started = time.time()
        native_path = output.with_name("independent-native-validation.json")
        native = validate_optimized([plan], model_dir, native_path,
            max_seconds=deadline_epoch-time.time()-5,
            validation_scope="preflight", model_session=session)
        phases["independent_validation_and_load"] = {"elapsed_seconds": time.time()-phase_started,
            "receipt_sha256": file_digest(native_path), "status": native["status"]}
        receipt["completed_native_comparisons"] = native.get("completed_expert_comparisons", 0)
        if native["status"] == "incomplete":
            raise TimeoutError("independent native validation incomplete before shared work deadline")
        if native["status"] != "pass" or native.get("completed_expert_comparisons") != 24:
            raise ValueError("complete 24-comparison independent validation required before timings")
        check()
        catalog = read_json(Path(plan)/"model-catalog.json")
        model = session.acquire(model_dir, catalog, settings["sampling"]["master_seed"])
        receipt["executor"] = {"model_catalog_sha256": digest(catalog),
            "runtime": runtime_identity(), "implementation_sha256": implementation_identity()}
        persist("warmup")
        phase_started = time.time()
        for event in measure_input(model, rows[0], optimized=True):
            _record(output.with_name("warmup-events.jsonl"), event)
        phases["warmup"] = {"elapsed_seconds": time.time()-phase_started, "inputs": 1, "events": 1}
        check()
        phase_started = time.time()
        for index, row in enumerate(rows, 1):
            check()
            persist(f"collector_input_{index}_of_3_domain_{row['domain']}")
            measured_start = time.monotonic()
            events = measure_input(model, row, optimized=True)
            for event in events:
                _record(output.with_name("measured-events.jsonl"), event)
                measured_events.append(event)
            elapsed = time.monotonic()-measured_start
            _record(output.with_name("input-timings.jsonl"), {
                "input_id": row["input_id"], "domain": row["domain"], "context_length": 4096,
                "layers": [15], "event_ids": [e["event_id"] for e in events],
                "input_total_seconds": events[0]["input_total_seconds"],
                "end_to_end_with_event_persistence_seconds": elapsed,
                "persistence_seconds": max(0,elapsed-events[0]["input_total_seconds"]),
                "executor": receipt["executor"], "diagnostic_only": True})
            receipt["completed_measured_inputs"] += 1
            receipt["completed_measured_events"] += len(events)
            persist(f"collector_input_{index}_durable_pass")
        phases["collector_and_persistence"] = {"elapsed_seconds": time.time()-phase_started,
            "inputs": receipt["completed_measured_inputs"], "events": receipt["completed_measured_events"]}
        check()
        receipt["status"] = "pass"
    except BaseException as exc:
        receipt.update(status="incomplete" if isinstance(exc, (TimeoutError, KeyboardInterrupt)) else "fail",
                       stop_reason=f"{type(exc).__name__}: {exc}")
    finally:
        persist("finished")
        receipt["critical_metrics"] = critical_metrics(measured_events, phases.get("collector_and_persistence", {}))
        receipt["raw_evidence_sha256"] = {p.name:file_digest(p) for p in output.parent.glob("*.jsonl")}
        atomic(output, canonical(receipt)+b"\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("plan", "model-dir", "output"):
        parser.add_argument("--"+arg, type=Path, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args(argv)
    if not math.isfinite(args.deadline_epoch) or not time.time() < args.deadline_epoch <= time.time()+1800:
        parser.error("work deadline must be finite and in the next 1800 seconds")
    return 0 if run(args.plan, args.model_dir, args.output, args.deadline_epoch)["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
