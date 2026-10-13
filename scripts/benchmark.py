"""Verify all technical windows independently and benchmark durable routing/effects."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import time

import numpy as np

from scripts.common import Journal, digest, event_key, load_prepared, parser, progress, records, sample_positions, settings, snapshot, window, write_json
from scripts.measurement import load_model, measure, memory, runtime, verify


def technical_cases(documents, research):
    for doc in documents:
        if doc["stage"] == "technical":
            for predictor in sample_positions(doc, research["sampling"]["technical_positions"], research["seed"], research["context"], technical=True):
                yield doc, predictor


def benchmark(research, execution, model=None, limit=None, loaded_seconds=None):
    prepared, documents, tokens = load_prepared(execution, research)
    cases = list(technical_cases(documents, research))
    for domain in research["domains"]:
        groups = {d["group"] for d in documents if d["stage"] == "technical" and d["domain"] == domain}
        if len(groups) != research["sampling"]["technical_groups"]:
            raise ValueError(f"Technical group coverage missing in {domain}")
    if limit is not None:
        cases = cases[:limit]
    if not cases:
        raise ValueError("No technical windows")
    folder = Path(execution["results_path"])
    snapshot(folder, research, execution)
    load_start = time.monotonic()
    model = model or load_model(research, execution)
    load_seconds = time.monotonic() - load_start if loaded_seconds is None else loaded_seconds
    name = "benchmark" if limit is None else "benchmark-diagnostic"
    journal = Journal(folder / f"{name}.sqlite", research, prepared, name, {"runtime": runtime(model)})
    started = time.monotonic()
    try:
        # Warm-up is explicit and never counted as technical/scientific observations.
        first_ids, first_target = window(*cases[0], tokens, research["context"])
        measure(model, first_ids, first_target, research)
        for index, (doc, predictor) in enumerate(cases, 1):
            key = event_key(doc, predictor)
            ids, target = window(doc, predictor, tokens, research["context"])
            if journal.get(key, ids, target) is None:
                try:
                    verification = verify(model, ids, target, research)
                    times = {}
                    route_state = None
                    # Alternate order by input to reduce fixed-order timing bias.
                    modes = [False, True] if index % 2 else [True, False]
                    for routing in modes:
                        then = time.monotonic()
                        result, state, _, _ = measure(model, ids, target, research, routing=routing)
                        if routing:
                            route_state = state
                        else:
                            effect = result
                        # Include an actual durable write in the timed path.
                        subkey = key + (":routing:" if routing else ":effects:") + str(time.time_ns())
                        journal.append(subkey, result, ids, state["ids"].cpu().numpy() if routing else None,
                                       state["weights"].cpu().numpy() if routing else None)
                        times["routing" if routing else "effects"] = time.monotonic() - then
                    payload = {**effect, "id": key, "domain": doc["domain"], "document": doc["id"],
                               "group": doc["group"], "predictor": predictor, "verification": verification,
                               "durable_seconds": times}
                    journal.append(key, payload, ids)
                except Exception as error:
                    journal.fail(key, error)
                    raise
            progress(name, index, len(cases), started, memory(model))
            if execution["max_seconds"] is not None and time.monotonic() - started >= execution["max_seconds"]:
                raise TimeoutError("Benchmark runtime limit reached; complete windows remain resumable")
    finally:
        journal.close()
    events = [r for key, r in records(folder / f"{name}.sqlite") if "verification" in r]
    times = defaultdict(lambda: defaultdict(list))
    for row in events:
        for mode, elapsed in row["durable_seconds"].items():
            times[row["domain"]][mode].append(elapsed)
    summary = {"complete": len(events) == len(cases), "diagnostic": limit is not None,
               "windows": len(events), "comparisons": sum(r["verification"]["comparisons"] for r in events),
               "cases": [row["id"] for row in events],
               "prepared_id": prepared["id"], "research": research, "runtime": runtime(model),
               "measurement_code": digest((Path(__file__).parent / "measurement.py").read_text()),
               "load_seconds": load_seconds, "elapsed_seconds": time.monotonic() - started,
               "rates": {d: {mode: {"count": len(v), "mean": float(np.mean(v)), "p95": float(np.quantile(v, .95)), "max": max(v)}
                             for mode, v in modes.items()} for d, modes in times.items()}, **memory(model)}
    write_json(folder / f"{name}.json", summary)
    return summary


def require_benchmark(research, execution, prepared, model=None):
    from scripts.common import read_json
    summary = read_json(Path(execution["results_path"]) / "benchmark.json")
    expected = digest((Path(__file__).parent / "measurement.py").read_text())
    _, documents, _ = load_prepared(execution, research)
    expected_cases = {event_key(doc, predictor) for doc, predictor in technical_cases(documents, research)}
    if (not summary["complete"] or summary["diagnostic"] or summary["research"] != research
            or summary["prepared_id"] != prepared["id"] or summary["measurement_code"] != expected
            or set(summary["cases"]) != expected_cases
            or summary["comparisons"] != summary["windows"] * research["model"]["active_experts"]):
        raise ValueError("Full fresh technical verification is required")
    if model is not None and summary["runtime"] != runtime(model):
        raise ValueError("GPU/runtime differs from the verified benchmark; benchmark on this runtime first")
    return summary


def main():
    p = parser(__doc__)
    p.add_argument("--limit", type=int, help="Diagnostic only; cannot substitute for full technical coverage")
    args = p.parse_args()
    research, execution = settings(args)
    print(benchmark(research, execution, limit=args.limit))


if __name__ == "__main__":
    main()
