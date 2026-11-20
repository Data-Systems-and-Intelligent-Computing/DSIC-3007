"""Verify all technical windows independently and benchmark durable routing/effects."""
from __future__ import annotations

from collections import defaultdict
from itertools import zip_longest
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


def diagnostic_cases(cases, domains, limit):
    """Interleave domains and source groups; leave full technical coverage intact."""
    if type(limit) is not int or limit < 1:
        raise ValueError("Diagnostic window limit must be a positive integer")
    by_domain = []
    for domain in domains:
        groups = defaultdict(list)
        for doc, predictor in cases:
            if doc["domain"] == domain:
                groups[doc["group"]].append((doc, predictor))
        by_domain.append([case for row in zip_longest(*groups.values()) for case in row if case is not None])
    return [case for row in zip_longest(*by_domain) for case in row if case is not None][:limit]


def runtime_projection(documents, research, rates, load_seconds):
    """Timing scenarios only: no power calculation, allocation freeze or GPU work."""
    from scripts.experiment import ordered_groups

    scenarios = []

    def add(label, windows, modes, **details):
        missing = [d for d, n in windows.items() if n and any(m not in rates.get(d, {}) for m in modes)]
        row = {"scenario": label, "windows": windows, **details}
        if missing:
            row.update(available=False, reason=f"Missing timing observations in {missing}")
        else:
            row.update(available=True,
                       work_seconds_mean=sum(n * sum(rates[d][m]["mean"] for m in modes) for d, n in windows.items() if n),
                       work_seconds_observed_max=sum(n * sum(rates[d][m]["max"] for m in modes) for d, n in windows.items() if n))
        scenarios.append(row)

    domains = research["domains"]
    technical = defaultdict(int)
    for doc, _ in technical_cases(documents, research):
        technical[doc["domain"]] += 1
    add("full_technical", dict(technical), ("verification", "routing", "effects"))
    probe = {d: sum(min(research["sampling"]["probe_positions"], doc["tokens"] - research["context"])
                    for doc in documents if doc["domain"] == d and doc["stage"] == "probe") for d in domains}
    for budget in research["calibration"]["ladder"]:
        counts = {}
        for domain in domains:
            selected = sorted((doc for doc in documents if doc["domain"] == domain and doc["stage"] == "calibration"),
                              key=lambda doc: doc["stage_order"])[:budget]
            # Nested budgets reuse saved inputs; repeated seed positions are read once.
            keys = {(doc["id"], p) for doc in selected for seed in research["calibration"]["seeds"]
                    for p in sample_positions(doc, 1, seed, research["context"])}
            counts[domain] = len(keys)
        freeze_eligible = budget != research["calibration"]["ladder"][0]
        add(f"calibration_{budget}" + ("_plus_probe" if freeze_eligible else ""),
            {d: counts[d] + (probe[d] if freeze_eligible else 0) for d in domains}, ("routing",),
            documents_per_domain=budget, freeze_eligible=freeze_eligible, includes_probe=freeze_eligible)

    def outcome(label, stage, units):
        selected, windows, group_counts = {}, {}, {}
        for domain in research["outcome_domains"]:
            groups = ordered_groups(documents, domain, stage, research)
            count = units[domain]
            if len(groups) < count:
                scenarios.append({"scenario": label, "available": False,
                                  "reason": f"Only {len(groups)} {domain} groups; {count} requested"})
                return
            selected[domain] = [doc for group in groups[:count] for doc in group]
            group_counts[domain] = count
            windows[domain] = sum(min(research["sampling"]["positions"], doc["tokens"] - research["context"])
                                  for doc in selected[domain])
        add(label, windows, ("effects",), groups=group_counts,
            documents={d: len(v) for d, v in selected.items()})

    for units in research["sampling"]["pilot_ladder"]:
        outcome(f"pilot_{units}", "pilot", {d: units for d in research["outcome_domains"]})
    outcome("main_minimum", "main", {"scientific": research["sampling"]["main_min_scientific_documents"],
                                     "code": research["sampling"]["main_min_code_groups"]})
    outcome("main_inventory", "main", {d: len(ordered_groups(documents, d, "main", research))
                                       for d in research["outcome_domains"]})
    return {"evidence": "Conditional timing extrapolation from sampled technical windows; not a main allocation or precision claim",
            "load_seconds": load_seconds,
            "excluded": "Provisioning, transfer, warm-up, final record writes, calibration tables, compression differences and offline analysis; add one model load per session",
            "observed_max_is_not_a_confidence_bound": True, "scenarios": scenarios}


def benchmark(research, execution, model=None, limit=None, loaded_seconds=None):
    prepared, documents, tokens = load_prepared(execution, research)
    cases = list(technical_cases(documents, research))
    for domain in research["domains"]:
        groups = {d["group"] for d in documents if d["stage"] == "technical" and d["domain"] == domain}
        if len(groups) != research["sampling"]["technical_groups"]:
            raise ValueError(f"Technical group coverage missing in {domain}")
    if limit is not None:
        cases = diagnostic_cases(cases, research["domains"], limit)
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
        warmup_start = time.monotonic()
        measure(model, first_ids, first_target, research)
        warmup_seconds = time.monotonic() - warmup_start
        for index, (doc, predictor) in enumerate(cases, 1):
            key = event_key(doc, predictor)
            ids, target = window(doc, predictor, tokens, research["context"])
            if journal.get(key, ids, target) is None:
                try:
                    verification_start = time.monotonic()
                    verification = verify(model, ids, target, research)
                    verification_seconds = time.monotonic() - verification_start
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
                               "durable_seconds": times, "verification_seconds": verification_seconds}
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
        if "verification_seconds" in row:
            times[row["domain"]]["verification"].append(row["verification_seconds"])
    summary = {"complete": len(events) == len(cases), "diagnostic": limit is not None,
               "windows": len(events), "comparisons": sum(r["verification"]["comparisons"] for r in events),
               "cases": [row["id"] for row in events],
               "prepared_id": prepared["id"], "research": research, "runtime": runtime(model),
               "measurement_code": digest((Path(__file__).parent / "measurement.py").read_text()),
               "load_seconds": load_seconds, "warmup_seconds": warmup_seconds,
               "elapsed_seconds": time.monotonic() - started,
               "rates": {d: {mode: {"count": len(v), "mean": float(np.mean(v)), "p95": float(np.quantile(v, .95)), "max": max(v)}
                             for mode, v in modes.items()} for d, modes in times.items()}, **memory(model)}
    summary["runtime_projection"] = runtime_projection(documents, research, summary["rates"], load_seconds)
    write_json(folder / f"{name}.json", summary)
    print("Runtime projections: estimated work only; add model load/startup/transfer. Observed maxima are not confidence bounds.", flush=True)
    for row in summary["runtime_projection"]["scenarios"]:
        if row["available"]:
            print(f"  {row['scenario']}: {sum(row['windows'].values())} windows; "
                  f"mean-rate estimate {row['work_seconds_mean']/3600:.2f} h; "
                  f"observed-max-rate estimate {row['work_seconds_observed_max']/3600:.2f} h", flush=True)
        else:
            print(f"  {row['scenario']}: unavailable ({row['reason']})", flush=True)
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
