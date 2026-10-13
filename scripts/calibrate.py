"""Collect nested native routing tables, freeze stable calibration, and run the probe."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from scripts.benchmark import require_benchmark
from scripts.common import Journal, digest, event_key, load_prepared, parser, progress, sample_positions, settings, snapshot, window, write_json
from scripts.measurement import load_model, measure, memory, runtime
from scripts.statistics import calibration_tables, pair, stability


def collect_routing(journal, model, doc, predictor, tokens, research):
    key = event_key(doc, predictor)
    ids, target = window(doc, predictor, tokens, research["context"])
    result = journal.get(key, ids, target)
    if result is None:
        try:
            result, state, _, _ = measure(model, ids, target, research, routing=True)
            routes = state["ids"].cpu().numpy()
            coefficients = state["weights"].cpu().numpy()
            counts = np.bincount(routes.ravel(), minlength=research["model"]["experts"]) / len(routes)
            result.update(id=key, document=doc["id"], group=doc["group"], domain=doc["domain"],
                          predictor=predictor, utilization=counts.tolist())
            journal.append(key, result, ids, routes, coefficients)
        except Exception as error:
            journal.fail(key, error)
            raise
    return result


def calibrate(research, execution, model=None):
    prepared, documents, tokens = load_prepared(execution, research)
    folder = Path(execution["results_path"])
    require_benchmark(research, execution, prepared)
    model = model or load_model(research, execution)
    require_benchmark(research, execution, prepared, model)
    snapshot(folder, research, execution)
    journal = Journal(folder / "calibration.sqlite", research, prepared, "calibration", {"runtime": runtime(model)})
    observations, previous, history, frozen = {}, None, [], None
    started = time.monotonic()
    try:
        for budget in research["calibration"]["ladder"]:
            members = [d for domain in research["domains"]
                       for d in sorted((r for r in documents if r["domain"] == domain and r["stage"] == "calibration"), key=lambda r: r["stage_order"])[:budget]]
            total = len(members) * len(research["calibration"]["seeds"])
            count = 0
            for doc in members:
                for seed in research["calibration"]["seeds"]:
                    predictor = sample_positions(doc, 1, seed, research["context"])[0]
                    row = collect_routing(journal, model, doc, predictor, tokens, research)
                    observations[doc["id"], seed] = row["utilization"]
                    count += 1
                    progress(f"calibration/{budget}", count, total, started, memory(model))
                    if execution["max_seconds"] is not None and time.monotonic() - started >= execution["max_seconds"]:
                        raise TimeoutError("Calibration limit reached; completed windows remain resumable")
            tables = calibration_tables(observations, documents, research, budget)
            check = None if previous is None else stability(previous, tables, research)
            history.append({"budget": budget, "tables": {d: [t.tolist() for t in values] for d, values in tables.items()}, "stability": check})
            write_json(folder / "calibration-history.json", history)
            if check and check["pass"]:
                domain_tables = {d: np.mean(v, axis=0).tolist() for d, v in tables.items()}
                frozen = {"budget": budget, "domains": domain_tables,
                          "M1": np.mean(list(domain_tables.values()), axis=0).tolist(),
                          "research": research, "prepared_id": prepared["id"], "history": history,
                          "runtime": runtime(model)}
                frozen["id"] = digest(frozen)
                break
            previous = tables
        if frozen is None:
            write_json(folder / "calibration-status.json", {"stable": False, "history": history})
            raise ValueError("No stable calibration candidate; outcomes must not proceed")
        agreement = {domain: [] for domain in research["domains"]}
        for doc in documents:
            if doc["stage"] != "probe":
                continue
            for predictor in sample_positions(doc, research["sampling"]["probe_positions"], research["seed"], research["context"]):
                row = collect_routing(journal, model, doc, predictor, tokens, research)
                ties = np.random.default_rng(int(digest([research["seed"], row["id"], "ties"])[:16], 16)).permutation(row["active"]).tolist()
                matches = pair(row["active"], frozen["M1"], ties) == pair(row["active"], frozen["domains"][doc["domain"]], ties)
                agreement[doc["domain"]].append(bool(matches))
                if execution["max_seconds"] is not None and time.monotonic() - started >= execution["max_seconds"]:
                    raise TimeoutError("Probe limit reached; completed inputs remain resumable")
        frozen["probe"] = {d: {"events": len(v), "M1_M2_pair_agreement": float(np.mean(v)) if v else None} for d, v in agreement.items()}
        # Bind rankings, seeds, budget and probe evidence to the actual saved object.
        frozen.pop("id")
        frozen["id"] = digest(frozen)
        write_json(folder / "calibration.json", frozen)
        write_json(folder / "calibration-status.json", {"stable": True, "budget": frozen["budget"]})
    finally:
        journal.close()
    return frozen


def main():
    args = parser(__doc__).parse_args()
    research, execution = settings(args)
    result = calibrate(research, execution)
    print(f"Frozen calibration at {result['budget']} documents/domain; {result['probe']}")


if __name__ == "__main__":
    main()
