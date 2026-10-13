"""Collect nested pilot or frozen main effects, automatically resuming complete inputs."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import time

from scripts.benchmark import require_benchmark
from scripts.common import Journal, canonical, digest, event_key, load_prepared, parser, progress, read_json, sample_positions, seed_for, settings, snapshot, window
from scripts.measurement import load_model, measure, memory, runtime


def calibration_for(research, execution, prepared):
    value = read_json(Path(execution["results_path"]) / "calibration.json")
    identity = value.pop("id")
    if digest(value) != identity or value["research"] != research or value["prepared_id"] != prepared["id"]:
        raise ValueError("Frozen calibration does not match this study/data")
    return {**value, "id": identity}


def ordered_groups(documents, domain, stage, research):
    groups = defaultdict(list)
    for doc in documents:
        if doc["domain"] == domain and doc["stage"] == stage:
            groups[doc["group"]].append(doc)
    return [sorted(groups[g], key=lambda d: d["id"]) for g in sorted(groups, key=lambda g: seed_for(research["seed"], domain, stage, g))]


def pilot_documents(documents, units, research):
    selected = []
    if units not in research["sampling"]["pilot_ladder"]:
        raise ValueError("Pilot units must be a configured ladder rung")
    for domain in research["outcome_domains"]:
        groups = ordered_groups(documents, domain, "pilot", research)
        if len(groups) < units:
            raise ValueError(f"Insufficient pilot groups in {domain}")
        selected.extend(doc for group in groups[:units] for doc in group)
    return selected


def collect(research, execution, stage, units=None, model=None):
    prepared, documents, tokens = load_prepared(execution, research)
    folder = Path(execution["results_path"])
    calibration = calibration_for(research, execution, prepared)
    require_benchmark(research, execution, prepared)
    extra = {"calibration": calibration}
    if stage == "pilot":
        units = units or research["sampling"]["pilot_ladder"][0]
        selected = pilot_documents(documents, units, research)
        cases = [(doc, p) for doc in selected for p in sample_positions(doc, research["sampling"]["positions"], research["seed"], research["context"])]
    elif stage == "main":
        plan = read_json(folder / "main-plan.json")
        identity = plan.pop("id")
        if (digest(plan) != identity or plan["prepared_id"] != prepared["id"]
                or plan["research"] != research or plan["calibration_id"] != calibration["id"]):
            raise ValueError("Main plan differs from the prospectively frozen study")
        lookup = {d["id"]: d for d in documents if d["stage"] == "main"}
        cases = [(lookup[item["document"]], predictor) for item in plan["selection"] for predictor in item["predictors"]]
        extra["main_plan"] = {**plan, "id": identity}
    else:
        raise ValueError("Only pilot and main outcomes are selected")
    model = model or load_model(research, execution)
    require_benchmark(research, execution, prepared, model)
    if runtime(model) != calibration["runtime"]:
        raise ValueError("Runtime differs from frozen calibration")
    extra["runtime"] = runtime(model)
    snapshot(folder, research, execution)
    journal = Journal(folder / f"{stage}.sqlite", research, prepared, stage, extra)
    target_keys = [event_key(doc, p) for doc, p in cases]
    with journal.db:
        journal.db.execute("INSERT OR REPLACE INTO metadata VALUES ('target',?)", (canonical({"keys": target_keys, "units": units}),))
    started = time.monotonic()
    try:
        for index, (doc, predictor) in enumerate(cases, 1):
            key = event_key(doc, predictor)
            ids, target = window(doc, predictor, tokens, research["context"])
            if journal.get(key, ids, target) is None:
                try:
                    result, state, _, _ = measure(model, ids, target, research)
                    result.update(id=key, domain=doc["domain"], document=doc["id"], group=doc["group"], predictor=predictor)
                    journal.append(key, result, ids)
                except Exception as error:
                    journal.fail(key, error)
                    raise
            progress(stage, index, len(cases), started, memory(model))
            if execution["max_seconds"] is not None and time.monotonic() - started >= execution["max_seconds"]:
                raise TimeoutError("Execution limit reached; completed inputs remain resumable")
        with journal.db:
            journal.db.execute("INSERT OR REPLACE INTO metadata VALUES ('complete', 'true')")
    finally:
        journal.close()
    return len(cases)


def main():
    p = parser(__doc__)
    p.add_argument("stage", choices=["pilot", "main"])
    p.add_argument("--units", type=int, help="Nested pilot rung; ignored for frozen main")
    args = p.parse_args()
    research, execution = settings(args)
    print(f"Completed {collect(research, execution, args.stage, args.units)} windows")


if __name__ == "__main__":
    main()
