"""Offline paired inference, pilot assessment, prospective main selection, and figures."""
from __future__ import annotations

import csv
import math
from pathlib import Path

from scripts.benchmark import require_benchmark
from scripts.common import digest, load_prepared, metadata, parser, read_json, records, sample_positions, settings, snapshot, write_json
from scripts.experiment import calibration_for, ordered_groups
from scripts.statistics import bootstrap, document_means


def complete_events(path, research, prepared, calibration):
    meta = metadata(path)
    binding = meta["binding"]
    if binding["research"] != research or binding["prepared_id"] != prepared["id"] or binding["calibration"] != calibration:
        raise ValueError("Analysis must use the actual collected settings, data, and frozen calibration")
    wanted = set(meta["target"]["keys"])
    found = {key: row for key, row in records(path) if key in wanted}
    if set(found) != wanted:
        raise ValueError(f"Incomplete stage: {len(found)}/{len(wanted)} windows; resume collection")
    for row in found.values():
        if not row.get("cleanup_exact") or len(row.get("effects", [])) != research["model"]["active_experts"]:
            raise ValueError("Incomplete or invalid expert event")
    return list(found.values()), meta


def pilot_assessment(rows, inference, research):
    reasons = []
    for domain in research["outcome_domains"]:
        result = inference[domain]
        if result["groups"] < research["analysis"]["minimum_pilot_groups"]:
            reasons.append(domain + ": fewer than ten groups")
        if result["ci"] is None:
            reasons.append(domain + ": invalid bootstrap interval")
            continue
        source_sd = math.sqrt(result["variance"] * result["groups"])
        halfwidth = (result["ci"][1] - result["ci"][0]) / 2
        if halfwidth > research["analysis"]["pilot_halfwidth_sd_fraction"] * source_sd:
            reasons.append(domain + ": pilot uncertainty exceeds source-SD threshold")
    return {"sufficient": not reasons, "reasons": reasons,
            "source_sd_definition": "sqrt(number_of_groups * cluster_variance), retaining unequal code group sizes"}


def plan_main(documents, pilot, assessment, benchmark, calibration, prepared, research, execution):
    if not assessment["sufficient"] or pilot["primary"]["critical"] is None:
        raise ValueError("Pilot is insufficient; extend at the next configured rung or report insufficiency")
    for key in ("remaining_usd", "remaining_wall_hours", "hourly_usd", "fixed_remaining_usd", "remaining_overhead_seconds", "remaining_offline_seconds"):
        if execution.get(key) is None:
            raise ValueError(f"Set reconciled execution.{key} before freezing main")
    groups = {d: ordered_groups(documents, d, "main", research) for d in research["outcome_domains"]}
    costs = {}
    for domain in research["outcome_domains"]:
        times = benchmark["rates"][domain]["effects"]
        costs[domain] = times["max"]
    minimum = research["sampling"]
    s_min, c_min = minimum["main_min_scientific_documents"], minimum["main_min_code_groups"]
    # Scientific preparation retains one audited version per paper component.
    if any(len(group) != 1 for group in groups["scientific"]):
        raise ValueError("Scientific main requires one selected document per source component")
    pos = research["sampling"]["positions"]
    counts = {domain: [sum(min(pos, d["tokens"] - research["context"]) for d in group) for group in values]
              for domain, values in groups.items()}
    cumulative = {d: [0] for d in groups}
    for domain in groups:
        for count in counts[domain]:
            cumulative[domain].append(cumulative[domain][-1] + count)
    pilot_s = pilot["scientific"]["variance"] * pilot["scientific"]["groups"]
    pilot_c = pilot["code"]["variance"] * pilot["code"]["groups"]
    critical = pilot["primary"]["critical"]
    target = research["analysis"]["target_halfwidth"]
    options = []
    for c in range(c_min, len(groups["code"]) + 1):
        available_var = 4 * (target / critical) ** 2 - pilot_c / c
        if available_var <= 0:
            continue
        s = max(s_min, math.ceil(pilot_s / available_var))
        if s <= len(groups["scientific"]):
            seconds = cumulative["scientific"][s] * costs["scientific"] + cumulative["code"][c] * costs["code"]
            options.append((seconds, s, c))
    if not options:
        raise ValueError("No available allocation meets the target precision; report insufficiency or decide a prospective fallback")
    seconds, s, c = min(options)
    forecast = (seconds + execution["remaining_overhead_seconds"]) * (1 + execution["reserve_fraction"])
    wall_seconds = forecast + execution["remaining_offline_seconds"]
    cost = forecast / 3600 * execution["hourly_usd"] + execution["fixed_remaining_usd"]
    if wall_seconds / 3600 > execution["remaining_wall_hours"] or cost > execution["remaining_usd"]:
        raise ValueError("Precision-qualified main does not fit reconciled remaining funds/time including reserve; fallback remains a scientific decision")
    selection = [{"document": doc["id"], "domain": domain, "group": doc["group"],
                  "predictors": sample_positions(doc, pos, research["seed"], research["context"])}
                 for domain, n in (("scientific", s), ("code", c)) for group in groups[domain][:n] for doc in group]
    plan = {"research": research, "prepared_id": prepared["id"], "calibration_id": calibration["id"],
            "selection": selection, "units": {"scientific": s, "code": c},
            "windows": sum(len(x["predictors"]) for x in selection),
            "projected_halfwidth": critical * math.sqrt((pilot_s / s + pilot_c / c) / 4),
            "remaining_work": {"main_seconds": seconds, "overhead_seconds": execution["remaining_overhead_seconds"],
                               "reserve_fraction": execution["reserve_fraction"], "total_hours": forecast / 3600,
                               "offline_seconds": execution["remaining_offline_seconds"], "wall_hours": wall_seconds / 3600,
                               "nominal_usd": cost, "evidence": "pilot cluster variance and full technical durable cohort maxima; conditional forecast, not achieved precision or invoice"}}
    plan["id"] = digest(plan)
    return plan


def write_artifacts(folder, rows, results, research):
    folder = Path(folder)
    with (folder / "document-means.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (folder / "estimates.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "population", "estimate_nats", "ci_low", "ci_high", "valid_draws"])
        for metric, inference in results.items():
            for domain in [*research["outcome_domains"], "primary"]:
                result = inference[domain]
                writer.writerow([metric, domain, result["estimate"], *(result["ci"] or ["", ""]), result["valid_draws"]])
    # Matplotlib is optional: analysis and numeric artifacts do not depend on it.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for y, domain in enumerate([*research["outcome_domains"], "primary"]):
        result = results["delta"][domain]
        if result["ci"] is not None:
            lo, hi = result["ci"]
            ax.plot([lo, hi], [y, y], color="tab:blue")
        ax.scatter([result["estimate"]], [y], color="tab:blue")
    ax.axvline(0, color="grey", lw=1)
    ax.set_yticks(range(3), [*research["outcome_domains"], "primary"])
    ax.set_xlabel("Paired M2 − M1 discrimination (nats)")
    fig.tight_layout()
    fig.savefig(folder / "paired-difference.pdf")
    fig.savefig(folder / "paired-difference.png", dpi=180)
    plt.close(fig)


def analyze(research, execution, stage, freeze_main=False):
    prepared, documents, _ = load_prepared(execution, research)
    folder = Path(execution["results_path"])
    calibration = calibration_for(research, execution, prepared)
    events, meta = complete_events(folder / f"{stage}.sqlite", research, prepared, calibration)
    rows = document_means(events, calibration, research)
    draws = research["analysis"]["planning_draws"] if stage == "pilot" else research["analysis"]["bootstrap_draws"]
    metrics = ["delta", "M1", "M2", "M0", "M4", *["ranking_" + d for d in research["domains"]]]
    results = {metric: bootstrap(rows, research, draws, metric) for metric in metrics}
    for metric, inference in results.items():
        inference["role"] = "primary_and_two_confirmatory_secondary" if metric == "delta" else "descriptive"
        if metric != "delta":
            inference.pop("holm")
            for domain in [*research["outcome_domains"], "primary"]:
                inference[domain].pop("p_two_sided")
    output = {"stage": stage, "calibration_id": calibration["id"], "conditional_on_frozen_calibration": True,
              "results": results, "randomization": {domain: {"M0_mc_se": math.sqrt(sum(row["M0_mc_variance"] for row in rows if row["domain"] == domain)) / sum(row["domain"] == domain for row in rows)}
                                                      for domain in research["outcome_domains"]}}
    output["randomization"]["primary"] = {"M0_mc_se": math.sqrt(sum(output["randomization"][d]["M0_mc_se"] ** 2 for d in research["outcome_domains"]) / 4)}
    primary = results["delta"]["primary"]
    output["achieved_halfwidth"] = None if primary["ci"] is None else (primary["ci"][1] - primary["ci"][0]) / 2
    output["precision_target_met"] = output["achieved_halfwidth"] is not None and output["achieved_halfwidth"] <= research["analysis"]["target_halfwidth"]
    target = folder / f"{stage}-analysis"
    snapshot(target, research, execution)
    target.mkdir(exist_ok=True)
    write_artifacts(target, rows, results, research)
    if stage == "pilot":
        assessment = pilot_assessment(rows, results["delta"], research)
        output["pilot"] = assessment
        output["pilot"]["units"] = meta["target"]["units"]
        output["pilot"]["next_rung"] = next((x for x in research["sampling"]["pilot_ladder"] if x > meta["target"]["units"]), None)
        write_json(target / "analysis.json", output)
        if freeze_main:
            plan = plan_main(documents, results["delta"], assessment, require_benchmark(research, execution, prepared), calibration, prepared, research, execution)
            destination = folder / "main-plan.json"
            if (folder / "main.sqlite").exists():
                raise ValueError("Main outcomes already exist; prospective allocation cannot be changed")
            if destination.exists() and read_json(destination) != plan:
                raise ValueError("An existing main freeze differs; choose a new study directory before changing it")
            write_json(destination, plan)
            output["main_plan"] = {"units": plan["units"], "windows": plan["windows"], "remaining_work": plan["remaining_work"]}
    write_json(target / "analysis.json", output)
    return output


def main():
    p = parser(__doc__)
    p.add_argument("stage", choices=["pilot", "main"])
    p.add_argument("--freeze-main", action="store_true", help="Choose and prospectively save precision/budget-qualified main")
    args = p.parse_args()
    research, execution = settings(args)
    if args.freeze_main and args.stage != "pilot":
        p.error("--freeze-main is only available after a pilot")
    result = analyze(research, execution, args.stage, args.freeze_main)
    print({key: result.get(key) for key in ("stage", "pilot", "achieved_halfwidth", "precision_target_met", "main_plan")})


if __name__ == "__main__":
    main()
