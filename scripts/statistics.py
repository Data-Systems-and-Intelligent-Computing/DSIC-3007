"""Document-weighted estimates and whole-group paired bootstrap-t inference."""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np

from scripts.common import seed_for


def pair(active, scores, tie_order):
    rank = {expert: index for index, expert in enumerate(tie_order)}
    ordered = sorted(active, key=lambda e: (float(scores[e]), rank[e]))
    return ordered[-1], ordered[0]


def event_metrics(event, calibration, research):
    active = event["active"]
    if len(active) != research["model"]["active_experts"] or len(event["effects"]) != len(active) or len(set(active)) != len(active):
        raise ValueError("All eight distinct expert effects are required")
    effects = dict(zip(active, event["effects"], strict=True))
    rng = np.random.default_rng(seed_for(research["seed"], event["id"], "ties"))
    tie_order = rng.permutation(active).tolist()
    scores = {"M1": calibration["M1"], "M2": calibration["domains"][event["domain"]]}
    coefficients = dict(zip(active, event["coefficients"], strict=True))
    scores["M4"] = [coefficients.get(e, 0.0) for e in range(research["model"]["experts"])]
    for domain, table in calibration["domains"].items():
        scores["ranking_" + domain] = table
    result = {}
    for name, values in scores.items():
        high, low = pair(active, values, tie_order)
        result[name] = effects[high] - effects[low]
    rng = np.random.default_rng(seed_for(research["seed"], event["id"], "M0"))
    draws = [effects[h] - effects[l] for h, l in (rng.choice(active, 2, replace=False) for _ in range(research["analysis"]["m0_draws"]))]
    result["M0"] = float(np.mean(draws))
    result["M0_mc_variance"] = float(np.var(draws, ddof=1) / len(draws))
    result["delta"] = result["M2"] - result["M1"]
    return result


def document_means(events, calibration, research):
    buckets = defaultdict(list)
    identity = {}
    for event in events:
        key = event["domain"], event["document"]
        if key in identity and identity[key] != event["group"]:
            raise ValueError("Document appears in inconsistent source groups")
        identity[key] = event["group"]
        buckets[key].append(event_metrics(event, calibration, research))
    output = []
    for (domain, document), values in buckets.items():
        row = {"domain": domain, "document": document, "group": identity[domain, document], "positions": len(values)}
        row.update({name: float(np.mean([v[name] for v in values])) for name in values[0] if name != "M0_mc_variance"})
        row["M0_mc_variance"] = sum(v["M0_mc_variance"] for v in values) / len(values) ** 2
        output.append(row)
    return output


def grouped(rows, metric="delta"):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(float(row[metric]))
    if not groups:
        raise ValueError("No source groups")
    values = [groups[key] for key in sorted(groups)]
    return np.asarray([sum(v) for v in values]), np.asarray([len(v) for v in values], dtype=float)


def estimate(sums, sizes, multiplicity=None):
    counts = np.ones(len(sums)) if multiplicity is None else np.asarray(multiplicity)
    groups = int(counts.sum())
    total = float(counts @ sizes)
    mean = float(counts @ sums / total)
    variance = math.nan if groups < 2 else float(groups / (groups - 1) * (counts @ ((sums - sizes * mean) ** 2)) / total ** 2)
    return mean, variance


def bootstrap(rows, research, draws=None, metric="delta"):
    domains = research["outcome_domains"]
    arrays = {domain: grouped([r for r in rows if r["domain"] == domain], metric) for domain in domains}
    observed = {domain: estimate(*arrays[domain]) for domain in domains}
    n = draws or research["analysis"]["bootstrap_draws"]
    rng = np.random.default_rng(seed_for(research["seed"], metric, "bootstrap", n))
    t_values = {domain: [] for domain in domains}
    t_values["primary"] = []
    primary = sum(x[0] for x in observed.values()) / 2
    primary_var = sum(x[1] for x in observed.values()) / 4
    for _ in range(n):
        samples = {}
        for domain in domains:
            sums, sizes = arrays[domain]
            multiplicity = rng.multinomial(len(sums), np.full(len(sums), 1 / len(sums)))
            mean, variance = estimate(sums, sizes, multiplicity)
            samples[domain] = mean, variance
            if variance > 0 and np.isfinite(variance):
                t_values[domain].append((mean - observed[domain][0]) / math.sqrt(variance))
        mean = sum(x[0] for x in samples.values()) / 2
        variance = sum(x[1] for x in samples.values()) / 4
        if variance > 0 and np.isfinite(variance):
            t_values["primary"].append((mean - primary) / math.sqrt(variance))

    def interval(mean, variance, values):
        valid = len(values)
        result = {"estimate": mean, "variance": variance if np.isfinite(variance) else None,
                  "draws": n, "valid_draws": valid, "degenerate_draws": n - valid,
                  "ci": None, "p_two_sided": None, "critical": None}
        if valid < research["analysis"]["valid_fraction"] * n or not variance > 0:
            return result
        alpha = 1 - research["analysis"]["confidence"]
        lower, upper = np.quantile(values, [alpha / 2, 1 - alpha / 2], method="linear")
        se = math.sqrt(variance)
        result["ci"] = [float(mean - upper * se), float(mean - lower * se)]
        result["critical"] = float(max(abs(lower), abs(upper)))
        observed_t = mean / se
        lower_tail = (sum(t <= observed_t for t in values) + 1) / (valid + 1)
        upper_tail = (sum(t >= observed_t for t in values) + 1) / (valid + 1)
        result["p_two_sided"] = min(1.0, 2 * min(lower_tail, upper_tail))
        return result

    output = {domain: interval(*observed[domain], t_values[domain]) for domain in domains}
    output["primary"] = interval(primary, primary_var, t_values["primary"])
    for domain in domains:
        output[domain].update(documents=int(arrays[domain][1].sum()), groups=len(arrays[domain][0]))
    ps = {d: output[d]["p_two_sided"] for d in domains}
    output["holm"] = holm(ps) if all(v is not None for v in ps.values()) else None
    return output


def holm(values):
    ordered = sorted(values, key=values.get)
    result, previous = {}, 0.0
    for index, key in enumerate(ordered):
        previous = max(previous, min(1.0, (len(ordered) - index) * values[key]))
        result[key] = previous
    return result


def calibration_tables(observations, documents, research, budget):
    tables = {}
    for domain in research["domains"]:
        members = sorted((r for r in documents if r["domain"] == domain and r["stage"] == "calibration"), key=lambda r: r["stage_order"])[:budget]
        if len(members) != budget:
            raise ValueError(f"Only {len(members)} calibration documents in {domain}; need {budget}")
        tables[domain] = []
        for seed in research["calibration"]["seeds"]:
            means = [observations[(row["id"], seed)] for row in members]
            table = np.mean(means, axis=0)
            if not np.isclose(table.sum(), research["model"]["active_experts"], atol=1e-12):
                raise ValueError("Utilization must sum to eight")
            tables[domain].append(table)
    return tables


def stability(previous, candidate, research):
    cfg = research["calibration"]
    k = research["model"]["active_experts"]

    def extremes(values, reverse=False):
        # Calibration stability ties resolve deterministically by expert identity.
        return set(sorted(range(len(values)), key=lambda e: ((-1 if reverse else 1) * values[e], e))[:k])

    def compare(left, right):
        jaccards = []
        for reverse in (False, True):
            a, b = extremes(left, reverse), extremes(right, reverse)
            jaccards.append(len(a & b) / len(a | b))
        drift = float(np.abs(left - right).sum())
        return {"l1": drift, "top_bottom_jaccard": jaccards,
                "pass": drift <= cfg["l1_max"] and min(jaccards) >= cfg["jaccard_min"]}

    families = {}
    count = len(research["calibration"]["seeds"])
    for a in range(count):
        for b in range(a + 1, count):
            families[f"seed_{a}_{b}"] = {d: compare(candidate[d][a], candidate[d][b]) for d in research["domains"]}
    families["consecutive_budget"] = {d: compare(np.mean(previous[d], axis=0), np.mean(candidate[d], axis=0)) for d in research["domains"]}
    # Every family must pass >=90% of its domain/layer comparisons.
    passed = all(sum(v["pass"] for v in comparisons.values()) / len(comparisons) >= cfg["pass_fraction"] for comparisons in families.values())
    return {"pass": passed, "families": families}
