"""Synthetic checks for DEC-011; never reads data or loads a model.

Requires NumPy. This is a method-validation harness, not the experiment pipeline.
"""

import argparse
import json
import math
import platform
from pathlib import Path

import numpy as np


def estimate(sums, sizes):
    """Document-weighted mean and finite-G cluster sandwich variance."""
    sums = np.asarray(sums, dtype=np.float64)
    sizes = np.asarray(sizes, dtype=np.float64)
    if len(sums) < 2 or np.any(sizes <= 0):
        raise ValueError("Need at least two nonempty clusters")
    mean = sums.sum() / sizes.sum()
    residual = sums - sizes * mean
    variance = len(sums) / (len(sums) - 1) * (residual @ residual) / sizes.sum() ** 2
    return mean, variance


def resample(sums, sizes, draws, rng):
    """Duplicate clusters are separate draw slots, not squared multiplicities."""
    count = len(sums)
    weights = rng.multinomial(count, np.full(count, 1 / count), size=draws)
    total = weights @ sizes
    mean = (weights @ sums) / total
    residual_sq = (
        weights @ (sums * sums)
        - 2 * mean * (weights @ (sums * sizes))
        + mean * mean * (weights @ (sizes * sizes))
    )
    # This algebra can produce tiny negative roundoff, not negative variance.
    if np.any(residual_sq < -1e-8):
        raise ArithmeticError("Negative bootstrap residual sum")
    variance = count / (count - 1) * np.maximum(residual_sq, 0) / total**2
    return mean, variance


def interval(sci, sums, sizes, draws, rng):
    sci_mean, sci_var = estimate(sci, np.ones(len(sci)))
    code_mean, code_var = estimate(sums, sizes)
    observed = 0.5 * (sci_mean + code_mean)
    se = math.sqrt(0.25 * (sci_var + code_var))
    sci_b, sci_vb = resample(sci, np.ones(len(sci)), draws, rng)
    code_b, code_vb = resample(sums, sizes, draws, rng)
    se_b = np.sqrt(0.25 * (sci_vb + code_vb))
    if se <= 0 or np.any(se_b <= 0) or not np.isfinite(se_b).all():
        raise ArithmeticError("Degenerate studentization; do not discard draws")
    t = (0.5 * (sci_b + code_b) - observed) / se_b
    low_t, high_t = np.quantile(t, [0.025, 0.975], method="linear")
    return float(observed - high_t * se), float(observed - low_t * se)


def checks():
    y = np.array([-2.0, 1.0, 3.0, 7.0])
    mean, var = estimate(y, np.ones(4))
    assert np.isclose(mean, y.mean())
    assert np.isclose(var, y.var(ddof=1) / 4)
    # Unequal groups: document-weighted mean is 2.5, equal-group mean is 5.
    assert estimate(np.array([10.0, 0.0]), np.array([1.0, 3.0]))[0] == 2.5
    # Explicit draw slots preserve multiplicity and variance independently.
    sums, sizes = np.array([2.0, 12.0, -1.0]), np.array([1.0, 3.0, 1.0])
    slots = np.array([1, 1, 2])
    explicit = estimate(sums[slots], sizes[slots])
    multiplicity = np.array([0.0, 2.0, 1.0])
    theta = (multiplicity @ sums) / (multiplicity @ sizes)
    variance = 1.5 * (multiplicity @ ((sums - sizes * theta) ** 2)) / (multiplicity @ sizes) ** 2
    assert np.allclose(explicit, [theta, variance])
    sci = np.linspace(-2, 3, 30)
    sizes = np.tile(np.array([1.0, 3.0, 1.0]), 10)
    sums = sizes * np.linspace(-1, 4, 30)
    ci = interval(sci, sums, sizes, 1999, np.random.default_rng(11))
    shifted = interval(sci + 0.7, sums + 0.7 * sizes, sizes, 1999, np.random.default_rng(11))
    assert np.allclose(np.array(ci) + 0.7, shifted)
    scaled = interval(sci * 3, sums * 3, sizes, 1999, np.random.default_rng(11))
    assert np.allclose(np.array(ci) * 3, scaled)
    try:
        interval(np.zeros(4), np.zeros(4), np.ones(4), 99, np.random.default_rng(1))
    except ArithmeticError:
        pass
    else:
        raise AssertionError("All-zero data must not produce a studentized CI")
    print("PASS: singleton SE, document weighting, duplicate draw slots, shift/scale invariance, degenerate SE", flush=True)


def sample(rng, scenario, units):
    sizes = rng.integers(1, 11, size=units).astype(float)
    if scenario == "normal":
        sci, group = rng.normal(size=(2, units))
    elif scenario == "skew":
        sigma = 0.7
        center = math.exp(sigma**2 / 2)
        scale = math.sqrt((math.exp(sigma**2) - 1) * math.exp(sigma**2))
        sci, group = (rng.lognormal(0, sigma, size=(2, units)) - center) / scale
    elif scenario == "informative_sizes":
        sci, group = rng.normal(size=(2, units))
        # Uniform integer sizes: E[n^2]/E[n] = 7; E[n*(n-7)] = 0.
        group = group + 0.3 * (sizes - 7)
    elif scenario == "heavy_t3":
        sci, group = rng.standard_t(3, size=(2, units)) / math.sqrt(3)
    else:
        raise ValueError(scenario)
    # Within-group ICC = 0.75 in the normal case; grouped sums have truth zero.
    sums = sizes * math.sqrt(0.75) * group + np.sqrt(0.25 * sizes) * rng.normal(size=units)
    return sci, sums, sizes


def wilson(successes, count):
    p = successes / count
    z = 1.959963984540054
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [center - radius, center + radius]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--replicates", type=int, default=1999)
    parser.add_argument("--units", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3007011)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.trials < 1 or args.replicates < 99 or args.units < 2:
        parser.error("Invalid simulation size")
    checks()
    rows = []
    for index, scenario in enumerate(["normal", "skew", "informative_sizes", "heavy_t3"]):
        rng = np.random.default_rng(args.seed + index)
        covered = invalid = 0
        widths = []
        for trial in range(args.trials):
            sci, sums, sizes = sample(rng, scenario, args.units)
            try:
                low, high = interval(sci, sums, sizes, args.replicates, rng)
            except ArithmeticError:
                invalid += 1
            else:
                covered += low <= 0 <= high
                widths.append(high - low)
            if (trial + 1) % 250 == 0:
                print(f"{scenario}: {trial + 1}/{args.trials}", flush=True)
        row = {
            "scenario": scenario, "coverage": covered / args.trials,
            "wilson_95": wilson(covered, args.trials), "invalid_trials": invalid,
            "mean_width": float(np.mean(widths)) if widths else None,
            "screen_pass": invalid == 0 and 0.93 <= covered / args.trials <= 0.98,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {
        "kind": "synthetic_method_screen_not_pretrained_or_pilot_evidence",
        "python": platform.python_version(), "numpy": np.__version__,
        "trials": args.trials, "replicates": args.replicates, "units_per_domain": args.units,
        "seed": args.seed, "rows": rows,
    }
    if args.output:
        with args.output.open("x") as output:
            json.dump(report, output, indent=2)
            output.write("\n")
    print("Complete. Pilot-matched design coverage remains a separate pre-main gate.", flush=True)


if __name__ == "__main__":
    main()
