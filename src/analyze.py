"""Statistical analysis utilities for causal discrimination."""

import numpy as np


def paired_difference(cd_domain, cd_global):
    a = np.asarray(cd_domain, dtype=float)
    b = np.asarray(cd_global, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Paired arrays must have identical shape")
    return a - b


def bootstrap_mean_ci(values, n_boot=5000, seed=42, ci=0.95):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = []
    n = len(values)
    for _ in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        means.append(float(np.mean(sample)))
    alpha = (1 - ci) / 2
    return (
        float(np.mean(values)),
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1 - alpha)),
    )
