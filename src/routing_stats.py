"""Observational routing metrics M1-M4 calculated on the calibration split only."""

import numpy as np


def global_utilization(active_expert_events, num_experts: int):
    counts = np.zeros(num_experts, dtype=np.float64)
    total = 0
    for active in active_expert_events:
        for e in active:
            counts[e] += 1
        total += 1
    if total == 0:
        return counts
    return counts / total


def domain_utilization(active_expert_events, num_experts: int):
    # Same estimator as utilization, but events must come from one source-defined domain.
    return global_utilization(active_expert_events, num_experts)


def domain_enrichment(u_domain, u_not_domain, eps: float = 1e-8):
    return np.log((np.asarray(u_domain) + eps) / (np.asarray(u_not_domain) + eps))


def per_token_routing_weight(router_weights):
    return np.asarray(router_weights)
