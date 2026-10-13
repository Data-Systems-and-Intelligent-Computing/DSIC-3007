"""Frozen routing tables and event-level metric selections.

Utilization is active-set membership frequency. Routing coefficients are kept
for the M4 control, but never enter M1 or M2.
"""

from collections import defaultdict
import numpy as np
from .artifacts import priority


def _ids(active, n_experts=64):
    values = [int(e) for e in active]
    if len(values) != 8 or len(set(values)) != 8 or any(e < 0 or e >= n_experts for e in values):
        raise ValueError("routing must contain eight distinct expert IDs")
    return values


def global_utilization(active_expert_events, num_experts: int = 64):
    """Membership frequencies; each event contributes eight units."""
    counts = np.zeros(num_experts, dtype=np.float64)
    total = 0
    for active in active_expert_events:
        counts[_ids(active, num_experts)] += 1
        total += 1
    return counts / total if total else counts


def document_weighted_utilization(events, num_experts: int = 64):
    """Average per-document membership frequencies with equal document weight."""
    by_document = defaultdict(list)
    for row in events:
        by_document[row["document_id"]].append(row["routing"]["expert_ids"])
    if not by_document:
        return np.zeros(num_experts, dtype=np.float64)
    tables = [global_utilization(active_rows, num_experts) for active_rows in by_document.values()]
    return np.mean(tables, axis=0)


def domain_utilization(active_expert_events, num_experts: int = 64):
    """Compatibility helper; caller supplies one domain's event stream."""
    return global_utilization(active_expert_events, num_experts)


def domain_enrichment(u_domain, u_not_domain, eps: float = 1e-8):
    return np.log((np.asarray(u_domain) + eps) / (np.asarray(u_not_domain) + eps))


def per_token_routing_weight(router_weights):
    return np.asarray(router_weights)


def decode_window_weights(window_routing):
    """Decode the explicitly recorded little-endian float32 coefficient matrix."""
    shape = window_routing["shape"]
    if window_routing.get("coefficient_dtype") != "float32":
        raise ValueError("window routing coefficients must be recorded as float32")
    if len(shape) != 2 or shape[1] != 8:
        raise ValueError("window routing weights must have shape [tokens, 8]")
    raw = bytes.fromhex(window_routing["weights_fp32_le_hex"])
    if len(raw) != shape[0] * shape[1] * 4:
        raise ValueError("packed window routing weight length disagrees with shape")
    values = np.frombuffer(raw, dtype="<f4").astype(np.float64).reshape(shape)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("invalid window routing coefficients")
    return values


def calibration_document_tables(rows, num_experts: int = 64):
    """Return seed/domain/layer utilization tables, equal-weighting documents.

    Expected rows are calibration_window events. Every token in each sampled
    calibration window contributes membership, while one window per document
    is first reduced to a document table before documents are averaged.
    """
    documents = defaultdict(list)
    for row in rows:
        wr = row["window_routing"]
        ids_by_position = wr["expert_ids"]
        if len(ids_by_position) != wr["shape"][0]:
            raise ValueError("window expert IDs disagree with declared shape")
        # Verify the coefficients are intact even though utilization ignores them.
        decode_window_weights(wr)
        flat = []
        for active in ids_by_position:
            flat.extend(_ids(active, num_experts))
        key = (row["sampling"]["seed"], row["domain"], row["layer"], row["document_id"])
        documents[key].append(global_utilization(np.asarray(flat).reshape(-1, 8), num_experts))
    by_table = defaultdict(list)
    for (seed, domain, layer, document_id), tables in documents.items():
        if len(tables) != 1:
            raise ValueError(f"calibration requires one sampled window per document/seed: {document_id}")
        by_table[seed, domain, layer].append(tables[0])
    return {key: np.mean(values, axis=0) for key, values in by_table.items()}


def average_seed_tables(seed_tables):
    """Average seed-specific domain tables after each seed's document mean."""
    grouped = defaultdict(list)
    for (seed, domain, layer), values in seed_tables.items():
        grouped[domain, layer].append(np.asarray(values, dtype=np.float64))
    return {key: np.mean(values, axis=0) for key, values in grouped.items()}


def m1_tables(domain_tables, domains=("c4", "scientific", "code")):
    """Equal-domain M1, keeping each domain's table equally weighted."""
    output = {}
    layers = sorted({layer for domain, layer in domain_tables})
    for layer in layers:
        missing = [domain for domain in domains if (domain, layer) not in domain_tables]
        if missing:
            raise ValueError(f"M1 missing calibration domains at layer {layer}: {missing}")
        output[layer] = np.mean([domain_tables[d, layer] for d in domains], axis=0)
    return output


def stable_tie_order(seed, purpose, identity, experts=range(64)):
    """One deterministic tie order shared by all metrics for the same event."""
    def key(expert):
        return priority(int(seed), purpose, [identity, int(expert)]), int(expert)
    return sorted(experts, key=key)


def _select_extremes(scores, active, tie_order):
    rank = {expert: index for index, expert in enumerate(tie_order)}
    ordered = sorted(active, key=lambda expert: (-float(scores[expert]), rank[expert]))
    return ordered[0], ordered[-1]


def deterministic_m0_pairs(seed, event_identity, active, draws=64):
    """Uniform seeded ordered pairs, sampled with replacement across draws."""
    active = _ids(active)
    seed_hash = priority(int(seed), "M0-pair-draws", event_identity)
    rng = np.random.Generator(np.random.PCG64(int(seed_hash[:32], 16)))
    output = []
    for _ in range(draws):
        a = int(rng.integers(len(active)))
        b = int(rng.integers(len(active) - 1))
        if b >= a:
            b += 1
        output.append((active[a], active[b]))
    return output


def select_event_metrics(row, m1, m2, seed=3007, m0_draws=64):
    """Select high/low pairs for M1, M2, M4 and retain all M0 draws."""
    active = _ids(row["routing"]["expert_ids"])
    layer = int(row["layer"])
    if layer not in m1 or row["domain"] not in m2 or layer not in m2[row["domain"]]:
        raise ValueError(f"frozen routing tables missing layer {layer}")
    identity = row.get("event_id", row.get("input_id"))
    tie = stable_tie_order(seed, "metric-selection", identity)
    m1_pair = _select_extremes(m1[layer], active, tie)
    m2_pair = _select_extremes(m2[row["domain"]][layer], active, tie)
    weights = row["routing"]["weights"]
    if len(weights) != 8:
        raise ValueError("M4 needs eight native routing weights")
    native = np.zeros(64, dtype=np.float64)
    for expert, weight in zip(active, weights):
        native[expert] = float(weight)
    m4_pair = _select_extremes(native, active, tie)
    return {"M1": list(m1_pair), "M2": list(m2_pair), "M4": list(m4_pair),
            "M0": [list(pair) for pair in deterministic_m0_pairs(seed, identity, active, m0_draws)],
            "tie_order": tie}


def causal_discrimination(pair, delta_by_expert):
    high, low = pair
    return float(delta_by_expert[high] - delta_by_expert[low])
