import json
import struct

import numpy as np
import pytest

from src.analyze import _bootstrap_domain, _group_variance, analyze_bundle, analyze_events
from src.artifacts import canonical, digest, seal, write_rows
from src.routing_stats import (calibration_document_tables, select_event_metrics)


def _effect_event(event_id="evt", domain="scientific", group="paper-a", doc="doc-a"):
    experts = list(range(8))
    weights = [1 / 8] * 8
    baseline = 2.0
    return {
        "kind": "effects", "event_id": event_id, "input_id": "input-a",
        "dataset": "dataset", "domain": domain, "source_group": group,
        "document_id": doc, "predictor_position": 4100, "context_length": 4096,
        "layer": 15, "sampling": {"seed": 3007, "group_rank": 0},
        "routing": {"expert_ids": experts, "weights": weights,
                    "weights_fp32_le_hex": struct.pack("<8f", *weights).hex()},
        "baseline_nll": baseline,
        "effects": [{"expert_id": e, "ablated_nll": baseline + e / 10,
                     "delta_nll": (baseline + e / 10) - baseline} for e in experts],
    }


def test_replay_keeps_tied_metric_selections_and_all_eight_effects():
    event = _effect_event()
    tied = {15: np.full(64, 0.125)}
    metrics = select_event_metrics(event, tied, {"scientific": tied}, seed=3007)
    assert metrics["M1"] == metrics["M2"]
    assert metrics["M1"][0] != metrics["M1"][1]
    replay, documents, groups = analyze_events([event], tied, {"scientific": tied}, m0_draws=64)
    assert len(replay[0]["delta_nll_by_expert"]) == 8
    assert replay[0]["delta_m2_minus_m1"] == 0.0
    assert len(replay[0]["m0_pairs"]) == len(replay[0]["m0_randomization_draws"]) == 64
    assert all(a != b for a, b in replay[0]["m0_pairs"])
    assert replay[0]["m0_randomization_variance"] >= 0
    assert documents[0]["mean_cd_m1"] == documents[0]["mean_cd_m2"]
    assert groups[0]["n_documents"] == 1


def test_document_weighted_cluster_variance_uses_unequal_group_sizes():
    # Group a has two documents and group b has one. The point estimate is 2;
    # cluster residual totals are +2 and -2, with G=2 and N=3.
    rows = [
        {"source_group": "a", "delta": 3.0},
        {"source_group": "a", "delta": 3.0},
        {"source_group": "b", "delta": 0.0},
    ]
    mean, variance, groups, documents = _group_variance(rows)
    assert mean == 2.0
    assert variance == pytest.approx((2 / 1) * (2**2 + (-2)**2) / 3**2)
    assert (groups, documents) == (2, 3)


def test_group_bootstrap_is_reproducible_and_preserves_group_draw_multiplicity():
    rows = [{"source_group": f"g{i}", "delta": value}
            for i, value in enumerate((0.0, 1.0, 3.0, 6.0))]
    first = _bootstrap_domain(rows, 300, seed=19, confidence=0.95)
    second = _bootstrap_domain(rows, 300, seed=19, confidence=0.95)
    assert first["mean"] == pytest.approx(2.5)
    assert first["groups"] == 4
    assert first["valid_replicates"] >= 285
    assert first["degenerate_replicates"] == 300 - first["valid_replicates"]
    assert first["ci"] == second["ci"]


def test_calibration_membership_uses_every_input_position_and_equal_document_weight():
    rows = []
    for doc, assignments in [("short", [list(range(8))]),
                             ("long", [list(range(8, 16))] * 3)]:
        rows.append({"sampling": {"seed": 3008}, "domain": "code", "layer": 15,
                     "document_id": doc,
                     "window_routing": {
                         "expert_ids": assignments,
                         "weights_fp32_le_hex": struct.pack("<" + "f" * (8 * len(assignments)), *([0.125] * 8 * len(assignments))).hex(),
                         "coefficient_dtype": "float32", "shape": [len(assignments), 8],
                     }})
    tables = calibration_document_tables(rows)
    table = tables[3008, "code", 15]
    assert table[:8] == pytest.approx(np.full(8, 0.5))
    assert table[8:16] == pytest.approx(np.full(8, 0.5))
    assert table.sum() == pytest.approx(8.0)


def test_bundle_with_missing_planned_event_is_rejected(tmp_path):
    plan = tmp_path / "plan"
    plan.mkdir()
    input_id = "input-a"
    write_rows(plan / "inputs.jsonl", [{"input_id": input_id, "layers": [15]}])
    (plan / "research.json").write_bytes(canonical({
        "sampling": {"master_seed": 3007, "primary_layer": 15},
        "controls": {"m0_draws_per_event": 64},
        "analysis": {"confidence": 0.95, "initial_replicates": 10000,
                     "minimum_valid_replicate_fraction": 0.95},
    }) + b"\n")
    seal(plan, "experiment-plan", {"stage": "main"})

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "plan").mkdir()
    for path in plan.iterdir():
        (bundle / "plan" / path.name).write_bytes(path.read_bytes())
    write_rows(bundle / "events.jsonl", [])
    seal(bundle, "measurements", {"stage": "main", "expected_event_ids": [digest([input_id, 15])]})

    with pytest.raises(ValueError, match="event coverage differs"):
        analyze_bundle(bundle)


def test_route_only_probe_directs_analysis_to_calibration_freeze(tmp_path):
    bundle = tmp_path / "probe"
    bundle.mkdir()
    write_rows(bundle / "events.jsonl", [])
    seal(bundle, "measurements", {"stage": "probe"})

    with pytest.raises(ValueError, match="route-only probe.*calibration_freeze"):
        analyze_bundle(bundle)
