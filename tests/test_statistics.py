import copy

import numpy as np
import pytest

from scripts.common import read_json, seed_for
from scripts.statistics import bootstrap, calibration_tables, document_means, estimate, event_metrics, holm, stability


@pytest.fixture
def research():
    return copy.deepcopy(read_json("config/research.json"))


def test_document_weighted_cluster_estimator_and_duplicate_draws():
    # Repositories A=[1,3], B=[9]. Equal-repository weighting would incorrectly give 5.5.
    mean, variance = estimate(np.array([4., 9.]), np.array([2., 1.]))
    assert mean == pytest.approx(13 / 3)
    assert variance == pytest.approx(2 * ((4 - 2 * mean) ** 2 + (9 - mean) ** 2) / 9)
    # Draw A twice and B once: duplicates are distinct resampled clusters.
    mean, variance = estimate(np.array([4., 9.]), np.array([2., 1.]), np.array([2, 1]))
    assert mean == pytest.approx(17 / 5)
    brute_groups = [[1., 3.], [1., 3.], [9.]]
    expected = 3 / 2 * sum((sum(group) - len(group) * mean) ** 2 for group in brute_groups) / 25
    assert variance == pytest.approx(expected)


def test_ties_and_identical_pairs_are_retained(research):
    event = {"id": "e", "domain": "code", "document": "d", "group": "g", "active": list(range(8)),
             "coefficients": [.1] * 8, "effects": list(range(8))}
    calibration = {"M1": [1.] * 64, "domains": {d: [1.] * 64 for d in research["domains"]}}
    metrics = event_metrics(event, calibration, research)
    assert metrics["delta"] == 0
    assert abs(metrics["M1"]) > 0  # shared tie order still selects distinct experts
    assert metrics == event_metrics(event, calibration, research)
    assert len(document_means([event], calibration, research)) == 1


def test_complete_effect_census_required(research):
    event = {"id": "e", "domain": "code", "active": list(range(8)), "effects": [1.] * 7}
    with pytest.raises(ValueError, match="eight"):
        event_metrics(event, {}, research)


def test_bootstrap_primary_equal_domains_and_unequal_code_groups(research):
    rows = []
    for domain in research["outcome_domains"]:
        for group in range(30):
            for file in range(1 + (group % 3 if domain == "code" else 0)):
                rows.append({"domain": domain, "group": str(group), "delta": group / 50 + file / 100})
    result = bootstrap(rows, research, draws=300)
    s = np.mean([r["delta"] for r in rows if r["domain"] == "scientific"])
    c = np.mean([r["delta"] for r in rows if r["domain"] == "code"])
    assert result["primary"]["estimate"] == pytest.approx((s + c) / 2)
    assert result["primary"]["variance"] == pytest.approx((result["scientific"]["variance"] + result["code"]["variance"]) / 4)
    assert result["code"]["groups"] == 30
    assert result["code"]["documents"] == 60
    assert result["primary"]["valid_draws"] == 300
    assert result["primary"]["ci"][0] < result["primary"]["estimate"] < result["primary"]["ci"][1]


def test_degenerate_draws_do_not_create_equivalence_claim(research):
    rows = [{"domain": d, "group": str(g), "delta": 0.} for d in research["outcome_domains"] for g in range(12)]
    result = bootstrap(rows, research, draws=100)
    assert result["primary"]["valid_draws"] == 0
    assert result["primary"]["ci"] is None
    assert result["primary"]["p_two_sided"] is None


def test_holm_monotonicity():
    assert holm({"scientific": .02, "code": .03}) == {"scientific": .04, "code": .04}


def test_calibration_equal_document_seed_domain_weighting(research):
    docs, observations = [], {}
    for domain in research["domains"]:
        for document in range(2):
            identity = domain + str(document)
            docs.append({"id": identity, "domain": domain, "stage": "calibration", "stage_order": document})
            for seed in research["calibration"]["seeds"]:
                observations[identity, seed] = np.array([1.] * 8 + [0.] * 56) if document == 0 else np.array([0.] * 8 + [1.] * 8 + [0.] * 48)
    tables = calibration_tables(observations, docs, research, 2)
    assert tables["scientific"][0].sum() == 8
    np.testing.assert_array_equal(tables["code"][1][:16], [.5] * 16)


def test_stability_drift_is_not_divided_by_eight(research):
    base = np.linspace(.01, .24, 64)
    previous = {d: [base.copy()] * 3 for d in research["domains"]}
    candidate = {d: [base.copy()] * 3 for d in research["domains"]}
    assert stability(previous, candidate, research)["pass"]
    candidate["code"] = [base + .002] * 3
    result = stability(previous, candidate, research)
    assert not result["pass"]
    assert result["families"]["consecutive_budget"]["code"]["l1"] == pytest.approx(.128)


def test_studentized_interval_against_explicit_resampled_group_copies(research):
    groups = {d: [[(g - 4) / 20 + f / 100 for f in range(1 + (g % 3 if d == "code" else 0))]
                  for g in range(10)] for d in research["outcome_domains"]}
    rows = [{"domain": domain, "group": str(g), "delta": value}
            for domain, values in groups.items() for g, group in enumerate(values) for value in group]

    def explicit(values):
        data = [value for group in values for value in group]
        mean = sum(data) / len(data)
        variance = len(values) / (len(values) - 1) * sum(sum(value - mean for value in group) ** 2 for group in values) / len(data) ** 2
        return mean, variance

    observed = [explicit(groups[d]) for d in research["outcome_domains"]]
    point = sum(x[0] for x in observed) / 2
    variance = sum(x[1] for x in observed) / 4
    draws = 200
    rng = np.random.default_rng(seed_for(research["seed"], "delta", "bootstrap", draws))
    studentized = []
    for _ in range(draws):
        samples = []
        for domain in research["outcome_domains"]:
            counts = rng.multinomial(10, np.ones(10) / 10)
            copied_groups = [group.copy() for count, group in zip(counts, groups[domain], strict=True) for _ in range(count)]
            samples.append(explicit(copied_groups))
        sample_point = sum(x[0] for x in samples) / 2
        sample_variance = sum(x[1] for x in samples) / 4
        if sample_variance > 0:
            studentized.append((sample_point - point) / np.sqrt(sample_variance))
    lower, upper = np.quantile(studentized, [.025, .975], method="linear")
    expected = [point - upper * np.sqrt(variance), point - lower * np.sqrt(variance)]
    actual = bootstrap(rows, research, draws=draws)["primary"]
    np.testing.assert_allclose(actual["ci"], expected, atol=1e-14, rtol=0)
    assert actual["valid_draws"] == len(studentized)
