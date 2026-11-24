import numpy as np
import pandas as pd
import pytest

from mediguard.audit.prioritization import (build_audit_curves, priority_tier, random_baseline, ranking_order,
                                            risk_level, score_providers, summarize_at_capacity)


@pytest.fixture()
def cohort():
    df = pd.DataFrame({
        "Provider": ["A", "B", "C", "D", "E", "F"],
        "total_reimbursed": [1000.0, 50000.0, 20000.0, 20000.0, 5000.0, 100.0],
    })
    prob = np.array([0.9, 0.1, 0.6, 0.6, 0.95, 0.99])
    scored = score_providers(df, prob)
    scored["label"] = [1, 0, 1, 0, 1, 1]
    return scored


def test_exposure_and_ranking(cohort):
    assert cohort["estimated_exposure"].tolist() == pytest.approx([900, 5000, 12000, 12000, 4750, 99])
    # ties (C and D) broken by provider ID
    order = cohort["Provider"].to_numpy()[ranking_order(cohort, "estimated_exposure")].tolist()
    assert order == ["C", "D", "B", "E", "A", "F"]
    assert cohort.set_index("Provider")["rank_exposure"].to_dict() == {"C": 1, "D": 2, "B": 3, "E": 4, "A": 5, "F": 6}
    assert cohort.set_index("Provider").loc["F", "rank_probability"] == 1
    assert cohort.set_index("Provider").loc["B", "rank_reimbursement"] == 1


def test_top_k_financials_and_denominator(cohort):
    curves = build_audit_curves(cohort, "label", n_sims=50, seed=0)
    total_fraud = 1000 + 20000 + 5000 + 100
    assert curves["total_known_fraud_reimbursement"] == total_fraud
    rows = {r["strategy"]: r for r in summarize_at_capacity(curves, 2)}
    exp = rows["exposure"]  # C (fraud, 20000) + D (not fraud)
    assert exp["fraud_labelled_identified"] == 1 and exp["precision_at_k"] == 0.5
    assert exp["known_fraud_reimbursement_captured"] == 20000
    assert exp["capture_rate"] == pytest.approx(20000 / total_fraud)
    assert exp["reimbursement_reviewed"] == 40000 and exp["estimated_exposure_selected"] == pytest.approx(24000)
    reimb = rows["reimbursement"]  # B (not fraud) + C (fraud): tie C/D broken by ID
    assert reimb["known_fraud_reimbursement_captured"] == 20000 and reimb["overlap_with_proposed"] == 1
    prob = rows["probability"]  # F (0.99) + E (0.95)
    assert prob["known_fraud_reimbursement_captured"] == 5100 and prob["recall_at_k"] == pytest.approx(0.5)
    # capacity beyond the cohort is clamped; everything is captured
    full = {r["strategy"]: r for r in summarize_at_capacity(curves, 99)}
    for r in full.values():
        assert r["providers_selected"] == 6 and r["capture_rate"] == pytest.approx(1.0)


def test_baselines_use_identical_cohort(cohort):
    curves = build_audit_curves(cohort, "label", n_sims=20, seed=0)
    n = len(cohort)
    for s in curves["strategies"].values():
        assert len(s["fraud_identified"]) == n
        assert s["fraud_identified"][-1] == pytest.approx(cohort["label"].sum())


def test_random_baseline_reproducible_and_unbiased():
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.1).astype(float)
    r = rng.lognormal(10, 1, 300)
    a = random_baseline(y, r, n_sims=500, seed=42)
    b = random_baseline(y, r, n_sims=500, seed=42)
    np.testing.assert_array_equal(a["known_fraud_reimbursement"], b["known_fraud_reimbursement"])
    k = 100
    assert a["fraud_identified"][k - 1] == pytest.approx(y.sum() * k / 300, rel=0.05)
    assert a["fraud_identified_p05"][k - 1] <= a["fraud_identified"][k - 1] <= a["fraud_identified_p95"][k - 1]
    c = random_baseline(y, r, n_sims=500, seed=7)
    assert not np.array_equal(a["known_fraud_reimbursement"], c["known_fraud_reimbursement"])


def test_risk_levels_and_priority_tiers():
    assert risk_level([0.6, 0.5, 0.2, 0.15, 0.01]).tolist() == ["high", "high", "medium", "medium", "low"]
    tiers = priority_tier(np.array([1, 5, 6, 15, 16, 100]), np.array(["low", "low", "low", "high", "medium", "low"]), 100)
    assert tiers.tolist() == ["P1", "P1", "P2", "P2", "P3", "Routine"]


def test_scoring_does_not_use_labels():
    df = pd.DataFrame({"Provider": ["A", "B"], "total_reimbursed": [1.0, 2.0]})
    out = score_providers(df, np.array([0.5, 0.5]))
    assert "label" not in out.columns and "PotentialFraud" not in out.columns
