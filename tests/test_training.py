"""End-to-end pipeline on synthetic data: leakage checks, artifacts, inference consistency."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from mediguard.models.inference import InputValidationError, build_feature_frame, predict_from_inputs, validate_inputs


def _report(paths, name):
    return json.loads((paths.reports_dir / f"{name}.json").read_text())


def test_all_leakage_checks_passed(trained):
    paths, _ = trained
    checks = _report(paths, "leakage_checks")
    assert len(checks) >= 10 and all(c["passed"] for c in checks)


def test_no_provider_overlap_and_test_set_isolation(trained):
    paths, _ = trained
    split = _report(paths, "split")
    scores = pd.read_parquet(paths.artifact_dir / "provider_scores.parquet")
    base = pd.read_parquet(paths.processed_dir / "provider_base_features.parquet")
    assert len(scores) == split["test_providers"]
    assert split["development_providers"] + split["test_providers"] == len(base)
    fp = joblib.load(paths.artifact_dir / "feature_pipeline.joblib")
    test_ids = set(scores["Provider"].astype(str))
    # every learned component of the final pipeline was fitted without the scoring cohort
    assert not set(map(str, fp.train_providers_)) & test_ids
    assert not set(map(str, fp.miner_.train_providers_)) & test_ids
    assert fp.miner_.n_transactions_ == split["development_claims"]
    assert fp.regression_oof_metrics_["n_claims"] == split["development_claims"]


def test_selection_and_calibration_use_development_data_only(trained):
    paths, _ = trained
    perf = _report(paths, "model_performance")
    # selection uses CV means; the selected family has the highest CV PR-AUC, regardless of test PR-AUC
    best_cv = max(perf["test"], key=lambda m: m["cv_pr_auc_mean"])
    assert best_cv["selected"]
    assert "out-of-fold" in perf["calibration_method"] and "out-of-fold" in perf["threshold_rule"]
    for m in perf["test"]:
        assert 0 < m["threshold"] < 1
        cm = m["confusion_matrix"]
        assert cm["tp"] + cm["fn"] == m["positives"] and sum(cm.values()) == m["n"]


def test_predictions_valid_and_feature_order_consistent(trained):
    paths, _ = trained
    bundle = joblib.load(paths.artifact_dir / "model_bundle.joblib")
    scores = pd.read_parquet(paths.artifact_dir / "provider_scores.parquet")
    p = bundle.predict(scores[bundle.feature_names])
    assert p.shape == (len(scores),) and np.all((p >= 0) & (p <= 1))
    np.testing.assert_allclose(p, scores["fraud_probability"].to_numpy(), rtol=1e-9)
    # column order is enforced: shuffled input gives identical predictions
    shuffled = scores[bundle.feature_names[::-1]]
    np.testing.assert_allclose(bundle.predict(shuffled), p)
    with pytest.raises(KeyError):
        bundle.predict(scores[bundle.feature_names[1:]])
    assert "retrospective_fraud_label" not in bundle.feature_names


def test_inference_reproduces_stored_scores(trained):
    """Calculator path (validation -> derived features -> clusterer -> imputer -> model -> calibrator)."""
    paths, _ = trained
    bundle = joblib.load(paths.artifact_dir / "model_bundle.joblib")
    scores = pd.read_parquet(paths.artifact_dir / "provider_scores.parquet")
    for _, row in scores.sample(5, random_state=0).iterrows():
        inputs = {f: float(row[f]) for f in bundle.input_features() if pd.notna(row[f])}
        out = predict_from_inputs(bundle, inputs)
        assert out["fraud_probability"] == pytest.approx(row["fraud_probability"], abs=1e-9)
        assert out["estimated_exposure"] == pytest.approx(row["estimated_exposure"], rel=1e-9)


def test_inference_imputes_missing_with_training_medians(trained):
    paths, _ = trained
    bundle = joblib.load(paths.artifact_dir / "model_bundle.joblib")
    X, imputed, computed = build_feature_frame(bundle, {"total_claims": 40, "total_reimbursed": 9000, "unique_beneficiaries": 20})
    assert list(X.columns) == bundle.feature_names
    assert computed["claims_per_beneficiary"] == 2 and computed["mean_reimbursed"] == 225
    assert "total_claims" not in imputed and "mean_residual" in imputed
    imputer = bundle.pipeline.named_steps["impute"]
    filled = imputer.transform(X)[0]
    j = bundle.feature_names.index("mean_residual")
    assert filled[j] == pytest.approx(imputer.statistics_[j])


@pytest.mark.parametrize("inputs,field", [
    ({"total_claims": 10, "total_reimbursed": -1, "unique_beneficiaries": 5}, "total_reimbursed"),
    ({"total_claims": 10, "total_reimbursed": 10, "unique_beneficiaries": 50}, "unique_beneficiaries"),
    ({"total_claims": 10, "total_reimbursed": 10, "unique_beneficiaries": 5, "mean_reimbursed": 1}, "mean_reimbursed"),
    ({"total_claims": 10, "total_reimbursed": 10, "unique_beneficiaries": 5, "bogus": 1}, "bogus"),
    ({"total_claims": 10, "total_reimbursed": 10, "unique_beneficiaries": 5, "positive_residual_ratio": 1.5}, "positive_residual_ratio"),
    ({"total_claims": 10, "total_reimbursed": float("nan"), "unique_beneficiaries": 5}, "total_reimbursed"),
    ({"total_reimbursed": 10, "unique_beneficiaries": 5}, "total_claims"),
])
def test_input_validation(trained, inputs, field):
    paths, _ = trained
    bundle = joblib.load(paths.artifact_dir / "model_bundle.joblib")
    with pytest.raises(InputValidationError) as e:
        validate_inputs(bundle, inputs)
    assert field in {d["field"] for d in e.value.errors}


def test_audit_curves_consistent_with_scores(trained):
    paths, _ = trained
    curves = _report(paths, "audit_curves")
    scores = pd.read_parquet(paths.artifact_dir / "provider_scores.parquet")
    y = scores["retrospective_fraud_label"].to_numpy()
    r = scores["total_reimbursed"].to_numpy()
    assert curves["total_known_fraud_reimbursement"] == pytest.approx(float((y * r).sum()))
    top = scores.sort_values("rank_exposure").head(5)
    assert curves["strategies"]["exposure"]["known_fraud_reimbursement"][4] == pytest.approx(
        float((top["retrospective_fraud_label"] * top["total_reimbursed"]).sum()))
