"""End-to-end training and evaluation.

Protocol
--------
1. Preprocess claims (row-local, deterministic) and aggregate label-free provider features.
2. Provider-level stratified split: 80% development / 20% **held-out test**. Labels are
   per provider and each provider's claims go to exactly one side.
3. On the development set, 5-fold provider-level stratified cross-validation. In every fold the
   *entire* ``FeaturePipeline`` (regression, clustering, association rules) is fitted on the
   training fold and applied to the validation fold, then every classifier configuration is fitted.
4. Model selection on mean validation PR-AUC (hyper-parameters per family, then family).
5. Platt calibration and the F1-optimal decision threshold are fitted on pooled out-of-fold
   validation predictions. Ablation and permutation importance are also computed on the folds.
6. The feature pipeline and the chosen configuration are refitted on all development providers.
7. The test set is used once, for final evaluation and the audit simulation. Nothing is selected
   or tuned on it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import time
from dataclasses import replace
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from mediguard.audit.prioritization import build_audit_curves, score_providers, summarize_at_capacity
from mediguard.config import AUDIT_CAPACITY_OPTIONS, PATHS, TRAINING, Paths, TrainingConfig
from mediguard.data.ingest import DIAG_COLS, PROC_COLS, load_raw_tables
from mediguard.data.preprocess import build_claims
from mediguard.features.catalog import GROUP_LABELS, catalog, describe
from mediguard.features.clustering import CLUSTER_INPUTS
from mediguard.features.pipeline import FeaturePipeline
from mediguard.features.provider_features import aggregate_provider_features
from mediguard.features.regression import CATEGORICAL_INPUTS, NUMERIC_INPUTS, TARGET
from mediguard.models.classifiers import MODEL_SPECS, config_label
from mediguard.models.evaluation import (PlattCalibrator, best_f1_threshold, classification_metrics, curve_points,
                                         expected_calibration_error, reliability_curve)
from mediguard.models.explain import permutation_importance_pr_auc, reference_substitution
from mediguard.models.inference import ModelBundle

log = logging.getLogger("mediguard.train")
CLAIM_COLUMNS_FOR_MODELLING = sorted(set(
    ["Provider", "ClaimID", "ClaimStartDt", TARGET, "is_inpatient", *NUMERIC_INPUTS, *CATEGORICAL_INPUTS.values(),
     *DIAG_COLS, *PROC_COLS]))
LABEL = "PotentialFraud"
ABLATION_SETS = {
    "base": ("Base provider features", ["base"]),
    "base_regression": ("Base + regression residuals", ["base", "regression"]),
    "base_clustering": ("Base + clustering", ["base", "clustering"]),
    "base_association": ("Base + association rules", ["base", "association"]),
    "full": ("Full pipeline (all techniques)", ["base", "regression", "clustering", "association"]),
}


class LeakageLog:
    def __init__(self):
        self.checks: list[dict] = []

    def check(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append({"check": name, "passed": bool(passed), "detail": detail})
        if not passed:
            raise AssertionError(f"Leakage check failed: {name} ({detail})")

    def disjoint(self, name: str, a, b) -> None:
        inter = set(a) & set(b)
        self.check(name, not inter, f"{len(set(a))} vs {len(set(b))} providers, overlap={len(inter)}")


def _group_of(col: str, groups: dict) -> str:
    for g, cols in groups.items():
        if col in cols:
            return g
    raise KeyError(col)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(type(o))


def _write_json(path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, default=_json_default, indent=1, allow_nan=False))


def _nan_to_none(obj):
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    return obj


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return None


def _precision_top_fraction(y: np.ndarray, p: np.ndarray, frac: float = 0.10) -> float:
    k = max(1, int(round(len(y) * frac)))
    return float(y[np.argsort(-p, kind="mergesort")[:k]].mean())


def run(paths: Paths = PATHS, cfg: TrainingConfig = TRAINING, quick: bool = False) -> dict:
    t_start = time.time()
    rs = cfg.random_state
    leak = LeakageLog()
    specs = MODEL_SPECS
    configs = {s.key: (s.configs()[:1] if quick else s.configs()) for s in specs}

    # 1. Data ------------------------------------------------------------------------------------
    claims, labels, data_report = build_claims(load_raw_tables(paths.raw_dir))
    base = aggregate_provider_features(claims)
    y_all = labels.set_index("Provider")[LABEL].sort_index()
    assert list(base.index) == list(y_all.index), "provider feature/label index mismatch"
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    base.join(y_all).to_parquet(paths.processed_dir / "provider_base_features.parquet")

    data_report["total_reimbursement"] = float(base["total_reimbursed"].sum())
    data_report["fraud_labelled_reimbursement"] = float((base["total_reimbursed"] * y_all).sum())
    # Keep only the claim columns used after provider aggregation (memory: ~558k rows).
    claims = claims[CLAIM_COLUMNS_FOR_MODELLING].copy()

    def claims_of(providers) -> pd.DataFrame:
        return claims[claims["Provider"].isin(set(providers))]

    # 2. Split -----------------------------------------------------------------------------------
    providers = y_all.index.to_numpy(dtype=object)
    dev, test = train_test_split(providers, test_size=cfg.test_size, stratify=y_all.to_numpy(), random_state=rs)
    dev, test = np.sort(dev), np.sort(test)
    y_dev, y_test = y_all.loc[dev].to_numpy(), y_all.loc[test].to_numpy()
    leak.disjoint("Development and test providers are disjoint", dev, test)
    dev_claims, test_claims = claims_of(dev), claims_of(test)
    leak.check("Every claim belongs to exactly one partition",
               len(dev_claims) + len(test_claims) == len(claims)
               and not set(dev_claims["ClaimID"]) & set(test_claims["ClaimID"]),
               f"{len(dev_claims):,} development + {len(test_claims):,} test = {len(claims):,} claims")
    split = {
        "development_providers": int(len(dev)), "test_providers": int(len(test)),
        "development_fraud": int(y_dev.sum()), "test_fraud": int(y_test.sum()),
        "development_claims": int(len(dev_claims)), "test_claims": int(len(test_claims)),
        "development_prevalence": float(y_dev.mean()), "test_prevalence": float(y_test.mean()),
        "cv_folds": cfg.cv_folds, "inner_folds": cfg.inner_folds, "random_state": rs,
    }
    log.info("Split: %s", split)

    # 3. Cross-validation --------------------------------------------------------------------------
    skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=rs)
    oof = {(s.key, config_label(p)): np.full(len(dev), np.nan) for s in specs for p in configs[s.key]}
    fold_scores = {k: [] for k in oof}
    folds = []
    for f, (tr_i, va_i) in enumerate(skf.split(dev, y_dev)):
        t0 = time.time()
        tr_p, va_p = dev[tr_i], dev[va_i]
        leak.disjoint(f"Fold {f + 1}: training and validation providers are disjoint", tr_p, va_p)
        fp = FeaturePipeline(cfg)
        Xtr = fp.fit_transform(claims_of(tr_p), base, y_all.loc[tr_p])
        Xva = fp.transform(claims_of(va_p), base)
        leak.check(f"Fold {f + 1}: association rules mined from training-fold providers only",
                   fp.miner_.train_providers_ == set(tr_p) and not fp.miner_.train_providers_ & set(va_p),
                   f"{len(fp.miner_.train_providers_)} providers, {len(fp.miner_.selected_)} rules")
        leak.check(f"Fold {f + 1}: regression, clustering and scaling fitted on training fold only",
                   fp.train_providers_ == frozenset(tr_p), f"fitted on {len(fp.train_providers_)} providers")
        ytr, yva = y_dev[tr_i], y_dev[va_i]
        for s in specs:
            for params in configs[s.key]:
                key = (s.key, config_label(params))
                p = s.fit(params, Xtr, ytr, rs).predict_proba(Xva)[:, 1]
                oof[key][va_i] = p
                fold_scores[key].append({"pr_auc": average_precision_score(yva, p), "roc_auc": roc_auc_score(yva, p)})
        folds.append({"tr_i": tr_i, "va_i": va_i, "Xtr": Xtr, "Xva": Xva, "ytr": ytr, "yva": yva,
                      "groups": fp.groups_, "regression_validation": fp.last_regression_metrics_,
                      "regression_oof_train": fp.regression_oof_metrics_, "n_rules": len(fp.miner_.selected_),
                      "n_clusters": fp.clusterer_.n_clusters_})
        log.info("Fold %d/%d done in %.0fs", f + 1, cfg.cv_folds, time.time() - t0)

    # 4. Selection -------------------------------------------------------------------------------
    cv_rows, best_cfg = [], {}
    for s in specs:
        for params in configs[s.key]:
            key = (s.key, config_label(params))
            pr = np.array([d["pr_auc"] for d in fold_scores[key]])
            roc = np.array([d["roc_auc"] for d in fold_scores[key]])
            cv_rows.append({"model": s.key, "model_name": s.name, "config": key[1], "params": params,
                            "cv_pr_auc_mean": pr.mean(), "cv_pr_auc_std": pr.std(ddof=1),
                            "cv_roc_auc_mean": roc.mean(), "cv_roc_auc_std": roc.std(ddof=1),
                            "fold_pr_auc": pr.tolist()})
        rows = [r for r in cv_rows if r["model"] == s.key]
        best_cfg[s.key] = max(rows, key=lambda r: r["cv_pr_auc_mean"])
    for r in cv_rows:
        r["selected_for_family"] = r is best_cfg[r["model"]]
    selected_key = max(best_cfg, key=lambda k: best_cfg[k]["cv_pr_auc_mean"])
    log.info("Selected model: %s (%s) CV PR-AUC %.4f", selected_key, best_cfg[selected_key]["config"],
             best_cfg[selected_key]["cv_pr_auc_mean"])

    # 5. Calibration + threshold per family (OOF) ------------------------------------------------
    calibrators, thresholds, cv_pooled = {}, {}, {}
    for s in specs:
        raw = oof[(s.key, best_cfg[s.key]["config"])]
        assert not np.isnan(raw).any()
        cal = PlattCalibrator().fit(raw, y_dev)
        p_cal = cal.transform(raw)
        calibrators[s.key] = cal
        thresholds[s.key] = best_f1_threshold(y_dev, p_cal)
        cv_pooled[s.key] = classification_metrics(y_dev, p_cal, thresholds[s.key])
    sel_spec = next(s for s in specs if s.key == selected_key)
    sel_params = best_cfg[selected_key]["params"]

    # 5b. Ablation and permutation importance on folds (selected configuration) ------------------
    abl_oof = {k: np.full(len(dev), np.nan) for k in ABLATION_SETS}
    abl_fold = {k: [] for k in ABLATION_SETS}
    importances = []
    for f, fd in enumerate(folds):
        for k, (_, groups) in ABLATION_SETS.items():
            cols = [c for c in fd["Xtr"].columns if _group_of(c, fd["groups"]) in groups]
            m = sel_spec.fit(sel_params, fd["Xtr"][cols], fd["ytr"], rs)
            p = m.predict_proba(fd["Xva"][cols])[:, 1]
            abl_oof[k][fd["va_i"]] = p
            abl_fold[k].append({"pr_auc": average_precision_score(fd["yva"], p),
                                "roc_auc": roc_auc_score(fd["yva"], p),
                                "precision_top10pct": _precision_top_fraction(fd["yva"], p)})
            if k == "full":
                imp = permutation_importance_pr_auc(m, fd["Xva"][cols], fd["yva"], n_repeats=3 if quick else 5,
                                                    random_state=rs + f)
                importances.append(imp)
        log.info("Ablation/importance fold %d done", f + 1)
    imp_df = pd.concat(importances, axis=1)
    importance = pd.DataFrame({"mean": imp_df.mean(axis=1), "std": imp_df.std(axis=1, ddof=1),
                               "folds_present": imp_df.notna().sum(axis=1)}).sort_values("mean", ascending=False)

    # 6. Final refit on all development providers -----------------------------------------------
    fp_final = FeaturePipeline(cfg)
    X_dev = fp_final.fit_transform(dev_claims, base, y_all.loc[dev])
    X_test = fp_final.transform(test_claims, base)
    leak.check("Final feature pipeline fitted on development providers only",
               fp_final.train_providers_ == frozenset(dev) and not fp_final.train_providers_ & set(test),
               f"{len(fp_final.train_providers_)} providers")
    leak.check("Final association rules mined without any test provider",
               not fp_final.miner_.train_providers_ & set(test), f"{len(fp_final.miner_.selected_)} rules")
    assert list(X_dev.index) == list(dev) and list(X_test.index) == list(test)
    feature_names = list(X_dev.columns)

    # 7. Test evaluation (once) ------------------------------------------------------------------
    test_results, test_probs, final_models = {}, {}, {}
    for s in specs:
        m = s.fit(best_cfg[s.key]["params"], X_dev, y_dev, rs)
        raw = m.predict_proba(X_test)[:, 1]
        p = calibrators[s.key].transform(raw)
        final_models[s.key] = m
        test_probs[s.key] = p
        met = classification_metrics(y_test, p, thresholds[s.key])
        met.update({
            "model": s.key, "model_name": s.name, "config": best_cfg[s.key]["config"],
            "cv_pr_auc_mean": best_cfg[s.key]["cv_pr_auc_mean"], "cv_pr_auc_std": best_cfg[s.key]["cv_pr_auc_std"],
            "cv_roc_auc_mean": best_cfg[s.key]["cv_roc_auc_mean"], "cv_roc_auc_std": best_cfg[s.key]["cv_roc_auc_std"],
            "selected": s.key == selected_key,
            "curves": curve_points(y_test, p),
            "calibration": {
                "raw_brier": float(np.mean((raw - y_test) ** 2)), "calibrated_brier": met["brier"],
                "raw_ece": expected_calibration_error(y_test, raw), "calibrated_ece": expected_calibration_error(y_test, p),
                "raw_reliability": reliability_curve(y_test, raw), "calibrated_reliability": reliability_curve(y_test, p),
                "platt_coef": calibrators[s.key].coef_, "platt_intercept": calibrators[s.key].intercept_,
            },
        })
        test_results[s.key] = met

    # Ablation on test (secondary, each variant calibrated with its own OOF predictions)
    abl_rows = []
    for k, (label, groups) in ABLATION_SETS.items():
        cols = [c for c in feature_names if _group_of(c, fp_final.groups_) in groups]
        cal = PlattCalibrator().fit(abl_oof[k], y_dev)
        m = sel_spec.fit(sel_params, X_dev[cols], y_dev, rs)
        p = cal.transform(m.predict_proba(X_test[cols])[:, 1])
        sc = score_providers(pd.DataFrame({"Provider": test, "total_reimbursed": X_test["total_reimbursed"].to_numpy()}), p)
        sc["label"] = y_test
        cur = build_audit_curves(sc, "label", n_sims=10, seed=rs)
        at100 = {r["strategy"]: r for r in summarize_at_capacity(cur, 100)}["exposure"]
        fold = abl_fold[k]
        abl_rows.append({
            "variant": k, "label": label, "groups": groups, "n_features": len(cols),
            "cv_pr_auc_mean": float(np.mean([d["pr_auc"] for d in fold])),
            "cv_pr_auc_std": float(np.std([d["pr_auc"] for d in fold], ddof=1)),
            "cv_roc_auc_mean": float(np.mean([d["roc_auc"] for d in fold])),
            "cv_roc_auc_std": float(np.std([d["roc_auc"] for d in fold], ddof=1)),
            "cv_precision_top10pct_mean": float(np.mean([d["precision_top10pct"] for d in fold])),
            "fold_pr_auc": [d["pr_auc"] for d in fold],
            "test_pr_auc": float(average_precision_score(y_test, p)), "test_roc_auc": float(roc_auc_score(y_test, p)),
            "test_capture_rate_at_100": at100["capture_rate"],
            "test_fraud_identified_at_100": at100["fraud_labelled_identified"],
        })
    full_mean = next(r for r in abl_rows if r["variant"] == "full")["cv_pr_auc_mean"]
    base_row = next(r for r in abl_rows if r["variant"] == "base")
    for r in abl_rows:
        r["delta_cv_pr_auc_vs_base"] = r["cv_pr_auc_mean"] - base_row["cv_pr_auc_mean"]
        # paired fold differences: same folds, so the spread of the difference is the right noise scale
        diffs = np.array(r["fold_pr_auc"]) - np.array(base_row["fold_pr_auc"])
        r["delta_fold_std"] = float(diffs.std(ddof=1)) if r is not base_row else 0.0
    log.info("Ablation done (full CV PR-AUC %.4f)", full_mean)

    # 8. Bundle ----------------------------------------------------------------------------------
    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    fingerprint = json.dumps({"cfg": cfg.as_dict(), "model": selected_key, "params": sel_params,
                              "features": feature_names, "quick": quick}, sort_keys=True, default=str)
    model_version = f"{datetime.now(timezone.utc):%Y%m%d}-{hashlib.sha1(fingerprint.encode()).hexdigest()[:8]}"
    bounds = {c: {"min": float(X_dev[c].min()), "max": float(X_dev[c].max()),
                  "p01": float(X_dev[c].quantile(0.01)), "p99": float(X_dev[c].quantile(0.99))}
              for c in feature_names}
    bundle = ModelBundle(
        model_key=selected_key, model_name=sel_spec.name, params=sel_params, pipeline=final_models[selected_key],
        calibrator=calibrators[selected_key], threshold=thresholds[selected_key], feature_names=feature_names,
        reference=X_dev.median(), bounds=bounds, clusterer=fp_final.clusterer_, model_version=model_version,
        trained_at=trained_at, selection_criterion="Highest mean 5-fold cross-validated PR-AUC on development providers",
        metadata={"n_development_providers": int(len(dev)), "feature_groups": fp_final.groups_, "quick": quick},
    )

    # 9. Score the held-out cohort, explanations, audit curves -----------------------------------
    p_test = test_probs[selected_key]
    assert np.allclose(p_test, bundle.predict(X_test)), "bundle predictions differ from evaluation predictions"
    assign = fp_final.clusterer_.assign(X_test)
    scored = X_test.reset_index()
    scored = score_providers(scored, p_test)
    scored["raw_model_score"] = bundle.predict_raw(X_test)
    scored["peer_group"] = assign["peer_group"].to_numpy()
    scored["pca_1"], scored["pca_2"] = assign["pca_1"].to_numpy(), assign["pca_2"].to_numpy()
    expl = reference_substitution(bundle.predict, X_test, bundle.reference, top_n=8)
    scored["explanation"] = [json.dumps(e) for e in expl]
    scored["retrospective_fraud_label"] = y_test
    audit_curves = build_audit_curves(scored, "retrospective_fraud_label", cfg.random_baseline_simulations, rs)
    audit_curves["providers"] = scored["Provider"].astype(str).tolist()

    # Provider profile material for the held-out cohort
    tc = test_claims.assign(month=test_claims["ClaimStartDt"].dt.strftime("%Y-%m"))
    monthly = tc.groupby(["Provider", "month"]).agg(
        claims=("ClaimID", "size"), reimbursed=("InscClaimAmtReimbursed", "sum"),
        inpatient_claims=("is_inpatient", "sum")).reset_index()
    buckets = pd.cut(tc["InscClaimAmtReimbursed"], [-0.1, 0, 99, 499, 1999, 9999, np.inf],
                     labels=["$0", "$1-99", "$100-499", "$500-1,999", "$2,000-9,999", "$10,000+"])
    mix = tc.assign(bucket=buckets.astype(str), setting=np.where(tc["is_inpatient"] == 1, "inpatient", "outpatient")) \
            .groupby(["Provider", "setting", "bucket"]).size().rename("claims").reset_index()
    rule_hits = fp_final.miner_.provider_rule_hits(test_claims)

    # 10. Technique reports -----------------------------------------------------------------------
    reg_sample = test_claims.sample(n=min(600, len(test_claims)), random_state=rs)
    reg_report = {
        "model": "HistGradientBoostingRegressor",
        "params": fp_final.reg_params,
        "development_out_of_fold": fp_final.regression_oof_metrics_,
        "test": fp_final.last_regression_metrics_,
        "cv_validation_folds": [fd["regression_validation"] for fd in folds],
        "scatter_sample": [[float(a), float(b)] for a, b in zip(reg_sample["InscClaimAmtReimbursed"],
                                                                fp_final.regressor_.predict(reg_sample))],
        "residual_distribution": {q: float(scored["mean_residual"].quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
    }
    profile_cols = ["total_claims", "total_reimbursed", "mean_reimbursed", "inpatient_ratio", "claims_per_beneficiary",
                    "unique_attending_physicians", "mean_claim_duration"]
    clusters = []
    for g, sub in scored.groupby("peer_group"):
        clusters.append({
            "peer_group": int(g), "providers": int(len(sub)), "dbscan_outliers": int(sub["dbscan_outlier"].sum()),
            "median_profile": {c: float(sub[c].median()) for c in profile_cols},
            "mean_fraud_probability": float(sub["fraud_probability"].mean()),
            "retrospective_fraud_rate": float(sub["retrospective_fraud_label"].mean()),
        })
    outl = scored["dbscan_outlier"] == 1
    cluster_report = {
        **fp_final.clusterer_.summary(),
        "test_cohort": clusters,
        "test_outliers": int(outl.sum()),
        "retrospective_fraud_rate_outliers": float(scored.loc[outl, "retrospective_fraud_label"].mean()) if outl.any() else None,
        "retrospective_fraud_rate_non_outliers": float(scored.loc[~outl, "retrospective_fraud_label"].mean()),
        "pca_points": [[round(a, 4), round(b, 4), int(g), int(o)] for a, b, g, o in
                       zip(scored["pca_1"], scored["pca_2"], scored["peer_group"], scored["dbscan_outlier"])],
        "fold_n_clusters": [fd["n_clusters"] for fd in folds],
        "profile_columns": profile_cols,
    }
    rules_report = fp_final.miner_.rule_report()
    rules_report["fold_rule_counts"] = [fd["n_rules"] for fd in folds]

    perf = {
        "selection_criterion": bundle.selection_criterion,
        "selected_model": selected_key, "selected_model_name": sel_spec.name, "selected_config": best_cfg[selected_key]["config"],
        "decision_threshold": thresholds[selected_key],
        "threshold_rule": "Maximises F1 on pooled out-of-fold calibrated predictions (development set)",
        "calibration_method": "Platt scaling fitted on pooled out-of-fold scores (development set)",
        "cv_configurations": cv_rows,
        "cv_pooled": cv_pooled,
        "test": [test_results[s.key] for s in specs],
        "feature_importance": [{**describe(f), "mean": float(r["mean"]), "std": float(r["std"]) if not np.isnan(r["std"]) else 0.0,
                                "folds_present": int(r["folds_present"])} for f, r in importance.head(25).iterrows()],
        "importance_method": "Permutation importance: mean drop in validation PR-AUC across CV folds (5 repeats)",
        "test_prevalence": float(y_test.mean()),
    }
    ablation = {"selected_model": sel_spec.name, "config": best_cfg[selected_key]["config"], "variants": abl_rows,
                "note": "CV metrics are the primary comparison (5 provider-level folds, same folds for every "
                        "variant). Test metrics are shown for completeness and were not used for any decision."}

    # 11. Persist ---------------------------------------------------------------------------------
    art = paths.artifact_dir
    rep = paths.reports_dir
    rep.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, art / "model_bundle.joblib", compress=3)
    joblib.dump(fp_final, art / "feature_pipeline.joblib", compress=3)
    scored.to_parquet(art / "provider_scores.parquet", index=False)
    monthly.to_parquet(art / "provider_monthly.parquet", index=False)
    mix.to_parquet(art / "provider_claim_mix.parquet", index=False)
    _write_json(rep / "provider_rule_hits.json", rule_hits)
    _write_json(rep / "data_report.json", data_report)
    _write_json(rep / "split.json", split)
    _write_json(rep / "model_performance.json", _nan_to_none(perf))
    _write_json(rep / "ablation.json", _nan_to_none(ablation))
    _write_json(rep / "regression.json", _nan_to_none(reg_report))
    _write_json(rep / "clustering.json", _nan_to_none(cluster_report))
    _write_json(rep / "association_rules.json", _nan_to_none(rules_report))
    _write_json(rep / "audit_curves.json", audit_curves)
    _write_json(rep / "leakage_checks.json", leak.checks)
    _write_json(rep / "feature_catalog.json", {"features": catalog(feature_names), "groups": GROUP_LABELS,
                                               "cluster_inputs": CLUSTER_INPUTS})
    summary_at = {k: summarize_at_capacity(audit_curves, k) for k in AUDIT_CAPACITY_OPTIONS}
    manifest = {
        "model_version": model_version, "trained_at": trained_at, "git_commit": _git_commit(),
        "selected_model": sel_spec.name, "selected_config": best_cfg[selected_key]["config"],
        "n_features": len(feature_names), "quick_mode": quick, "config": cfg.as_dict(),
        "runtime_seconds": round(time.time() - t_start, 1),
        "headline": {
            "test_pr_auc": test_results[selected_key]["pr_auc"], "test_roc_auc": test_results[selected_key]["roc_auc"],
            "capture_at_100": {r["strategy"]: r["capture_rate"] for r in summary_at[100]},
        },
    }
    _write_json(art / "manifest.json", manifest)
    log.info("Training complete in %.0fs. Model %s (%s). Test PR-AUC %.4f", time.time() - t_start,
             model_version, sel_spec.name, test_results[selected_key]["pr_auc"])
    return manifest


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Train and evaluate the MediGuard pipeline")
    ap.add_argument("--quick", action="store_true", help="one configuration per model family (smoke runs)")
    ap.add_argument("--cv-folds", type=int, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    cfg = TRAINING if args.cv_folds is None else replace(TRAINING, cv_folds=args.cv_folds)
    manifest = run(PATHS, cfg, quick=args.quick)
    print(json.dumps(manifest["headline"], indent=1))


if __name__ == "__main__":
    main()
