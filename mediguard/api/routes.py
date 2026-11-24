"""HTTP API. All numbers come from persisted training artifacts; nothing is hardcoded."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from flask import Blueprint, current_app, jsonify, request

from mediguard import __version__
from mediguard.api.schemas import (CapacityQuery, PredictRequest, ProviderDetailQuery, ProviderListQuery,
                                   StatusUpdate)
from mediguard.audit.prioritization import STRATEGIES, priority_tier, summarize_at_capacity
from mediguard.config import AUDIT_CAPACITY_OPTIONS, INVESTIGATION_STATUSES
from mediguard.features.catalog import describe
from mediguard.models.inference import predict_from_inputs

api = Blueprint("api", __name__, url_prefix="/api")

DEFAULT_CAPACITY = 100
RECOMMENDATIONS = {
    "P1": ("Prioritise for full audit", "Ranks in the top 5% of estimated financial exposure in the scoring cohort."),
    "P2": ("Schedule desk review", "Ranks in the top 5-15% of estimated financial exposure."),
    "P3": ("Monitor", "Elevated fraud probability but comparatively low financial exposure."),
    "Routine": ("Routine monitoring", "Low predicted probability and exposure relative to the cohort."),
}
PROFILE_FEATURES = ["total_claims", "total_reimbursed", "mean_reimbursed", "unique_beneficiaries",
                    "claims_per_beneficiary", "reimbursement_per_beneficiary", "inpatient_ratio",
                    "mean_claim_duration", "max_claim_duration", "unique_attending_physicians",
                    "mean_residual", "positive_residual_ratio", "total_positive_deviation",
                    "suspicious_itemset_claim_ratio"]


class ApiError(Exception):
    def __init__(self, status: int, message: str, details=None):
        super().__init__(message)
        self.status, self.message, self.details = status, message, details


# -- helpers ----------------------------------------------------------------------------------------
def store():
    s = current_app.extensions.get("mediguard_store")
    if s is None:
        raise ApiError(503, current_app.extensions.get("mediguard_store_error") or "Model artifacts are not loaded")
    return s


def repo():
    return current_app.extensions["mediguard_repo"]


def py(v):
    """JSON-safe Python scalar."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def args_dict() -> dict:
    return {k: v for k, v in request.args.items() if v != ""}


def recommendation(tier: str) -> dict:
    action, why = RECOMMENDATIONS[tier]
    return {"tier": tier, "action": action, "rationale": why,
            "disclaimer": "Application-generated recommendation based on model output. It is not an "
                          "investigation finding; investigator decisions are recorded separately."}


def indicators(expl: list[dict], n: int = 3) -> list[str]:
    return [describe(e["feature"])["label"] for e in expl if e["contribution"] > 0][:n]


def queue_row(r: pd.Series, statuses: dict, retrospective: bool) -> dict:
    row = {
        "rank": int(r["rank_exposure"]), "provider_id": r["Provider"],
        "fraud_probability": py(r["fraud_probability"]), "total_reimbursed": py(r["total_reimbursed"]),
        "estimated_exposure": py(r["estimated_exposure"]), "total_claims": int(r["total_claims"]),
        "unique_beneficiaries": int(r["unique_beneficiaries"]), "risk_level": r["risk_level"],
        "audit_priority": r["audit_priority"], "main_indicators": indicators(r["explanation"]),
        "investigation_status": statuses.get(r["Provider"], "not_reviewed"),
    }
    if retrospective:
        row["retrospective_fraud_label"] = int(r["retrospective_fraud_label"])
    return row


def cohort_summary(s) -> dict:
    c = s.reports["audit_curves"]
    return {"providers": c["n_providers"], "fraud_labelled_providers": c["n_fraud_labelled"],
            "known_fraud_reimbursement": c["total_known_fraud_reimbursement"],
            "total_reimbursement": c["total_reimbursement"],
            "description": "Held-out test providers: excluded from all model fitting, tuning, calibration and "
                           "threshold selection."}


def capture_curves(s, max_k: int | None = None) -> dict:
    c = s.reports["audit_curves"]
    total = c["total_known_fraud_reimbursement"] or 1.0
    n = c["n_providers"] if max_k is None else min(max_k, c["n_providers"])
    out = {"k": list(range(1, n + 1))}
    for key in STRATEGIES:
        out[key] = [v / total for v in c["strategies"][key]["known_fraud_reimbursement"][:n]]
    out["random_p05"] = [v / total for v in c["strategies"]["random"]["known_fraud_reimbursement_p05"][:n]]
    out["random_p95"] = [v / total for v in c["strategies"]["random"]["known_fraud_reimbursement_p95"][:n]]
    out["identified"] = {key: c["strategies"][key]["fraud_identified"][:n] for key in STRATEGIES}
    out["identified"]["random_p05"] = c["strategies"]["random"]["fraud_identified_p05"][:n]
    out["identified"]["random_p95"] = c["strategies"]["random"]["fraud_identified_p95"][:n]
    return out


def model_info(s) -> dict:
    m, b = s.manifest, s.bundle
    sel = next(t for t in s.reports["model_performance"]["test"] if t["selected"])
    return {"name": b.model_name, "version": m["model_version"], "trained_at": m["trained_at"],
            "config": m["selected_config"], "n_features": m["n_features"], "test_pr_auc": sel["pr_auc"],
            "test_roc_auc": sel["roc_auc"], "cv_pr_auc": sel["cv_pr_auc_mean"], "cv_pr_auc_std": sel["cv_pr_auc_std"],
            "decision_threshold": b.threshold, "selection_criterion": b.selection_criterion}


def capacity_arg(s) -> int:
    q = CapacityQuery(**args_dict())
    n = s.reports["audit_curves"]["n_providers"]
    if q.capacity > n:
        raise ApiError(422, f"capacity cannot exceed the {n} providers in the scoring cohort",
                       [{"field": "capacity", "message": f"must be <= {n}"}])
    return q.capacity


# -- endpoints --------------------------------------------------------------------------------------
@api.get("/health")
def health():
    s = current_app.extensions.get("mediguard_store")
    return jsonify({
        "status": "ok" if s else "degraded", "app_version": __version__, "model_loaded": s is not None,
        "model_version": s.manifest["model_version"] if s else None,
        "model_name": s.bundle.model_name if s else None,
        "artifacts_error": current_app.extensions.get("mediguard_store_error"),
    }), (200 if s else 503)


@api.get("/overview")
def overview():
    s = store()
    d = s.reports["data_report"]
    return jsonify({
        "data": {
            "claims": d["claims"], "inpatient_claims": d["inpatient_claims"], "outpatient_claims": d["outpatient_claims"],
            "providers": d["providers"], "beneficiaries": d["beneficiaries"],
            "fraud_labelled_providers": d["fraud_labelled_providers"], "fraud_prevalence": d["fraud_prevalence"],
            "total_reimbursement": d.get("total_reimbursement"),
            "fraud_labelled_reimbursement": d.get("fraud_labelled_reimbursement"),
            "claim_period": {"start": d["claim_start_min"], "end": d["claim_start_max"]},
        },
        "cohort": cohort_summary(s),
        "model": model_info(s),
        "default_capacity": DEFAULT_CAPACITY,
        "capacity_options": list(AUDIT_CAPACITY_OPTIONS),
        "strategies_at_default": summarize_at_capacity(s.reports["audit_curves"], DEFAULT_CAPACITY),
        "curves": capture_curves(s),
        "strategy_labels": STRATEGIES,
    })


@api.get("/audit-simulation")
def audit_simulation():
    s = store()
    k = capacity_arg(s)
    include = CapacityQuery(**args_dict()).include_curves
    body = {"capacity": k, "cohort": cohort_summary(s), "strategy_labels": STRATEGIES,
            "strategies": summarize_at_capacity(s.reports["audit_curves"], k),
            "random_method": f"{s.reports['audit_curves']['random_simulations']} seeded random orderings "
                             f"(seed {s.reports['audit_curves']['random_seed']}); mean with 5th-95th percentile"}
    if include:
        body["curves"] = capture_curves(s)
    return jsonify(body)


@api.get("/audit-priorities")
def audit_priorities():
    s = store()
    k = capacity_arg(s)
    top = s.scores.iloc[:k]
    statuses = repo().statuses()
    rows = [queue_row(r, statuses, False) for _, r in top.iterrows()]
    total_exp = float(s.scores["estimated_exposure"].sum())
    return jsonify({
        "capacity": k,
        "queue": rows,
        "selected_estimated_exposure": float(top["estimated_exposure"].sum()),
        "selected_reimbursement": float(top["total_reimbursed"].sum()),
        "share_of_cohort_exposure": float(top["estimated_exposure"].sum() / total_exp) if total_exp else 0.0,
        "risk_mix": top["risk_level"].value_counts().to_dict(),
        "status_mix": pd.Series([r["investigation_status"] for r in rows]).value_counts().to_dict(),
        "formula": "Estimated exposure = calibrated fraud probability x total reimbursement",
    })


@api.get("/providers")
def list_providers():
    s = store()
    q = ProviderListQuery(**args_dict())
    df = s.scores
    statuses = repo().statuses()
    status_col = df["Provider"].map(lambda p: statuses.get(p, "not_reviewed"))
    mask = np.ones(len(df), dtype=bool)
    if q.search:
        mask &= df["Provider"].str.contains(q.search, case=False, regex=False).to_numpy()
    if q.risk_level:
        mask &= (df["risk_level"] == q.risk_level).to_numpy()
    if q.priority:
        mask &= (df["audit_priority"] == q.priority).to_numpy()
    if q.status:
        mask &= (status_col == q.status).to_numpy()
    if q.min_probability is not None:
        mask &= (df["fraud_probability"] >= q.min_probability).to_numpy()
    if q.max_probability is not None:
        mask &= (df["fraud_probability"] <= q.max_probability).to_numpy()
    if q.min_exposure is not None:
        mask &= (df["estimated_exposure"] >= q.min_exposure).to_numpy()
    if q.max_exposure is not None:
        mask &= (df["estimated_exposure"] <= q.max_exposure).to_numpy()
    if q.within_capacity is not None:
        mask &= (df["rank_exposure"] <= q.within_capacity).to_numpy()
    sub = df[mask].sort_values([q.sort, "rank_exposure"], ascending=[q.order == "asc", True], kind="mergesort")
    total = int(len(sub))
    pages = max(1, math.ceil(total / q.per_page))
    page = min(q.page, pages)
    chunk = sub.iloc[(page - 1) * q.per_page: page * q.per_page]
    retro = q.evaluation_mode == "retrospective"
    return jsonify({
        "total": total, "page": page, "per_page": q.per_page, "pages": pages, "sort": q.sort, "order": q.order,
        "evaluation_mode": q.evaluation_mode,
        "providers": [queue_row(r, statuses, retro) for _, r in chunk.iterrows()],
    })


def _percentile(series: pd.Series, value: float) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float((series < value).mean() + 0.5 * (series == value).mean())


@api.get("/providers/<provider_id>")
def provider_detail(provider_id: str):
    s = store()
    q = ProviderDetailQuery(**args_dict())
    r = s.provider(provider_id)
    if r is None:
        raise ApiError(404, f"Provider {provider_id!r} is not in the scoring cohort")
    pid = r["Provider"]

    months = s.reports["data_report"]
    trend = s.monthly_by.get(pid, pd.DataFrame(columns=["month", "claims", "reimbursed", "inpatient_claims"]))
    trend = trend.sort_values("month")
    mix = s.mix_by.get(pid, pd.DataFrame(columns=["setting", "bucket", "claims"]))
    comps = []
    for f in PROFILE_FEATURES:
        if f in s.scores.columns:
            col = s.scores[f]
            comps.append({**describe(f), "value": py(r[f]), "cohort_median": py(col.median()),
                          "percentile": _percentile(col, py(r[f]))})
    expl = [{**e, "label": describe(e["feature"])["label"], "kind": describe(e["feature"])["kind"]} for e in r["explanation"]]
    body = {
        "provider_id": pid,
        "model_version": s.manifest["model_version"],
        "scores": {
            "fraud_probability": py(r["fraud_probability"]), "raw_model_score": py(r["raw_model_score"]),
            "estimated_exposure": py(r["estimated_exposure"]), "risk_level": r["risk_level"],
            "audit_priority": r["audit_priority"], "rank_exposure": int(r["rank_exposure"]),
            "rank_probability": int(r["rank_probability"]), "rank_reimbursement": int(r["rank_reimbursement"]),
            "cohort_size": int(len(s.scores)), "decision_threshold": s.bundle.threshold,
            "above_decision_threshold": bool(r["fraud_probability"] >= s.bundle.threshold),
        },
        "recommendation": recommendation(r["audit_priority"]),
        "billing": {f: py(r[f]) for f in [
            "total_claims", "inpatient_claims", "total_reimbursed", "mean_reimbursed", "median_reimbursed",
            "max_reimbursed", "cv_reimbursed", "total_deductible_paid", "total_claim_days", "mean_claim_duration",
            "unique_beneficiaries", "claims_per_beneficiary", "reimbursement_per_beneficiary", "active_months",
            "unique_attending_physicians", "inpatient_mean_reimbursed", "outpatient_mean_reimbursed"] if f in r},
        "comparisons": comps,
        "trend": [{"month": m, "claims": int(c), "reimbursed": float(v), "inpatient_claims": int(i)}
                  for m, c, v, i in zip(trend["month"], trend["claims"], trend["reimbursed"], trend["inpatient_claims"])],
        "claim_mix": [{"setting": a, "bucket": b, "claims": int(c)} for a, b, c in zip(mix["setting"], mix["bucket"], mix["claims"])],
        "anomaly": {
            "regression": {f: py(r[f]) for f in ["mean_residual", "median_residual", "positive_residual_ratio",
                                                  "total_positive_deviation", "excess_reimbursement_ratio"]},
            "clustering": {"peer_group": int(r["peer_group"]), "dbscan_outlier": bool(r["dbscan_outlier"]),
                           "dbscan_core_distance": py(r["dbscan_core_distance"]),
                           "cluster_centroid_distance": py(r["cluster_centroid_distance"]),
                           "pca": [py(r["pca_1"]), py(r["pca_2"])]},
            "association": {"suspicious_itemset_claims": int(r["suspicious_itemset_claims"]),
                            "suspicious_itemset_variety": int(r["suspicious_itemset_variety"]),
                            "suspicious_itemset_claim_ratio": py(r["suspicious_itemset_claim_ratio"]),
                            "itemsets": s.reports["provider_rule_hits"].get(pid, [])},
        },
        "explanation": expl,
        "explanation_method": "Change in calibrated probability when each feature is replaced by the training "
                              "median (one feature at a time). Positive values raise the estimate.",
        "investigation": repo().get(pid),
        "evaluation_mode": q.evaluation_mode,
        "claim_period": {"start": months["claim_start_min"], "end": months["claim_start_max"]},
    }
    if q.evaluation_mode == "retrospective":
        body["retrospective"] = {
            "fraud_label": int(r["retrospective_fraud_label"]),
            "note": "Historical provider-level label from the dataset. Shown only in retrospective evaluation "
                    "mode; it is never an input to the model or the audit ranking.",
        }
    return jsonify(body)


@api.get("/providers/<provider_id>/status")
def get_status(provider_id: str):
    s = store()
    if s.provider(provider_id) is None:
        raise ApiError(404, f"Provider {provider_id!r} is not in the scoring cohort")
    return jsonify(repo().get(provider_id))


@api.route("/providers/<provider_id>/status", methods=["PATCH", "PUT"])
def set_status(provider_id: str):
    s = store()
    if s.provider(provider_id) is None:
        raise ApiError(404, f"Provider {provider_id!r} is not in the scoring cohort")
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise ApiError(400, "Request body must be a JSON object")
    upd = StatusUpdate(**body)
    return jsonify(repo().set_status(provider_id, upd.status, upd.note, s.manifest["model_version"]))


@api.get("/model-performance")
def model_performance():
    s = store()
    return jsonify({"model": model_info(s), "split": s.reports["split"], **s.reports["model_performance"],
                    "ablation": s.reports["ablation"]})


@api.get("/analysis")
def analysis():
    s = store()
    return jsonify({"regression": s.reports["regression"], "clustering": s.reports["clustering"],
                    "association_rules": s.reports["association_rules"]})


@api.get("/methodology")
def methodology():
    s = store()
    return jsonify({"manifest": s.manifest, "data_report": s.reports["data_report"], "split": s.reports["split"],
                    "leakage_checks": s.reports["leakage_checks"], "feature_catalog": s.reports["feature_catalog"],
                    "statuses": list(INVESTIGATION_STATUSES)})


def _example_inputs(s) -> list[dict]:
    feats = s.bundle.input_features()
    sc = s.scores
    picks = [("Highest estimated exposure", sc.iloc[0]),
             ("Median estimated exposure", sc.iloc[len(sc) // 2]),
             ("Lowest fraud probability", sc.loc[sc["fraud_probability"].idxmin()])]
    out = []
    for title, r in picks:
        out.append({"title": title, "provider_id": r["Provider"],
                    "note": f"Feature values of held-out provider {r['Provider']} (an actual record), for illustration.",
                    "features": {f: (None if py(r[f]) is None else round(py(r[f]), 4)) for f in feats}})
    return out


@api.get("/predict/schema")
def predict_schema():
    s = store()
    b = s.bundle
    return jsonify({"model_version": s.manifest["model_version"], "model_name": b.model_name,
                    "decision_threshold": b.threshold, "features": b.input_schema(),
                    "computed_server_side": [describe(f) for f in b.feature_names if f not in b.input_features()],
                    "examples": _example_inputs(s)})


@api.post("/predict")
def predict():
    s = store()
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise ApiError(400, "Request body must be a JSON object of the form {\"features\": {...}}")
    req = PredictRequest(**body)
    out = predict_from_inputs(s.bundle, req.features)
    rank = s.exposure_rank(out["estimated_exposure"])
    n = len(s.scores)
    tier = str(priority_tier(np.array([rank]), np.array([out["risk_level"]]), n)[0])
    out.update({"cohort_rank": rank, "cohort_size": n, "audit_priority": tier,
                "recommendation": recommendation(tier)})
    return jsonify({k: py(v) if not isinstance(v, (list, dict)) else v for k, v in out.items()})
