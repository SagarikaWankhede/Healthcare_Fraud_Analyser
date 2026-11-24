"""Inference with the persisted model bundle.

The bundle holds everything inference needs, all fitted during training:
the classifier ``Pipeline`` (median imputer -> [transform] -> model), the Platt calibrator, the
decision threshold, the ordered feature schema, training medians (explanation reference) and the
fitted provider clusterer. ``predict_from_inputs`` validates user inputs, recomputes derived
ratios with the same function used in training, derives clustering features with the fitted
clusterer and leaves anything not supplied as NaN so the *training-fitted* imputer fills it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mediguard.audit.prioritization import risk_level
from mediguard.features.catalog import DERIVED, describe
from mediguard.features.clustering import ProviderClusterer
from mediguard.features.provider_features import add_derived_features
from mediguard.models.explain import reference_substitution

CLUSTER_OUTPUTS_PREFIX = ("dbscan_outlier", "dbscan_core_distance", "cluster_centroid_distance", "peer_group_")
REQUIRED_INPUTS = ("total_claims", "total_reimbursed", "unique_beneficiaries")


class InputValidationError(ValueError):
    def __init__(self, errors: list[dict]):
        super().__init__("; ".join(f"{e['field']}: {e['message']}" for e in errors))
        self.errors = errors


@dataclass
class ModelBundle:
    model_key: str
    model_name: str
    params: dict
    pipeline: object
    calibrator: object
    threshold: float
    feature_names: list[str]
    reference: pd.Series
    bounds: dict
    clusterer: ProviderClusterer
    model_version: str
    trained_at: str
    selection_criterion: str
    metadata: dict = field(default_factory=dict)

    # -- prediction -----------------------------------------------------------------------------
    def frame(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [f for f in self.feature_names if f not in X.columns]
        if missing:
            raise KeyError(f"missing model features: {missing}")
        return X[self.feature_names].astype(float)

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(self.frame(X))[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.calibrator.transform(self.predict_raw(X))

    # -- schema for the API / UI ------------------------------------------------------------------
    def input_features(self) -> list[str]:
        return [f for f in self.feature_names
                if f not in DERIVED and not f.startswith(CLUSTER_OUTPUTS_PREFIX)]

    def input_schema(self) -> list[dict]:
        out = []
        for f in self.input_features():
            d = describe(f)
            b = self.bounds.get(f, {})
            d.update({"required": f in REQUIRED_INPUTS, "train_min": b.get("min"), "train_max": b.get("max"),
                      "train_median": _clean(self.reference.get(f))})
            out.append(d)
        return out


def _clean(v):
    if v is None:
        return None
    v = float(v)
    return None if math.isnan(v) else v


def validate_inputs(bundle: ModelBundle, inputs: dict) -> tuple[dict, list[str]]:
    """Return (clean numeric inputs, warnings) or raise InputValidationError."""
    errors, warnings, clean = [], [], {}
    allowed = set(bundle.input_features())
    for key, raw in inputs.items():
        if key in DERIVED or key.startswith(CLUSTER_OUTPUTS_PREFIX):
            errors.append({"field": key, "message": "is computed by the server and cannot be supplied"})
            continue
        if key not in allowed:
            errors.append({"field": key, "message": "is not a model input feature"})
            continue
        if raw is None or raw == "":
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            errors.append({"field": key, "message": "must be a number"})
            continue
        if not math.isfinite(v):
            errors.append({"field": key, "message": "must be finite"})
            continue
        kind = describe(key)["kind"]
        if kind in ("count", "amount", "days", "score") and v < 0:
            errors.append({"field": key, "message": "must be >= 0"})
        elif kind == "count" and abs(v - round(v)) > 1e-9:
            errors.append({"field": key, "message": "must be a whole number"})
        elif kind == "ratio" and not 0 <= v <= 1:
            errors.append({"field": key, "message": "must be between 0 and 1"})
        elif kind == "binary" and v not in (0, 1):
            errors.append({"field": key, "message": "must be 0 or 1"})
        elif kind == "years" and not 0 <= v <= 120:
            errors.append({"field": key, "message": "must be between 0 and 120"})
        clean[key] = v
    for req in REQUIRED_INPUTS:
        if req not in clean and not any(e["field"] == req for e in errors):
            errors.append({"field": req, "message": "is required"})
    if "total_claims" in clean and clean["total_claims"] < 1:
        errors.append({"field": "total_claims", "message": "must be at least 1"})
    n = clean.get("total_claims")
    if n is not None:
        for f in ("unique_beneficiaries", "inpatient_claims", "suspicious_itemset_claims",
                  "max_claims_single_beneficiary", "deductible_missing_claims"):
            if f in clean and clean[f] > n:
                errors.append({"field": f, "message": "cannot exceed total claims"})
    tr = clean.get("total_reimbursed")
    if tr is not None:
        for f in ("max_reimbursed", "median_reimbursed", "total_positive_deviation"):
            if f in clean and clean[f] > tr + 1e-6:
                errors.append({"field": f, "message": "cannot exceed total reimbursement"})
    if errors:
        raise InputValidationError(errors)
    for f, v in clean.items():
        b = bundle.bounds.get(f)
        if b and b.get("min") is not None and (v < b["min"] or v > b["max"]):
            warnings.append(f"{describe(f)['label']} ({v:g}) is outside the range seen in training "
                            f"[{b['min']:g}, {b['max']:g}]; the prediction is an extrapolation.")
    return clean, warnings


def build_feature_frame(bundle: ModelBundle, clean: dict) -> tuple[pd.DataFrame, list[str], dict]:
    row = pd.DataFrame([{f: np.nan for f in bundle.feature_names}])
    for k, v in clean.items():
        row[k] = v
    row = add_derived_features(row)
    # Clustering features from the fitted clusterer (inputs it needs are imputed with its own
    # training medians when absent).
    cl = bundle.clusterer.transform(row)
    for c in cl.columns:
        if c in row.columns:
            row[c] = cl[c].to_numpy()
    assigned = bundle.clusterer.assign(row).iloc[0]
    computed = {
        "peer_group": int(assigned["peer_group"]), "dbscan_outlier": int(assigned["dbscan_outlier"]),
        "dbscan_core_distance": float(assigned["dbscan_core_distance"]),
    }
    for f in DERIVED:
        if f in row.columns and not np.isnan(row.at[0, f]):
            computed[f] = float(row.at[0, f])
    row = row[bundle.feature_names]
    imputed = [f for f in bundle.feature_names if np.isnan(row.at[0, f])]
    return row, imputed, computed


def predict_from_inputs(bundle: ModelBundle, inputs: dict) -> dict:
    clean, warnings = validate_inputs(bundle, inputs)
    X, imputed, computed = build_feature_frame(bundle, clean)
    prob = float(bundle.predict(X)[0])
    exposure = prob * clean["total_reimbursed"]
    contributions = reference_substitution(bundle.predict, X, bundle.reference, top_n=8)[0]
    for c in contributions:
        c["label"] = describe(c["feature"])["label"]
        c["kind"] = describe(c["feature"])["kind"]
    return {
        "fraud_probability": prob,
        "raw_model_score": float(bundle.predict_raw(X)[0]),
        "estimated_exposure": exposure,
        "risk_level": str(risk_level(prob)[0]),
        "above_decision_threshold": prob >= bundle.threshold,
        "decision_threshold": bundle.threshold,
        "computed_features": computed,
        "imputed_features": imputed,
        "warnings": warnings,
        "contributions": contributions,
        "model_version": bundle.model_version,
        "model_name": bundle.model_name,
    }
