"""Technique 1 - Regression: reimbursement deviation analysis.

A claim-level regressor estimates the reimbursement *expected* for a claim given the patient and
treatment information on it (inpatient flag, DRG, primary diagnosis category, length of stay,
coding intensity, chronic conditions, age, coverage months). Then

    residual = actual reimbursement - predicted reimbursement

is aggregated per provider. A persistently positive residual means a provider is paid more than
comparable claims elsewhere; it is an anomaly signal, **not proof of fraud** (it also reflects
case-mix the model cannot see).

Leakage controls
----------------
* Inputs exclude the target and anything computed from it (e.g. beneficiary annual reimbursement
  totals, which include the claim itself) and exclude the provider ID and fraud label.
* The regressor and its category encoders are fitted on training providers' claims only.
* Residuals for *training* providers come from out-of-fold predictions (GroupKFold by provider),
  so they are out-of-sample just like residuals for validation/test providers. Otherwise the
  classifier would learn from optimistically small in-sample residuals.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from mediguard.data.ingest import CHRONIC_COLS

log = logging.getLogger(__name__)

NUMERIC_INPUTS = [
    "is_inpatient", "ClaimDurationDays", "StayDurationDays", "DeductibleAmtPaid", "deductible_missing",
    "AgeAtClaim", "RenalDiseaseIndicator", "chronic_condition_count", "diag_count", "proc_count",
    "has_procedure", "NoOfMonths_PartACov", "NoOfMonths_PartBCov", *CHRONIC_COLS,
]
CATEGORICAL_INPUTS = {
    "drg": "DiagnosisGroupCode",               # inpatient payment group
    "dx_category": "ClmDiagnosisCode_1",       # ICD-9 3-character category of the primary diagnosis
    "admit_dx_category": "ClmAdmitDiagnosisCode",
}
TARGET = "InscClaimAmtReimbursed"
RESIDUAL_FEATURES = [
    "mean_residual", "median_residual", "positive_residual_ratio",
    "total_positive_deviation", "excess_reimbursement_ratio",
]


class _TopCategoryEncoder:
    """Maps a string column to integer codes for its most frequent categories (fitted on train)."""

    def __init__(self, max_categories: int = 250):
        self.max_categories = max_categories
        self.mapping_: dict[str, int] = {}

    def fit(self, s: pd.Series) -> "_TopCategoryEncoder":
        top = s.dropna().value_counts().index[: self.max_categories - 1]
        self.mapping_ = {cat: i for i, cat in enumerate(top)}
        return self

    def transform(self, s: pd.Series) -> np.ndarray:
        out = pd.to_numeric(s.astype(object).map(self.mapping_), errors="coerce").to_numpy(dtype=float, copy=True)
        unseen = s.notna().to_numpy(dtype=bool) & np.isnan(out)
        out[unseen] = len(self.mapping_)                            # unseen category -> "other"
        return out                                                  # missing stays NaN


def _category_source(claims: pd.DataFrame, col: str) -> pd.Series:
    s = claims[col]
    if col in ("ClmDiagnosisCode_1", "ClmAdmitDiagnosisCode"):
        s = s.str.slice(0, 3)
    return s


class ReimbursementRegressor:
    def __init__(self, max_iter: int = 300, learning_rate: float = 0.08, max_leaf_nodes: int = 63,
                 random_state: int = 42):
        self.params = dict(max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
                           random_state=random_state)
        self.encoders_: dict[str, _TopCategoryEncoder] = {}
        self.model_: HistGradientBoostingRegressor | None = None

    def _matrix(self, claims: pd.DataFrame) -> pd.DataFrame:
        X = claims[NUMERIC_INPUTS].astype(float).copy()
        for name, col in CATEGORICAL_INPUTS.items():
            X[name] = self.encoders_[name].transform(_category_source(claims, col))
        return X

    def fit(self, claims: pd.DataFrame) -> "ReimbursementRegressor":
        for name, col in CATEGORICAL_INPUTS.items():
            self.encoders_[name] = _TopCategoryEncoder().fit(_category_source(claims, col))
        X = self._matrix(claims)
        cat_mask = [c in CATEGORICAL_INPUTS for c in X.columns]
        self.model_ = HistGradientBoostingRegressor(categorical_features=cat_mask, **self.params)
        self.model_.fit(X, claims[TARGET].to_numpy(dtype=float))
        return self

    def predict(self, claims: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("ReimbursementRegressor is not fitted")
        return np.clip(self.model_.predict(self._matrix(claims)), 0.0, None)

    @staticmethod
    def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
            "n_claims": int(len(y_true)),
        }


def out_of_fold_predictions(claims: pd.DataFrame, n_splits: int, **params) -> np.ndarray:
    """Predictions for every claim from a model that never saw that claim's provider."""
    oof = np.zeros(len(claims))
    groups = claims["Provider"].to_numpy()
    for train_idx, hold_idx in GroupKFold(n_splits=n_splits).split(claims, groups=groups):
        reg = ReimbursementRegressor(**params).fit(claims.iloc[train_idx])
        oof[hold_idx] = reg.predict(claims.iloc[hold_idx])
    return oof


def provider_residual_features(claims: pd.DataFrame, predicted: np.ndarray) -> pd.DataFrame:
    actual = claims[TARGET].to_numpy(dtype=float)
    resid = actual - predicted
    frame = pd.DataFrame({
        "Provider": claims["Provider"].to_numpy(),
        "residual": resid,
        "positive": (resid > 0).astype(float),
        "positive_amount": np.clip(resid, 0, None),
        "actual": actual,
    })
    g = frame.groupby("Provider", sort=True)
    out = pd.DataFrame({
        "mean_residual": g["residual"].mean(),
        "median_residual": g["residual"].median(),
        "positive_residual_ratio": g["positive"].mean(),
        "total_positive_deviation": g["positive_amount"].sum(),
    })
    out["excess_reimbursement_ratio"] = (out["total_positive_deviation"] / g["actual"].sum().clip(lower=1.0)).clip(0, 1)
    return out[RESIDUAL_FEATURES]
