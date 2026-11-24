"""Provider-level aggregation of claims.

The fraud label is defined per provider, so the modelling unit is one row per provider.
Everything in this module is a deterministic aggregation of that provider's own claims and
**never touches the fraud label**. Ratio features are produced by :func:`add_derived_features`,
which the prediction API reuses so that user-entered inputs are transformed identically.

Race and gender are deliberately excluded from the model features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mediguard.data.ingest import CHRONIC_COLS, DIAG_COLS

# Derived features: name -> (numerator, denominator). Denominators are clipped at 1 (counts) so
# a provider with e.g. zero recorded physicians does not produce inf.
DERIVED_RATIOS: dict[str, tuple[str, str]] = {
    "mean_reimbursed": ("total_reimbursed", "total_claims"),
    "inpatient_ratio": ("inpatient_claims", "total_claims"),
    "claims_per_beneficiary": ("total_claims", "unique_beneficiaries"),
    "reimbursement_per_beneficiary": ("total_reimbursed", "unique_beneficiaries"),
    "claims_per_attending_physician": ("total_claims", "unique_attending_physicians"),
    "claims_per_active_month": ("total_claims", "active_months"),
    "suspicious_itemset_claim_ratio": ("suspicious_itemset_claims", "total_claims"),
}


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ratio features from primary aggregates (shared by training and inference)."""
    df = df.copy()
    for name, (num, den) in DERIVED_RATIOS.items():
        if num in df.columns and den in df.columns:
            df[name] = df[num] / df[den].clip(lower=1)
    if {"std_reimbursed", "mean_reimbursed"} <= set(df.columns):
        df["cv_reimbursed"] = df["std_reimbursed"] / df["mean_reimbursed"].clip(lower=1.0)
    return df


def aggregate_provider_features(claims: pd.DataFrame) -> pd.DataFrame:
    """One row per provider, indexed by ``Provider``. Label-free."""
    c = claims
    g = c.groupby("Provider", sort=True)

    out = pd.DataFrame(index=g.size().index)
    out["total_claims"] = g.size()
    out["inpatient_claims"] = g["is_inpatient"].sum()
    # Billing
    out["total_reimbursed"] = g["InscClaimAmtReimbursed"].sum()
    out["median_reimbursed"] = g["InscClaimAmtReimbursed"].median()
    out["std_reimbursed"] = g["InscClaimAmtReimbursed"].std().fillna(0.0)
    out["max_reimbursed"] = g["InscClaimAmtReimbursed"].max()
    out["zero_reimbursement_ratio"] = g["zero_reimbursement"].mean()
    ip = c[c["is_inpatient"] == 1].groupby("Provider")["InscClaimAmtReimbursed"]
    op = c[c["is_inpatient"] == 0].groupby("Provider")["InscClaimAmtReimbursed"]
    out["inpatient_mean_reimbursed"] = ip.mean()          # NaN when the provider has no inpatient claims
    out["outpatient_mean_reimbursed"] = op.mean()         # NaN when no outpatient claims
    out["total_deductible_paid"] = g["DeductibleAmtPaid"].sum(min_count=1).fillna(0.0)
    out["deductible_missing_claims"] = g["deductible_missing"].sum()
    # Utilisation / duration
    out["total_claim_days"] = g["ClaimDurationDays"].sum()
    out["mean_claim_duration"] = g["ClaimDurationDays"].mean()
    out["max_claim_duration"] = g["ClaimDurationDays"].max()
    out["mean_inpatient_stay"] = c[c["is_inpatient"] == 1].groupby("Provider")["StayDurationDays"].mean()
    month = c["ClaimStartDt"].dt.to_period("M")
    out["active_months"] = month.groupby(c["Provider"]).nunique()
    # Beneficiaries
    out["unique_beneficiaries"] = g["BeneID"].nunique()
    per_bene = c.groupby(["Provider", "BeneID"]).size()
    out["max_claims_single_beneficiary"] = per_bene.groupby(level=0).max()
    out["mean_patient_age"] = g["AgeAtClaim"].mean()
    out["deceased_patient_ratio"] = g["is_deceased"].mean()
    out["renal_disease_ratio"] = g["RenalDiseaseIndicator"].mean()
    out["mean_chronic_conditions"] = g["chronic_condition_count"].mean()
    for col in CHRONIC_COLS:
        out[f"ratio_{col.replace('ChronicCond_', '').lower()}"] = g[col].mean()
    # Physicians
    out["unique_attending_physicians"] = g["AttendingPhysician"].nunique()
    out["unique_operating_physicians"] = g["OperatingPhysician"].nunique()
    out["unique_other_physicians"] = g["OtherPhysician"].nunique()
    out["missing_attending_ratio"] = g["missing_attending_physician"].mean()
    out["same_attending_operating_ratio"] = g["same_attending_operating"].mean()
    out["physician_shared_ratio"] = _physician_overlap(c)
    # Clinical coding intensity
    out["mean_diag_count"] = g["diag_count"].mean()
    out["mean_proc_count"] = g["proc_count"].mean()
    out["procedure_claim_ratio"] = g["has_procedure"].mean()
    out["unique_diagnosis_codes"] = _unique_codes_per_provider(c, DIAG_COLS)

    out = add_derived_features(out)
    out = out.astype(float)
    out.index.name = "Provider"
    return out


def _unique_codes_per_provider(c: pd.DataFrame, cols: list[str]) -> pd.Series:
    stacked = c[["Provider", *cols]].set_index("Provider")[cols].stack()
    return stacked.groupby(level=0).nunique()


def _physician_overlap(c: pd.DataFrame) -> pd.Series:
    """Share of a provider's attending physicians who also appear on claims of another provider.

    A label-free network indicator computed from claims only (no fraud labels).
    """
    pairs = c[["Provider", "AttendingPhysician"]].dropna().drop_duplicates()
    providers_per_doc = pairs.groupby("AttendingPhysician")["Provider"].nunique()
    pairs = pairs.assign(shared=pairs["AttendingPhysician"].map(providers_per_doc).gt(1).astype(float))
    return pairs.groupby("Provider")["shared"].mean()
