"""Claim-level preprocessing.

Every transformation here is deterministic and row-local (no statistics are learned from the
data), so it is safe to apply to all claims before the provider-level train/test split.
Anything that *learns* from data (scalers, imputers, regressors, clustering, rule mining)
lives in ``mediguard.features`` and is fitted on training providers only.

Documented transformations
--------------------------
* Labels: ``PotentialFraud`` Yes/No -> 1/0. Unknown values raise.
* Claims: inpatient and outpatient tables are stacked with an ``is_inpatient`` flag.
  Duplicate ``ClaimID`` rows are dropped (count reported).
* Reimbursement: never imputed (no missing values in the source). Zero-dollar claims are kept
  and flagged; negative values would be rejected as invalid.
* Deductible: 899 inpatient claims have no ``DeductibleAmtPaid``. All other inpatient claims
  carry exactly the 2009 Part A deductible ($1,068), so the value is *unknown*, not zero. It is
  left as NaN (gradient-boosted models handle missing natively) and a
  ``deductible_missing`` indicator is added. It is never silently replaced with 0.
* Dates: parsed; ``ClaimDurationDays`` = end - start + 1. Claims with end < start are flagged.
  ``StayDurationDays`` = discharge - admission + 1 for inpatient claims, 0 for outpatient
  (outpatient claims have no inpatient stay by definition).
* Codes: diagnosis/procedure codes are treated as strings (``"7092.0"`` -> ``"7092"``).
* Beneficiaries: chronic-condition flags 1/2 -> 1/0, renal ``Y``/``0`` -> 1/0, age at claim from
  DOB, ``claim_after_death`` when a claim starts after the recorded date of death.
  ``IP/OPAnnualReimbursementAmt`` are dropped: they are annual totals that *include the claim's
  own reimbursement*, so using them to predict reimbursement would leak the target.
  ``Race``/``Gender`` are retained for data-quality reporting but excluded from all models.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from mediguard.data.ingest import CHRONIC_COLS, DIAG_COLS, PROC_COLS, RawTables

log = logging.getLogger(__name__)

LEAKY_BENEFICIARY_COLS = ["IPAnnualReimbursementAmt", "IPAnnualDeductibleAmt",
                          "OPAnnualReimbursementAmt", "OPAnnualDeductibleAmt"]


def clean_labels(labels: pd.DataFrame) -> pd.DataFrame:
    labels = labels[["Provider", "PotentialFraud"]].copy()
    mapped = labels["PotentialFraud"].map({"Yes": 1, "No": 0})
    if mapped.isna().any():
        bad = labels.loc[mapped.isna(), "PotentialFraud"].unique().tolist()
        raise ValueError(f"Unexpected PotentialFraud values: {bad}")
    if labels["Provider"].duplicated().any():
        raise ValueError("Duplicate provider IDs in label file")
    labels["PotentialFraud"] = mapped.astype(int)
    return labels.reset_index(drop=True)


def _normalise_code(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    return s.mask(s == "", pd.NA)


def clean_beneficiaries(bene: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    bene = bene.copy()
    report = {
        "beneficiaries": int(len(bene)),
        "duplicate_bene_ids": int(bene["BeneID"].duplicated().sum()),
        "negative_annual_reimbursement_rows": int(
            sum((bene[c] < 0).sum() for c in ["IPAnnualReimbursementAmt", "OPAnnualReimbursementAmt"] if c in bene)
        ),
        "deceased_beneficiaries": int(bene["DOD"].notna().sum()),
    }
    bene = bene.drop_duplicates("BeneID")
    bene["DOB"] = pd.to_datetime(bene["DOB"], errors="coerce")
    bene["DOD"] = pd.to_datetime(bene["DOD"], errors="coerce")
    bene["is_deceased"] = bene["DOD"].notna().astype(np.int8)
    bene["RenalDiseaseIndicator"] = (bene["RenalDiseaseIndicator"].astype(str).str.upper() == "Y").astype(np.int8)
    for col in CHRONIC_COLS:
        bene[col] = (bene[col] == 1).astype(np.int8)
    bene["chronic_condition_count"] = bene[CHRONIC_COLS].sum(axis=1).astype(np.int8)
    bene = bene.drop(columns=[c for c in LEAKY_BENEFICIARY_COLS if c in bene.columns])
    return bene, report


def build_claims(raw: RawTables) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (claims, labels, data_quality_report)."""
    labels = clean_labels(raw.labels)

    ip = raw.inpatient.copy()
    op = raw.outpatient.copy()
    ip["is_inpatient"] = np.int8(1)
    op["is_inpatient"] = np.int8(0)
    claims = pd.concat([ip, op], ignore_index=True, sort=False)

    report: dict = {
        "inpatient_claims": int(len(ip)),
        "outpatient_claims": int(len(op)),
        "raw_claims": int(len(claims)),
        "providers_in_label_file": int(len(labels)),
        "fraud_labelled_providers": int(labels["PotentialFraud"].sum()),
    }

    dupes = claims["ClaimID"].duplicated()
    report["duplicate_claim_ids_dropped"] = int(dupes.sum())
    claims = claims.loc[~dupes].copy()

    # Referential integrity
    unknown_providers = set(claims["Provider"]) - set(labels["Provider"])
    report["claims_with_unlabelled_provider"] = int(claims["Provider"].isin(unknown_providers).sum())
    claims = claims[~claims["Provider"].isin(unknown_providers)]
    report["labelled_providers_without_claims"] = int(len(set(labels["Provider"]) - set(claims["Provider"])))

    # Reimbursement
    reimb = claims["InscClaimAmtReimbursed"]
    report["missing_reimbursement"] = int(reimb.isna().sum())
    report["negative_reimbursement_dropped"] = int((reimb < 0).sum())
    claims = claims[~(reimb < 0) & reimb.notna()].copy()
    claims["zero_reimbursement"] = (claims["InscClaimAmtReimbursed"] == 0).astype(np.int8)
    report["zero_reimbursement_claims"] = int(claims["zero_reimbursement"].sum())

    # Deductible: unknown, not zero
    claims["deductible_missing"] = claims["DeductibleAmtPaid"].isna().astype(np.int8)
    report["missing_deductible_claims"] = int(claims["deductible_missing"].sum())
    report["missing_deductible_inpatient"] = int(claims.loc[claims["is_inpatient"] == 1, "deductible_missing"].sum())

    # Dates
    for col in ["ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt"]:
        claims[col] = pd.to_datetime(claims[col], errors="coerce")
    report["unparseable_claim_dates"] = int(claims["ClaimStartDt"].isna().sum() + claims["ClaimEndDt"].isna().sum())
    claims["ClaimDurationDays"] = (claims["ClaimEndDt"] - claims["ClaimStartDt"]).dt.days + 1
    claims["invalid_claim_dates"] = (claims["ClaimDurationDays"] < 1).astype(np.int8)
    report["claims_end_before_start"] = int(claims["invalid_claim_dates"].sum())
    claims["ClaimDurationDays"] = claims["ClaimDurationDays"].clip(lower=1)
    stay = (claims["DischargeDt"] - claims["AdmissionDt"]).dt.days + 1
    claims["StayDurationDays"] = np.where(claims["is_inpatient"] == 1, stay.clip(lower=1), 0).astype(float)
    report["claim_start_min"] = str(claims["ClaimStartDt"].min().date())
    report["claim_start_max"] = str(claims["ClaimStartDt"].max().date())

    # Codes
    for col in DIAG_COLS + PROC_COLS + ["ClmAdmitDiagnosisCode", "DiagnosisGroupCode"]:
        claims[col] = _normalise_code(claims[col])
    claims["diag_count"] = claims[DIAG_COLS].notna().sum(axis=1).astype(np.int8)
    claims["proc_count"] = claims[PROC_COLS].notna().sum(axis=1).astype(np.int8)
    claims["has_procedure"] = (claims["proc_count"] > 0).astype(np.int8)
    claims["missing_attending_physician"] = claims["AttendingPhysician"].isna().astype(np.int8)
    claims["same_attending_operating"] = (
        claims["AttendingPhysician"].notna() & (claims["AttendingPhysician"] == claims["OperatingPhysician"])
    ).fillna(False).astype(np.int8)

    # Beneficiaries
    bene, bene_report = clean_beneficiaries(raw.beneficiary)
    report.update(bene_report)
    before = len(claims)
    claims = claims.merge(bene, on="BeneID", how="left", validate="many_to_one")
    assert len(claims) == before, "beneficiary join changed the number of claims"
    report["claims_without_beneficiary_record"] = int(claims["DOB"].isna().sum())
    claims["AgeAtClaim"] = ((claims["ClaimStartDt"] - claims["DOB"]).dt.days / 365.25).clip(lower=0, upper=120)
    claims["claim_after_death"] = (claims["DOD"].notna() & (claims["ClaimStartDt"] > claims["DOD"])).astype(np.int8)
    report["claims_after_beneficiary_death"] = int(claims["claim_after_death"].sum())

    claims = claims.reset_index(drop=True)
    report["claims"] = int(len(claims))
    report["providers"] = int(claims["Provider"].nunique())
    report["fraud_prevalence"] = float(labels["PotentialFraud"].mean())
    log.info("Preprocessed %d claims across %d providers", report["claims"], report["providers"])
    return claims, labels, report
