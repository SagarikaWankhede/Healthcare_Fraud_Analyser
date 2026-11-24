"""Human-readable feature catalog: label, group, kind (drives validation and formatting)."""
from __future__ import annotations

from mediguard.features.provider_features import DERIVED_RATIOS

# kind: count | amount | signed_amount | ratio | days | years | binary | score
_CATALOG: dict[str, tuple[str, str, str]] = {
    # Billing behaviour
    "total_claims": ("Total claims", "billing", "count"),
    "inpatient_claims": ("Inpatient claims", "billing", "count"),
    "inpatient_ratio": ("Inpatient share of claims", "billing", "ratio"),
    "total_reimbursed": ("Total reimbursement", "billing", "amount"),
    "mean_reimbursed": ("Mean reimbursement per claim", "billing", "amount"),
    "median_reimbursed": ("Median reimbursement per claim", "billing", "amount"),
    "std_reimbursed": ("Std. dev. of claim reimbursement", "billing", "amount"),
    "cv_reimbursed": ("Billing variability (coefficient of variation)", "billing", "score"),
    "max_reimbursed": ("Largest single claim", "billing", "amount"),
    "zero_reimbursement_ratio": ("Share of $0 claims", "billing", "ratio"),
    "inpatient_mean_reimbursed": ("Mean inpatient claim", "billing", "amount"),
    "outpatient_mean_reimbursed": ("Mean outpatient claim", "billing", "amount"),
    "total_deductible_paid": ("Total deductible paid", "billing", "amount"),
    "deductible_missing_claims": ("Claims with unknown deductible", "billing", "count"),
    "reimbursement_per_beneficiary": ("Reimbursement per beneficiary", "billing", "amount"),
    "total_claim_days": ("Total claim days", "billing", "days"),
    "mean_claim_duration": ("Mean claim duration", "billing", "days"),
    "max_claim_duration": ("Longest claim duration", "billing", "days"),
    "mean_inpatient_stay": ("Mean inpatient stay", "billing", "days"),
    "active_months": ("Months with claims", "billing", "count"),
    "claims_per_active_month": ("Claims per active month", "billing", "score"),
    # Patients
    "unique_beneficiaries": ("Unique beneficiaries", "patients", "count"),
    "claims_per_beneficiary": ("Claims per beneficiary", "patients", "score"),
    "max_claims_single_beneficiary": ("Most claims for one beneficiary", "patients", "count"),
    "mean_patient_age": ("Mean patient age", "patients", "years"),
    "deceased_patient_ratio": ("Share of claims for deceased patients", "patients", "ratio"),
    "renal_disease_ratio": ("Share with renal disease", "patients", "ratio"),
    "mean_chronic_conditions": ("Mean chronic conditions per patient", "patients", "score"),
    "ratio_alzheimer": ("Share with Alzheimer's", "patients", "ratio"),
    "ratio_heartfailure": ("Share with heart failure", "patients", "ratio"),
    "ratio_kidneydisease": ("Share with kidney disease", "patients", "ratio"),
    "ratio_cancer": ("Share with cancer", "patients", "ratio"),
    "ratio_obstrpulmonary": ("Share with COPD", "patients", "ratio"),
    "ratio_depression": ("Share with depression", "patients", "ratio"),
    "ratio_diabetes": ("Share with diabetes", "patients", "ratio"),
    "ratio_ischemicheart": ("Share with ischemic heart disease", "patients", "ratio"),
    "ratio_osteoporasis": ("Share with osteoporosis", "patients", "ratio"),
    "ratio_rheumatoidarthritis": ("Share with rheumatoid arthritis", "patients", "ratio"),
    "ratio_stroke": ("Share with stroke", "patients", "ratio"),
    # Physicians
    "unique_attending_physicians": ("Unique attending physicians", "physicians", "count"),
    "unique_operating_physicians": ("Unique operating physicians", "physicians", "count"),
    "unique_other_physicians": ("Unique other physicians", "physicians", "count"),
    "claims_per_attending_physician": ("Claims per attending physician", "physicians", "score"),
    "missing_attending_ratio": ("Share of claims without attending physician", "physicians", "ratio"),
    "same_attending_operating_ratio": ("Share where attending = operating physician", "physicians", "ratio"),
    "physician_shared_ratio": ("Share of physicians also billing elsewhere", "physicians", "ratio"),
    # Coding
    "mean_diag_count": ("Mean diagnosis codes per claim", "coding", "score"),
    "mean_proc_count": ("Mean procedure codes per claim", "coding", "score"),
    "procedure_claim_ratio": ("Share of claims with a procedure", "coding", "ratio"),
    "unique_diagnosis_codes": ("Distinct diagnosis codes billed", "coding", "count"),
    # Regression
    "mean_residual": ("Mean reimbursement residual", "regression", "signed_amount"),
    "median_residual": ("Median reimbursement residual", "regression", "signed_amount"),
    "positive_residual_ratio": ("Share of claims above predicted", "regression", "ratio"),
    "total_positive_deviation": ("Total reimbursement above predicted", "regression", "amount"),
    "excess_reimbursement_ratio": ("Above-predicted share of reimbursement", "regression", "ratio"),
    # Clustering
    "dbscan_outlier": ("DBSCAN outlier", "clustering", "binary"),
    "dbscan_core_distance": ("Distance to nearest dense region (x eps)", "clustering", "score"),
    "cluster_centroid_distance": ("Distance to peer-group centre", "clustering", "score"),
    # Association rules
    "suspicious_itemset_claims": ("Claims with flagged code combinations", "association", "count"),
    "suspicious_itemset_variety": ("Distinct flagged code combinations", "association", "count"),
    "suspicious_itemset_claim_ratio": ("Share of claims with flagged combinations", "association", "ratio"),
}

GROUP_LABELS = {
    "billing": "Billing behaviour", "patients": "Patient mix", "physicians": "Physician patterns",
    "coding": "Clinical coding", "regression": "Reimbursement deviation (regression)",
    "clustering": "Peer-group anomaly (clustering)", "association": "Code combinations (association rules)",
}

DERIVED = set(DERIVED_RATIOS) | {"cv_reimbursed"}


def describe(feature: str) -> dict:
    if feature.startswith("peer_group_"):
        k = feature.rsplit("_", 1)[1]
        return {"feature": feature, "label": f"Peer group {k} membership", "group": "clustering",
                "kind": "binary", "derived": False}
    label, group, kind = _CATALOG.get(feature, (feature.replace("_", " ").capitalize(), "other", "score"))
    return {"feature": feature, "label": label, "group": group, "kind": kind, "derived": feature in DERIVED}


def catalog(features: list[str]) -> list[dict]:
    return [describe(f) for f in features]
