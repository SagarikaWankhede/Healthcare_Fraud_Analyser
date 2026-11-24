import numpy as np
import pandas as pd
import pytest

from mediguard.data.ingest import RAW_FILES, SchemaError, load_raw_tables
from mediguard.data.preprocess import build_claims, clean_labels
from mediguard.features.catalog import catalog
from mediguard.features.provider_features import DERIVED_RATIOS, add_derived_features, aggregate_provider_features


def test_load_and_counts(synthetic_paths, claims_and_labels):
    claims, labels, report = claims_and_labels
    raw = load_raw_tables(synthetic_paths.raw_dir)
    assert report["claims"] == len(raw.inpatient) + len(raw.outpatient) == len(claims)
    assert report["providers"] == labels["Provider"].nunique() == claims["Provider"].nunique()
    assert set(labels["PotentialFraud"].unique()) <= {0, 1}
    assert report["fraud_labelled_providers"] == int(labels["PotentialFraud"].sum())


def test_missing_raw_file_and_schema_errors(tmp_path, synthetic_paths):
    with pytest.raises(FileNotFoundError):
        load_raw_tables(tmp_path)
    for name, fname in RAW_FILES.items():
        df = pd.read_csv(synthetic_paths.raw_dir / fname)
        df.to_csv(tmp_path / fname, index=False)
    pd.read_csv(tmp_path / RAW_FILES["labels"]).drop(columns="PotentialFraud").to_csv(tmp_path / RAW_FILES["labels"], index=False)
    with pytest.raises(SchemaError):
        load_raw_tables(tmp_path)


def test_label_mapping_rejects_unknown_values():
    with pytest.raises(ValueError):
        clean_labels(pd.DataFrame({"Provider": ["A"], "PotentialFraud": ["Maybe"]}))
    with pytest.raises(ValueError):
        clean_labels(pd.DataFrame({"Provider": ["A", "A"], "PotentialFraud": ["Yes", "No"]}))


def test_join_preserves_claims_and_beneficiary_fields(claims_and_labels):
    claims, _, report = claims_and_labels
    assert claims["ClaimID"].is_unique
    assert report["claims_without_beneficiary_record"] == 0
    assert claims["DOB"].notna().all() and claims["AgeAtClaim"].between(0, 120).all()
    # leaky beneficiary totals are dropped before modelling
    assert not any(c.startswith(("IPAnnual", "OPAnnual")) for c in claims.columns)


def test_missing_deductible_is_not_zero_filled(claims_and_labels):
    claims, _, report = claims_and_labels
    missing = claims["deductible_missing"] == 1
    assert report["missing_deductible_claims"] == int(missing.sum()) > 0
    assert claims.loc[missing, "DeductibleAmtPaid"].isna().all()
    assert (claims.loc[claims["is_inpatient"] == 0, "StayDurationDays"] == 0).all()
    assert (claims["ClaimDurationDays"] >= 1).all()


def test_duplicate_claims_dropped(synthetic_paths, tmp_path):
    for fname in RAW_FILES.values():
        pd.read_csv(synthetic_paths.raw_dir / fname).to_csv(tmp_path / fname, index=False)
    op = pd.read_csv(tmp_path / RAW_FILES["outpatient"])
    pd.concat([op, op.head(5)]).to_csv(tmp_path / RAW_FILES["outpatient"], index=False)
    claims, _, report = build_claims(load_raw_tables(tmp_path))
    assert report["duplicate_claim_ids_dropped"] == 5 and claims["ClaimID"].is_unique


def test_provider_aggregation_by_hand():
    t = pd.Timestamp("2009-03-01")
    claims = pd.DataFrame({
        "Provider": ["P1", "P1", "P1", "P2"], "BeneID": ["B1", "B1", "B2", "B3"], "ClaimID": list("abcd"),
        "InscClaimAmtReimbursed": [100.0, 300.0, 0.0, 50.0], "is_inpatient": [1, 0, 0, 0],
        "zero_reimbursement": [0, 0, 1, 0], "DeductibleAmtPaid": [np.nan, 10.0, 0.0, 0.0], "deductible_missing": [1, 0, 0, 0],
        "ClaimDurationDays": [3, 1, 1, 2], "StayDurationDays": [3.0, 0.0, 0.0, 0.0],
        "ClaimStartDt": [t, t, t + pd.Timedelta(days=40), t], "AgeAtClaim": [70.0, 70.0, 80.0, 60.0],
        "is_deceased": [0, 0, 1, 0], "RenalDiseaseIndicator": [1, 1, 0, 0], "chronic_condition_count": [2, 2, 4, 0],
        **{c: [0, 0, 0, 0] for c in __import__("mediguard.data.ingest", fromlist=["CHRONIC_COLS"]).CHRONIC_COLS},
        "AttendingPhysician": ["D1", "D1", "D2", "D2"], "OperatingPhysician": ["D1", None, None, None], "OtherPhysician": [None] * 4,
        "missing_attending_physician": [0, 0, 0, 0], "same_attending_operating": [1, 0, 0, 0],
        "diag_count": [2, 1, 1, 1], "proc_count": [1, 0, 0, 0], "has_procedure": [1, 0, 0, 0],
        **{f"ClmDiagnosisCode_{i}": (["X", "Y", "X", "Z"] if i == 1 else [None] * 4) for i in range(1, 11)},
    })
    f = aggregate_provider_features(claims)
    p1 = f.loc["P1"]
    assert p1["total_claims"] == 3 and p1["total_reimbursed"] == 400 and p1["unique_beneficiaries"] == 2
    assert p1["mean_reimbursed"] == pytest.approx(400 / 3)
    assert p1["claims_per_beneficiary"] == pytest.approx(1.5)
    assert p1["inpatient_ratio"] == pytest.approx(1 / 3)
    assert p1["total_deductible_paid"] == 10 and p1["deductible_missing_claims"] == 1
    assert p1["active_months"] == 2 and p1["max_claims_single_beneficiary"] == 2
    assert p1["unique_diagnosis_codes"] == 2
    assert p1["physician_shared_ratio"] == pytest.approx(0.5)  # D2 also bills under P2, D1 does not
    assert np.isnan(f.loc["P2", "inpatient_mean_reimbursed"])  # no inpatient claims -> unknown, not 0


def test_feature_schema_has_no_label_or_identifiers(claims_and_labels):
    claims, _, _ = claims_and_labels
    f = aggregate_provider_features(claims)
    banned = {"PotentialFraud", "Provider", "Race", "Gender", "BeneID"}
    assert not banned & set(f.columns)
    assert f.index.is_unique and f.index.name == "Provider"
    assert all(d["group"] != "other" for d in catalog(list(f.columns))), "every feature is documented"


def test_derived_features_match_definition():
    df = pd.DataFrame({"total_claims": [10.0], "total_reimbursed": [500.0], "unique_beneficiaries": [0.0],
                       "inpatient_claims": [2.0], "unique_attending_physicians": [5.0], "active_months": [4.0],
                       "suspicious_itemset_claims": [3.0], "std_reimbursed": [25.0]})
    out = add_derived_features(df)
    for name, (num, den) in DERIVED_RATIOS.items():
        assert out.at[0, name] == pytest.approx(df.at[0, num] / max(df.at[0, den], 1))
    assert out.at[0, "cv_reimbursed"] == pytest.approx(25 / 50)
