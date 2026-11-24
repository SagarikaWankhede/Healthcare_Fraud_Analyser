"""Raw data ingestion with schema validation.

The Kaggle "Healthcare Provider Fraud Detection Analysis" training files are read as-is from
``data/raw``. Raw files are never modified; everything derived is written elsewhere.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

RAW_FILES = {
    "labels": "Train-1542865627584.csv",
    "inpatient": "Train_Inpatientdata-1542865627584.csv",
    "outpatient": "Train_Outpatientdata-1542865627584.csv",
    "beneficiary": "Train_Beneficiarydata-1542865627584.csv",
}

DIAG_COLS = [f"ClmDiagnosisCode_{i}" for i in range(1, 11)]
PROC_COLS = [f"ClmProcedureCode_{i}" for i in range(1, 7)]
CHRONIC_COLS = [
    "ChronicCond_Alzheimer", "ChronicCond_Heartfailure", "ChronicCond_KidneyDisease",
    "ChronicCond_Cancer", "ChronicCond_ObstrPulmonary", "ChronicCond_Depression",
    "ChronicCond_Diabetes", "ChronicCond_IschemicHeart", "ChronicCond_Osteoporasis",
    "ChronicCond_rheumatoidarthritis", "ChronicCond_stroke",
]

_CLAIM_COMMON = [
    "BeneID", "ClaimID", "ClaimStartDt", "ClaimEndDt", "Provider", "InscClaimAmtReimbursed",
    "AttendingPhysician", "OperatingPhysician", "OtherPhysician", "ClmAdmitDiagnosisCode",
    "DeductibleAmtPaid", *DIAG_COLS, *PROC_COLS,
]
REQUIRED_COLUMNS = {
    "labels": ["Provider", "PotentialFraud"],
    "inpatient": _CLAIM_COMMON + ["AdmissionDt", "DischargeDt", "DiagnosisGroupCode"],
    "outpatient": _CLAIM_COMMON,
    "beneficiary": ["BeneID", "DOB", "DOD", "Gender", "Race", "RenalDiseaseIndicator",
                    "NoOfMonths_PartACov", "NoOfMonths_PartBCov", *CHRONIC_COLS],
}

# Codes are identifiers, not numbers: read them as strings so "0389" keeps its leading zero.
_STRING_COLS = {c: "string" for c in [
    "BeneID", "ClaimID", "Provider", "AttendingPhysician", "OperatingPhysician", "OtherPhysician",
    "ClmAdmitDiagnosisCode", "DiagnosisGroupCode", *DIAG_COLS, *PROC_COLS,
]}


class SchemaError(ValueError):
    """Raised when a raw file does not match the expected schema."""


@dataclass
class RawTables:
    labels: pd.DataFrame
    inpatient: pd.DataFrame
    outpatient: pd.DataFrame
    beneficiary: pd.DataFrame


def validate_columns(df: pd.DataFrame, table: str) -> None:
    missing = [c for c in REQUIRED_COLUMNS[table] if c not in df.columns]
    if missing:
        raise SchemaError(f"{table}: missing required columns {missing}")


def _read(path: Path, table: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Raw file not found: {path}. Download the Kaggle dataset "
            "'Healthcare Provider Fraud Detection Analysis' into data/raw/ (see README)."
        )
    header = pd.read_csv(path, nrows=0).columns
    dtypes = {c: t for c, t in _STRING_COLS.items() if c in header}
    df = pd.read_csv(path, dtype=dtypes, low_memory=False)
    validate_columns(df, table)
    log.info("Loaded %-11s %s rows=%d cols=%d", table, path.name, len(df), df.shape[1])
    return df


def load_raw_tables(raw_dir: Path) -> RawTables:
    raw_dir = Path(raw_dir)
    tables = {name: _read(raw_dir / fname, name) for name, fname in RAW_FILES.items()}
    return RawTables(**tables)
