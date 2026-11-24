"""Synthetic raw data in the exact Kaggle file layout, with a planted fraud signal.

Lets the full pipeline, API and browser tests run in CI without downloading the dataset.
Fraud-labelled providers bill more per claim, have longer claims and over-use a specific
diagnosis pair, so a working pipeline must find signal; a broken one will not.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mediguard.data.ingest import CHRONIC_COLS, RAW_FILES

DX = [f"{c}" for c in ["4019", "25000", "2724", "41401", "4280", "5990", "486", "42731", "2449", "V5869",
                       "2859", "53081", "496", "5849", "2720", "71590", "311", "7802", "78650", "73300"]]
PROC = ["9904", "3893", "8154", "4516", "3995"]


def write_synthetic_raw(raw_dir: Path, n_providers: int = 150, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    raw_dir.mkdir(parents=True, exist_ok=True)
    providers = [f"PRV{60000 + i}" for i in range(n_providers)]
    fraud = (rng.random(n_providers) < 0.2).astype(int)
    fraud[:3] = 1  # guarantee enough positives for stratified splits
    # Labels are noisy (10% flipped) so the task is learnable but not perfectly separable.
    observed = np.where(rng.random(n_providers) < 0.10, 1 - fraud, fraud)
    observed[:3] = 1
    labels = pd.DataFrame({"Provider": providers, "PotentialFraud": np.where(observed == 1, "Yes", "No")})

    n_bene = 900
    benes = [f"BENE{100000 + i}" for i in range(n_bene)]
    dob = pd.to_datetime("1925-01-01") + pd.to_timedelta(rng.integers(0, 365 * 30, n_bene), unit="D")
    dod = np.where(rng.random(n_bene) < 0.02, "2009-11-01", None)
    bene = pd.DataFrame({
        "BeneID": benes, "DOB": dob.strftime("%Y-%m-%d"), "DOD": dod,
        "Gender": rng.integers(1, 3, n_bene), "Race": rng.choice([1, 2, 3, 5], n_bene),
        "RenalDiseaseIndicator": rng.choice(["0", "Y"], n_bene, p=[0.85, 0.15]),
        "State": rng.integers(1, 50, n_bene), "County": rng.integers(1, 900, n_bene),
        "NoOfMonths_PartACov": 12, "NoOfMonths_PartBCov": 12,
        **{c: rng.choice([1, 2], n_bene, p=[0.35, 0.65]) for c in CHRONIC_COLS},
        "IPAnnualReimbursementAmt": rng.integers(-100, 20000, n_bene), "IPAnnualDeductibleAmt": 1068,
        "OPAnnualReimbursementAmt": rng.integers(0, 5000, n_bene), "OPAnnualDeductibleAmt": rng.integers(0, 500, n_bene),
    })

    ip_rows, op_rows = [], []
    cid = 0
    for p, f in zip(providers, fraud):
        n = int(np.clip(rng.lognormal(3.4 + 0.25 * f, 0.7), 5, 250))
        my_benes = rng.choice(benes, size=max(3, n // (2 + 2 * f)), replace=False)
        docs = [f"PHY{rng.integers(1000, 1400)}" for _ in range(rng.integers(1, 6))]
        for _ in range(n):
            cid += 1
            inpatient = rng.random() < (0.10 + 0.15 * f)
            start = pd.Timestamp("2009-01-01") + pd.Timedelta(days=int(rng.integers(0, 360)))
            dur = int(rng.integers(1, 10 if inpatient else 2) + (rng.integers(0, 8) if f and inpatient else 0))
            end = start + pd.Timedelta(days=dur - 1)
            codes = list(rng.choice(DX, size=rng.integers(1, 6), replace=False))
            if f and rng.random() < 0.2:
                codes = ["4280", "5849"] + [c for c in codes if c not in ("4280", "5849")][:3]
            base = (6000 if inpatient else 120) * (1 + 0.15 * len(codes)) * (1.2 if f else 1.0)
            amt = float(np.round(base * rng.lognormal(0, 0.4), -1))
            row = {
                "BeneID": rng.choice(my_benes), "ClaimID": f"CLM{cid}", "ClaimStartDt": start.strftime("%Y-%m-%d"),
                "ClaimEndDt": end.strftime("%Y-%m-%d"), "Provider": p, "InscClaimAmtReimbursed": amt,
                "AttendingPhysician": rng.choice(docs) if rng.random() > 0.01 else None,
                "OperatingPhysician": rng.choice(docs) if rng.random() < 0.3 else None,
                "OtherPhysician": rng.choice(docs) if rng.random() < 0.3 else None,
                "ClmAdmitDiagnosisCode": codes[0] if rng.random() < 0.5 else None,
                **{f"ClmDiagnosisCode_{i + 1}": (codes[i] if i < len(codes) else None) for i in range(10)},
                **{f"ClmProcedureCode_{i + 1}": (float(rng.choice(PROC)) if i == 0 and inpatient and rng.random() < 0.6 else None) for i in range(6)},
            }
            if inpatient:
                row.update({"AdmissionDt": row["ClaimStartDt"], "DischargeDt": row["ClaimEndDt"],
                            "DeductibleAmtPaid": None if rng.random() < 0.03 else 1068.0,
                            "DiagnosisGroupCode": str(rng.integers(100, 140))})
                ip_rows.append(row)
            else:
                row["DeductibleAmtPaid"] = float(rng.choice([0, 0, 0, 10, 50]))
                op_rows.append(row)

    labels.to_csv(raw_dir / RAW_FILES["labels"], index=False)
    pd.DataFrame(ip_rows).to_csv(raw_dir / RAW_FILES["inpatient"], index=False)
    pd.DataFrame(op_rows).to_csv(raw_dir / RAW_FILES["outpatient"], index=False)
    bene.to_csv(raw_dir / RAW_FILES["beneficiary"], index=False)
    return {"providers": n_providers, "fraud": int(observed.sum()), "inpatient": len(ip_rows), "outpatient": len(op_rows)}
