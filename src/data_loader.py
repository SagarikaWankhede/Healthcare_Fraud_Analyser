import os
import pandas as pd
import numpy as np

CHRONIC_COND_COLS = [
    'ChronicCond_Alzheimer', 'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease',
    'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression',
    'ChronicCond_Diabetes', 'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis',
    'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke'
]

def load_raw_data(data_dir: str = 'data/raw'):
    """
    Loads raw CSV files for inpatient, outpatient, beneficiary, and provider labels.
    Performs data cleaning, feature extraction (dates, durations, flags), and merges
    claims with beneficiary demographics.

    Returns:
        claims_df (pd.DataFrame): Consolidated claims with patient features.
        labels_df (pd.DataFrame): Provider IDs with binary target 'PotentialFraud' (0/1).
    """
    labels_path = os.path.join(data_dir, 'Train-1542865627584.csv')
    inpatient_path = os.path.join(data_dir, 'Train_Inpatientdata-1542865627584.csv')
    outpatient_path = os.path.join(data_dir, 'Train_Outpatientdata-1542865627584.csv')
    beneficiary_path = os.path.join(data_dir, 'Train_Beneficiarydata-1542865627584.csv')

    print(f"Loading provider labels from {labels_path}...")
    labels_df = pd.read_csv(labels_path)
    labels_df['PotentialFraud'] = labels_df['PotentialFraud'].map({'Yes': 1, 'No': 0}).astype(int)

    print(f"Loading inpatient claims from {inpatient_path}...")
    inpatient_df = pd.read_csv(inpatient_path)
    inpatient_df['is_inpatient'] = 1

    print(f"Loading outpatient claims from {outpatient_path}...")
    outpatient_df = pd.read_csv(outpatient_path)
    outpatient_df['is_inpatient'] = 0

    # Align columns between inpatient and outpatient
    claims_df = pd.concat([inpatient_df, outpatient_df], ignore_index=True)
    claims_df['InscClaimAmtReimbursed'] = claims_df['InscClaimAmtReimbursed'].fillna(0.0)
    claims_df['DeductibleAmtPaid'] = claims_df['DeductibleAmtPaid'].fillna(0.0)

    # Convert date fields
    date_cols = ['ClaimStartDt', 'ClaimEndDt', 'AdmissionDt', 'DischargeDt']
    for col in date_cols:
        if col in claims_df.columns:
            claims_df[col] = pd.to_datetime(claims_df[col], errors='coerce')

    claims_df['ClaimDurationDays'] = (claims_df['ClaimEndDt'] - claims_df['ClaimStartDt']).dt.days + 1
    claims_df['ClaimDurationDays'] = claims_df['ClaimDurationDays'].fillna(1).clip(lower=1)

    if 'AdmissionDt' in claims_df.columns and 'DischargeDt' in claims_df.columns:
        claims_df['StayDurationDays'] = (claims_df['DischargeDt'] - claims_df['AdmissionDt']).dt.days + 1
        claims_df['StayDurationDays'] = claims_df['StayDurationDays'].fillna(0).clip(lower=0)
    else:
        claims_df['StayDurationDays'] = 0

    print(f"Loading beneficiary data from {beneficiary_path}...")
    bene_df = pd.read_csv(beneficiary_path)
    bene_df['DOB'] = pd.to_datetime(bene_df['DOB'], errors='coerce')
    bene_df['DOD'] = pd.to_datetime(bene_df['DOD'], errors='coerce')
    bene_df['is_deceased'] = bene_df['DOD'].notna().astype(int)
    bene_df['RenalDiseaseIndicator'] = (bene_df['RenalDiseaseIndicator'].astype(str) == 'Y').astype(int)

    # In CMS data: 1 = Yes, 2 = No
    for col in CHRONIC_COND_COLS:
        if col in bene_df.columns:
            bene_df[col] = (bene_df[col] == 1).astype(int)

    bene_df['ChronicCond_Count'] = bene_df[CHRONIC_COND_COLS].sum(axis=1)

    print("Merging claims with beneficiary details...")
    claims_df = claims_df.merge(bene_df, on='BeneID', how='left')

    # Precompute diagnosis and procedure counts
    diag_cols = [f'ClmDiagnosisCode_{i}' for i in range(1, 11) if f'ClmDiagnosisCode_{i}' in claims_df.columns]
    proc_cols = [f'ClmProcedureCode_{i}' for i in range(1, 7) if f'ClmProcedureCode_{i}' in claims_df.columns]
    claims_df['diag_count'] = claims_df[diag_cols].notna().sum(axis=1)
    claims_df['proc_count'] = claims_df[proc_cols].notna().sum(axis=1)
    claims_df['has_procedure'] = (claims_df['proc_count'] > 0).astype(int)

    # Calculate Patient Age at time of claim
    claims_df['AgeAtClaim'] = ((claims_df['ClaimStartDt'] - claims_df['DOB']).dt.days / 365.25).fillna(65.0).clip(lower=0, upper=115)

    print(f"Data loading complete: {len(claims_df):,} claims across {claims_df['Provider'].nunique():,} unique providers.")
    return claims_df, labels_df

if __name__ == '__main__':
    claims, labels = load_raw_data()
    print("Labels distribution:\n", labels['PotentialFraud'].value_counts())
