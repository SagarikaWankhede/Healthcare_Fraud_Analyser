import pandas as pd
import numpy as np
from typing import Tuple

CHRONIC_COND_COLS = [
    'ChronicCond_Alzheimer', 'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease',
    'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression',
    'ChronicCond_Diabetes', 'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis',
    'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke'
]

DIAG_COLS = [f'ClmDiagnosisCode_{i}' for i in range(1, 11)]
PROC_COLS = [f'ClmProcedureCode_{i}' for i in range(1, 7)]

def extract_provider_features(claims_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms claim-level data into a comprehensive provider-level feature matrix.
    Computes billing volume, reimbursement distribution, patient demographics,
    physician involvement, diagnostic/procedure intensity, and duration metrics.

    Args:
        claims_df (pd.DataFrame): Merged claims and beneficiary records.
        labels_df (pd.DataFrame): Provider IDs with target 'PotentialFraud'.

    Returns:
        provider_df (pd.DataFrame): Provider-level feature dataset.
    """
    print("Engineering provider-level aggregated features...")
    df = claims_df.copy()

    # Precompute claim-level counts
    df['diag_count'] = df[DIAG_COLS].notna().sum(axis=1)
    df['proc_count'] = df[PROC_COLS].notna().sum(axis=1)
    df['has_procedure'] = (df['proc_count'] > 0).astype(int)
    
    # Same physician indicator
    df['same_physician'] = (
        (df['AttendingPhysician'].notna()) &
        (df['AttendingPhysician'] == df['OperatingPhysician'])
    ).astype(int)

    # 1. Volume & Financial Aggregations
    agg_dict = {
        'ClaimID': ['count'],
        'is_inpatient': ['sum', 'mean'],
        'InscClaimAmtReimbursed': ['sum', 'mean', 'median', 'std', 'max'],
        'DeductibleAmtPaid': ['sum', 'mean'],
        'ClaimDurationDays': ['mean', 'max', 'sum'],
        'StayDurationDays': ['mean', 'max'],
        'AgeAtClaim': ['mean', 'std'],
        'is_deceased': ['sum', 'mean'],
        'RenalDiseaseIndicator': ['mean'],
        'ChronicCond_Count': ['mean', 'std'],
        'diag_count': ['mean', 'max'],
        'proc_count': ['mean', 'max'],
        'has_procedure': ['mean'],
        'same_physician': ['sum', 'mean']
    }

    # Add chronic conditions means
    for col in CHRONIC_COND_COLS:
        if col in df.columns:
            agg_dict[col] = ['mean']

    grouped = df.groupby('Provider').agg(agg_dict)
    
    # Flatten multi-level columns
    grouped.columns = ['_'.join(c).strip() for c in grouped.columns.values]
    grouped = grouped.reset_index()

    # Rename key columns for readability
    rename_map = {
        'ClaimID_count': 'total_claims',
        'is_inpatient_sum': 'inpatient_claims',
        'is_inpatient_mean': 'inpatient_ratio',
        'InscClaimAmtReimbursed_sum': 'total_reimbursed',
        'InscClaimAmtReimbursed_mean': 'mean_reimbursed',
        'InscClaimAmtReimbursed_median': 'median_reimbursed',
        'InscClaimAmtReimbursed_std': 'std_reimbursed',
        'InscClaimAmtReimbursed_max': 'max_reimbursed',
        'DeductibleAmtPaid_sum': 'total_deductible_paid',
        'DeductibleAmtPaid_mean': 'mean_deductible_paid',
        'ClaimDurationDays_mean': 'mean_claim_duration',
        'ClaimDurationDays_max': 'max_claim_duration',
        'ClaimDurationDays_sum': 'total_claim_days',
        'StayDurationDays_mean': 'mean_stay_duration',
        'StayDurationDays_max': 'max_stay_duration',
        'AgeAtClaim_mean': 'mean_patient_age',
        'AgeAtClaim_std': 'std_patient_age',
        'is_deceased_sum': 'deceased_patients_count',
        'is_deceased_mean': 'deceased_patients_ratio',
        'RenalDiseaseIndicator_mean': 'renal_disease_ratio',
        'ChronicCond_Count_mean': 'mean_chronic_conditions',
        'ChronicCond_Count_std': 'std_chronic_conditions',
        'diag_count_mean': 'mean_diag_count',
        'diag_count_max': 'max_diag_count',
        'proc_count_mean': 'mean_proc_count',
        'proc_count_max': 'max_proc_count',
        'has_procedure_mean': 'procedure_claim_ratio',
        'same_physician_sum': 'same_physician_claims',
        'same_physician_mean': 'same_physician_ratio'
    }
    grouped = grouped.rename(columns=rename_map)

    # 2. Distinct Entity Counts (Beneficiaries, Physicians)
    entity_counts = df.groupby('Provider').agg({
        'BeneID': 'nunique',
        'AttendingPhysician': 'nunique',
        'OperatingPhysician': 'nunique',
        'OtherPhysician': 'nunique'
    }).reset_index().rename(columns={
        'BeneID': 'unique_beneficiaries',
        'AttendingPhysician': 'unique_attending_physicians',
        'OperatingPhysician': 'unique_operating_physicians',
        'OtherPhysician': 'unique_other_physicians'
    })

    provider_df = grouped.merge(entity_counts, on='Provider', how='left')

    # 3. Ratio Features
    provider_df['claims_per_beneficiary'] = provider_df['total_claims'] / provider_df['unique_beneficiaries'].clip(lower=1)
    provider_df['claims_per_attending_doctor'] = provider_df['total_claims'] / provider_df['unique_attending_physicians'].clip(lower=1)
    provider_df['daily_reimbursement_rate'] = provider_df['total_reimbursed'] / provider_df['total_claim_days'].clip(lower=1)
    provider_df['reimbursement_per_patient'] = provider_df['total_reimbursed'] / provider_df['unique_beneficiaries'].clip(lower=1)

    # Clean missing values resulting from single-claim std
    provider_df['std_reimbursed'] = provider_df['std_reimbursed'].fillna(0.0)
    provider_df['std_patient_age'] = provider_df['std_patient_age'].fillna(0.0)
    provider_df['std_chronic_conditions'] = provider_df['std_chronic_conditions'].fillna(0.0)

    # Merge labels
    provider_df = provider_df.merge(labels_df, on='Provider', how='left')

    print(f"Engineered {provider_df.shape[1] - 2} features for {len(provider_df)} providers.")
    return provider_df

if __name__ == '__main__':
    from src.data_loader import load_raw_data
    claims, labels = load_raw_data()
    prov_features = extract_provider_features(claims, labels)
    print("Features preview:\n", prov_features.head(3))
