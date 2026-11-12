import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Tuple, Dict

CLAIM_REGRESSION_FEATURES = [
    'is_inpatient', 'ClaimDurationDays', 'StayDurationDays', 'DeductibleAmtPaid',
    'AgeAtClaim', 'RenalDiseaseIndicator', 'ChronicCond_Count',
    'diag_count', 'proc_count', 'has_procedure',
    'ChronicCond_Alzheimer', 'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease',
    'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression',
    'ChronicCond_Diabetes', 'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis',
    'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke'
]

def train_reimbursement_regressor(train_claims: pd.DataFrame) -> Tuple[HistGradientBoostingRegressor, Dict[str, float]]:
    """
    Trains a regression model on claim-level data to predict standard expected
    reimbursement amounts given patient demographics and medical service intensity.

    Args:
        train_claims (pd.DataFrame): Claims belonging to training providers.

    Returns:
        model: Fitted HistGradientBoostingRegressor.
        metrics: Dictionary of training performance metrics (R2, RMSE, MAE).
    """
    print(f"Training claim reimbursement regression model on {len(train_claims):,} claims...")
    X_train = train_claims[CLAIM_REGRESSION_FEATURES].fillna(0.0).copy()
    y_train = train_claims['InscClaimAmtReimbursed'].copy()

    regressor = HistGradientBoostingRegressor(
        max_iter=100,
        learning_rate=0.1,
        max_depth=8,
        random_state=42
    )
    regressor.fit(X_train, y_train)

    preds = regressor.predict(X_train)
    r2 = r2_score(y_train, preds)
    rmse = np.sqrt(mean_squared_error(y_train, preds))
    mae = mean_absolute_error(y_train, preds)

    metrics = {'R2': float(r2), 'RMSE': float(rmse), 'MAE': float(mae)}
    print(f"Claim Reimbursement Regressor trained - R2: {r2:.4f}, RMSE: ${rmse:,.2f}, MAE: ${mae:,.2f}")
    return regressor, metrics


def compute_provider_residuals(claims_df: pd.DataFrame, regressor: HistGradientBoostingRegressor) -> pd.DataFrame:
    """
    Predicts expected claim amounts and computes overbilling residuals at the provider level.

    Args:
        claims_df (pd.DataFrame): Claims dataset (train or test).
        regressor: Trained reimbursement model.

    Returns:
        provider_residuals (pd.DataFrame): Provider ID mapped to overbilling features.
    """
    X = claims_df[CLAIM_REGRESSION_FEATURES].fillna(0.0).copy()
    y_actual = claims_df['InscClaimAmtReimbursed'].values
    y_pred = regressor.predict(X)

    residuals = y_actual - y_pred
    positive_residuals = np.maximum(0, residuals)

    res_df = pd.DataFrame({
        'Provider': claims_df['Provider'],
        'residual': residuals,
        'is_positive_residual': (residuals > 0).astype(int),
        'positive_residual_amount': positive_residuals,
        'actual_reimbursed': y_actual
    })

    grouped = res_df.groupby('Provider').agg({
        'residual': ['mean', 'median', 'max'],
        'is_positive_residual': ['mean'],
        'positive_residual_amount': ['sum'],
        'actual_reimbursed': ['sum']
    })

    grouped.columns = ['_'.join(c).strip() for c in grouped.columns.values]
    grouped = grouped.reset_index()

    grouped = grouped.rename(columns={
        'residual_mean': 'mean_claim_residual',
        'residual_median': 'median_claim_residual',
        'residual_max': 'max_claim_residual',
        'is_positive_residual_mean': 'positive_residual_ratio',
        'positive_residual_amount_sum': 'total_excess_reimbursement',
        'actual_reimbursed_sum': 'provider_reimbursed_sum'
    })

    grouped['excess_reimbursement_ratio'] = (
        grouped['total_excess_reimbursement'] / grouped['provider_reimbursed_sum'].clip(lower=1.0)
    ).clip(lower=0.0, upper=1.0)

    # Drop temporary sum column to avoid collision
    grouped = grouped.drop(columns=['provider_reimbursed_sum'])
    return grouped
