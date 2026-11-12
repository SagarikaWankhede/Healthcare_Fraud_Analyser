import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from typing import Dict, Tuple, List

def evaluate_audit_strategies(
    test_df: pd.DataFrame,
    fraud_probabilities: np.ndarray,
    budget_k: int = 100,
    random_trials: int = 100,
    output_dir: str = 'outputs'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Simulates and compares competing provider audit strategies under a fixed budget K:
    1. Proposed ML Expected-Recovery System (Fraud Prob * Total Reimbursement)
    2. Pure Risk Ranking (Fraud Prob only)
    3. Highest Billers First (Traditional Heuristic)
    4. Random Selection (Chance Baseline)

    Args:
        test_df (pd.DataFrame): Test providers with 'Provider', 'total_reimbursed', 'PotentialFraud'.
        fraud_probabilities (np.ndarray): Predicted fraud risk for each test provider.
        budget_k (int): Number of providers the audit team can investigate (default: 100).
        random_trials (int): Number of Monte Carlo runs for random selection.

    Returns:
        manifest_df: Top K audit recommendations for the proposed system.
        comparison_df: Benchmark metrics across the 4 strategies at budget K.
        curves_df: Cumulative recovery data across budget sizes [1..N].
    """
    df = test_df.copy().reset_index(drop=True)
    df['fraud_prob'] = fraud_probabilities
    df['expected_fraud_dollars'] = df['fraud_prob'] * df['total_reimbursed']
    
    # Total fraud dollars in the entire held-out test partition
    total_fraud_dollars = (df['total_reimbursed'] * df['PotentialFraud']).sum()
    total_fraud_providers = df['PotentialFraud'].sum()
    total_providers_test = len(df)

    print(f"\nHeld-Out Test Evaluation Set:")
    print(f"Total Providers: {total_providers_test:,} | Fraudulent Providers: {total_fraud_providers:,} ({total_fraud_providers/total_providers_test:.1%})")
    print(f"Total Test Reimbursement: ${df['total_reimbursed'].sum():,.2f}")
    print(f"Total Known Fraudulent Reimbursement: ${total_fraud_dollars:,.2f} ({(total_fraud_dollars/df['total_reimbursed'].sum()):.1%} of test money)")

    # 1. Proposed Strategy: Expected Fraud Recovery (Prob * Dollars)
    df_expected = df.sort_values(by='expected_fraud_dollars', ascending=False).reset_index(drop=True)
    
    # 2. Probability-Only Strategy
    df_prob = df.sort_values(by='fraud_prob', ascending=False).reset_index(drop=True)

    # 3. Highest Billers Strategy
    df_billers = df.sort_values(by='total_reimbursed', ascending=False).reset_index(drop=True)

    # Calculate cumulative recoveries across all budget thresholds
    max_k = min(250, total_providers_test)
    ks = np.arange(1, max_k + 1)

    rec_expected = [
        (df_expected.iloc[:k]['total_reimbursed'] * df_expected.iloc[:k]['PotentialFraud']).sum()
        for k in ks
    ]
    rec_prob = [
        (df_prob.iloc[:k]['total_reimbursed'] * df_prob.iloc[:k]['PotentialFraud']).sum()
        for k in ks
    ]
    rec_billers = [
        (df_billers.iloc[:k]['total_reimbursed'] * df_billers.iloc[:k]['PotentialFraud']).sum()
        for k in ks
    ]

    # Monte Carlo simulation for random selection
    np.random.seed(42)
    random_recs = np.zeros((random_trials, max_k))
    for t in range(random_trials):
        rand_order = np.random.permutation(total_providers_test)
        df_rand = df.iloc[rand_order]
        cum_rand = (df_rand['total_reimbursed'].values * df_rand['PotentialFraud'].values).cumsum()
        random_recs[t, :] = cum_rand[:max_k]
    
    rec_random_mean = random_recs.mean(axis=0)

    # Build comparison at exact budget K
    def summarize_strategy(name, subset_df):
        fraud_caught = subset_df['PotentialFraud'].sum()
        dollars_rec = (subset_df['total_reimbursed'] * subset_df['PotentialFraud']).sum()
        total_dollars_audited = subset_df['total_reimbursed'].sum()
        capture_rate = (dollars_rec / total_fraud_dollars) * 100.0
        precision_at_k = (fraud_caught / len(subset_df)) * 100.0
        dollar_yield = (dollars_rec / max(total_dollars_audited, 1.0)) * 100.0

        return {
            'Strategy': name,
            f'Fraud Providers Caught (of {total_fraud_providers})': f"{int(fraud_caught)} / {total_fraud_providers}",
            'Precision @ K': f"{precision_at_k:.1f}%",
            'Fraud Dollars Recovered': f"${dollars_rec:,.2f}",
            'Fraud Dollar Capture Rate': f"{capture_rate:.2f}%",
            'Total Audited Payout': f"${total_dollars_audited:,.2f}",
            'Audit Efficiency Yield': f"{dollar_yield:.1f}%"
        }

    comp_rows = []
    comp_rows.append(summarize_strategy("Proposed ML System (Expected Recovery)", df_expected.iloc[:budget_k]))
    comp_rows.append(summarize_strategy("Highest Billers First (Naive Business Heuristic)", df_billers.iloc[:budget_k]))
    comp_rows.append(summarize_strategy("Pure Fraud Risk Only (Probability-Based)", df_prob.iloc[:budget_k]))

    # Random baseline at budget K
    rand_dollars = rec_random_mean[budget_k - 1]
    rand_capture = (rand_dollars / total_fraud_dollars) * 100.0
    rand_providers = (total_fraud_providers / total_providers_test) * budget_k
    comp_rows.append({
        'Strategy': 'Random Selection (Chance Baseline)',
        f'Fraud Providers Caught (of {total_fraud_providers})': f"{rand_providers:.1f} / {total_fraud_providers}",
        'Precision @ K': f"{(rand_providers/budget_k)*100:.1f}%",
        'Fraud Dollars Recovered': f"${rand_dollars:,.2f}",
        'Fraud Dollar Capture Rate': f"{rand_capture:.2f}%",
        'Total Audited Payout': f"${(df['total_reimbursed'].mean() * budget_k):,.2f}",
        'Audit Efficiency Yield': f"{(rand_dollars / (df['total_reimbursed'].mean() * budget_k))*100:.1f}%"
    })

    comparison_df = pd.DataFrame(comp_rows)

    # Plot Audit Recovery Curves
    plt.figure(figsize=(11, 7))
    plt.plot(ks, [(r / total_fraud_dollars) * 100 for r in rec_expected], 'royalblue', lw=3, label='Proposed ML Prioritization (Expected Recovery)')
    plt.plot(ks, [(r / total_fraud_dollars) * 100 for r in rec_billers], 'darkorange', lw=2.2, linestyle='--', label='Highest Billers First (Volume Heuristic)')
    plt.plot(ks, [(r / total_fraud_dollars) * 100 for r in rec_prob], 'mediumseagreen', lw=2.2, linestyle='-.', label='Pure Risk Ranking (Probability-Only)')
    plt.plot(ks, [(r / total_fraud_dollars) * 100 for r in rec_random_mean], 'gray', lw=1.8, linestyle=':', label='Random Selection (Chance Baseline)')

    plt.axvline(x=budget_k, color='red', linestyle=':', lw=1.5, label=f'Audit Budget = {budget_k} Providers')
    plt.title(f"Cumulative Fraud Dollars Recovered vs. Audit Budget (K Providers)", fontsize=13, fontweight='bold')
    plt.xlabel("Number of Healthcare Providers Audited (Budget K)", fontsize=11)
    plt.ylabel("% of Total Fraudulent Reimbursement Recovered", fontsize=11)
    plt.ylim(0, 105)
    plt.xlim(1, max_k)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    curve_path = os.path.join(output_dir, 'figures', 'recovery_curve_comparison.png')
    os.makedirs(os.path.dirname(curve_path), exist_ok=True)
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(f"Saved recovery curve comparison to {curve_path}")

    # Build Top K Actionable Manifest
    manifest_df = df_expected.iloc[:budget_k].copy()
    manifest_df['AuditRank'] = range(1, len(manifest_df) + 1)
    
    # Assign primary anomaly drivers / reason codes
    def determine_reason_code(row):
        reasons = []
        if row.get('dbscan_outlier_flag', 0) == 1:
            reasons.append("Behavioral Density Outlier (DBSCAN)")
        if row.get('positive_residual_ratio', 0) > 0.6:
            reasons.append("Persistent Overbilling Residuals")
        if row.get('suspicious_code_claim_count', 0) > 5:
            reasons.append("Suspicious Diagnosis/Proc Combos (Apriori)")
        if row.get('same_physician_ratio', 0) > 0.5:
            reasons.append("High Same-Physician Billing Ratio")
        if not reasons:
            reasons.append("High Aggregated Risk & Claim Volume")
        return "; ".join(reasons)

    manifest_df['PrimaryAnomalyDrivers'] = manifest_df.apply(determine_reason_code, axis=1)

    export_cols = [
        'AuditRank', 'Provider', 'fraud_prob', 'total_reimbursed',
        'expected_fraud_dollars', 'total_claims', 'unique_beneficiaries',
        'mean_claim_residual', 'dbscan_outlier_flag', 'suspicious_code_claim_count',
        'PotentialFraud', 'PrimaryAnomalyDrivers'
    ]
    available_cols = [c for c in export_cols if c in manifest_df.columns]
    manifest_export = manifest_df[available_cols]

    manifest_path = os.path.join(output_dir, 'reports', f'top_{budget_k}_audit_manifest.csv')
    manifest_export.to_csv(manifest_path, index=False)
    print(f"Saved actionable Top {budget_k} audit manifest to {manifest_path}")

    comp_path = os.path.join(output_dir, 'reports', 'audit_strategy_comparison.csv')
    comparison_df.to_csv(comp_path, index=False)
    print(f"Saved audit strategy comparison to {comp_path}")

    curves_df = pd.DataFrame({
        'Budget_K': ks,
        'Proposed_ML_Capture_Pct': [(r / total_fraud_dollars) * 100 for r in rec_expected],
        'Highest_Billers_Capture_Pct': [(r / total_fraud_dollars) * 100 for r in rec_billers],
        'Probability_Only_Capture_Pct': [(r / total_fraud_dollars) * 100 for r in rec_prob],
        'Random_Capture_Pct': [(r / total_fraud_dollars) * 100 for r in rec_random_mean]
    })

    return manifest_export, comparison_df, curves_df
