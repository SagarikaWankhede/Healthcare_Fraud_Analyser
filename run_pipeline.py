"""
Master Execution Pipeline: Healthcare Provider Fraud Detection & Audit Prioritization System
"""
import os
import sys
import time

# Configure local matplotlib cache directory
os.environ['MPLCONFIGDIR'] = os.path.join(os.getcwd(), '.matplotlib_cache')
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Import project modules
from src.data_loader import load_raw_data
from src.feature_engineering import extract_provider_features
from src.regression_residuals import train_reimbursement_regressor, compute_provider_residuals
from src.clustering import ProviderClusterAnalyzer
from src.apriori_mining import SuspiciousCodeMiner
from src.classifiers import train_and_evaluate_classifiers
from src.audit_ranking import evaluate_audit_strategies
from src.ablation import run_ablation_study

def main():
    start_time = time.time()
    print("=" * 80)
    print("HEALTHCARE PROVIDER FRAUD DETECTION & AUDIT PRIORITIZATION SYSTEM")
    print("=" * 80)

    # 1. Load Data
    claims_df, labels_df = load_raw_data('data/raw')
    print(f"Total Claims: {len(claims_df):,} | Total Providers: {len(labels_df):,}")

    # 2. Strict Provider-Level Train/Test Split (80/20)
    print("\n--- Step 1: Leakage-Free Provider-Level Train/Test Partition ---")
    train_labels, test_labels = train_test_split(
        labels_df,
        test_size=0.20,
        stratify=labels_df['PotentialFraud'],
        random_state=42
    )
    train_providers = set(train_labels['Provider'])
    test_providers = set(test_labels['Provider'])
    print(f"Train Providers: {len(train_providers):,} (Fraud: {train_labels['PotentialFraud'].sum():,})")
    print(f"Test Providers:  {len(test_providers):,} (Fraud: {test_labels['PotentialFraud'].sum():,})")

    train_claims = claims_df[claims_df['Provider'].isin(train_providers)].copy()
    test_claims = claims_df[claims_df['Provider'].isin(test_providers)].copy()
    print(f"Train Claims: {len(train_claims):,} | Test Claims: {len(test_claims):,}")

    # 3. Base Provider-Level Feature Engineering
    print("\n--- Step 2: Extracting Base Provider Aggregated Features ---")
    train_base_features = extract_provider_features(train_claims, train_labels)
    test_base_features = extract_provider_features(test_claims, test_labels)

    # 4. Technique A: Claim Reimbursement Regression & Overbilling Residuals
    print("\n--- Step 3: Technique A - Regression (Claim Cost & Overbilling Residuals) ---")
    regressor, reg_metrics = train_reimbursement_regressor(train_claims)
    train_residuals = compute_provider_residuals(train_claims, regressor)
    test_residuals = compute_provider_residuals(test_claims, regressor)

    # 5. Technique B: Provider Clustering (DBSCAN Outliers & Hierarchical Typology)
    print("\n--- Step 4: Technique B - Clustering (Hierarchical & DBSCAN) ---")
    cluster_analyzer = ProviderClusterAnalyzer(n_clusters=4, dbscan_eps=2.5, dbscan_min_samples=5)
    cluster_analyzer.fit(train_base_features)
    train_clusters = cluster_analyzer.transform(train_base_features)
    test_clusters = cluster_analyzer.transform(test_base_features)
    cluster_analyzer.plot_clusters(train_base_features, 'outputs/figures/clustering_outliers_tsne.png')

    # 6. Technique C: Association Rule Mining (Apriori on Diagnosis & Procedure Codes)
    print("\n--- Step 5: Technique C - Association Rules (Apriori Code Pairs) ---")
    code_miner = SuspiciousCodeMiner(min_fraud_support=0.002, min_enrichment_ratio=1.8, top_k_rules=30)
    code_miner.fit(train_claims, train_labels)
    train_apriori = code_miner.transform(train_claims, list(train_providers))
    test_apriori = code_miner.transform(test_claims, list(test_providers))

    # 7. Merge All Enriched Features
    print("\n--- Step 6: Consolidating Multi-Technique Feature Space ---")
    def merge_all_features(base_df, res_df, clust_df, apriori_df):
        m = base_df.merge(res_df, on='Provider', how='left')
        m = m.merge(clust_df, on='Provider', how='left')
        m = m.merge(apriori_df, on='Provider', how='left')
        return m

    train_enriched = merge_all_features(train_base_features, train_residuals, train_clusters, train_apriori)
    test_enriched = merge_all_features(test_base_features, test_residuals, test_clusters, test_apriori)

    # Save processed datasets
    os.makedirs('data/processed', exist_ok=True)
    train_enriched.to_csv('data/processed/train_providers_enriched.csv', index=False)
    test_enriched.to_csv('data/processed/test_providers_enriched.csv', index=False)
    print(f"Saved processed feature tables ({train_enriched.shape[1]} columns) to data/processed/")

    # Define Feature Sets for Modeling and Ablation
    ignore_cols = {'Provider', 'PotentialFraud'}
    all_feature_cols = [c for c in train_enriched.columns if c not in ignore_cols]
    
    residual_cols = list(train_residuals.columns.drop('Provider'))
    cluster_cols = list(train_clusters.columns.drop('Provider'))
    apriori_cols = list(train_apriori.columns.drop('Provider'))
    base_cols = [c for c in train_base_features.columns if c not in ignore_cols]

    feature_sets = {
        '1. Base Billing & Demographics Only': base_cols,
        '2. Base + Regression Overbilling Residuals': base_cols + residual_cols,
        '3. Base + Regression + Clustering (DBSCAN/Hierarchical)': base_cols + residual_cols + cluster_cols,
        '4. Full Pipeline (+ Apriori Suspicious Codes)': all_feature_cols
    }

    # 8. Train and Benchmark Classification Models
    print("\n--- Step 7: Classification Models Benchmark ---")
    X_train = train_enriched[all_feature_cols].fillna(0.0)
    y_train = train_enriched['PotentialFraud'].values
    X_test = test_enriched[all_feature_cols].fillna(0.0)
    y_test = test_enriched['PotentialFraud'].values

    models, benchmark_df, probabilities = train_and_evaluate_classifiers(
        X_train, y_train, X_test, y_test, output_dir='outputs'
    )
    print("\nModel Benchmark Performance on Held-Out Test Set:")
    print(benchmark_df.to_string(index=False))

    # 9. Audit Prioritization & Expected Dollar Recovery Simulation
    print("\n--- Step 8: Audit Prioritization & Dollar Recovery Showdown (Novelty) ---")
    best_model_name = benchmark_df.iloc[0]['Model']
    best_probs = probabilities[best_model_name]

    manifest_df, strategy_comp_df, curves_df = evaluate_audit_strategies(
        test_enriched,
        best_probs,
        budget_k=100,
        random_trials=100,
        output_dir='outputs'
    )
    print("\nAudit Strategy Comparison under 100-Provider Fixed Budget:")
    print(strategy_comp_df.to_string(index=False))

    # 10. Run Ablation Study
    print("\n--- Step 9: Multi-Technique Ablation Study ---")
    ablation_df = run_ablation_study(
        train_enriched,
        test_enriched,
        feature_sets,
        budget_k=100,
        output_dir='outputs'
    )
    print("\nAblation Study Summary:")
    print(ablation_df.to_string(index=False))

    # Extract Key Numbers for Resume Bullet
    proposed_row = strategy_comp_df[strategy_comp_df['Strategy'].str.contains('Proposed ML')].iloc[0]
    random_row = strategy_comp_df[strategy_comp_df['Strategy'].str.contains('Random')].iloc[0]
    billers_row = strategy_comp_df[strategy_comp_df['Strategy'].str.contains('Highest Billers')].iloc[0]

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETE (Elapsed: {:.1f}s)".format(elapsed))
    print("=" * 80)
    print(f"\nRESUME BULLET METRICS:")
    print(f"Proposed ML System:  {proposed_row['Fraud Dollar Capture Rate']} of fraud dollars captured ({proposed_row['Fraud Dollars Recovered']})")
    print(f"Highest Billers:     {billers_row['Fraud Dollar Capture Rate']} of fraud dollars captured ({billers_row['Fraud Dollars Recovered']})")
    print(f"Random Selection:    {random_row['Fraud Dollar Capture Rate']} of fraud dollars captured ({random_row['Fraud Dollars Recovered']})")
    print(f"Top 100 Precision:   {proposed_row['Precision @ K']} true fraud providers caught in top 100")
    print("=" * 80)

if __name__ == '__main__':
    main()
