import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

def run_ablation_study(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_sets: Dict[str, List[str]],
    budget_k: int = 100,
    output_dir: str = 'outputs'
) -> pd.DataFrame:
    """
    Executes a multi-stage ablation study measuring the incremental impact of each syllabus technique:
    1. Base Aggregated Claims Features
    2. Base + Regression Overbilling Residuals
    3. Base + Regression + Clustering (DBSCAN & Hierarchical)
    4. Full System (+ Apriori Suspicious Code Combinations)

    Evaluates both statistical performance (PR-AUC, ROC-AUC) and business outcome (Top-K Fraud Dollar Capture Rate).
    """
    print("\n--- Running Multi-Technique Ablation Study ---")
    y_train = train_df['PotentialFraud'].values
    y_test = test_df['PotentialFraud'].values
    total_fraud_dollars = (test_df['total_reimbursed'] * test_df['PotentialFraud']).sum()

    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    scale_weight = float(neg_count / max(pos_count, 1))

    ablation_results = []

    for stage_name, cols in feature_sets.items():
        print(f"Evaluating {stage_name} ({len(cols)} features)...")
        # Ensure available columns
        valid_cols = [c for c in cols if c in train_df.columns and c in test_df.columns]
        X_train_subset = train_df[valid_cols].fillna(0.0)
        X_test_subset = test_df[valid_cols].fillna(0.0)

        clf = XGBClassifier(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train_subset, y_train)

        probs = clf.predict_proba(X_test_subset)[:, 1]
        pr_auc = average_precision_score(y_test, probs)
        roc_auc = roc_auc_score(y_test, probs)

        # Expected recovery ranking
        test_eval = test_df[['Provider', 'total_reimbursed', 'PotentialFraud']].copy()
        test_eval['fraud_prob'] = probs
        test_eval['expected_dollars'] = test_eval['fraud_prob'] * test_eval['total_reimbursed']
        test_sorted = test_eval.sort_values(by='expected_dollars', ascending=False).reset_index(drop=True)

        top_k = test_sorted.iloc[:budget_k]
        fraud_caught = int(top_k['PotentialFraud'].sum())
        dollars_rec = float((top_k['total_reimbursed'] * top_k['PotentialFraud']).sum())
        capture_pct = float((dollars_rec / total_fraud_dollars) * 100.0)

        ablation_results.append({
            'Pipeline Configuration': stage_name,
            'Num Features': len(valid_cols),
            'PR-AUC': pr_auc,
            'ROC-AUC': roc_auc,
            f'Top-{budget_k} Fraud Providers Caught': fraud_caught,
            f'Top-{budget_k} Fraud Dollars Recovered': f"${dollars_rec:,.2f}",
            f'Top-{budget_k} Dollar Capture Rate': f"{capture_pct:.2f}%"
        })

    ablation_df = pd.DataFrame(ablation_results)
    ablation_path = os.path.join(output_dir, 'reports', 'ablation_study_results.csv')
    os.makedirs(os.path.dirname(ablation_path), exist_ok=True)
    ablation_df.to_csv(ablation_path, index=False)
    print(f"Saved ablation study results to {ablation_path}")
    return ablation_df
