import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import joblib
from typing import Dict, Tuple, Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_curve, precision_recall_curve
)

def build_model_pipelines(pos_weight: float = 10.0) -> Dict[str, Any]:
    """
    Constructs machine learning classification models with appropriate
    regularization, class imbalance compensation, and feature scaling.
    """
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
        ]),
        'Random Forest': RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
    }
    return models

def train_and_evaluate_classifiers(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str = 'outputs'
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Trains all candidate classifiers, records performance metrics on the held-out test set,
    plots comparative ROC and Precision-Recall curves, and saves the best model.
    """
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    scale_weight = float(neg_count / max(pos_count, 1))

    models = build_model_pipelines(pos_weight=scale_weight)
    results = []
    probabilities = {}

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        # Predict probabilities
        if hasattr(model, 'predict_proba'):
            y_probs = model.predict_proba(X_test)[:, 1]
        else:
            y_probs = model.decision_function(X_test)
            y_probs = 1 / (1 + np.exp(-y_probs))

        probabilities[name] = y_probs

        # Choose threshold that balances precision and recall or standard 0.5
        y_pred = (y_probs >= 0.5).astype(int)

        roc_auc = roc_auc_score(y_test, y_probs)
        pr_auc = average_precision_score(y_test, y_probs)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        results.append({
            'Model': name,
            'ROC-AUC': roc_auc,
            'PR-AUC (Avg Precision)': pr_auc,
            'F1 Score': f1,
            'Precision': prec,
            'Recall': rec,
            'True Positives': int(tp),
            'False Positives': int(fp),
            'False Negatives': int(fn)
        })

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        ax_roc.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

        # Plot PR curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_probs)
        ax_pr.plot(recall_curve, precision_curve, lw=2, label=f"{name} (PR-AUC = {pr_auc:.3f})")

    # Finalize ROC plot
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7)
    ax_roc.set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight='bold')
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True, alpha=0.3)

    # Finalize PR plot
    baseline_pr = float(y_test.sum() / len(y_test))
    ax_pr.plot([0, 1], [baseline_pr, baseline_pr], 'k--', lw=1.5, alpha=0.7, label=f"No-Skill Baseline ({baseline_pr:.3f})")
    ax_pr.set_title("Precision-Recall (PR) Curve", fontsize=13, fontweight='bold')
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(loc="upper right")
    ax_pr.grid(True, alpha=0.3)

    plt.tight_layout()
    roc_pr_path = os.path.join(output_dir, 'figures', 'roc_pr_curves.png')
    os.makedirs(os.path.dirname(roc_pr_path), exist_ok=True)
    plt.savefig(roc_pr_path, dpi=300)
    plt.close()
    print(f"Saved ROC & PR curves to {roc_pr_path}")

    results_df = pd.DataFrame(results).sort_values(by='PR-AUC (Avg Precision)', ascending=False).reset_index(drop=True)
    
    # Save model benchmark metrics
    metrics_path = os.path.join(output_dir, 'reports', 'model_benchmark_results.csv')
    results_df.to_csv(metrics_path, index=False)
    print(f"Saved model benchmark results to {metrics_path}")

    # Plot Feature Importance from the top tree model (XGBoost)
    plot_feature_importance(models['XGBoost'], X_train.columns, os.path.join(output_dir, 'figures', 'feature_importance.png'))

    # Serialize best model
    best_model_name = results_df.iloc[0]['Model']
    best_model = models[best_model_name]
    model_save_path = os.path.join(output_dir, 'models', 'best_fraud_model.joblib')
    joblib.dump(best_model, model_save_path)
    print(f"Best Model: '{best_model_name}' saved to {model_save_path}")

    return models, results_df, probabilities

def plot_feature_importance(model: Any, feature_names: pd.Index, output_path: str, top_n: int = 15):
    """Plots and saves the top feature importances from a tree-based model."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_scores = importances[indices]

        plt.figure(figsize=(10, 6))
        plt.barh(range(top_n), top_scores[::-1], color='teal', alpha=0.85)
        plt.yticks(range(top_n), top_features[::-1], fontsize=10)
        plt.xlabel("Feature Importance (Gain / Gini Impurity)")
        plt.title(f"Top {top_n} Predictive Features in Provider Fraud Detection", fontsize=12, fontweight='bold')
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Saved feature importance plot to {output_path}")
