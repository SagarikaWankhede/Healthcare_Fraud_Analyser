"""Model explanations.

* **Global**: permutation importance measured as the drop in PR-AUC on *validation folds* when a
  feature column is shuffled (never on the held-out test set).
* **Local**: a model-agnostic, per-provider "reference substitution" attribution. For each
  feature we replace the provider's value with the training median and record how much the
  calibrated fraud probability changes. Positive = the provider's actual value raises risk relative
  to a typical provider. These are one-at-a-time sensitivities (interactions are not split out),
  which is stated wherever they are shown.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def permutation_importance_pr_auc(model, X: pd.DataFrame, y: np.ndarray, n_repeats: int = 5,
                                  random_state: int = 0) -> pd.Series:
    rng = np.random.default_rng(random_state)
    base = average_precision_score(y, model.predict_proba(X)[:, 1])
    drops = {}
    for col in X.columns:
        scores = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            scores.append(base - average_precision_score(y, model.predict_proba(Xp)[:, 1]))
        drops[col] = float(np.mean(scores))
    return pd.Series(drops)


def reference_substitution(predict_fn, X: pd.DataFrame, reference: pd.Series, top_n: int = 6) -> list[list[dict]]:
    """Per-row top contributing features via median substitution.

    ``predict_fn`` maps a feature DataFrame to calibrated probabilities.
    """
    cols = list(X.columns)
    base = predict_fn(X)
    n, m = X.shape
    # Build all substituted rows in one batch: block j has column j set to its reference value.
    big = pd.concat([X] * m, ignore_index=True)
    for j, col in enumerate(cols):
        big.iloc[j * n:(j + 1) * n, j] = reference[col]
    sub = predict_fn(big).reshape(m, n).T             # (n rows, m features)
    delta = base[:, None] - sub                          # >0: actual value increases risk
    out = []
    for i in range(n):
        order = np.argsort(-np.abs(delta[i]))[:top_n]
        out.append([{
            "feature": cols[j], "value": _num(X.iat[i, j]), "reference": _num(reference[cols[j]]),
            "contribution": round(float(delta[i, j]), 5),
        } for j in order if abs(delta[i, j]) > 1e-6])
    return out


def _num(v):
    v = float(v)
    return None if np.isnan(v) else round(v, 4)
