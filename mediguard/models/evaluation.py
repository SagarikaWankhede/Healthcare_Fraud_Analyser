"""Metrics, calibration and threshold selection."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score, log_loss,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class PlattCalibrator:
    """Sigmoid (Platt) calibration fitted on out-of-fold scores.

    Chosen over isotonic regression because it is monotone *and strictly increasing*, so it never
    introduces ties that would change the audit ranking, and it is stable with ~400 positives.
    """

    def fit(self, raw_scores: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(raw_scores).reshape(-1, 1), y)
        self.coef_ = float(self.lr_.coef_[0, 0])
        self.intercept_ = float(self.lr_.intercept_[0])
        return self

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        z = self.coef_ * _logit(np.asarray(raw_scores, dtype=float)) + self.intercept_
        return 1.0 / (1.0 + np.exp(-z))


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    return float(thr[int(np.argmax(f1[:-1]))])


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def reliability_curve(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    """Quantile-binned reliability diagram (equal-count bins are readable with 9% prevalence)."""
    order = np.argsort(p)
    out = []
    for chunk in np.array_split(order, bins):
        if len(chunk):
            out.append({"mean_predicted": float(p[chunk].mean()), "observed_rate": float(y[chunk].mean()),
                        "n": int(len(chunk))})
    return out


def classification_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, EPS, 1 - EPS), labels=[0, 1])),
        "threshold": float(threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": int(len(y)), "positives": int(y.sum()),
    }


def _downsample(x: np.ndarray, y: np.ndarray, n: int = 120) -> list[list[float]]:
    if len(x) > n:
        idx = np.unique(np.linspace(0, len(x) - 1, n).astype(int))
        x, y = x[idx], y[idx]
    return [[round(float(a), 5), round(float(b), 5)] for a, b in zip(x, y)]


def curve_points(y: np.ndarray, p: np.ndarray) -> dict:
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    return {"roc": _downsample(fpr, tpr), "pr": _downsample(rec[::-1], prec[::-1])}
