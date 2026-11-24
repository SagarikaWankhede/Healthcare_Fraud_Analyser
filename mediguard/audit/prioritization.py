"""Audit prioritization under limited investigation capacity.

Before an audit (operational quantities)
    * calibrated fraud probability  p_i  (model output)
    * total reimbursement           R_i  (billing on record)
    * estimated financial exposure  E_i = p_i * R_i
      An *exposure proxy*: the reimbursement volume weighted by how likely the provider is to be
      fraud-labelled. It is not an estimate of fraudulent dollars or of money recoverable.

After retrospective evaluation (held-out providers with historical labels)
    * fraud-labelled providers identified within the top K
    * known fraudulent reimbursement captured = sum of R_i over fraud-labelled providers selected.
      Labels are provider-level, so this counts *all* reimbursement of a fraud-labelled provider;
      it is reimbursement associated with fraud-labelled providers in the historical data, not
      verified fraudulent claims and not money recovered.
    * capture rate = captured / total reimbursement of all fraud-labelled providers in the cohort.

Strategies compared on the identical held-out cohort:
    exposure     - rank by E_i (proposed)
    reimbursement- rank by R_i
    probability  - rank by p_i
    random       - uniformly random order, repeated with fixed seeds (mean and 5th-95th pct.)
Ties are broken by provider ID so every ranking is deterministic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mediguard.config import RISK_LEVELS

STRATEGIES = {
    "exposure": "Risk-weighted exposure (proposed)",
    "reimbursement": "Highest reimbursement first",
    "probability": "Fraud probability only",
    "random": "Random selection",
}
RANK_KEYS = {"exposure": "estimated_exposure", "reimbursement": "total_reimbursed", "probability": "fraud_probability"}


def risk_level(p: np.ndarray | float) -> np.ndarray:
    p = np.atleast_1d(np.asarray(p, dtype=float))
    out = np.full(p.shape, RISK_LEVELS[-1][0], dtype=object)
    for name, cut in reversed(RISK_LEVELS):
        out[p >= cut] = name
    return out


def ranking_order(df: pd.DataFrame, column: str) -> np.ndarray:
    """Positional order, highest value first, ties broken by provider ID ascending."""
    tmp = pd.DataFrame({"v": df[column].to_numpy(dtype=float), "p": df["Provider"].astype(str).to_numpy()})
    return tmp.sort_values(["v", "p"], ascending=[False, True], kind="mergesort").index.to_numpy()


def priority_tier(exposure_rank: np.ndarray, level: np.ndarray, n: int) -> np.ndarray:
    """Audit priority from rank by exposure: P1 top 5% of cohort, P2 next 10%, P3 any other
    provider with medium/high risk, otherwise routine monitoring."""
    pct = exposure_rank / n
    tier = np.where(pct <= 0.05, "P1", np.where(pct <= 0.15, "P2", np.where(level != "low", "P3", "Routine")))
    return tier


def score_providers(df: pd.DataFrame, probability: np.ndarray) -> pd.DataFrame:
    """Add probability, exposure, risk level, ranks and priority tier (no labels used)."""
    out = df.copy()
    out["fraud_probability"] = probability
    out["estimated_exposure"] = out["fraud_probability"] * out["total_reimbursed"]
    out["risk_level"] = risk_level(out["fraud_probability"].to_numpy())
    for key, col in RANK_KEYS.items():
        rank = np.empty(len(out), dtype=int)
        rank[ranking_order(out, col)] = np.arange(1, len(out) + 1)
        out[f"rank_{key}"] = rank
    out["audit_priority"] = priority_tier(out["rank_exposure"].to_numpy(), out["risk_level"].to_numpy(), len(out))
    return out


def random_baseline(is_fraud: np.ndarray, reimb: np.ndarray, n_sims: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(is_fraud)
    fraud_dollars = reimb * is_fraud
    cum_f = np.empty((n_sims, n))
    cum_d = np.empty((n_sims, n))
    cum_r = np.empty((n_sims, n))
    for s in range(n_sims):
        perm = rng.permutation(n)
        cum_f[s] = np.cumsum(is_fraud[perm])
        cum_d[s] = np.cumsum(fraud_dollars[perm])
        cum_r[s] = np.cumsum(reimb[perm])
    return {
        "fraud_identified": cum_f.mean(0), "fraud_identified_p05": np.percentile(cum_f, 5, 0),
        "fraud_identified_p95": np.percentile(cum_f, 95, 0),
        "known_fraud_reimbursement": cum_d.mean(0), "known_fraud_reimbursement_p05": np.percentile(cum_d, 5, 0),
        "known_fraud_reimbursement_p95": np.percentile(cum_d, 95, 0),
        "reimbursement_reviewed": cum_r.mean(0),
    }


def build_audit_curves(scored: pd.DataFrame, label_col: str, n_sims: int, seed: int) -> dict:
    """Cumulative outcomes for every K = 1..N under each strategy (held-out cohort only)."""
    y = scored[label_col].to_numpy(dtype=float)
    reimb = scored["total_reimbursed"].to_numpy(dtype=float)
    exposure = scored["estimated_exposure"].to_numpy(dtype=float)
    curves: dict = {
        "n_providers": int(len(scored)),
        "n_fraud_labelled": int(y.sum()),
        "total_known_fraud_reimbursement": float((reimb * y).sum()),
        "total_reimbursement": float(reimb.sum()),
        "random_simulations": n_sims, "random_seed": seed,
        "strategies": {},
        "orders": {},
    }
    for key, col in RANK_KEYS.items():
        order = ranking_order(scored, col)
        curves["orders"][key] = order.tolist()
        curves["strategies"][key] = {
            "fraud_identified": np.cumsum(y[order]).tolist(),
            "known_fraud_reimbursement": np.cumsum((reimb * y)[order]).tolist(),
            "reimbursement_reviewed": np.cumsum(reimb[order]).tolist(),
            "estimated_exposure": np.cumsum(exposure[order]).tolist(),
        }
    rb = random_baseline(y, reimb, n_sims, seed)
    curves["strategies"]["random"] = {k: v.tolist() for k, v in rb.items()}
    return curves


def summarize_at_capacity(curves: dict, k: int) -> list[dict]:
    """Strategy comparison at audit capacity K (O(1) lookups into precomputed curves)."""
    n = curves["n_providers"]
    k = int(min(max(k, 1), n))
    total_fraud = curves["total_known_fraud_reimbursement"]
    total_fraud_providers = curves["n_fraud_labelled"]
    proposed = set(curves["orders"]["exposure"][:k])
    rows = []
    for key, label in STRATEGIES.items():
        s = curves["strategies"][key]
        i = k - 1
        found = float(s["fraud_identified"][i])
        captured = float(s["known_fraud_reimbursement"][i])
        row = {
            "strategy": key, "label": label, "providers_selected": k,
            "fraud_labelled_identified": found,
            "precision_at_k": found / k,
            "recall_at_k": found / total_fraud_providers if total_fraud_providers else 0.0,
            "known_fraud_reimbursement_captured": captured,
            "capture_rate": captured / total_fraud if total_fraud else 0.0,
            "reimbursement_reviewed": float(s["reimbursement_reviewed"][i]),
        }
        if key == "random":
            row.update({
                "fraud_labelled_identified_p05": float(s["fraud_identified_p05"][i]),
                "fraud_labelled_identified_p95": float(s["fraud_identified_p95"][i]),
                "capture_rate_p05": float(s["known_fraud_reimbursement_p05"][i]) / total_fraud if total_fraud else 0.0,
                "capture_rate_p95": float(s["known_fraud_reimbursement_p95"][i]) / total_fraud if total_fraud else 0.0,
                "overlap_with_proposed": None,
            })
        else:
            row["estimated_exposure_selected"] = float(s["estimated_exposure"][i])
            row["overlap_with_proposed"] = len(proposed.intersection(curves["orders"][key][:k]))
        rows.append(row)
    return rows
