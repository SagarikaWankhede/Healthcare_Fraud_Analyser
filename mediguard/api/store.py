"""Loads persisted training artifacts once and serves read-only views of them.

Nothing here trains or recomputes models: the API only reads artifacts produced by
``python run_pipeline.py``. If artifacts are missing the store reports exactly which ones, and the
API answers 503 with that message instead of failing at import time.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from mediguard.config import Paths

log = logging.getLogger(__name__)

REQUIRED = [
    "manifest.json", "model_bundle.joblib", "provider_scores.parquet", "provider_monthly.parquet",
    "provider_claim_mix.parquet",
    *[f"reports/{n}.json" for n in ("data_report", "split", "model_performance", "ablation", "regression",
                                    "clustering", "association_rules", "audit_curves", "leakage_checks",
                                    "feature_catalog", "provider_rule_hits")],
]


class ArtifactsUnavailable(RuntimeError):
    pass


@dataclass
class ArtifactStore:
    root: Path
    manifest: dict
    bundle: object
    scores: pd.DataFrame
    monthly: pd.DataFrame
    claim_mix: pd.DataFrame
    reports: dict

    @classmethod
    def load(cls, paths: Paths) -> "ArtifactStore":
        root = paths.artifact_dir
        missing = [r for r in REQUIRED if not (root / r).exists()]
        if missing:
            raise ArtifactsUnavailable(
                f"Model artifacts missing in {root}: {', '.join(missing)}. Run `python run_pipeline.py` first.")
        reports = {p.stem: json.loads(p.read_text()) for p in (root / "reports").glob("*.json")}
        scores = pd.read_parquet(root / "provider_scores.parquet")
        scores["Provider"] = scores["Provider"].astype(str)
        scores["explanation"] = scores["explanation"].map(json.loads)
        store = cls(root=root, manifest=json.loads((root / "manifest.json").read_text()),
                    bundle=joblib.load(root / "model_bundle.joblib"), scores=scores,
                    monthly=pd.read_parquet(root / "provider_monthly.parquet"),
                    claim_mix=pd.read_parquet(root / "provider_claim_mix.parquet"), reports=reports)
        store._index()
        log.info("Loaded artifacts: model %s, %d scored providers", store.manifest["model_version"], len(scores))
        return store

    def _index(self) -> None:
        self.scores = self.scores.sort_values("rank_exposure").reset_index(drop=True)
        self.by_id = {p: i for i, p in enumerate(self.scores["Provider"])}
        self.sorted_exposure = np.sort(self.scores["estimated_exposure"].to_numpy())[::-1]
        self.monthly["Provider"] = self.monthly["Provider"].astype(str)
        self.claim_mix["Provider"] = self.claim_mix["Provider"].astype(str)
        self.monthly_by = {p: g for p, g in self.monthly.groupby("Provider")}
        self.mix_by = {p: g for p, g in self.claim_mix.groupby("Provider")}

    def provider(self, provider_id: str) -> pd.Series | None:
        i = self.by_id.get(provider_id)
        return None if i is None else self.scores.iloc[i]

    def exposure_rank(self, exposure: float) -> int:
        """Where a hypothetical provider with this exposure would rank in the held-out cohort."""
        return int((self.sorted_exposure > exposure).sum()) + 1
