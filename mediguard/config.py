"""Central configuration. Every path and tunable can be overridden by environment variables
so that nothing environment-specific lives in source code."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).resolve() if value else default


@dataclass(frozen=True)
class Paths:
    raw_dir: Path = field(default_factory=lambda: _env_path("MEDIGUARD_RAW_DIR", PROJECT_ROOT / "data" / "raw"))
    processed_dir: Path = field(default_factory=lambda: _env_path("MEDIGUARD_PROCESSED_DIR", PROJECT_ROOT / "data" / "processed"))
    artifact_dir: Path = field(default_factory=lambda: _env_path("MEDIGUARD_ARTIFACT_DIR", PROJECT_ROOT / "artifacts"))
    db_path: Path = field(default_factory=lambda: _env_path("MEDIGUARD_DB_PATH", PROJECT_ROOT / "data" / "investigations.db"))

    @property
    def reports_dir(self) -> Path:
        return self.artifact_dir / "reports"


@dataclass(frozen=True)
class TrainingConfig:
    """Hyper-parameters of the end-to-end pipeline. Changing any of these changes the model version."""
    random_state: int = 42
    test_size: float = 0.20            # share of providers locked away for final evaluation
    cv_folds: int = 5                  # provider-level stratified folds on the development set
    inner_folds: int = 3               # out-of-fold folds for label-/fit-dependent training features
    # Regression
    regression_max_iter: int = 300
    regression_learning_rate: float = 0.08
    regression_max_leaf_nodes: int = 63
    # Clustering
    cluster_k_range: tuple[int, ...] = (3, 4, 5, 6, 7, 8)
    dbscan_min_samples: int = 10
    dbscan_eps_quantile: float = 0.95  # eps = this quantile of k-distance (label-free heuristic)
    # Association rules
    apriori_min_support: float = 0.0015
    apriori_max_len: int = 3
    rule_min_claims: int = 50          # an itemset must appear in at least this many claims
    rule_min_providers: int = 15       # ...billed by at least this many distinct providers
    rule_min_lift: float = 1.3         # lift of {itemset} -> {fraud-labelled provider}
    rule_top_k: int = 40
    # Audit
    random_baseline_simulations: int = 1000

    def as_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


PATHS = Paths()
TRAINING = TrainingConfig()

# Calibrated-probability cut-offs used for the risk level shown to investigators.
RISK_LEVELS = (("high", 0.50), ("medium", 0.15), ("low", 0.0))
INVESTIGATION_STATUSES = ("not_reviewed", "under_review", "escalated", "closed")
AUDIT_CAPACITY_OPTIONS = (10, 25, 50, 100, 150, 200)
