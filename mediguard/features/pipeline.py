"""Composes the provider feature space from all analytical techniques.

``fit_transform`` is called with *training providers only* and returns their features;
``transform`` applies the frozen, fitted components to any other providers. The class is the unit
that is re-fitted inside every cross-validation fold, which is what keeps regression, clustering
and rule mining inside the fold boundary.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from mediguard.config import TrainingConfig
from mediguard.features.association_rules import APRIORI_FEATURES, CodeAssociationMiner
from mediguard.features.clustering import ProviderClusterer
from mediguard.features.provider_features import add_derived_features
from mediguard.features.regression import (RESIDUAL_FEATURES, ReimbursementRegressor, TARGET,
                                           out_of_fold_predictions, provider_residual_features)

log = logging.getLogger(__name__)


def _stage(name: str, t0: float) -> float:
    """Log stage duration and resident memory (diagnostics for long training runs)."""
    try:
        import psutil
        mem = f", rss={psutil.Process().memory_info().rss / 2**30:.2f} GiB"
    except ImportError:  # psutil is optional
        mem = ""
    log.info("  %-24s %6.1fs%s", name, time.time() - t0, mem)
    return time.time()

ASSOCIATION_FEATURES = APRIORI_FEATURES + ["suspicious_itemset_claim_ratio"]


class FeaturePipeline:
    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg
        self.reg_params = dict(max_iter=cfg.regression_max_iter, learning_rate=cfg.regression_learning_rate,
                               max_leaf_nodes=cfg.regression_max_leaf_nodes, random_state=cfg.random_state)

    def fit_transform(self, train_claims: pd.DataFrame, base: pd.DataFrame, train_labels: pd.Series) -> pd.DataFrame:
        t0 = time.time()
        providers = np.array(sorted(train_labels.index))
        assert set(train_claims["Provider"].unique()) == set(providers), "claims/labels provider mismatch"
        self.train_providers_ = frozenset(providers)
        base_tr = base.loc[providers]

        # Regression: out-of-fold residuals for training providers, final model for everyone else.
        oof_pred = out_of_fold_predictions(train_claims, self.cfg.inner_folds, **self.reg_params)
        self.regression_oof_metrics_ = ReimbursementRegressor.metrics(train_claims[TARGET].to_numpy(), oof_pred)
        self.regressor_ = ReimbursementRegressor(**self.reg_params).fit(train_claims)
        residuals = provider_residual_features(train_claims, oof_pred)
        t = _stage("regression", t0)

        # Clustering (label-free) on training providers.
        self.clusterer_ = ProviderClusterer(k_range=self.cfg.cluster_k_range, min_samples=self.cfg.dbscan_min_samples,
                                            eps_quantile=self.cfg.dbscan_eps_quantile,
                                            random_state=self.cfg.random_state).fit(base_tr)
        clusters = self.clusterer_.transform(base_tr)
        t = _stage("clustering", t)

        # Association rules (labels of training providers; out-of-fold features).
        self.miner_ = CodeAssociationMiner(
            min_support=self.cfg.apriori_min_support, max_len=self.cfg.apriori_max_len,
            min_claims=self.cfg.rule_min_claims, min_providers=self.cfg.rule_min_providers,
            min_lift=self.cfg.rule_min_lift, top_k=self.cfg.rule_top_k,
            inner_folds=self.cfg.inner_folds, random_state=self.cfg.random_state)
        rules = self.miner_.fit_transform(train_claims, train_labels)
        _stage("association rules", t)

        out = self._combine(base_tr, residuals, clusters, rules)
        self.feature_names_ = list(out.columns)
        self.groups_ = {
            "base": list(base.columns),
            "regression": RESIDUAL_FEATURES,
            "clustering": self.clusterer_.feature_names(),
            "association": ASSOCIATION_FEATURES,
        }
        log.info("Feature pipeline fitted on %d providers in %.1fs (%d features, %d rules)",
                 len(providers), time.time() - t0, len(self.feature_names_), len(self.miner_.selected_))
        return out

    def transform(self, claims: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
        providers = np.array(sorted(claims["Provider"].unique()))
        overlap = self.train_providers_.intersection(providers)
        if overlap:
            raise ValueError(f"transform() called on {len(overlap)} training providers; use fit_transform output")
        base_p = base.loc[providers]
        pred = self.regressor_.predict(claims)
        self.last_regression_metrics_ = ReimbursementRegressor.metrics(claims[TARGET].to_numpy(), pred)
        out = self._combine(base_p, provider_residual_features(claims, pred),
                            self.clusterer_.transform(base_p), self.miner_.transform(claims))
        return out[self.feature_names_]

    @staticmethod
    def _combine(*parts: pd.DataFrame) -> pd.DataFrame:
        out = pd.concat([p.sort_index() for p in parts], axis=1)
        out = add_derived_features(out)
        out.index.name = "Provider"
        return out.astype(float)
