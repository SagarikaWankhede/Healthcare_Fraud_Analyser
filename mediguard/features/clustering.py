"""Technique 2 - Clustering: unusual provider behaviour.

* **Hierarchical (Ward) clustering** groups providers with similar billing behaviour into peer
  groups. The number of clusters is chosen on training providers by silhouette score.
* **DBSCAN** marks providers that sit in low-density regions of behaviour space as outliers.
  ``eps`` is set from the k-distance distribution of training providers (a standard label-free
  heuristic) rather than tuned against fraud labels.

Both are unsupervised: an outlier is a provider that bills *differently*, not a confirmed
fraudster. Neither step sees the fraud label.

Inductive assignment for unseen providers (validation/test/API):
* peer group  = nearest Ward centroid;
* DBSCAN      = outlier iff no DBSCAN core sample lies within ``eps``. For training providers this
  reproduces DBSCAN's own noise labels exactly (core and border points have a core within eps).

Heavy-tailed counts and dollar amounts are log1p-transformed before standardisation so a handful
of very large providers do not dominate Euclidean distances.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

CLUSTER_INPUTS = [
    "total_claims", "inpatient_ratio", "mean_reimbursed", "cv_reimbursed", "total_reimbursed",
    "unique_beneficiaries", "claims_per_beneficiary", "unique_attending_physicians",
    "claims_per_attending_physician", "mean_claim_duration", "mean_chronic_conditions",
    "procedure_claim_ratio", "mean_diag_count",
]
LOG_INPUTS = {"total_claims", "mean_reimbursed", "total_reimbursed", "unique_beneficiaries",
              "claims_per_beneficiary", "unique_attending_physicians", "claims_per_attending_physician",
              "mean_claim_duration"}


class ProviderClusterer:
    def __init__(self, k_range=(3, 4, 5, 6, 7, 8), min_samples: int = 10, eps_quantile: float = 0.95,
                 random_state: int = 42):
        self.k_range = tuple(k_range)
        self.min_samples = min_samples
        self.eps_quantile = eps_quantile
        self.random_state = random_state
        self.scaler_ = StandardScaler()
        self.medians_: pd.Series | None = None

    # -- helpers --------------------------------------------------------------------------------
    def _prepare(self, provider_features: pd.DataFrame) -> np.ndarray:
        X = provider_features[CLUSTER_INPUTS].copy()
        X = X.fillna(self.medians_)
        for col in LOG_INPUTS:
            X[col] = np.log1p(X[col].clip(lower=0))
        return X.to_numpy(dtype=float)

    def feature_names(self) -> list[str]:
        return ["dbscan_outlier", "dbscan_core_distance", "cluster_centroid_distance",
                *[f"peer_group_{k}" for k in range(self.n_clusters_)]]

    # -- fit / transform ------------------------------------------------------------------------
    def fit(self, train_features: pd.DataFrame) -> "ProviderClusterer":
        self.medians_ = train_features[CLUSTER_INPUTS].median()
        Z = self.scaler_.fit_transform(self._prepare(train_features))

        # Ward clustering; choose k by silhouette on the training providers.
        rng = np.random.default_rng(self.random_state)
        sample = rng.choice(len(Z), size=min(len(Z), 3000), replace=False)
        self.silhouette_by_k_ = {}
        best = None
        for k in self.k_range:
            labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)
            score = float(silhouette_score(Z[sample], labels[sample]))
            self.silhouette_by_k_[k] = score
            if best is None or score > best[0]:
                best = (score, k, labels)
        _, self.n_clusters_, ward_labels = best
        self.centroids_ = np.vstack([Z[ward_labels == k].mean(axis=0) for k in range(self.n_clusters_)])

        # DBSCAN with eps from the k-distance curve.
        nn = NearestNeighbors(n_neighbors=self.min_samples).fit(Z)
        kdist = nn.kneighbors(Z)[0][:, -1]
        self.eps_ = float(np.quantile(kdist, self.eps_quantile))
        db = DBSCAN(eps=self.eps_, min_samples=self.min_samples).fit(Z)
        core = Z[db.core_sample_indices_]
        self.core_nn_ = NearestNeighbors(n_neighbors=1).fit(core) if len(core) else None
        self.n_dbscan_clusters_ = int(len(set(db.labels_)) - (1 if -1 in db.labels_ else 0))
        self.train_outlier_rate_ = float(np.mean(db.labels_ == -1))

        self.pca_ = PCA(n_components=2, random_state=self.random_state).fit(Z)
        return self

    def _scaled(self, provider_features: pd.DataFrame) -> np.ndarray:
        return self.scaler_.transform(self._prepare(provider_features))

    def assign(self, provider_features: pd.DataFrame) -> pd.DataFrame:
        Z = self._scaled(provider_features)
        d = np.linalg.norm(Z[:, None, :] - self.centroids_[None, :, :], axis=2)
        group = d.argmin(axis=1)
        if self.core_nn_ is not None:
            core_dist = self.core_nn_.kneighbors(Z)[0].ravel()
        else:
            core_dist = np.full(len(Z), np.inf)
        pcs = self.pca_.transform(Z)
        return pd.DataFrame({
            "peer_group": group,
            "dbscan_outlier": (core_dist > self.eps_).astype(int),
            "dbscan_core_distance": core_dist / self.eps_,     # >1 means outside every dense region
            "cluster_centroid_distance": d[np.arange(len(Z)), group],
            "pca_1": pcs[:, 0],
            "pca_2": pcs[:, 1],
        }, index=provider_features.index)

    def transform(self, provider_features: pd.DataFrame) -> pd.DataFrame:
        a = self.assign(provider_features)
        out = a[["dbscan_outlier", "dbscan_core_distance", "cluster_centroid_distance"]].astype(float).copy()
        for k in range(self.n_clusters_):
            out[f"peer_group_{k}"] = (a["peer_group"] == k).astype(float)
        return out

    def summary(self) -> dict:
        return {
            "n_clusters": int(self.n_clusters_),
            "silhouette_by_k": {str(k): round(v, 4) for k, v in self.silhouette_by_k_.items()},
            "dbscan_eps": self.eps_,
            "dbscan_min_samples": self.min_samples,
            "dbscan_eps_quantile": self.eps_quantile,
            "dbscan_dense_regions": self.n_dbscan_clusters_,
            "train_outlier_rate": self.train_outlier_rate_,
            "pca_explained_variance": [float(v) for v in self.pca_.explained_variance_ratio_],
            "inputs": CLUSTER_INPUTS,
        }
