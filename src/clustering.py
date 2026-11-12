import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors, NearestCentroid
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from typing import Tuple, Dict

CLUSTERING_FEATURES = [
    'total_claims', 'inpatient_ratio', 'mean_reimbursed', 'std_reimbursed',
    'total_reimbursed', 'unique_beneficiaries', 'claims_per_beneficiary',
    'unique_attending_physicians', 'claims_per_attending_doctor',
    'mean_claim_duration', 'renal_disease_ratio', 'mean_chronic_conditions'
]

class ProviderClusterAnalyzer:
    """
    Performs unsupervised clustering on provider operational behavior:
    1. Hierarchical Clustering (Agglomerative) for discovering provider typology (e.g. clinic scale).
    2. DBSCAN for density-based anomaly/outlier detection without using fraud labels.
    """
    def __init__(self, n_clusters: int = 4, dbscan_eps: float = 2.5, dbscan_min_samples: int = 5):
        self.n_clusters = n_clusters
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.scaler = StandardScaler()
        self.hierarchical_model = AgglomerativeClustering(n_clusters=self.n_clusters, linkage='ward')
        self.centroid_classifier = NearestCentroid()
        self.dbscan_model = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        self.dbscan_core_samples_ = None
        self.dbscan_core_labels_ = None
        self.dbscan_nn_ = None
        self.pca = PCA(n_components=2, random_state=42)

    def fit(self, train_provider_df: pd.DataFrame):
        """Fits scaler, Agglomerative clustering, and DBSCAN on training providers."""
        print(f"Fitting clustering models on {len(train_provider_df)} providers (features={len(CLUSTERING_FEATURES)})...")
        X_train = train_provider_df[CLUSTERING_FEATURES].fillna(0.0).values
        X_scaled = self.scaler.fit_transform(X_train)

        # 1. Hierarchical Clustering
        hier_labels = self.hierarchical_model.fit_predict(X_scaled)
        self.centroid_classifier.fit(X_scaled, hier_labels)

        # 2. DBSCAN
        db_labels = self.dbscan_model.fit_predict(X_scaled)
        core_mask = np.zeros_like(db_labels, dtype=bool)
        core_mask[self.dbscan_model.core_sample_indices_] = True

        if core_mask.sum() > 0:
            self.dbscan_core_samples_ = X_scaled[core_mask]
            self.dbscan_core_labels_ = db_labels[core_mask]
            self.dbscan_nn_ = NearestNeighbors(n_neighbors=1).fit(self.dbscan_core_samples_)
        else:
            self.dbscan_nn_ = None

        self.pca.fit(X_scaled)

        outlier_count = (db_labels == -1).sum()
        print(f"Clustering fit complete: Hierarchical={self.n_clusters} clusters, DBSCAN outliers={outlier_count:,} ({outlier_count/len(X_train)*100:.1f}%)")
        return self

    def transform(self, provider_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms provider records to add:
        - One-hot encoded hierarchical cluster assignments (provider types)
        - DBSCAN outlier flag (1 = anomaly, 0 = core/dense group)
        """
        X = provider_df[CLUSTERING_FEATURES].fillna(0.0).values
        X_scaled = self.scaler.transform(X)

        # Hierarchical cluster prediction via nearest centroid
        hier_preds = self.centroid_classifier.predict(X_scaled)

        # DBSCAN prediction via inductive distance to core samples
        if self.dbscan_nn_ is not None:
            distances, indices = self.dbscan_nn_.kneighbors(X_scaled)
            is_outlier = (distances.ravel() > self.dbscan_eps).astype(int)
        else:
            is_outlier = np.zeros(len(X), dtype=int)

        features_df = pd.DataFrame({'Provider': provider_df['Provider']})
        features_df['dbscan_outlier_flag'] = is_outlier

        for c in range(self.n_clusters):
            features_df[f'hierarchical_cluster_{c}'] = (hier_preds == c).astype(int)

        return features_df

    def plot_clusters(self, train_provider_df: pd.DataFrame, output_path: str = 'outputs/figures/clustering_outliers_tsne.png'):
        """Generates and saves a 2D PCA projection plot of DBSCAN outliers and clusters."""
        X_train = train_provider_df[CLUSTERING_FEATURES].fillna(0.0).values
        X_scaled = self.scaler.transform(X_train)
        X_pca = self.pca.transform(X_scaled)
        
        hier_preds = self.centroid_classifier.predict(X_scaled)
        if self.dbscan_nn_ is not None:
            distances, _ = self.dbscan_nn_.kneighbors(X_scaled)
            outliers = (distances.ravel() > self.dbscan_eps)
        else:
            outliers = np.zeros(len(X_train), dtype=bool)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Hierarchical Clusters
        scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=hier_preds, cmap='tab10', alpha=0.6, s=18)
        ax1.set_title(f"Hierarchical Clustering ({self.n_clusters} Operational Types)", fontsize=13, fontweight='bold')
        ax1.set_xlabel("PCA Component 1")
        ax1.set_ylabel("PCA Component 2")
        fig.colorbar(scatter1, ax=ax1, label='Cluster ID')

        # DBSCAN Outliers
        colors = np.where(outliers, 'crimson', 'royalblue')
        ax2.scatter(X_pca[~outliers, 0], X_pca[~outliers, 1], c='royalblue', alpha=0.4, s=15, label='Normal Density Core')
        ax2.scatter(X_pca[outliers, 0], X_pca[outliers, 1], c='crimson', alpha=0.9, s=35, marker='x', label=f'DBSCAN Outliers ({outliers.sum():,})')
        ax2.set_title("DBSCAN Anomaly Detection", fontsize=13, fontweight='bold')
        ax2.set_xlabel("PCA Component 1")
        ax2.set_ylabel("PCA Component 2")
        ax2.legend(loc='upper right')

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Saved clustering visualization to {output_path}")
