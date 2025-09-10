"""Clustering engine for adaptive customer segmentation.

Features:
 - Fit static clustering methods (KMeans, GaussianMixture).
 - Incremental updates via MiniBatchKMeans.partial_fit.
 - Optional dimensionality reduction (PCA / UMAP) for visualization.
 - Maintains mapping customer_id -> cluster label.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

try:
	import umap  # type: ignore
except ImportError:  # pragma: no cover
	umap = None  # noqa


ClusteringAlgo = Literal["kmeans", "gmm", "minibatch_kmeans", "dbscan"]


@dataclass
class ClusterResult:
	labels: pd.Series
	model: object
	embeddings: Optional[pd.DataFrame] = None


def _select_features(df: pd.DataFrame, feature_cols: Optional[list[str]] = None) -> pd.DataFrame:
	if feature_cols is None:
		feature_cols = [c for c in df.columns if c.endswith("_scaled")]
		if not feature_cols:  # fallback
			feature_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
	return df[feature_cols].copy(), feature_cols


def fit_static(
	features: pd.DataFrame,
	algo: ClusteringAlgo = "kmeans",
	n_clusters: int = 6,
	feature_cols: Optional[list[str]] = None,
	reduce: Optional[Literal["pca", "umap"]] = None,
	random_state: int = 42,
	dbscan_eps: float = 0.7,
	dbscan_min_samples: int = 5,
) -> ClusterResult:
	X, used_cols = _select_features(features, feature_cols)

	if algo == "kmeans":
		model = KMeans(n_clusters=n_clusters, n_init='auto', random_state=random_state)
		labels = model.fit_predict(X)
	elif algo == "gmm":
		model = GaussianMixture(n_components=n_clusters, random_state=random_state)
		labels = model.fit_predict(X)
	elif algo == "minibatch_kmeans":
		model = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=256)
		labels = model.fit_predict(X)
	elif algo == "dbscan":
		model = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, n_jobs=-1)
		labels = model.fit_predict(X)
	else:
		raise ValueError(f"Unsupported algorithm: {algo}")

	embeddings = None
	if reduce:
		if reduce == "pca":
			p = PCA(n_components=2, random_state=random_state)
			arr = p.fit_transform(X)
			embeddings = pd.DataFrame(arr, columns=["PC1", "PC2"], index=features.index)
		elif reduce == "umap":
			if umap is None:
				raise ImportError("umap-learn not installed; cannot use reduce='umap'")
			reducer = umap.UMAP(n_components=2, random_state=random_state)
			arr = reducer.fit_transform(X)
			embeddings = pd.DataFrame(arr, columns=["UMAP1", "UMAP2"], index=features.index)
		else:  # pragma: no cover
			raise ValueError("reduce must be one of None, 'pca', 'umap'")

	return ClusterResult(labels=pd.Series(labels, index=features.index, name="cluster"), model=model, embeddings=embeddings)


class IncrementalClusterer:
	"""Incremental clustering with MiniBatchKMeans.

	Usage:
		ic = IncrementalClusterer(n_clusters=6)
		ic.partial_fit(batch_df)
		labels = ic.predict(full_feature_df)
	"""

	def __init__(self, n_clusters: int = 6, feature_cols: Optional[list[str]] = None, random_state: int = 42):
		self.n_clusters = n_clusters
		self.feature_cols = feature_cols
		self.random_state = random_state
		self.model = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=512)
		self._initialized = False

	def _prepare_X(self, df: pd.DataFrame) -> np.ndarray:
		X, cols = _select_features(df, self.feature_cols)
		if self.feature_cols is None:
			self.feature_cols = cols
		return X.values

	def partial_fit(self, batch: pd.DataFrame) -> None:
		X = self._prepare_X(batch)
		self.model.partial_fit(X)
		self._initialized = True

	def predict(self, df: pd.DataFrame) -> pd.Series:
		if not self._initialized:
			raise RuntimeError("Model not fitted yet. Call partial_fit first.")
		X, _ = _select_features(df, self.feature_cols)
		labels = self.model.predict(X)
		return pd.Series(labels, index=df.index, name="cluster")


__all__ = [
	"fit_static",
	"IncrementalClusterer",
	"ClusterResult",
]

