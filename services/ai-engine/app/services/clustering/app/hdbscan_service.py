"""
HDBSCAN clustering utilities with parameter optimisation and validation metrics.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

import hdbscan
import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from .executor import executor
from .logging import logger


class HDBSCANClusterer:
    """HDBSCAN clustering with parameter optimisation and validation metrics."""

    def __init__(self) -> None:
        self.clusterers: Dict[str, hdbscan.HDBSCAN] = {}
        self.cluster_metrics: Dict[str, Dict[str, Any]] = {}

    def _optimize_parameters(
        self,
        embeddings: np.ndarray,
        n_samples: int,
        embedding_model: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Optimise HDBSCAN parameters based on dataset characteristics.

        Returns:
            Dictionary of optimised parameters.
        """
        # Base parameter optimisation based on dataset size
        if n_samples < 100:
            min_cluster_size = max(3, n_samples // 20)
            min_samples = max(2, min_cluster_size // 2)
        elif n_samples < 1000:
            min_cluster_size = max(5, n_samples // 50)
            min_samples = max(3, min_cluster_size // 2)
        elif n_samples < 10000:
            min_cluster_size = max(10, n_samples // 100)
            min_samples = max(5, min_cluster_size // 2)
        else:
            min_cluster_size = max(20, n_samples // 200)
            min_samples = max(10, min_cluster_size // 2)

        # Model-specific adjustments
        model_adjustments = {
            "nomic-embed-text": {"alpha": 1.0, "cluster_selection_epsilon": 0.1},
            "all-minilm": {"alpha": 1.2, "cluster_selection_epsilon": 0.15},
            "mxbai-embed-large": {"alpha": 0.8, "cluster_selection_epsilon": 0.05},
        }

        base_params = model_adjustments.get(
            embedding_model,
            {
                "alpha": 1.0,
                "cluster_selection_epsilon": 0.1,
            },
        )

        # Dimensionality-based metric selection.
        n_dims = embeddings.shape[1]
        if n_dims > 100:
            metric = "manhattan"
        elif n_dims > 50:
            metric = "manhattan"
        else:
            metric = "euclidean"

        return {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "metric": metric,
            "alpha": base_params["alpha"],
            "cluster_selection_epsilon": base_params["cluster_selection_epsilon"],
            "cluster_selection_method": "eom",  # Excess of Mass
        }

    async def cluster_embeddings(
        self,
        embeddings: np.ndarray,
        session_id: str,
        custom_params: Optional[Dict[str, Any]] = None,
        auto_optimize: bool = True,
        embedding_model: str = "unknown",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform HDBSCAN clustering with parameter optimisation.

        Returns:
            Tuple of (cluster_labels, clustering_metrics).
        """
        start_time = time.time()
        n_samples, n_dims = embeddings.shape

        logger.info(
            "Starting HDBSCAN clustering",
            session_id=session_id,
            n_samples=n_samples,
            n_dims=n_dims,
            embedding_model=embedding_model,
            auto_optimize=auto_optimize,
        )

        # Get optimised parameters
        if auto_optimize:
            params = self._optimize_parameters(embeddings, n_samples, embedding_model)
        else:
            params = {
                "min_cluster_size": 5,
                "min_samples": 3,
                "metric": "euclidean",
                "alpha": 1.0,
                "cluster_selection_epsilon": 0.0,
                "cluster_selection_method": "eom",
            }

        # Override with custom parameters if provided
        if custom_params:
            params.update(custom_params)

        # Create HDBSCAN clusterer
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=params["min_cluster_size"],
            min_samples=params["min_samples"],
            metric=params["metric"],
            alpha=params["alpha"],
            cluster_selection_epsilon=params["cluster_selection_epsilon"],
            cluster_selection_method=params["cluster_selection_method"],
            prediction_data=True,
        )

        # Perform clustering
        cluster_labels = await asyncio.get_event_loop().run_in_executor(executor, clusterer.fit_predict, embeddings)

        # Calculate clustering metrics
        metrics = await self._calculate_cluster_metrics(embeddings, cluster_labels, clusterer, params)

        # Add timing information
        processing_time = time.time() - start_time
        metrics.update(
            {
                "processing_time_seconds": processing_time,
                "session_id": session_id,
                "embedding_model": embedding_model,
                "parameters_used": params,
                "samples_per_second": n_samples / processing_time if processing_time > 0 else 0,
            }
        )

        # Cache clusterer for potential reuse
        clusterer_key = f"{session_id}_{embedding_model}_{int(time.time())}"
        self.clusterers[clusterer_key] = clusterer
        self.cluster_metrics[clusterer_key] = metrics

        logger.info(
            "HDBSCAN clustering completed",
            session_id=session_id,
            n_clusters=metrics["n_clusters"],
            n_noise_points=metrics["n_noise_points"],
            silhouette_score=metrics["silhouette_score"],
            processing_time=processing_time,
        )

        return cluster_labels, metrics

    async def _calculate_cluster_metrics(
        self,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        clusterer: hdbscan.HDBSCAN,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate comprehensive clustering validation metrics."""

        # Basic cluster statistics
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise_points = int(np.sum(cluster_labels == -1))

        # Cluster sizes (excluding noise)
        cluster_sizes: List[int] = []
        for label in unique_labels:
            if label != -1:
                cluster_sizes.append(int(np.sum(cluster_labels == label)))

        metrics: Dict[str, Any] = {
            "n_clusters": n_clusters,
            "n_noise_points": n_noise_points,
            "cluster_sizes": cluster_sizes,
            "noise_ratio": n_noise_points / len(cluster_labels),
        }

        # Calculate validation metrics (only if we have clusters)
        if n_clusters > 1:
            # Silhouette score (excluding noise points)
            non_noise_mask = cluster_labels != -1
            if int(np.sum(non_noise_mask)) > 1:
                try:
                    silhouette = await asyncio.get_event_loop().run_in_executor(
                        executor,
                        silhouette_score,
                        embeddings[non_noise_mask],
                        cluster_labels[non_noise_mask],
                    )
                    metrics["silhouette_score"] = float(silhouette)
                except Exception as error:  # pragma: no cover - logging branch
                    logger.warning("Failed to calculate silhouette score", error=str(error))
                    metrics["silhouette_score"] = 0.0

                # Calinski-Harabasz score
                try:
                    ch_score = await asyncio.get_event_loop().run_in_executor(
                        executor,
                        calinski_harabasz_score,
                        embeddings[non_noise_mask],
                        cluster_labels[non_noise_mask],
                    )
                    metrics["calinski_harabasz_score"] = float(ch_score)
                except Exception as error:  # pragma: no cover - logging branch
                    logger.warning("Failed to calculate Calinski-Harabasz score", error=str(error))
                    metrics["calinski_harabasz_score"] = 0.0

                # Davies-Bouldin score
                try:
                    db_score = await asyncio.get_event_loop().run_in_executor(
                        executor,
                        davies_bouldin_score,
                        embeddings[non_noise_mask],
                        cluster_labels[non_noise_mask],
                    )
                    metrics["davies_bouldin_score"] = float(db_score)
                except Exception as error:  # pragma: no cover - logging branch
                    logger.warning("Failed to calculate Davies-Bouldin score", error=str(error))
                    metrics["davies_bouldin_score"] = float("inf")
            else:
                metrics.update(
                    {
                        "silhouette_score": 0.0,
                        "calinski_harabasz_score": 0.0,
                        "davies_bouldin_score": float("inf"),
                    }
                )
        else:
            metrics.update(
                {
                    "silhouette_score": 0.0,
                    "calinski_harabasz_score": 0.0,
                    "davies_bouldin_score": float("inf"),
                }
            )

        # HDBSCAN-specific metrics
        if hasattr(clusterer, "cluster_persistence_"):
            metrics["cluster_persistence"] = clusterer.cluster_persistence_.tolist()

        if hasattr(clusterer, "probabilities_"):
            # Calculate stability score as mean probability
            non_noise_probs = clusterer.probabilities_[cluster_labels != -1]
            if len(non_noise_probs) > 0:
                metrics["stability_score"] = float(np.mean(non_noise_probs))
            else:
                metrics["stability_score"] = 0.0

        # Cluster quality assessment
        if cluster_sizes:
            metrics["cluster_balance"] = {
                "min_size": min(cluster_sizes),
                "max_size": max(cluster_sizes),
                "mean_size": float(np.mean(cluster_sizes)),
                "std_size": float(np.std(cluster_sizes)),
            }

        return metrics

    def get_cluster_hierarchy(self, clusterer_key: str) -> Optional[Dict[str, Any]]:
        """Get hierarchical clustering information if available."""
        if clusterer_key not in self.clusterers:
            return None

        clusterer = self.clusterers[clusterer_key]

        if not hasattr(clusterer, "condensed_tree_"):
            return None

        hierarchy_info = {
            "condensed_tree": clusterer.condensed_tree_.to_pandas().to_dict("records"),
            "cluster_hierarchy": (
                clusterer.cluster_hierarchy_.to_pandas().to_dict("records")
                if hasattr(clusterer, "cluster_hierarchy_")
                else None
            ),
        }

        return hierarchy_info

    def predict_cluster(self, clusterer_key: str, new_embeddings: np.ndarray) -> Optional[np.ndarray]:
        """Predict cluster labels for new embeddings using cached clusterer."""
        if clusterer_key not in self.clusterers:
            return None

        clusterer = self.clusterers[clusterer_key]

        try:
            labels, _strengths = hdbscan.approximate_predict(clusterer, new_embeddings)
            return labels
        except Exception as error:  # pragma: no cover - logging branch
            logger.error("Failed to predict clusters for new embeddings", error=str(error))
            return None


__all__ = ["HDBSCANClusterer"]
