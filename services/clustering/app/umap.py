"""
UMAP dimensionality reduction utilities with model-aware optimisation.
"""

from __future__ import annotations

import asyncio
import gc
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import umap

from .executor import executor
from .logging import logger


class ModelAwareUMAPConfig:
    """Model-aware UMAP configuration based on embedding model characteristics."""

    EMBEDDING_MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
        "nomic-embed-text": {
            "dimensions": 768,
            "optimal_neighbors": 15,
            "min_dist": 0.1,
            "metric": "cosine",
            "batch_size": 800,
            "memory_factor": 1.2,
        },
        "all-minilm": {
            "dimensions": 384,
            "optimal_neighbors": 12,
            "min_dist": 0.15,
            "metric": "cosine",
            "batch_size": 1200,
            "memory_factor": 0.8,
        },
        "mxbai-embed-large": {
            "dimensions": 1024,
            "optimal_neighbors": 20,
            "min_dist": 0.05,
            "metric": "cosine",
            "batch_size": 600,
            "memory_factor": 1.5,
        },
    }

    @classmethod
    def get_config(cls, embedding_model: str, n_samples: int, memory_limit_mb: int) -> Dict[str, Any]:
        """Get optimised UMAP configuration for a specific embedding model."""
        base_config = cls.EMBEDDING_MODEL_CONFIGS.get(
            embedding_model,
            cls.EMBEDDING_MODEL_CONFIGS["nomic-embed-text"],  # Default fallback
        )

        # Adjust parameters based on dataset size
        n_neighbors = base_config["optimal_neighbors"]
        if n_samples < 100:
            n_neighbors = min(n_neighbors, max(5, n_samples // 10))
        elif n_samples > 10000:
            n_neighbors = min(n_neighbors + 5, 30)

        # Adjust batch size based on memory limit and model characteristics
        memory_factor = base_config["memory_factor"]
        adjusted_batch_size = min(
            base_config["batch_size"],
            int(memory_limit_mb / (base_config["dimensions"] * memory_factor * 0.001)),
        )

        return {
            "n_neighbors": n_neighbors,
            "min_dist": base_config["min_dist"],
            "metric": base_config["metric"],
            "batch_size": max(adjusted_batch_size, 100),  # Minimum batch size
            "expected_memory_mb": base_config["dimensions"]
            * memory_factor
            * 0.001
            * adjusted_batch_size,
        }


class DimensionalityReducer:
    """UMAP-based dimensionality reduction with model-aware optimisation."""

    def __init__(self) -> None:
        self.umap_models: Dict[str, umap.UMAP] = {}
        self.performance_metrics: Dict[str, Dict[str, Any]] = {}

    async def reduce_embeddings(
        self,
        embeddings: np.ndarray,
        embedding_model: str,
        n_components: int = 2,
        custom_params: Optional[Dict[str, Any]] = None,
        batch_size: Optional[int] = None,
        memory_limit_mb: int = 2048,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reduce embedding dimensions using UMAP with model-aware optimisation.

        Returns:
            Tuple of (reduced_embeddings, performance_metrics).
        """
        start_time = time.time()
        n_samples, original_dims = embeddings.shape

        logger.info(
            "Starting UMAP dimensionality reduction",
            embedding_model=embedding_model,
            n_samples=n_samples,
            original_dims=original_dims,
            target_dims=n_components,
        )

        # Get model-aware configuration
        config = ModelAwareUMAPConfig.get_config(embedding_model, n_samples, memory_limit_mb)

        # Override with custom parameters if provided
        if custom_params:
            config.update(custom_params)

        if batch_size:
            config["batch_size"] = batch_size

        # Create UMAP model with optimised parameters
        umap_model = umap.UMAP(
            n_components=n_components,
            n_neighbors=config["n_neighbors"],
            min_dist=config["min_dist"],
            metric=config["metric"],
            random_state=42,
            n_jobs=1,  # Control parallelism explicitly
            verbose=True,
        )

        # Process in batches if dataset is large
        if n_samples > config["batch_size"]:
            reduced_embeddings = await self._batch_reduce(embeddings, umap_model, config["batch_size"])
        else:
            # Process all at once for smaller datasets
            reduced_embeddings = await asyncio.get_event_loop().run_in_executor(
                executor, umap_model.fit_transform, embeddings
            )

        # Calculate performance metrics
        processing_time = time.time() - start_time
        memory_used_mb = embeddings.nbytes / (1024 * 1024)

        metrics = {
            "processing_time_seconds": processing_time,
            "memory_used_mb": memory_used_mb,
            "original_dimensions": original_dims,
            "reduced_dimensions": n_components,
            "n_samples": n_samples,
            "embedding_model": embedding_model,
            "umap_parameters": {
                "n_neighbors": config["n_neighbors"],
                "min_dist": config["min_dist"],
                "metric": config["metric"],
            },
            "batch_size_used": config["batch_size"],
            "samples_per_second": n_samples / processing_time if processing_time > 0 else 0,
        }

        # Store model for potential reuse
        model_key = f"{embedding_model}_{n_components}_{n_samples}"
        self.umap_models[model_key] = umap_model
        self.performance_metrics[model_key] = metrics

        logger.info("UMAP reduction completed", **metrics)

        return reduced_embeddings, metrics

    async def _batch_reduce(
        self,
        embeddings: np.ndarray,
        umap_model: umap.UMAP,
        batch_size: int,
    ) -> np.ndarray:
        """Process large embedding datasets in batches to optimise memory usage."""
        n_samples = embeddings.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size

        logger.info(
            "Processing embeddings in batches",
            n_samples=n_samples,
            batch_size=batch_size,
            n_batches=n_batches,
        )

        # Fit UMAP on a representative sample first
        sample_size = min(batch_size * 2, n_samples)
        sample_indices = np.random.choice(n_samples, sample_size, replace=False)
        sample_embeddings = embeddings[sample_indices]

        # Fit the model
        await asyncio.get_event_loop().run_in_executor(executor, umap_model.fit, sample_embeddings)

        # Transform all data in batches
        reduced_batches: List[np.ndarray] = []
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch_embeddings = embeddings[start_idx:end_idx]

            # Transform batch
            reduced_batch = await asyncio.get_event_loop().run_in_executor(
                executor, umap_model.transform, batch_embeddings
            )
            reduced_batches.append(reduced_batch)

            # Force garbage collection to manage memory
            if i % 5 == 0:
                gc.collect()

            logger.debug(
                "Processed batch",
                batch_num=i + 1,
                total_batches=n_batches,
                batch_size=len(batch_embeddings),
            )

        # Combine all batches
        reduced_embeddings = np.vstack(reduced_batches)

        # Final garbage collection
        gc.collect()

        return reduced_embeddings

    def get_performance_metrics(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Get performance metrics for a specific model configuration."""
        return self.performance_metrics.get(model_key)

    def list_cached_models(self) -> List[str]:
        """List all cached UMAP models."""
        return list(self.umap_models.keys())


__all__ = ["ModelAwareUMAPConfig", "DimensionalityReducer"]
