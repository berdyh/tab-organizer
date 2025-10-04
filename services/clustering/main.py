"""Clustering Service - Groups similar content using advanced algorithms."""

import time
import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import structlog
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import umap
import hdbscan
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import DBSCAN, KMeans
import asyncio
from concurrent.futures import ThreadPoolExecutor
import gc

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

app = FastAPI(
    title="Clustering Service",
    description="Groups similar content using UMAP and HDBSCAN algorithms with model-aware optimization",
    version="1.0.0"
)

# Initialize Qdrant client
qdrant_client = QdrantClient(
    host=os.getenv("QDRANT_HOST", "qdrant"),
    port=int(os.getenv("QDRANT_PORT", "6333"))
)

# Thread pool for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=4)

# Data Models
class UMAPRequest(BaseModel):
    session_id: str
    embedding_model: str
    n_components: int = 2
    n_neighbors: Optional[int] = None
    min_dist: Optional[float] = None
    metric: Optional[str] = None
    batch_size: int = 1000
    memory_limit_mb: int = 2048

class UMAPResponse(BaseModel):
    job_id: str
    status: str
    message: str
    reduced_dimensions: Optional[int] = None
    original_dimensions: Optional[int] = None
    n_samples: Optional[int] = None

class VisualizationRequest(BaseModel):
    session_id: str
    plot_type: str = "2d"  # "2d" or "3d"
    color_by: str = "cluster"  # "cluster", "model", "quality"
    include_metrics: bool = True

class HDBSCANRequest(BaseModel):
    session_id: str
    min_cluster_size: Optional[int] = None
    min_samples: Optional[int] = None
    cluster_selection_epsilon: Optional[float] = None
    alpha: Optional[float] = None
    metric: Optional[str] = None
    auto_optimize: bool = True
    use_reduced_embeddings: bool = True

class HDBSCANResponse(BaseModel):
    job_id: str
    status: str
    message: str
    n_clusters: Optional[int] = None
    n_noise_points: Optional[int] = None
    silhouette_score: Optional[float] = None

class ClusterValidationMetrics(BaseModel):
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    n_clusters: int
    n_noise_points: int
    cluster_sizes: List[int]
    stability_score: Optional[float] = None

class SimilaritySearchRequest(BaseModel):
    session_id: str
    query_text: Optional[str] = None
    query_embedding: Optional[List[float]] = None
    top_k: int = 10
    similarity_threshold: float = 0.7
    filter_cluster_id: Optional[int] = None
    use_reduced_embeddings: bool = False

class SimilaritySearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    query_info: Dict[str, Any]
    search_metadata: Dict[str, Any]

class RecommendationRequest(BaseModel):
    session_id: str
    user_interactions: List[Dict[str, Any]]  # List of user interactions (clicks, views, etc.)
    recommendation_type: str = "content_based"  # "content_based", "collaborative", "hybrid"
    top_k: int = 10
    diversity_factor: float = 0.3  # Balance between relevance and diversity

class RecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    recommendation_metadata: Dict[str, Any]

class ModelAwareUMAPConfig:
    """Model-aware UMAP configuration based on embedding model characteristics."""
    
    EMBEDDING_MODEL_CONFIGS = {
        "nomic-embed-text": {
            "dimensions": 768,
            "optimal_neighbors": 15,
            "min_dist": 0.1,
            "metric": "cosine",
            "batch_size": 800,
            "memory_factor": 1.2
        },
        "all-minilm": {
            "dimensions": 384,
            "optimal_neighbors": 12,
            "min_dist": 0.15,
            "metric": "cosine", 
            "batch_size": 1200,
            "memory_factor": 0.8
        },
        "mxbai-embed-large": {
            "dimensions": 1024,
            "optimal_neighbors": 20,
            "min_dist": 0.05,
            "metric": "cosine",
            "batch_size": 600,
            "memory_factor": 1.5
        }
    }
    
    @classmethod
    def get_config(cls, embedding_model: str, n_samples: int, memory_limit_mb: int) -> Dict[str, Any]:
        """Get optimized UMAP configuration for specific embedding model."""
        base_config = cls.EMBEDDING_MODEL_CONFIGS.get(
            embedding_model, 
            cls.EMBEDDING_MODEL_CONFIGS["nomic-embed-text"]  # Default fallback
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
            int(memory_limit_mb / (base_config["dimensions"] * memory_factor * 0.001))
        )
        
        return {
            "n_neighbors": n_neighbors,
            "min_dist": base_config["min_dist"],
            "metric": base_config["metric"],
            "batch_size": max(adjusted_batch_size, 100),  # Minimum batch size
            "expected_memory_mb": base_config["dimensions"] * memory_factor * 0.001 * adjusted_batch_size
        }

class DimensionalityReducer:
    """UMAP-based dimensionality reduction with model-aware optimization."""
    
    def __init__(self):
        self.umap_models = {}  # Cache UMAP models for reuse
        self.performance_metrics = {}
        
    async def reduce_embeddings(
        self, 
        embeddings: np.ndarray, 
        embedding_model: str,
        n_components: int = 2,
        custom_params: Optional[Dict[str, Any]] = None,
        batch_size: Optional[int] = None,
        memory_limit_mb: int = 2048
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reduce embedding dimensions using UMAP with model-aware optimization.
        
        Args:
            embeddings: Input embeddings array
            embedding_model: Name of the embedding model used
            n_components: Target number of dimensions
            custom_params: Override default parameters
            batch_size: Batch size for processing
            memory_limit_mb: Memory limit in MB
            
        Returns:
            Tuple of (reduced_embeddings, performance_metrics)
        """
        start_time = time.time()
        n_samples, original_dims = embeddings.shape
        
        logger.info(
            "Starting UMAP dimensionality reduction",
            embedding_model=embedding_model,
            n_samples=n_samples,
            original_dims=original_dims,
            target_dims=n_components
        )
        
        # Get model-aware configuration
        config = ModelAwareUMAPConfig.get_config(embedding_model, n_samples, memory_limit_mb)
        
        # Override with custom parameters if provided
        if custom_params:
            config.update(custom_params)
            
        if batch_size:
            config["batch_size"] = batch_size
            
        # Create UMAP model with optimized parameters
        umap_model = umap.UMAP(
            n_components=n_components,
            n_neighbors=config["n_neighbors"],
            min_dist=config["min_dist"],
            metric=config["metric"],
            random_state=42,
            n_jobs=1,  # Control parallelism explicitly
            verbose=True
        )
        
        # Process in batches if dataset is large
        if n_samples > config["batch_size"]:
            reduced_embeddings = await self._batch_reduce(
                embeddings, umap_model, config["batch_size"]
            )
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
                "metric": config["metric"]
            },
            "batch_size_used": config["batch_size"],
            "samples_per_second": n_samples / processing_time if processing_time > 0 else 0
        }
        
        # Store model for potential reuse
        model_key = f"{embedding_model}_{n_components}_{n_samples}"
        self.umap_models[model_key] = umap_model
        self.performance_metrics[model_key] = metrics
        
        logger.info(
            "UMAP reduction completed",
            **metrics
        )
        
        return reduced_embeddings, metrics
    
    async def _batch_reduce(
        self, 
        embeddings: np.ndarray, 
        umap_model: umap.UMAP, 
        batch_size: int
    ) -> np.ndarray:
        """Process large embedding datasets in batches to optimize memory usage."""
        n_samples = embeddings.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        logger.info(
            "Processing embeddings in batches",
            n_samples=n_samples,
            batch_size=batch_size,
            n_batches=n_batches
        )
        
        # Fit UMAP on a representative sample first
        sample_size = min(batch_size * 2, n_samples)
        sample_indices = np.random.choice(n_samples, sample_size, replace=False)
        sample_embeddings = embeddings[sample_indices]
        
        # Fit the model
        await asyncio.get_event_loop().run_in_executor(
            executor, umap_model.fit, sample_embeddings
        )
        
        # Transform all data in batches
        reduced_batches = []
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
                batch_size=len(batch_embeddings)
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

class VisualizationGenerator:
    """Generate interactive visualizations for dimensionality-reduced embeddings."""
    
    def __init__(self):
        self.plot_cache = {}
        
    async def create_cluster_plot(
        self,
        reduced_embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
        plot_type: str = "2d",
        color_by: str = "cluster",
        include_metrics: bool = True,
        performance_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create interactive cluster visualization with model performance metrics.
        
        Args:
            reduced_embeddings: UMAP-reduced embeddings
            metadata: Metadata for each point (cluster labels, quality scores, etc.)
            plot_type: "2d" or "3d"
            color_by: What to color points by ("cluster", "model", "quality")
            include_metrics: Whether to include performance metrics in plot
            performance_metrics: UMAP performance metrics
            
        Returns:
            Dictionary containing plot HTML and metadata
        """
        n_samples, n_dims = reduced_embeddings.shape
        
        if plot_type == "3d" and n_dims < 3:
            raise ValueError("3D plot requires at least 3 dimensions in reduced embeddings")
            
        # Prepare data for plotting
        plot_data = {
            "x": reduced_embeddings[:, 0],
            "y": reduced_embeddings[:, 1],
        }
        
        if plot_type == "3d" and n_dims >= 3:
            plot_data["z"] = reduced_embeddings[:, 2]
            
        # Add color information
        if color_by == "cluster":
            plot_data["color"] = [item.get("cluster_id", -1) for item in metadata]
            color_title = "Cluster ID"
            color_discrete = False
        elif color_by == "model":
            # Convert categorical data to numeric for plotly
            unique_models = list(set(item.get("embedding_model", "unknown") for item in metadata))
            model_to_num = {model: i for i, model in enumerate(unique_models)}
            plot_data["color"] = [model_to_num.get(item.get("embedding_model", "unknown"), 0) for item in metadata]
            color_title = "Embedding Model"
            color_discrete = True
        elif color_by == "quality":
            plot_data["color"] = [item.get("quality_score", 0.0) for item in metadata]
            color_title = "Quality Score"
            color_discrete = False
        else:
            plot_data["color"] = [0] * n_samples  # Use numeric instead of color name
            color_title = "Data Points"
            color_discrete = False
            
        # Add hover information
        hover_text = []
        for i, item in enumerate(metadata):
            hover_info = [
                f"Point {i}",
                f"Cluster: {item.get('cluster_id', 'N/A')}",
                f"Model: {item.get('embedding_model', 'N/A')}",
                f"Quality: {item.get('quality_score', 'N/A'):.3f}",
            ]
            if "title" in item:
                hover_info.append(f"Title: {item['title'][:50]}...")
            hover_text.append("<br>".join(hover_info))
            
        plot_data["hover_text"] = hover_text
        
        # Create plot
        if plot_type == "3d":
            fig = go.Figure(data=go.Scatter3d(
                x=plot_data["x"],
                y=plot_data["y"], 
                z=plot_data["z"],
                mode='markers',
                marker=dict(
                    size=5,
                    color=plot_data["color"],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title=color_title)
                ),
                text=plot_data["hover_text"],
                hovertemplate='%{text}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f"3D UMAP Visualization - {color_title}",
                scene=dict(
                    xaxis_title="UMAP 1",
                    yaxis_title="UMAP 2", 
                    zaxis_title="UMAP 3"
                )
            )
        else:
            fig = go.Figure(data=go.Scatter(
                x=plot_data["x"],
                y=plot_data["y"],
                mode='markers',
                marker=dict(
                    size=8,
                    color=plot_data["color"],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title=color_title)
                ),
                text=plot_data["hover_text"],
                hovertemplate='%{text}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f"2D UMAP Visualization - {color_title}",
                xaxis_title="UMAP 1",
                yaxis_title="UMAP 2"
            )
        
        # Add performance metrics annotation if requested
        if include_metrics and performance_metrics:
            metrics_text = [
                f"Processing Time: {performance_metrics.get('processing_time_seconds', 0):.2f}s",
                f"Samples/Second: {performance_metrics.get('samples_per_second', 0):.1f}",
                f"Memory Used: {performance_metrics.get('memory_used_mb', 0):.1f}MB",
                f"Original Dims: {performance_metrics.get('original_dimensions', 0)}",
                f"Reduced Dims: {performance_metrics.get('reduced_dimensions', 0)}",
                f"Model: {performance_metrics.get('embedding_model', 'N/A')}"
            ]
            
            fig.add_annotation(
                text="<br>".join(metrics_text),
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                xanchor="left", yanchor="top",
                showarrow=False,
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1
            )
        
        # Convert to HTML
        plot_html = fig.to_html(include_plotlyjs=True)
        
        return {
            "plot_html": plot_html,
            "plot_type": plot_type,
            "n_points": n_samples,
            "color_by": color_by,
            "performance_metrics": performance_metrics
        }

class HDBSCANClusterer:
    """HDBSCAN clustering with parameter optimization and validation metrics."""
    
    def __init__(self):
        self.clusterers = {}  # Cache clusterers for reuse
        self.cluster_metrics = {}
        
    def _optimize_parameters(
        self, 
        embeddings: np.ndarray, 
        n_samples: int,
        embedding_model: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Optimize HDBSCAN parameters based on dataset characteristics.
        
        Args:
            embeddings: Input embeddings for clustering
            n_samples: Number of samples
            embedding_model: Name of embedding model used
            
        Returns:
            Dictionary of optimized parameters
        """
        # Base parameter optimization based on dataset size
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
            "mxbai-embed-large": {"alpha": 0.8, "cluster_selection_epsilon": 0.05}
        }
        
        base_params = model_adjustments.get(embedding_model, {
            "alpha": 1.0, 
            "cluster_selection_epsilon": 0.1
        })
        
        # Dimensionality-based metric selection
        # HDBSCAN supports: euclidean, manhattan, chebyshev, minkowski, hamming, jaccard
        n_dims = embeddings.shape[1]
        if n_dims > 100:
            metric = "manhattan"  # Use manhattan for high dimensions instead of cosine
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
            "cluster_selection_method": "eom"  # Excess of Mass
        }
    
    async def cluster_embeddings(
        self,
        embeddings: np.ndarray,
        session_id: str,
        custom_params: Optional[Dict[str, Any]] = None,
        auto_optimize: bool = True,
        embedding_model: str = "unknown"
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform HDBSCAN clustering with parameter optimization.
        
        Args:
            embeddings: Input embeddings for clustering
            session_id: Session identifier
            custom_params: Custom parameters to override defaults
            auto_optimize: Whether to auto-optimize parameters
            embedding_model: Name of embedding model used
            
        Returns:
            Tuple of (cluster_labels, clustering_metrics)
        """
        start_time = time.time()
        n_samples, n_dims = embeddings.shape
        
        logger.info(
            "Starting HDBSCAN clustering",
            session_id=session_id,
            n_samples=n_samples,
            n_dims=n_dims,
            embedding_model=embedding_model,
            auto_optimize=auto_optimize
        )
        
        # Get optimized parameters
        if auto_optimize:
            params = self._optimize_parameters(embeddings, n_samples, embedding_model)
        else:
            params = {
                "min_cluster_size": 5,
                "min_samples": 3,
                "metric": "euclidean",
                "alpha": 1.0,
                "cluster_selection_epsilon": 0.0,
                "cluster_selection_method": "eom"
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
            prediction_data=True  # Enable prediction for new points
        )
        
        # Perform clustering
        cluster_labels = await asyncio.get_event_loop().run_in_executor(
            executor, clusterer.fit_predict, embeddings
        )
        
        # Calculate clustering metrics
        metrics = await self._calculate_cluster_metrics(
            embeddings, cluster_labels, clusterer, params
        )
        
        # Add timing information
        processing_time = time.time() - start_time
        metrics.update({
            "processing_time_seconds": processing_time,
            "session_id": session_id,
            "embedding_model": embedding_model,
            "parameters_used": params,
            "samples_per_second": n_samples / processing_time if processing_time > 0 else 0
        })
        
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
            processing_time=processing_time
        )
        
        return cluster_labels, metrics
    
    async def _calculate_cluster_metrics(
        self,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        clusterer: hdbscan.HDBSCAN,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate comprehensive clustering validation metrics."""
        
        # Basic cluster statistics
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise_points = np.sum(cluster_labels == -1)
        
        # Cluster sizes (excluding noise)
        cluster_sizes = []
        for label in unique_labels:
            if label != -1:
                cluster_sizes.append(np.sum(cluster_labels == label))
        
        metrics = {
            "n_clusters": n_clusters,
            "n_noise_points": n_noise_points,
            "cluster_sizes": cluster_sizes,
            "noise_ratio": n_noise_points / len(cluster_labels)
        }
        
        # Calculate validation metrics (only if we have clusters)
        if n_clusters > 1:
            # Silhouette score (excluding noise points)
            non_noise_mask = cluster_labels != -1
            if np.sum(non_noise_mask) > 1:
                try:
                    silhouette = await asyncio.get_event_loop().run_in_executor(
                        executor,
                        silhouette_score,
                        embeddings[non_noise_mask],
                        cluster_labels[non_noise_mask]
                    )
                    metrics["silhouette_score"] = float(silhouette)
                except Exception as e:
                    logger.warning("Failed to calculate silhouette score", error=str(e))
                    metrics["silhouette_score"] = 0.0
                
                # Calinski-Harabasz score
                try:
                    ch_score = await asyncio.get_event_loop().run_in_executor(
                        executor,
                        calinski_harabasz_score,
                        embeddings[non_noise_mask],
                        cluster_labels[non_noise_mask]
                    )
                    metrics["calinski_harabasz_score"] = float(ch_score)
                except Exception as e:
                    logger.warning("Failed to calculate Calinski-Harabasz score", error=str(e))
                    metrics["calinski_harabasz_score"] = 0.0
                
                # Davies-Bouldin score
                try:
                    db_score = await asyncio.get_event_loop().run_in_executor(
                        executor,
                        davies_bouldin_score,
                        embeddings[non_noise_mask],
                        cluster_labels[non_noise_mask]
                    )
                    metrics["davies_bouldin_score"] = float(db_score)
                except Exception as e:
                    logger.warning("Failed to calculate Davies-Bouldin score", error=str(e))
                    metrics["davies_bouldin_score"] = float('inf')
            else:
                metrics.update({
                    "silhouette_score": 0.0,
                    "calinski_harabasz_score": 0.0,
                    "davies_bouldin_score": float('inf')
                })
        else:
            metrics.update({
                "silhouette_score": 0.0,
                "calinski_harabasz_score": 0.0,
                "davies_bouldin_score": float('inf')
            })
        
        # HDBSCAN-specific metrics
        if hasattr(clusterer, 'cluster_persistence_'):
            metrics["cluster_persistence"] = clusterer.cluster_persistence_.tolist()
        
        if hasattr(clusterer, 'probabilities_'):
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
                "mean_size": np.mean(cluster_sizes),
                "std_size": np.std(cluster_sizes)
            }
        
        return metrics
    
    def get_cluster_hierarchy(self, clusterer_key: str) -> Optional[Dict[str, Any]]:
        """Get hierarchical clustering information if available."""
        if clusterer_key not in self.clusterers:
            return None
        
        clusterer = self.clusterers[clusterer_key]
        
        if not hasattr(clusterer, 'condensed_tree_'):
            return None
        
        # Extract hierarchy information
        hierarchy_info = {
            "condensed_tree": clusterer.condensed_tree_.to_pandas().to_dict('records'),
            "cluster_hierarchy": clusterer.cluster_hierarchy_.to_pandas().to_dict('records') if hasattr(clusterer, 'cluster_hierarchy_') else None
        }
        
        return hierarchy_info
    
    def predict_cluster(self, clusterer_key: str, new_embeddings: np.ndarray) -> Optional[np.ndarray]:
        """Predict cluster labels for new embeddings using cached clusterer."""
        if clusterer_key not in self.clusterers:
            return None
        
        clusterer = self.clusterers[clusterer_key]
        
        try:
            # Use approximate prediction for new points
            labels, strengths = hdbscan.approximate_predict(clusterer, new_embeddings)
            return labels
        except Exception as e:
            logger.error("Failed to predict clusters for new embeddings", error=str(e))
            return None

class SimilaritySearchEngine:
    """Vector similarity search and recommendation engine."""
    
    def __init__(self):
        self.user_interaction_history = {}  # Store user interaction patterns
        self.content_profiles = {}  # Store content-based profiles
        
    async def vector_similarity_search(
        self,
        session_id: str,
        query_embedding: np.ndarray,
        top_k: int = 10,
        similarity_threshold: float = 0.7,
        filter_cluster_id: Optional[int] = None,
        use_reduced_embeddings: bool = False
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Perform vector similarity search using embeddings.
        
        Args:
            session_id: Session identifier
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity threshold
            filter_cluster_id: Optional cluster ID to filter results
            use_reduced_embeddings: Whether to use reduced embeddings for search
            
        Returns:
            Tuple of (search_results, search_metadata)
        """
        start_time = time.time()
        
        logger.info(
            "Starting vector similarity search",
            session_id=session_id,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            filter_cluster_id=filter_cluster_id,
            use_reduced_embeddings=use_reduced_embeddings
        )
        
        # Load embeddings from Qdrant
        collection_name = f"session_{session_id}"
        
        # Retrieve all points with embeddings
        search_result = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: qdrant_client.scroll(
                collection_name=collection_name,
                limit=10000,  # Adjust based on needs
                with_payload=True,
                with_vectors=True
            )
        )
        
        points = search_result[0]
        
        if not points:
            return [], {"message": "No embeddings found in session"}
        
        # Extract embeddings and metadata
        embeddings = []
        metadata = []
        
        for point in points:
            # Apply cluster filter if specified
            if filter_cluster_id is not None:
                point_cluster_id = point.payload.get("cluster_id", -1)
                if point_cluster_id != filter_cluster_id:
                    continue
            
            # Use appropriate embeddings
            if use_reduced_embeddings and "reduced_embedding" in point.payload:
                embeddings.append(point.payload["reduced_embedding"])
            elif not use_reduced_embeddings and point.vector:
                embeddings.append(point.vector)
            else:
                continue
                
            metadata.append({
                "point_id": point.id,
                "title": point.payload.get("title", ""),
                "url": point.payload.get("url", ""),
                "cluster_id": point.payload.get("cluster_id", -1),
                "quality_score": point.payload.get("quality_score", 0.0),
                "embedding_model": point.payload.get("embedding_model", "unknown"),
                "content_type": point.payload.get("content_type", "unknown"),
                "timestamp": point.payload.get("timestamp", 0)
            })
        
        if not embeddings:
            return [], {"message": "No suitable embeddings found for search"}
        
        embeddings = np.array(embeddings)
        
        # Calculate cosine similarities
        similarities = await self._calculate_cosine_similarities(
            query_embedding, embeddings
        )
        
        # Filter by similarity threshold and get top-k
        valid_indices = np.where(similarities >= similarity_threshold)[0]
        
        if len(valid_indices) == 0:
            return [], {
                "message": "No results above similarity threshold",
                "max_similarity": float(np.max(similarities)),
                "threshold": similarity_threshold
            }
        
        # Sort by similarity and take top-k
        valid_similarities = similarities[valid_indices]
        sorted_indices = np.argsort(valid_similarities)[::-1][:top_k]
        
        # Prepare results
        results = []
        for idx in sorted_indices:
            original_idx = valid_indices[idx]
            result = metadata[original_idx].copy()
            result["similarity_score"] = float(valid_similarities[idx])
            results.append(result)
        
        # Calculate search metadata
        processing_time = time.time() - start_time
        search_metadata = {
            "processing_time_seconds": processing_time,
            "total_candidates": len(embeddings),
            "results_returned": len(results),
            "similarity_threshold": similarity_threshold,
            "average_similarity": float(np.mean([r["similarity_score"] for r in results])) if results else 0.0,
            "max_similarity": float(np.max([r["similarity_score"] for r in results])) if results else 0.0,
            "min_similarity": float(np.min([r["similarity_score"] for r in results])) if results else 0.0,
            "filter_cluster_id": filter_cluster_id,
            "use_reduced_embeddings": use_reduced_embeddings
        }
        
        logger.info(
            "Vector similarity search completed",
            session_id=session_id,
            results_count=len(results),
            processing_time=processing_time
        )
        
        return results, search_metadata
    
    async def _calculate_cosine_similarities(
        self, 
        query_embedding: np.ndarray, 
        embeddings: np.ndarray
    ) -> np.ndarray:
        """Calculate cosine similarities between query and all embeddings."""
        
        def compute_similarities():
            # Normalize embeddings
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # Calculate cosine similarities
            similarities = np.dot(embeddings_norm, query_norm)
            return similarities
        
        # Run in executor for CPU-intensive computation
        similarities = await asyncio.get_event_loop().run_in_executor(
            executor, compute_similarities
        )
        
        return similarities
    
    async def content_based_recommendations(
        self,
        session_id: str,
        user_interactions: List[Dict[str, Any]],
        top_k: int = 10,
        diversity_factor: float = 0.3
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Generate content-based recommendations based on user interactions.
        
        Args:
            session_id: Session identifier
            user_interactions: List of user interactions with content
            top_k: Number of recommendations to return
            diversity_factor: Balance between relevance and diversity (0-1)
            
        Returns:
            Tuple of (recommendations, recommendation_metadata)
        """
        start_time = time.time()
        
        logger.info(
            "Generating content-based recommendations",
            session_id=session_id,
            interactions_count=len(user_interactions),
            top_k=top_k,
            diversity_factor=diversity_factor
        )
        
        if not user_interactions:
            return [], {"message": "No user interactions provided"}
        
        # Build user profile from interactions
        user_profile = await self._build_user_profile(session_id, user_interactions)
        
        if user_profile is None:
            return [], {"message": "Could not build user profile"}
        
        # Find similar content
        similar_items, search_metadata = await self.vector_similarity_search(
            session_id=session_id,
            query_embedding=user_profile["profile_embedding"],
            top_k=top_k * 3,  # Get more candidates for diversity filtering
            similarity_threshold=0.1,  # Lower threshold for more candidates
            use_reduced_embeddings=False
        )
        
        if not similar_items:
            return [], {"message": "No similar content found"}
        
        # Apply diversity filtering
        diverse_recommendations = await self._apply_diversity_filtering(
            similar_items, top_k, diversity_factor
        )
        
        # Add recommendation scores and reasons
        recommendations = []
        for item in diverse_recommendations:
            recommendation = item.copy()
            recommendation["recommendation_score"] = self._calculate_recommendation_score(
                item, user_profile
            )
            recommendation["recommendation_reason"] = self._generate_recommendation_reason(
                item, user_profile
            )
            recommendations.append(recommendation)
        
        # Calculate recommendation metadata
        processing_time = time.time() - start_time
        recommendation_metadata = {
            "processing_time_seconds": processing_time,
            "user_profile_summary": {
                "preferred_clusters": user_profile.get("preferred_clusters", []),
                "interaction_count": len(user_interactions),
                "content_types": user_profile.get("content_types", [])
            },
            "diversity_factor": diversity_factor,
            "candidates_considered": len(similar_items),
            "recommendations_returned": len(recommendations),
            "average_recommendation_score": np.mean([r["recommendation_score"] for r in recommendations]) if recommendations else 0.0
        }
        
        logger.info(
            "Content-based recommendations completed",
            session_id=session_id,
            recommendations_count=len(recommendations),
            processing_time=processing_time
        )
        
        return recommendations, recommendation_metadata
    
    async def _build_user_profile(
        self, 
        session_id: str, 
        user_interactions: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Build user profile from interaction history."""
        
        # Load content embeddings for interacted items
        collection_name = f"session_{session_id}"
        
        interacted_embeddings = []
        interaction_weights = []
        cluster_preferences = {}
        content_type_preferences = {}
        
        for interaction in user_interactions:
            point_id = interaction.get("point_id")
            interaction_type = interaction.get("type", "view")  # view, click, like, etc.
            timestamp = interaction.get("timestamp", time.time())
            
            if not point_id:
                continue
            
            # Get point data from Qdrant
            try:
                point_data = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda: qdrant_client.retrieve(
                        collection_name=collection_name,
                        ids=[point_id],
                        with_payload=True,
                        with_vectors=True
                    )
                )
                
                if not point_data:
                    continue
                
                point = point_data[0]
                
                if point.vector:
                    interacted_embeddings.append(point.vector)
                    
                    # Calculate interaction weight based on type and recency
                    weight = self._calculate_interaction_weight(interaction_type, timestamp)
                    interaction_weights.append(weight)
                    
                    # Track cluster preferences
                    cluster_id = point.payload.get("cluster_id", -1)
                    if cluster_id != -1:
                        cluster_preferences[cluster_id] = cluster_preferences.get(cluster_id, 0) + weight
                    
                    # Track content type preferences
                    content_type = point.payload.get("content_type", "unknown")
                    content_type_preferences[content_type] = content_type_preferences.get(content_type, 0) + weight
                    
            except Exception as e:
                logger.warning(f"Failed to retrieve interaction point {point_id}", error=str(e))
                continue
        
        if not interacted_embeddings:
            return None
        
        # Create weighted average embedding as user profile
        interacted_embeddings = np.array(interacted_embeddings)
        interaction_weights = np.array(interaction_weights)
        
        # Normalize weights
        interaction_weights = interaction_weights / np.sum(interaction_weights)
        
        # Calculate weighted average embedding
        profile_embedding = np.average(interacted_embeddings, axis=0, weights=interaction_weights)
        
        # Get top preferred clusters and content types
        preferred_clusters = sorted(cluster_preferences.items(), key=lambda x: x[1], reverse=True)[:3]
        preferred_content_types = sorted(content_type_preferences.items(), key=lambda x: x[1], reverse=True)[:3]
        
        user_profile = {
            "profile_embedding": profile_embedding,
            "preferred_clusters": [cluster_id for cluster_id, _ in preferred_clusters],
            "content_types": [content_type for content_type, _ in preferred_content_types],
            "interaction_count": len(user_interactions),
            "total_weight": float(np.sum(interaction_weights))
        }
        
        # Cache user profile
        self.content_profiles[session_id] = user_profile
        
        return user_profile
    
    def _calculate_interaction_weight(self, interaction_type: str, timestamp: float) -> float:
        """Calculate weight for user interaction based on type and recency."""
        
        # Base weights for different interaction types
        type_weights = {
            "view": 1.0,
            "click": 2.0,
            "like": 3.0,
            "share": 4.0,
            "bookmark": 5.0,
            "download": 3.0
        }
        
        base_weight = type_weights.get(interaction_type, 1.0)
        
        # Apply recency decay (interactions in last 24 hours get full weight)
        current_time = time.time()
        hours_ago = (current_time - timestamp) / 3600
        
        if hours_ago <= 24:
            recency_factor = 1.0
        elif hours_ago <= 168:  # 1 week
            recency_factor = 0.8
        elif hours_ago <= 720:  # 1 month
            recency_factor = 0.6
        else:
            recency_factor = 0.4
        
        return base_weight * recency_factor
    
    async def _apply_diversity_filtering(
        self,
        candidates: List[Dict[str, Any]],
        top_k: int,
        diversity_factor: float
    ) -> List[Dict[str, Any]]:
        """Apply diversity filtering to recommendation candidates."""
        
        if diversity_factor <= 0 or len(candidates) <= top_k:
            return candidates[:top_k]
        
        selected = []
        remaining = candidates.copy()
        
        # Always select the top candidate
        if remaining:
            selected.append(remaining.pop(0))
        
        # Select remaining candidates balancing relevance and diversity
        while len(selected) < top_k and remaining:
            best_candidate = None
            best_score = -1
            best_idx = -1
            
            for idx, candidate in enumerate(remaining):
                # Calculate diversity score (average distance from selected items)
                diversity_score = self._calculate_diversity_score(candidate, selected)
                
                # Combine relevance and diversity
                relevance_score = candidate["similarity_score"]
                combined_score = (1 - diversity_factor) * relevance_score + diversity_factor * diversity_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = candidate
                    best_idx = idx
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.pop(best_idx)
        
        return selected
    
    def _calculate_diversity_score(
        self, 
        candidate: Dict[str, Any], 
        selected: List[Dict[str, Any]]
    ) -> float:
        """Calculate diversity score for a candidate relative to selected items."""
        
        if not selected:
            return 1.0
        
        # Use cluster diversity as a proxy for content diversity
        candidate_cluster = candidate.get("cluster_id", -1)
        selected_clusters = [item.get("cluster_id", -1) for item in selected]
        
        # Higher score if candidate is from a different cluster
        if candidate_cluster not in selected_clusters:
            return 1.0
        else:
            # Lower score based on how many items from same cluster are already selected
            same_cluster_count = selected_clusters.count(candidate_cluster)
            return 1.0 / (1 + same_cluster_count)
    
    def _calculate_recommendation_score(
        self, 
        item: Dict[str, Any], 
        user_profile: Dict[str, Any]
    ) -> float:
        """Calculate final recommendation score for an item."""
        
        base_score = item.get("similarity_score", 0.0)
        
        # Boost score if item is from preferred cluster
        cluster_id = item.get("cluster_id", -1)
        if cluster_id in user_profile.get("preferred_clusters", []):
            cluster_boost = 0.1
        else:
            cluster_boost = 0.0
        
        # Boost score if item is of preferred content type
        content_type = item.get("content_type", "unknown")
        if content_type in user_profile.get("content_types", []):
            content_type_boost = 0.05
        else:
            content_type_boost = 0.0
        
        # Quality score boost
        quality_boost = item.get("quality_score", 0.0) * 0.1
        
        final_score = base_score + cluster_boost + content_type_boost + quality_boost
        return min(final_score, 1.0)  # Cap at 1.0
    
    def _generate_recommendation_reason(
        self, 
        item: Dict[str, Any], 
        user_profile: Dict[str, Any]
    ) -> str:
        """Generate human-readable reason for recommendation."""
        
        reasons = []
        
        # Similarity-based reason
        similarity = item.get("similarity_score", 0.0)
        if similarity > 0.8:
            reasons.append("highly similar to your interests")
        elif similarity > 0.6:
            reasons.append("similar to your interests")
        else:
            reasons.append("related to your interests")
        
        # Cluster-based reason
        cluster_id = item.get("cluster_id", -1)
        if cluster_id in user_profile.get("preferred_clusters", []):
            reasons.append("from your preferred topic area")
        
        # Content type reason
        content_type = item.get("content_type", "unknown")
        if content_type in user_profile.get("content_types", []):
            reasons.append(f"matches your preference for {content_type} content")
        
        # Quality reason
        quality_score = item.get("quality_score", 0.0)
        if quality_score > 0.8:
            reasons.append("high quality content")
        
        if not reasons:
            return "recommended based on your activity"
        
        return "Recommended because it's " + " and ".join(reasons)
    
    async def collaborative_filtering_recommendations(
        self,
        session_id: str,
        user_interactions: List[Dict[str, Any]],
        top_k: int = 10
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Generate collaborative filtering recommendations.
        Note: This is a simplified implementation for demonstration.
        In production, you'd want a more sophisticated collaborative filtering system.
        """
        
        # For now, return content-based recommendations with a note
        # In a full implementation, this would analyze patterns across multiple users
        recommendations, metadata = await self.content_based_recommendations(
            session_id, user_interactions, top_k, diversity_factor=0.2
        )
        
        metadata["recommendation_type"] = "collaborative_filtering_fallback"
        metadata["note"] = "Using content-based filtering as collaborative data is limited"
        
        return recommendations, metadata

# Initialize core components
dimensionality_reducer = DimensionalityReducer()
visualization_generator = VisualizationGenerator()
hdbscan_clusterer = HDBSCANClusterer()
similarity_search_engine = SimilaritySearchEngine()

# Job tracking
active_jobs = {}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "clustering",
        "timestamp": time.time(),
        "qdrant_connected": await _check_qdrant_connection()
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Clustering Service",
        "version": "1.0.0",
        "status": "running",
        "features": [
            "UMAP dimensionality reduction",
            "Model-aware optimization", 
            "Batch processing",
            "Interactive visualizations",
            "Performance metrics"
        ]
    }

@app.post("/umap/reduce", response_model=UMAPResponse)
async def reduce_dimensions(request: UMAPRequest, background_tasks: BackgroundTasks):
    """
    Reduce embedding dimensions using UMAP with model-aware optimization.
    """
    job_id = f"umap_{request.session_id}_{int(time.time())}"
    
    # Initialize job status
    active_jobs[job_id] = {
        "status": "started",
        "message": "Initializing UMAP reduction",
        "start_time": time.time()
    }
    
    # Start background processing
    background_tasks.add_task(
        _process_umap_reduction,
        job_id,
        request
    )
    
    return UMAPResponse(
        job_id=job_id,
        status="started",
        message="UMAP reduction job started"
    )

@app.get("/umap/status/{job_id}")
async def get_umap_status(job_id: str):
    """Get status of UMAP reduction job."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return active_jobs[job_id]

@app.get("/umap/models/{embedding_model}/config")
async def get_model_config(embedding_model: str, n_samples: int = 1000, memory_limit_mb: int = 2048):
    """Get optimized UMAP configuration for specific embedding model."""
    config = ModelAwareUMAPConfig.get_config(embedding_model, n_samples, memory_limit_mb)
    
    return {
        "embedding_model": embedding_model,
        "recommended_config": config,
        "n_samples": n_samples,
        "memory_limit_mb": memory_limit_mb
    }

@app.get("/umap/models/supported")
async def get_supported_models():
    """Get list of supported embedding models with their characteristics."""
    return {
        "supported_models": ModelAwareUMAPConfig.EMBEDDING_MODEL_CONFIGS,
        "default_model": "nomic-embed-text"
    }

@app.post("/visualize/create")
async def create_visualization(request: VisualizationRequest):
    """Create interactive visualization of dimensionality-reduced embeddings."""
    try:
        # Retrieve reduced embeddings from Qdrant
        collection_name = f"session_{request.session_id}"
        
        # Check if collection exists
        collections = await asyncio.get_event_loop().run_in_executor(
            executor, qdrant_client.get_collections
        )
        
        collection_exists = any(
            col.name == collection_name 
            for col in collections.collections
        )
        
        if not collection_exists:
            raise HTTPException(
                status_code=404, 
                detail=f"Session {request.session_id} not found"
            )
        
        # Retrieve points with reduced embeddings
        search_result = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: qdrant_client.scroll(
                collection_name=collection_name,
                limit=10000,  # Adjust based on needs
                with_payload=True,
                with_vectors=True
            )
        )
        
        points = search_result[0]
        
        if not points:
            raise HTTPException(
                status_code=404,
                detail="No embeddings found for visualization"
            )
        
        # Extract reduced embeddings and metadata
        reduced_embeddings = []
        metadata = []
        
        for point in points:
            if "reduced_embedding" in point.payload:
                reduced_embeddings.append(point.payload["reduced_embedding"])
                metadata.append({
                    "cluster_id": point.payload.get("cluster_id", -1),
                    "embedding_model": point.payload.get("embedding_model", "unknown"),
                    "quality_score": point.payload.get("quality_score", 0.0),
                    "title": point.payload.get("title", ""),
                    "url": point.payload.get("url", "")
                })
        
        if not reduced_embeddings:
            raise HTTPException(
                status_code=404,
                detail="No reduced embeddings found. Run UMAP reduction first."
            )
        
        reduced_embeddings = np.array(reduced_embeddings)
        
        # Get performance metrics if available
        performance_metrics = None
        if points and "umap_metrics" in points[0].payload:
            performance_metrics = points[0].payload["umap_metrics"]
        
        # Create visualization
        plot_result = await visualization_generator.create_cluster_plot(
            reduced_embeddings=reduced_embeddings,
            metadata=metadata,
            plot_type=request.plot_type,
            color_by=request.color_by,
            include_metrics=request.include_metrics,
            performance_metrics=performance_metrics
        )
        
        return plot_result
        
    except Exception as e:
        logger.error("Error creating visualization", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance/metrics/{job_id}")
async def get_performance_metrics(job_id: str):
    """Get detailed performance metrics for a UMAP reduction job."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = active_jobs[job_id]
    
    if "performance_metrics" not in job_data:
        return {"message": "Performance metrics not yet available"}
    
    return job_data["performance_metrics"]

@app.get("/performance/models")
async def get_model_performance():
    """Get performance comparison across different embedding models."""
    cached_models = dimensionality_reducer.list_cached_models()
    
    performance_data = {}
    for model_key in cached_models:
        metrics = dimensionality_reducer.get_performance_metrics(model_key)
        if metrics:
            performance_data[model_key] = metrics
    
    return {
        "model_performance": performance_data,
        "total_models_cached": len(cached_models)
    }

@app.post("/hdbscan/cluster", response_model=HDBSCANResponse)
async def cluster_with_hdbscan(request: HDBSCANRequest, background_tasks: BackgroundTasks):
    """
    Perform HDBSCAN clustering with parameter optimization.
    """
    job_id = f"hdbscan_{request.session_id}_{int(time.time())}"
    
    # Initialize job status
    active_jobs[job_id] = {
        "status": "started",
        "message": "Initializing HDBSCAN clustering",
        "start_time": time.time()
    }
    
    # Start background processing
    background_tasks.add_task(
        _process_hdbscan_clustering,
        job_id,
        request
    )
    
    return HDBSCANResponse(
        job_id=job_id,
        status="started",
        message="HDBSCAN clustering job started"
    )

@app.get("/hdbscan/status/{job_id}")
async def get_hdbscan_status(job_id: str):
    """Get status of HDBSCAN clustering job."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return active_jobs[job_id]

@app.get("/hdbscan/parameters/optimize")
async def get_optimized_parameters(
    session_id: str,
    embedding_model: str = "unknown",
    n_samples: Optional[int] = None
):
    """Get optimized HDBSCAN parameters for a dataset."""
    try:
        # If n_samples not provided, try to get from session
        if not n_samples:
            collection_name = f"session_{session_id}"
            
            # Get collection info
            collections = await asyncio.get_event_loop().run_in_executor(
                executor, qdrant_client.get_collections
            )
            
            collection_exists = any(
                col.name == collection_name 
                for col in collections.collections
            )
            
            if not collection_exists:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Session {session_id} not found"
                )
            
            # Get collection info to estimate size
            collection_info = await asyncio.get_event_loop().run_in_executor(
                executor, qdrant_client.get_collection, collection_name
            )
            n_samples = collection_info.points_count
        
        # Create dummy embeddings for parameter optimization
        dummy_embeddings = np.random.randn(min(n_samples, 1000), 768)  # Use reasonable dimensions
        
        # Get optimized parameters
        params = hdbscan_clusterer._optimize_parameters(
            dummy_embeddings, n_samples, embedding_model
        )
        
        return {
            "session_id": session_id,
            "embedding_model": embedding_model,
            "n_samples": n_samples,
            "optimized_parameters": params,
            "parameter_explanation": {
                "min_cluster_size": "Minimum number of points required to form a cluster",
                "min_samples": "Minimum number of points in neighborhood for core point",
                "metric": "Distance metric used for clustering",
                "alpha": "Controls cluster selection strictness",
                "cluster_selection_epsilon": "Distance threshold for cluster merging"
            }
        }
        
    except Exception as e:
        logger.error("Error optimizing HDBSCAN parameters", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/hdbscan/metrics/{job_id}")
async def get_cluster_metrics(job_id: str):
    """Get detailed clustering validation metrics."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = active_jobs[job_id]
    
    if "clustering_metrics" not in job_data:
        return {"message": "Clustering metrics not yet available"}
    
    return job_data["clustering_metrics"]

@app.get("/hdbscan/hierarchy/{clusterer_key}")
async def get_cluster_hierarchy(clusterer_key: str):
    """Get hierarchical clustering information."""
    hierarchy = hdbscan_clusterer.get_cluster_hierarchy(clusterer_key)
    
    if hierarchy is None:
        raise HTTPException(
            status_code=404, 
            detail="Clusterer not found or hierarchy not available"
        )
    
    return hierarchy

@app.post("/hdbscan/predict/{clusterer_key}")
async def predict_clusters(clusterer_key: str, embeddings: List[List[float]]):
    """Predict cluster labels for new embeddings."""
    try:
        new_embeddings = np.array(embeddings)
        predictions = hdbscan_clusterer.predict_cluster(clusterer_key, new_embeddings)
        
        if predictions is None:
            raise HTTPException(
                status_code=404,
                detail="Clusterer not found or prediction failed"
            )
        
        return {
            "clusterer_key": clusterer_key,
            "predictions": predictions.tolist(),
            "n_predictions": len(predictions)
        }
        
    except Exception as e:
        logger.error("Error predicting clusters", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions
async def _check_qdrant_connection() -> bool:
    """Check if Qdrant is accessible."""
    try:
        await asyncio.get_event_loop().run_in_executor(
            executor, qdrant_client.get_collections
        )
        return True
    except Exception:
        return False

async def _process_umap_reduction(job_id: str, request: UMAPRequest):
    """Background task to process UMAP dimensionality reduction."""
    try:
        # Update job status
        active_jobs[job_id]["status"] = "loading_data"
        active_jobs[job_id]["message"] = "Loading embeddings from Qdrant"
        
        # Load embeddings from Qdrant
        collection_name = f"session_{request.session_id}"
        
        # Check if collection exists
        collections = await asyncio.get_event_loop().run_in_executor(
            executor, qdrant_client.get_collections
        )
        
        collection_exists = any(
            col.name == collection_name 
            for col in collections.collections
        )
        
        if not collection_exists:
            active_jobs[job_id]["status"] = "error"
            active_jobs[job_id]["message"] = f"Session {request.session_id} not found"
            return
        
        # Retrieve embeddings
        search_result = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: qdrant_client.scroll(
                collection_name=collection_name,
                limit=10000,  # Adjust based on needs
                with_payload=True,
                with_vectors=True
            )
        )
        
        points = search_result[0]
        
        if not points:
            active_jobs[job_id]["status"] = "error"
            active_jobs[job_id]["message"] = "No embeddings found in session"
            return
        
        # Extract embeddings and filter by model if specified
        embeddings = []
        point_ids = []
        
        for point in points:
            if point.vector and (
                not request.embedding_model or 
                point.payload.get("embedding_model") == request.embedding_model
            ):
                embeddings.append(point.vector)
                point_ids.append(point.id)
        
        if not embeddings:
            active_jobs[job_id]["status"] = "error"
            active_jobs[job_id]["message"] = f"No embeddings found for model {request.embedding_model}"
            return
        
        embeddings = np.array(embeddings)
        
        # Update job status
        active_jobs[job_id]["status"] = "processing"
        active_jobs[job_id]["message"] = f"Reducing {len(embeddings)} embeddings"
        
        # Perform UMAP reduction
        custom_params = {}
        if request.n_neighbors:
            custom_params["n_neighbors"] = request.n_neighbors
        if request.min_dist:
            custom_params["min_dist"] = request.min_dist
        if request.metric:
            custom_params["metric"] = request.metric
        
        reduced_embeddings, performance_metrics = await dimensionality_reducer.reduce_embeddings(
            embeddings=embeddings,
            embedding_model=request.embedding_model,
            n_components=request.n_components,
            custom_params=custom_params if custom_params else None,
            batch_size=request.batch_size,
            memory_limit_mb=request.memory_limit_mb
        )
        
        # Update job status
        active_jobs[job_id]["status"] = "storing"
        active_jobs[job_id]["message"] = "Storing reduced embeddings"
        
        # Store reduced embeddings back to Qdrant
        for i, point_id in enumerate(point_ids):
            await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda i=i, point_id=point_id: qdrant_client.set_payload(
                    collection_name=collection_name,
                    payload={
                        "reduced_embedding": reduced_embeddings[i].tolist(),
                        "umap_metrics": performance_metrics,
                        "reduction_timestamp": time.time()
                    },
                    points=[point_id]
                )
            )
        
        # Complete job
        active_jobs[job_id]["status"] = "completed"
        active_jobs[job_id]["message"] = "UMAP reduction completed successfully"
        active_jobs[job_id]["performance_metrics"] = performance_metrics
        active_jobs[job_id]["reduced_dimensions"] = request.n_components
        active_jobs[job_id]["original_dimensions"] = embeddings.shape[1]
        active_jobs[job_id]["n_samples"] = len(embeddings)
        active_jobs[job_id]["end_time"] = time.time()
        
        logger.info(
            "UMAP reduction job completed",
            job_id=job_id,
            **performance_metrics
        )
        
    except Exception as e:
        active_jobs[job_id]["status"] = "error"
        active_jobs[job_id]["message"] = f"Error: {str(e)}"
        active_jobs[job_id]["error_details"] = str(e)
        
        logger.error(
            "UMAP reduction job failed",
            job_id=job_id,
            error=str(e)
        )

async def _process_hdbscan_clustering(job_id: str, request: HDBSCANRequest):
    """Background task to process HDBSCAN clustering."""
    try:
        # Update job status
        active_jobs[job_id]["status"] = "loading_data"
        active_jobs[job_id]["message"] = "Loading embeddings from Qdrant"
        
        # Load embeddings from Qdrant
        collection_name = f"session_{request.session_id}"
        
        # Check if collection exists
        collections = await asyncio.get_event_loop().run_in_executor(
            executor, qdrant_client.get_collections
        )
        
        collection_exists = any(
            col.name == collection_name 
            for col in collections.collections
        )
        
        if not collection_exists:
            active_jobs[job_id]["status"] = "error"
            active_jobs[job_id]["message"] = f"Session {request.session_id} not found"
            return
        
        # Retrieve embeddings
        search_result = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: qdrant_client.scroll(
                collection_name=collection_name,
                limit=10000,  # Adjust based on needs
                with_payload=True,
                with_vectors=True
            )
        )
        
        points = search_result[0]
        
        if not points:
            active_jobs[job_id]["status"] = "error"
            active_jobs[job_id]["message"] = "No embeddings found in session"
            return
        
        # Determine which embeddings to use
        embeddings = []
        point_ids = []
        embedding_model = "unknown"
        
        for point in points:
            # Use reduced embeddings if requested and available
            if request.use_reduced_embeddings and "reduced_embedding" in point.payload:
                embeddings.append(point.payload["reduced_embedding"])
                point_ids.append(point.id)
                embedding_model = point.payload.get("embedding_model", "unknown")
            elif not request.use_reduced_embeddings and point.vector:
                embeddings.append(point.vector)
                point_ids.append(point.id)
                embedding_model = point.payload.get("embedding_model", "unknown")
        
        if not embeddings:
            active_jobs[job_id]["status"] = "error"
            active_jobs[job_id]["message"] = (
                "No reduced embeddings found. Run UMAP reduction first." 
                if request.use_reduced_embeddings 
                else "No embeddings found in session"
            )
            return
        
        embeddings = np.array(embeddings)
        
        # Update job status
        active_jobs[job_id]["status"] = "processing"
        active_jobs[job_id]["message"] = f"Clustering {len(embeddings)} embeddings with HDBSCAN"
        
        # Prepare custom parameters
        custom_params = {}
        if request.min_cluster_size:
            custom_params["min_cluster_size"] = request.min_cluster_size
        if request.min_samples:
            custom_params["min_samples"] = request.min_samples
        if request.cluster_selection_epsilon:
            custom_params["cluster_selection_epsilon"] = request.cluster_selection_epsilon
        if request.alpha:
            custom_params["alpha"] = request.alpha
        if request.metric:
            custom_params["metric"] = request.metric
        
        # Perform HDBSCAN clustering
        cluster_labels, clustering_metrics = await hdbscan_clusterer.cluster_embeddings(
            embeddings=embeddings,
            session_id=request.session_id,
            custom_params=custom_params if custom_params else None,
            auto_optimize=request.auto_optimize,
            embedding_model=embedding_model
        )
        
        # Update job status
        active_jobs[job_id]["status"] = "storing"
        active_jobs[job_id]["message"] = "Storing cluster assignments"
        
        # Store cluster assignments back to Qdrant
        for i, point_id in enumerate(point_ids):
            await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda i=i, point_id=point_id: qdrant_client.set_payload(
                    collection_name=collection_name,
                    payload={
                        "cluster_id": int(cluster_labels[i]),
                        "clustering_metrics": clustering_metrics,
                        "clustering_timestamp": time.time(),
                        "clustering_method": "hdbscan"
                    },
                    points=[point_id]
                )
            )
        
        # Complete job
        active_jobs[job_id]["status"] = "completed"
        active_jobs[job_id]["message"] = "HDBSCAN clustering completed successfully"
        active_jobs[job_id]["clustering_metrics"] = clustering_metrics
        active_jobs[job_id]["n_clusters"] = clustering_metrics["n_clusters"]
        active_jobs[job_id]["n_noise_points"] = clustering_metrics["n_noise_points"]
        active_jobs[job_id]["silhouette_score"] = clustering_metrics.get("silhouette_score", 0.0)
        active_jobs[job_id]["end_time"] = time.time()
        
        logger.info(
            "HDBSCAN clustering job completed",
            job_id=job_id,
            **clustering_metrics
        )
        
    except Exception as e:
        active_jobs[job_id]["status"] = "error"
        active_jobs[job_id]["message"] = f"Error: {str(e)}"
        active_jobs[job_id]["error_details"] = str(e)
        
        logger.error(
            "HDBSCAN clustering job failed",
            job_id=job_id,
            error=str(e)
        )

@app.post("/similarity/search", response_model=SimilaritySearchResponse)
async def similarity_search(request: SimilaritySearchRequest):
    """
    Perform vector similarity search to find similar content.
    """
    try:
        # Validate request
        if not request.query_embedding and not request.query_text:
            raise HTTPException(
                status_code=400,
                detail="Either query_embedding or query_text must be provided"
            )
        
        # For now, we'll focus on query_embedding. In a full implementation,
        # query_text would be converted to embedding using the same model used for the session
        if not request.query_embedding:
            raise HTTPException(
                status_code=400,
                detail="query_text to embedding conversion not implemented. Please provide query_embedding."
            )
        
        query_embedding = np.array(request.query_embedding)
        
        # Perform similarity search
        results, search_metadata = await similarity_search_engine.vector_similarity_search(
            session_id=request.session_id,
            query_embedding=query_embedding,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            filter_cluster_id=request.filter_cluster_id,
            use_reduced_embeddings=request.use_reduced_embeddings
        )
        
        query_info = {
            "session_id": request.session_id,
            "embedding_dimensions": len(request.query_embedding),
            "top_k": request.top_k,
            "similarity_threshold": request.similarity_threshold,
            "filter_cluster_id": request.filter_cluster_id,
            "use_reduced_embeddings": request.use_reduced_embeddings
        }
        
        return SimilaritySearchResponse(
            results=results,
            query_info=query_info,
            search_metadata=search_metadata
        )
        
    except Exception as e:
        logger.error("Error performing similarity search", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommendations/content-based", response_model=RecommendationResponse)
async def get_content_based_recommendations(request: RecommendationRequest):
    """
    Generate content-based recommendations based on user interactions.
    """
    try:
        if not request.user_interactions:
            raise HTTPException(
                status_code=400,
                detail="user_interactions cannot be empty"
            )
        
        # Generate recommendations based on type
        if request.recommendation_type == "content_based":
            recommendations, metadata = await similarity_search_engine.content_based_recommendations(
                session_id=request.session_id,
                user_interactions=request.user_interactions,
                top_k=request.top_k,
                diversity_factor=request.diversity_factor
            )
        elif request.recommendation_type == "collaborative":
            recommendations, metadata = await similarity_search_engine.collaborative_filtering_recommendations(
                session_id=request.session_id,
                user_interactions=request.user_interactions,
                top_k=request.top_k
            )
        elif request.recommendation_type == "hybrid":
            # Combine content-based and collaborative
            content_recs, content_meta = await similarity_search_engine.content_based_recommendations(
                session_id=request.session_id,
                user_interactions=request.user_interactions,
                top_k=request.top_k // 2,
                diversity_factor=request.diversity_factor
            )
            
            collab_recs, collab_meta = await similarity_search_engine.collaborative_filtering_recommendations(
                session_id=request.session_id,
                user_interactions=request.user_interactions,
                top_k=request.top_k // 2
            )
            
            # Merge recommendations
            recommendations = content_recs + collab_recs
            recommendations = recommendations[:request.top_k]  # Limit to requested count
            
            metadata = {
                "recommendation_type": "hybrid",
                "content_based_metadata": content_meta,
                "collaborative_metadata": collab_meta,
                "total_recommendations": len(recommendations)
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported recommendation_type: {request.recommendation_type}"
            )
        
        recommendation_metadata = {
            "session_id": request.session_id,
            "recommendation_type": request.recommendation_type,
            "user_interactions_count": len(request.user_interactions),
            "top_k": request.top_k,
            "diversity_factor": request.diversity_factor,
            **metadata
        }
        
        return RecommendationResponse(
            recommendations=recommendations,
            recommendation_metadata=recommendation_metadata
        )
        
    except Exception as e:
        logger.error("Error generating recommendations", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/similarity/search/{session_id}/clusters/{cluster_id}")
async def get_cluster_similar_items(
    session_id: str,
    cluster_id: int,
    top_k: int = 10,
    similarity_threshold: float = 0.5
):
    """Get items similar to a specific cluster centroid."""
    try:
        # Get all items in the cluster to calculate centroid
        collection_name = f"session_{session_id}"
        
        # Retrieve points with cluster filter
        search_result = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="cluster_id",
                            match=MatchValue(value=cluster_id)
                        )
                    ]
                ),
                limit=1000,
                with_payload=True,
                with_vectors=True
            )
        )
        
        cluster_points = search_result[0]
        
        if not cluster_points:
            raise HTTPException(
                status_code=404,
                detail=f"No points found in cluster {cluster_id}"
            )
        
        # Calculate cluster centroid
        cluster_embeddings = []
        for point in cluster_points:
            if point.vector:
                cluster_embeddings.append(point.vector)
        
        if not cluster_embeddings:
            raise HTTPException(
                status_code=404,
                detail=f"No embeddings found for cluster {cluster_id}"
            )
        
        cluster_centroid = np.mean(cluster_embeddings, axis=0)
        
        # Find similar items using centroid
        results, search_metadata = await similarity_search_engine.vector_similarity_search(
            session_id=session_id,
            query_embedding=cluster_centroid,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            filter_cluster_id=None,  # Don't filter by cluster to find cross-cluster similarities
            use_reduced_embeddings=False
        )
        
        return {
            "cluster_id": cluster_id,
            "cluster_size": len(cluster_points),
            "centroid_dimensions": len(cluster_centroid),
            "similar_items": results,
            "search_metadata": search_metadata
        }
        
    except Exception as e:
        logger.error("Error finding cluster similar items", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommendations/trending/{session_id}")
async def get_trending_recommendations(
    session_id: str,
    time_window_hours: int = 24,
    top_k: int = 10
):
    """Get trending content recommendations based on recent activity patterns."""
    try:
        # This is a simplified trending algorithm
        # In production, you'd track actual user engagement metrics
        
        collection_name = f"session_{session_id}"
        current_time = time.time()
        time_threshold = current_time - (time_window_hours * 3600)
        
        # Get recent content (simulated by using timestamp if available)
        search_result = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: qdrant_client.scroll(
                collection_name=collection_name,
                limit=1000,
                with_payload=True,
                with_vectors=False
            )
        )
        
        points = search_result[0]
        
        # Score items based on recency and quality
        trending_items = []
        for point in points:
            timestamp = point.payload.get("timestamp", current_time)
            quality_score = point.payload.get("quality_score", 0.5)
            cluster_id = point.payload.get("cluster_id", -1)
            
            # Calculate trending score (recency + quality)
            recency_score = max(0, 1 - (current_time - timestamp) / (time_window_hours * 3600))
            trending_score = 0.6 * recency_score + 0.4 * quality_score
            
            trending_items.append({
                "point_id": point.id,
                "title": point.payload.get("title", ""),
                "url": point.payload.get("url", ""),
                "cluster_id": cluster_id,
                "quality_score": quality_score,
                "trending_score": trending_score,
                "recency_score": recency_score,
                "timestamp": timestamp
            })
        
        # Sort by trending score and return top-k
        trending_items.sort(key=lambda x: x["trending_score"], reverse=True)
        top_trending = trending_items[:top_k]
        
        metadata = {
            "session_id": session_id,
            "time_window_hours": time_window_hours,
            "total_items_considered": len(trending_items),
            "top_k": top_k,
            "average_trending_score": np.mean([item["trending_score"] for item in top_trending]) if top_trending else 0.0,
            "algorithm": "recency_quality_weighted"
        }
        
        return {
            "trending_recommendations": top_trending,
            "metadata": metadata
        }
        
    except Exception as e:
        logger.error("Error generating trending recommendations", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))