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
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score
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

# Initialize core components
dimensionality_reducer = DimensionalityReducer()
visualization_generator = VisualizationGenerator()

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