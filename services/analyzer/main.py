"""Content Analyzer Service - Generates embeddings and summaries with configurable models."""

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import uuid

import numpy as np
import psutil
import structlog
import tiktoken
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import torch

# Configure structured logging
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
    title="Content Analyzer Service",
    description="Generates embeddings and summaries for scraped content with configurable models",
    version="1.0.0"
)

# Pydantic models for API
class ContentItem(BaseModel):
    id: str
    content: str
    title: Optional[str] = None
    url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EmbeddingRequest(BaseModel):
    content_items: List[ContentItem]
    session_id: str
    embedding_model: Optional[str] = None
    chunk_size: int = Field(default=512, ge=100, le=2048)
    chunk_overlap: int = Field(default=50, ge=0, le=200)

class EmbeddingResponse(BaseModel):
    job_id: str
    status: str
    message: str

class ModelSwitchRequest(BaseModel):
    embedding_model: str

class HardwareInfo(BaseModel):
    ram_gb: float
    cpu_count: int
    has_gpu: bool
    gpu_memory_gb: float
    gpu_name: str
    available_ram_gb: float
    ram_usage_percent: float

class ModelRecommendation(BaseModel):
    recommended_model: str
    reason: str
    alternatives: List[str]
    performance_estimate: Dict[str, Any]

# Global components
hardware_detector = None
embedding_generator = None
text_chunker = None
embedding_cache = None
model_manager = None

class HardwareDetector:
    """Detect and monitor system hardware capabilities."""
    
    def __init__(self):
        self.logger = structlog.get_logger("hardware_detector")
    
    def detect_hardware(self) -> Dict[str, Any]:
        """Detect comprehensive hardware capabilities."""
        try:
            # Basic system info
            ram_gb = psutil.virtual_memory().total / (1024**3)
            cpu_count = psutil.cpu_count()
            
            # GPU detection
            has_gpu = False
            gpu_memory_gb = 0.0
            gpu_name = "None"
            
            try:
                if torch.cuda.is_available():
                    has_gpu = True
                    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    gpu_name = torch.cuda.get_device_name(0)
            except Exception as e:
                self.logger.warning("GPU detection failed", error=str(e))
            
            # Current resource usage
            memory = psutil.virtual_memory()
            available_ram_gb = memory.available / (1024**3)
            
            return {
                "ram_gb": ram_gb,
                "cpu_count": cpu_count,
                "has_gpu": has_gpu,
                "gpu_memory_gb": gpu_memory_gb,
                "gpu_name": gpu_name,
                "available_ram_gb": available_ram_gb,
                "ram_usage_percent": memory.percent,
                "cpu_usage_percent": psutil.cpu_percent(interval=0.1)
            }
        except Exception as e:
            self.logger.error("Hardware detection failed", error=str(e))
            # Return safe defaults
            return {
                "ram_gb": 8.0,
                "cpu_count": 4,
                "has_gpu": False,
                "gpu_memory_gb": 0.0,
                "gpu_name": "Unknown",
                "available_ram_gb": 4.0,
                "ram_usage_percent": 50.0,
                "cpu_usage_percent": 20.0
            }

class ModelManager:
    """Manage embedding model configurations and recommendations."""
    
    def __init__(self):
        self.logger = structlog.get_logger("model_manager")
        self.models_config = self._load_models_config()
    
    def _load_models_config(self) -> Dict[str, Any]:
        """Load models configuration from JSON file."""
        config_path = Path("/app/config/models.json")
        if not config_path.exists():
            # Fallback to relative path
            config_path = Path("../../config/models.json")
        
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning("Could not load models config", error=str(e))
            # Return minimal default config
            return {
                "embedding_models": {
                    "all-minilm": {
                        "name": "All-MiniLM",
                        "size": "90MB",
                        "dimensions": 384,
                        "quality": "good",
                        "min_ram_gb": 0.5,
                        "description": "Lightweight embedding model",
                        "model_name": "all-MiniLM-L6-v2",
                        "max_sequence_length": 512,
                        "recommended": True
                    }
                }
            }
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get all available embedding models."""
        return self.models_config.get("embedding_models", {})
    
    def recommend_model(self, hardware_info: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend best embedding model based on hardware."""
        available_ram = hardware_info.get("available_ram_gb", 4.0)
        has_gpu = hardware_info.get("has_gpu", False)
        
        suitable_models = []
        
        for model_id, config in self.models_config.get("embedding_models", {}).items():
            if config["min_ram_gb"] <= available_ram * 0.9:  # 90% safety margin
                score = self._calculate_model_score(config, has_gpu)
                suitable_models.append((model_id, config, score))
        
        if not suitable_models:
            # Fallback to most lightweight model
            return {
                "recommended_model": "all-minilm",
                "reason": "Fallback to lightweight model due to resource constraints",
                "alternatives": [],
                "performance_estimate": {"embeddings_per_sec": 10}
            }
        
        # Sort by score and select best
        suitable_models.sort(key=lambda x: x[2], reverse=True)
        best_model_id, best_config, _ = suitable_models[0]
        
        alternatives = [model[0] for model in suitable_models[1:3]]  # Top 2 alternatives
        
        return {
            "recommended_model": best_model_id,
            "reason": f"Best fit for {available_ram:.1f}GB available RAM",
            "alternatives": alternatives,
            "performance_estimate": self._estimate_performance(best_config, has_gpu)
        }
    
    def _calculate_model_score(self, config: Dict[str, Any], has_gpu: bool) -> float:
        """Calculate suitability score for a model."""
        score = 0.0
        
        # Quality score
        quality_scores = {"good": 2, "high": 3, "highest": 4}
        score += quality_scores.get(config["quality"], 2) * 10
        
        # Dimension bonus (higher dimensions often better for clustering)
        score += config["dimensions"] / 100
        
        # GPU bonus
        if has_gpu:
            score += 5
        
        # Recommended bonus
        if config.get("recommended", False):
            score += 8
        
        return score
    
    def _estimate_performance(self, config: Dict[str, Any], has_gpu: bool) -> Dict[str, Any]:
        """Estimate embedding generation performance."""
        base_speed = 100 if has_gpu else 50
        dimension_factor = config["dimensions"] / 384  # Normalize to all-minilm
        
        embeddings_per_sec = max(5, base_speed / dimension_factor)
        
        return {
            "embeddings_per_sec": round(embeddings_per_sec, 1),
            "dimensions": config["dimensions"],
            "suitable_for_batch": True
        }

class TextChunker:
    """Handle text chunking with overlap preservation."""
    
    def __init__(self):
        self.logger = structlog.get_logger("text_chunker")
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            self.logger.warning("Could not load tiktoken, using character-based chunking", error=str(e))
            self.tokenizer = None
    
    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks."""
        if not text or not text.strip():
            return []
        
        try:
            if self.tokenizer:
                return self._chunk_by_tokens(text, chunk_size, overlap)
            else:
                return self._chunk_by_characters(text, chunk_size * 4, overlap * 4)  # Rough char estimate
        except Exception as e:
            self.logger.error("Text chunking failed", error=str(e))
            # Fallback to simple splitting
            return [{"text": text, "chunk_index": 0, "token_count": len(text.split())}]
    
    def _chunk_by_tokens(self, text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """Chunk text by token count using tiktoken."""
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        start = 0
        chunk_index = 0
        
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            chunks.append({
                "text": chunk_text,
                "chunk_index": chunk_index,
                "token_count": len(chunk_tokens),
                "start_token": start,
                "end_token": end
            })
            
            # Move start position with overlap
            start = end - overlap
            chunk_index += 1
            
            if end >= len(tokens):
                break
        
        return chunks
    
    def _chunk_by_characters(self, text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """Fallback chunking by character count."""
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            
            chunks.append({
                "text": chunk_text,
                "chunk_index": chunk_index,
                "token_count": len(chunk_text.split()),
                "start_char": start,
                "end_char": end
            })
            
            start = end - overlap
            chunk_index += 1
            
            if end >= len(text):
                break
        
        return chunks

class EmbeddingCache:
    """Model-specific caching for embeddings."""
    
    def __init__(self):
        self.logger = structlog.get_logger("embedding_cache")
        self.cache_dir = Path("/app/cache/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}
        self.max_memory_items = 1000
    
    def _get_cache_key(self, text: str, model_id: str) -> str:
        """Generate cache key for text and model combination."""
        content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"{model_id}_{content_hash}"
    
    def get_embedding(self, text: str, model_id: str) -> Optional[np.ndarray]:
        """Retrieve cached embedding if available."""
        cache_key = self._get_cache_key(text, model_id)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.npy"
        if cache_file.exists():
            try:
                embedding = np.load(cache_file)
                # Add to memory cache
                if len(self.memory_cache) < self.max_memory_items:
                    self.memory_cache[cache_key] = embedding
                return embedding
            except Exception as e:
                self.logger.warning("Failed to load cached embedding", cache_key=cache_key, error=str(e))
        
        return None
    
    def store_embedding(self, text: str, model_id: str, embedding: np.ndarray):
        """Store embedding in cache."""
        cache_key = self._get_cache_key(text, model_id)
        
        # Store in memory cache
        if len(self.memory_cache) < self.max_memory_items:
            self.memory_cache[cache_key] = embedding
        
        # Store in disk cache
        cache_file = self.cache_dir / f"{cache_key}.npy"
        try:
            np.save(cache_file, embedding)
        except Exception as e:
            self.logger.warning("Failed to cache embedding", cache_key=cache_key, error=str(e))
    
    def clear_model_cache(self, model_id: str):
        """Clear cache for specific model."""
        # Clear memory cache
        keys_to_remove = [k for k in self.memory_cache.keys() if k.startswith(f"{model_id}_")]
        for key in keys_to_remove:
            del self.memory_cache[key]
        
        # Clear disk cache
        for cache_file in self.cache_dir.glob(f"{model_id}_*.npy"):
            try:
                cache_file.unlink()
            except Exception as e:
                self.logger.warning("Failed to remove cache file", file=str(cache_file), error=str(e))

class EmbeddingGenerator:
    """Generate embeddings with configurable models and dynamic switching."""
    
    def __init__(self, model_manager: ModelManager, embedding_cache: EmbeddingCache):
        self.logger = structlog.get_logger("embedding_generator")
        self.model_manager = model_manager
        self.embedding_cache = embedding_cache
        self.current_model_id = None
        self.current_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize with default model
        self._initialize_default_model()
    
    def _initialize_default_model(self):
        """Initialize with a default embedding model."""
        try:
            # Try to get hardware-based recommendation
            hardware_info = hardware_detector.detect_hardware() if hardware_detector else {}
            recommendation = self.model_manager.recommend_model(hardware_info)
            default_model = recommendation["recommended_model"]
        except Exception:
            default_model = "all-minilm"  # Safe fallback
        
        self.switch_model(default_model)
    
    def switch_model(self, model_id: str) -> bool:
        """Switch to a different embedding model without service restart."""
        if model_id == self.current_model_id and self.current_model is not None:
            return True  # Already using this model
        
        models_config = self.model_manager.get_available_models()
        if model_id not in models_config:
            self.logger.error("Unknown embedding model", model_id=model_id)
            return False
        
        model_config = models_config[model_id]
        
        try:
            # Get the actual model name for SentenceTransformers
            model_name = model_config.get("model_name", model_id)
            
            # Map our model IDs to actual model names
            model_name_mapping = {
                "all-minilm": "all-MiniLM-L6-v2",
                "nomic-embed-text": "nomic-ai/nomic-embed-text-v1",
                "mxbai-embed-large": "mixedbread-ai/mxbai-embed-large-v1"
            }
            
            actual_model_name = model_name_mapping.get(model_id, model_name)
            
            self.logger.info("Loading embedding model", model_id=model_id, model_name=actual_model_name)
            
            # Load new model
            new_model = SentenceTransformer(actual_model_name, device=self.device)
            
            # Clear old model from memory
            if self.current_model is not None:
                del self.current_model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Update current model
            self.current_model = new_model
            self.current_model_id = model_id
            
            self.logger.info("Successfully switched embedding model", 
                           model_id=model_id, 
                           device=self.device,
                           dimensions=model_config["dimensions"])
            return True
            
        except Exception as e:
            self.logger.error("Failed to switch embedding model", 
                            model_id=model_id, 
                            error=str(e))
            return False
    
    def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a list of texts."""
        if not self.current_model:
            raise RuntimeError("No embedding model loaded")
        
        if not texts:
            return []
        
        embeddings = []
        cache_hits = 0
        
        for text in texts:
            # Check cache first
            cached_embedding = self.embedding_cache.get_embedding(text, self.current_model_id)
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
                cache_hits += 1
                continue
            
            # Generate new embedding
            try:
                embedding = self.current_model.encode([text], convert_to_numpy=True)[0]
                embeddings.append(embedding)
                
                # Cache the embedding
                self.embedding_cache.store_embedding(text, self.current_model_id, embedding)
                
            except Exception as e:
                self.logger.error("Failed to generate embedding", error=str(e))
                # Return zero vector as fallback
                dimensions = self.model_manager.get_available_models()[self.current_model_id]["dimensions"]
                embeddings.append(np.zeros(dimensions))
        
        self.logger.info("Generated embeddings", 
                        total_texts=len(texts),
                        cache_hits=cache_hits,
                        new_embeddings=len(texts) - cache_hits)
        
        return embeddings
    
    def get_current_model_info(self) -> Dict[str, Any]:
        """Get information about currently loaded model."""
        if not self.current_model_id:
            return {"error": "No model loaded"}
        
        model_config = self.model_manager.get_available_models().get(self.current_model_id, {})
        return {
            "model_id": self.current_model_id,
            "model_name": model_config.get("name", "Unknown"),
            "dimensions": model_config.get("dimensions", 0),
            "device": self.device,
            "description": model_config.get("description", "")
        }

# Initialize global components
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global hardware_detector, embedding_generator, text_chunker, embedding_cache, model_manager
    
    logger.info("Initializing Content Analyzer Service")
    
    try:
        # Initialize components
        hardware_detector = HardwareDetector()
        model_manager = ModelManager()
        embedding_cache = EmbeddingCache()
        text_chunker = TextChunker()
        embedding_generator = EmbeddingGenerator(model_manager, embedding_cache)
        
        logger.info("Content Analyzer Service initialized successfully")
        
    except Exception as e:
        logger.error("Failed to initialize Content Analyzer Service", error=str(e))
        raise

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        model_info = embedding_generator.get_current_model_info() if embedding_generator else {}
        hardware_info = hardware_detector.detect_hardware() if hardware_detector else {}
        
        return {
            "status": "healthy",
            "service": "analyzer",
            "timestamp": time.time(),
            "current_model": model_info,
            "hardware": {
                "available_ram_gb": hardware_info.get("available_ram_gb", 0),
                "ram_usage_percent": hardware_info.get("ram_usage_percent", 0),
                "has_gpu": hardware_info.get("has_gpu", False)
            }
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "service": "analyzer",
            "timestamp": time.time(),
            "error": str(e)
        }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Content Analyzer Service",
        "version": "1.0.0",
        "status": "running",
        "features": [
            "Configurable embedding models",
            "Dynamic model switching",
            "Hardware-aware recommendations",
            "Text chunking with overlap",
            "Model-specific caching"
        ]
    }

@app.get("/hardware", response_model=HardwareInfo)
async def get_hardware_info():
    """Get current hardware information."""
    if not hardware_detector:
        raise HTTPException(status_code=503, detail="Hardware detector not initialized")
    
    hardware_info = hardware_detector.detect_hardware()
    return HardwareInfo(**hardware_info)

@app.get("/models/available")
async def get_available_models():
    """Get all available embedding models."""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    return model_manager.get_available_models()

@app.get("/models/current")
async def get_current_model():
    """Get information about currently loaded model."""
    if not embedding_generator:
        raise HTTPException(status_code=503, detail="Embedding generator not initialized")
    
    return embedding_generator.get_current_model_info()

@app.get("/models/recommend", response_model=ModelRecommendation)
async def get_model_recommendation():
    """Get hardware-based model recommendation."""
    if not hardware_detector or not model_manager:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    hardware_info = hardware_detector.detect_hardware()
    recommendation = model_manager.recommend_model(hardware_info)
    
    return ModelRecommendation(**recommendation)

@app.post("/models/switch")
async def switch_model(request: ModelSwitchRequest):
    """Switch to a different embedding model."""
    if not embedding_generator:
        raise HTTPException(status_code=503, detail="Embedding generator not initialized")
    
    success = embedding_generator.switch_model(request.embedding_model)
    
    if success:
        return {
            "status": "success",
            "message": f"Switched to model: {request.embedding_model}",
            "current_model": embedding_generator.get_current_model_info()
        }
    else:
        raise HTTPException(
            status_code=400, 
            detail=f"Failed to switch to model: {request.embedding_model}"
        )

@app.post("/embeddings/generate", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest, background_tasks: BackgroundTasks):
    """Generate embeddings for content items."""
    if not embedding_generator or not text_chunker:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    job_id = str(uuid.uuid4())
    
    # Switch model if requested
    if request.embedding_model:
        success = embedding_generator.switch_model(request.embedding_model)
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to switch to model: {request.embedding_model}"
            )
    
    # Start background processing
    background_tasks.add_task(
        process_embeddings_job,
        job_id,
        request.content_items,
        request.session_id,
        request.chunk_size,
        request.chunk_overlap
    )
    
    return EmbeddingResponse(
        job_id=job_id,
        status="processing",
        message=f"Started embedding generation for {len(request.content_items)} items"
    )

@app.get("/embeddings/status/{job_id}")
async def get_embedding_status(job_id: str):
    """Get status of embedding generation job."""
    # This would typically check a job queue/database
    # For now, return a simple response
    return {
        "job_id": job_id,
        "status": "completed",  # Simplified for this implementation
        "message": "Embedding generation completed"
    }

async def process_embeddings_job(
    job_id: str,
    content_items: List[ContentItem],
    session_id: str,
    chunk_size: int,
    chunk_overlap: int
):
    """Background task to process embeddings."""
    logger.info("Starting embedding job", job_id=job_id, items_count=len(content_items))
    
    try:
        # Initialize Qdrant client
        qdrant_client = QdrantClient(host="qdrant", port=6333)
        
        # Ensure collection exists
        collection_name = f"session_{session_id}"
        try:
            qdrant_client.get_collection(collection_name)
        except Exception:
            # Create collection if it doesn't exist
            model_info = embedding_generator.get_current_model_info()
            dimensions = model_info.get("dimensions", 384)
            
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dimensions, distance=Distance.COSINE)
            )
        
        # Process each content item
        points = []
        
        for item in content_items:
            # Chunk the content
            chunks = text_chunker.chunk_text(
                item.content,
                chunk_size=chunk_size,
                overlap=chunk_overlap
            )
            
            # Generate embeddings for chunks
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = embedding_generator.generate_embeddings(chunk_texts)
            
            # Create Qdrant points
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point_id = f"{item.id}_chunk_{i}"
                
                payload = {
                    "content_id": item.id,
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                    "title": item.title,
                    "url": item.url,
                    "token_count": chunk["token_count"],
                    "embedding_model": embedding_generator.current_model_id,
                    **item.metadata
                }
                
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=payload
                ))
        
        # Store in Qdrant
        if points:
            qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )
        
        logger.info("Embedding job completed", 
                   job_id=job_id, 
                   points_created=len(points))
        
    except Exception as e:
        logger.error("Embedding job failed", job_id=job_id, error=str(e))