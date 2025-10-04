"""Content Analyzer Service - Generates embeddings and summaries with configurable models."""

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import uuid
import re
from datetime import datetime

import numpy as np
import psutil
import structlog
import tiktoken
import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
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

class AnalysisRequest(BaseModel):
    content_items: List[ContentItem]
    session_id: str
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None
    generate_summary: bool = True
    extract_keywords: bool = True
    assess_quality: bool = True

class AnalysisResponse(BaseModel):
    job_id: str
    status: str
    message: str

class ModelPerformanceMetrics(BaseModel):
    model_id: str
    model_type: str  # "llm" or "embedding"
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    average_tokens_per_second: Optional[float] = None
    last_used: datetime
    resource_usage: Dict[str, float]

class QdrantConnectionInfo(BaseModel):
    host: str
    port: int
    collection_name: str
    vector_size: int
    distance_metric: str
    points_count: int

# Global components
hardware_detector = None
embedding_generator = None
text_chunker = None
embedding_cache = None
model_manager = None
ollama_client = None
qdrant_manager = None
performance_monitor = None

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

class OllamaClient:
    """Client for interacting with Ollama LLM service with configurable models and fallback strategies."""
    
    def __init__(self, base_url: str = "http://ollama:11434"):
        self.base_url = base_url
        self.logger = structlog.get_logger("ollama_client")
        self.current_model = None
        self.available_models = []
        self.fallback_chains = {}
        self.retry_config = {
            "max_retries": 3,
            "base_delay": 1.0,
            "max_delay": 30.0,
            "backoff_factor": 2.0
        }
        
    async def initialize(self):
        """Initialize Ollama client and discover available models."""
        try:
            await self._discover_available_models()
            await self._setup_fallback_chains()
            self.logger.info("Ollama client initialized", available_models=len(self.available_models))
        except Exception as e:
            self.logger.error("Failed to initialize Ollama client", error=str(e))
    
    async def _discover_available_models(self):
        """Discover which models are available in Ollama."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    self.available_models = [model["name"] for model in data.get("models", [])]
                    self.logger.info("Discovered Ollama models", models=self.available_models)
                else:
                    self.logger.warning("Failed to discover models", status_code=response.status_code)
        except Exception as e:
            self.logger.error("Error discovering Ollama models", error=str(e))
    
    async def _setup_fallback_chains(self):
        """Setup intelligent fallback chains for different model families."""
        # Define fallback chains based on model families and resource requirements
        self.fallback_chains = {
            "llama3.2:8b": ["llama3.2:3b", "llama3.2:1b"],
            "llama3.2:3b": ["llama3.2:1b"],
            "qwen3:8b": ["qwen3:4b", "qwen3:1.7b", "qwen3:0.6b"],
            "qwen3:4b": ["qwen3:1.7b", "qwen3:0.6b"],
            "qwen3:1.7b": ["qwen3:0.6b"],
            "phi4:3.8b": ["gemma3n:e4b", "gemma3n:e2b"],
            "mistral:7b": ["llama3.2:3b", "llama3.2:1b"],
            "codellama:7b": ["phi4:3.8b", "qwen3:4b"]
        }
    
    async def ensure_model_available(self, model_id: str) -> bool:
        """Ensure model is available, pull if necessary."""
        if model_id in self.available_models:
            return True
        
        try:
            self.logger.info("Pulling model from Ollama", model_id=model_id)
            async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minute timeout for model pulling
                response = await client.post(
                    f"{self.base_url}/api/pull",
                    json={"name": model_id}
                )
                if response.status_code == 200:
                    self.available_models.append(model_id)
                    self.logger.info("Successfully pulled model", model_id=model_id)
                    return True
                else:
                    self.logger.error("Failed to pull model", model_id=model_id, status_code=response.status_code)
                    return False
        except Exception as e:
            self.logger.error("Error pulling model", model_id=model_id, error=str(e))
            return False
    
    async def generate_with_fallback(self, prompt: str, model_id: str, **kwargs) -> Dict[str, Any]:
        """Generate text with automatic fallback to smaller models on failure."""
        models_to_try = [model_id] + self.fallback_chains.get(model_id, [])
        
        for attempt_model in models_to_try:
            try:
                result = await self._generate_with_retry(prompt, attempt_model, **kwargs)
                if result["success"]:
                    if attempt_model != model_id:
                        self.logger.info("Used fallback model", 
                                       requested=model_id, 
                                       used=attempt_model)
                    return result
            except Exception as e:
                self.logger.warning("Model generation failed", 
                                  model=attempt_model, 
                                  error=str(e))
                continue
        
        # All models failed
        return {
            "success": False,
            "error": f"All models failed for prompt generation: {models_to_try}",
            "response": "",
            "model_used": None
        }
    
    async def _generate_with_retry(self, prompt: str, model_id: str, **kwargs) -> Dict[str, Any]:
        """Generate text with retry mechanism."""
        if not await self.ensure_model_available(model_id):
            raise Exception(f"Model {model_id} not available")
        
        for attempt in range(self.retry_config["max_retries"]):
            try:
                start_time = time.time()
                
                async with httpx.AsyncClient(timeout=120.0) as client:
                    payload = {
                        "model": model_id,
                        "prompt": prompt,
                        "stream": False,
                        **kwargs
                    }
                    
                    response = await client.post(
                        f"{self.base_url}/api/generate",
                        json=payload
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        response_time = time.time() - start_time
                        
                        return {
                            "success": True,
                            "response": data.get("response", ""),
                            "model_used": model_id,
                            "response_time": response_time,
                            "tokens_evaluated": data.get("eval_count", 0),
                            "tokens_per_second": data.get("eval_count", 0) / response_time if response_time > 0 else 0
                        }
                    else:
                        raise Exception(f"HTTP {response.status_code}: {response.text}")
                        
            except Exception as e:
                if attempt < self.retry_config["max_retries"] - 1:
                    delay = min(
                        self.retry_config["base_delay"] * (self.retry_config["backoff_factor"] ** attempt),
                        self.retry_config["max_delay"]
                    )
                    self.logger.warning("Retrying after error", 
                                      attempt=attempt + 1, 
                                      delay=delay, 
                                      error=str(e))
                    await asyncio.sleep(delay)
                else:
                    raise e
    
    async def summarize_content(self, content: str, model_id: str = "llama3.2:3b") -> Dict[str, Any]:
        """Generate a summary of the content."""
        prompt = f"""Please provide a concise summary of the following content. Focus on the main points and key information:

Content:
{content[:4000]}  # Limit content length

Summary:"""
        
        return await self.generate_with_fallback(prompt, model_id)
    
    async def extract_keywords(self, content: str, model_id: str = "llama3.2:3b") -> Dict[str, Any]:
        """Extract keywords and key phrases from content."""
        prompt = f"""Extract the most important keywords and key phrases from the following content. Provide them as a comma-separated list:

Content:
{content[:4000]}

Keywords:"""
        
        return await self.generate_with_fallback(prompt, model_id)
    
    async def assess_content_quality(self, content: str, model_id: str = "llama3.2:3b") -> Dict[str, Any]:
        """Assess the quality and characteristics of content."""
        prompt = f"""Analyze the following content and provide a quality assessment. Consider factors like:
- Clarity and coherence
- Information density
- Relevance and usefulness
- Writing quality

Provide a brief assessment and a quality score from 1-10:

Content:
{content[:4000]}

Assessment:"""
        
        return await self.generate_with_fallback(prompt, model_id)

class QdrantManager:
    """Enhanced Qdrant client with model-specific metadata and advanced querying."""
    
    def __init__(self, host: str = "qdrant", port: int = 6333):
        self.host = host
        self.port = port
        self.client = None
        self.logger = structlog.get_logger("qdrant_manager")
        
    async def initialize(self):
        """Initialize Qdrant client."""
        try:
            self.client = QdrantClient(host=self.host, port=self.port)
            self.logger.info("Qdrant client initialized", host=self.host, port=self.port)
        except Exception as e:
            self.logger.error("Failed to initialize Qdrant client", error=str(e))
            raise
    
    async def ensure_collection_exists(self, collection_name: str, vector_size: int, distance: Distance = Distance.COSINE):
        """Ensure collection exists with proper configuration."""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if collection_name not in collection_names:
                self.logger.info("Creating new collection", collection_name=collection_name, vector_size=vector_size)
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=distance)
                )
            else:
                # Verify vector size matches
                collection_info = self.client.get_collection(collection_name)
                existing_size = collection_info.config.params.vectors.size
                if existing_size != vector_size:
                    self.logger.warning("Vector size mismatch", 
                                      collection=collection_name,
                                      expected=vector_size,
                                      existing=existing_size)
                    # Could recreate collection or handle mismatch
                    
        except Exception as e:
            self.logger.error("Error ensuring collection exists", 
                            collection_name=collection_name, 
                            error=str(e))
            raise
    
    async def store_analyzed_content(self, 
                                   collection_name: str,
                                   content_items: List[Dict[str, Any]],
                                   embedding_model: str,
                                   llm_model: Optional[str] = None) -> int:
        """Store analyzed content with model-specific metadata."""
        try:
            points = []
            
            for item in content_items:
                # Create comprehensive metadata
                metadata = {
                    "content_id": item["content_id"],
                    "chunk_index": item.get("chunk_index", 0),
                    "text": item["text"],
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "token_count": item.get("token_count", 0),
                    
                    # Model information
                    "embedding_model": embedding_model,
                    "llm_model": llm_model,
                    "analysis_timestamp": datetime.now().isoformat(),
                    
                    # Analysis results
                    "summary": item.get("summary"),
                    "keywords": item.get("keywords"),
                    "quality_score": item.get("quality_score"),
                    "quality_assessment": item.get("quality_assessment"),
                    
                    # Performance metrics
                    "embedding_generation_time": item.get("embedding_generation_time"),
                    "llm_processing_time": item.get("llm_processing_time"),
                    
                    # Original metadata
                    **item.get("original_metadata", {})
                }
                
                point_id = f"{item['content_id']}_chunk_{item.get('chunk_index', 0)}"
                
                points.append(PointStruct(
                    id=point_id,
                    vector=item["embedding"],
                    payload=metadata
                ))
            
            # Batch upsert
            if points:
                self.client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                
                self.logger.info("Stored analyzed content", 
                               collection=collection_name,
                               points_count=len(points),
                               embedding_model=embedding_model,
                               llm_model=llm_model)
            
            return len(points)
            
        except Exception as e:
            self.logger.error("Error storing analyzed content", 
                            collection_name=collection_name,
                            error=str(e))
            raise
    
    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get detailed information about a collection."""
        try:
            collection_info = self.client.get_collection(collection_name)
            
            # Get points count
            points_count = self.client.count(collection_name).count
            
            # Get sample of metadata to understand structure
            sample_points = self.client.scroll(
                collection_name=collection_name,
                limit=5,
                with_payload=True,
                with_vectors=False
            )[0]
            
            # Extract model usage statistics
            model_stats = {}
            for point in sample_points:
                embedding_model = point.payload.get("embedding_model")
                llm_model = point.payload.get("llm_model")
                
                if embedding_model:
                    model_stats.setdefault("embedding_models", set()).add(embedding_model)
                if llm_model:
                    model_stats.setdefault("llm_models", set()).add(llm_model)
            
            # Convert sets to lists for JSON serialization
            for key, value in model_stats.items():
                if isinstance(value, set):
                    model_stats[key] = list(value)
            
            return {
                "collection_name": collection_name,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance.name,
                "points_count": points_count,
                "model_usage": model_stats,
                "status": collection_info.status.name
            }
            
        except Exception as e:
            self.logger.error("Error getting collection info", 
                            collection_name=collection_name,
                            error=str(e))
            raise
    
    async def search_similar_content(self, 
                                   collection_name: str,
                                   query_vector: List[float],
                                   limit: int = 10,
                                   embedding_model_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for similar content with optional model filtering."""
        try:
            search_filter = None
            if embedding_model_filter:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="embedding_model",
                            match=MatchValue(value=embedding_model_filter)
                        )
                    ]
                )
            
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=limit,
                with_payload=True
            )
            
            return [
                {
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                }
                for result in results
            ]
            
        except Exception as e:
            self.logger.error("Error searching similar content", 
                            collection_name=collection_name,
                            error=str(e))
            raise

class PerformanceMonitor:
    """Monitor AI model performance, resource usage, and availability."""
    
    def __init__(self):
        self.logger = structlog.get_logger("performance_monitor")
        self.metrics = {}
        self.resource_history = []
        self.max_history_size = 1000
        
    def record_model_performance(self, 
                               model_id: str,
                               model_type: str,
                               success: bool,
                               response_time: float,
                               tokens_per_second: Optional[float] = None,
                               resource_usage: Optional[Dict[str, float]] = None):
        """Record performance metrics for a model."""
        if model_id not in self.metrics:
            self.metrics[model_id] = {
                "model_type": model_type,
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "response_times": [],
                "tokens_per_second_history": [],
                "last_used": None,
                "resource_usage_history": []
            }
        
        metrics = self.metrics[model_id]
        metrics["total_requests"] += 1
        metrics["last_used"] = datetime.now()
        
        if success:
            metrics["successful_requests"] += 1
            metrics["response_times"].append(response_time)
            
            if tokens_per_second is not None:
                metrics["tokens_per_second_history"].append(tokens_per_second)
            
            if resource_usage:
                metrics["resource_usage_history"].append({
                    "timestamp": datetime.now().isoformat(),
                    **resource_usage
                })
        else:
            metrics["failed_requests"] += 1
        
        # Keep history size manageable
        for key in ["response_times", "tokens_per_second_history", "resource_usage_history"]:
            if len(metrics[key]) > self.max_history_size:
                metrics[key] = metrics[key][-self.max_history_size:]
    
    def get_model_metrics(self, model_id: str) -> Optional[ModelPerformanceMetrics]:
        """Get performance metrics for a specific model."""
        if model_id not in self.metrics:
            return None
        
        metrics = self.metrics[model_id]
        
        # Calculate averages
        avg_response_time = (
            sum(metrics["response_times"]) / len(metrics["response_times"])
            if metrics["response_times"] else 0.0
        )
        
        avg_tokens_per_second = (
            sum(metrics["tokens_per_second_history"]) / len(metrics["tokens_per_second_history"])
            if metrics["tokens_per_second_history"] else None
        )
        
        # Calculate average resource usage
        resource_usage = {}
        if metrics["resource_usage_history"]:
            resource_keys = set()
            for usage in metrics["resource_usage_history"]:
                resource_keys.update(usage.keys())
            resource_keys.discard("timestamp")
            
            for key in resource_keys:
                values = [usage.get(key, 0) for usage in metrics["resource_usage_history"] if key in usage]
                if values:
                    resource_usage[key] = sum(values) / len(values)
        
        return ModelPerformanceMetrics(
            model_id=model_id,
            model_type=metrics["model_type"],
            total_requests=metrics["total_requests"],
            successful_requests=metrics["successful_requests"],
            failed_requests=metrics["failed_requests"],
            average_response_time=avg_response_time,
            average_tokens_per_second=avg_tokens_per_second,
            last_used=metrics["last_used"],
            resource_usage=resource_usage
        )
    
    def get_all_metrics(self) -> List[ModelPerformanceMetrics]:
        """Get performance metrics for all models."""
        return [
            self.get_model_metrics(model_id)
            for model_id in self.metrics.keys()
        ]
    
    def record_system_resources(self):
        """Record current system resource usage."""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            resource_snapshot = {
                "timestamp": datetime.now().isoformat(),
                "ram_usage_percent": memory.percent,
                "available_ram_gb": memory.available / (1024**3),
                "cpu_usage_percent": cpu_percent
            }
            
            # Add GPU info if available
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_stats()
                    resource_snapshot.update({
                        "gpu_memory_allocated_gb": gpu_memory.get("allocated_bytes.all.current", 0) / (1024**3),
                        "gpu_memory_reserved_gb": gpu_memory.get("reserved_bytes.all.current", 0) / (1024**3)
                    })
            except Exception:
                pass
            
            self.resource_history.append(resource_snapshot)
            
            # Keep history size manageable
            if len(self.resource_history) > self.max_history_size:
                self.resource_history = self.resource_history[-self.max_history_size:]
                
        except Exception as e:
            self.logger.error("Error recording system resources", error=str(e))
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get summary of recent resource usage."""
        if not self.resource_history:
            return {}
        
        recent_history = self.resource_history[-100:]  # Last 100 measurements
        
        # Calculate averages
        avg_ram_usage = sum(r["ram_usage_percent"] for r in recent_history) / len(recent_history)
        avg_cpu_usage = sum(r["cpu_usage_percent"] for r in recent_history) / len(recent_history)
        
        summary = {
            "average_ram_usage_percent": avg_ram_usage,
            "average_cpu_usage_percent": avg_cpu_usage,
            "current_available_ram_gb": recent_history[-1]["available_ram_gb"],
            "measurements_count": len(recent_history)
        }
        
        # Add GPU info if available
        gpu_measurements = [r for r in recent_history if "gpu_memory_allocated_gb" in r]
        if gpu_measurements:
            summary["average_gpu_memory_allocated_gb"] = (
                sum(r["gpu_memory_allocated_gb"] for r in gpu_measurements) / len(gpu_measurements)
            )
        
        return summary

# Initialize global components
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global hardware_detector, embedding_generator, text_chunker, embedding_cache, model_manager, ollama_client, qdrant_manager, performance_monitor
    
    logger.info("Initializing Content Analyzer Service")
    
    try:
        # Initialize components
        hardware_detector = HardwareDetector()
        model_manager = ModelManager()
        embedding_cache = EmbeddingCache()
        text_chunker = TextChunker()
        embedding_generator = EmbeddingGenerator(model_manager, embedding_cache)
        
        # Initialize new components
        ollama_client = OllamaClient()
        await ollama_client.initialize()
        
        qdrant_manager = QdrantManager()
        await qdrant_manager.initialize()
        
        performance_monitor = PerformanceMonitor()
        
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

@app.post("/analysis/complete", response_model=AnalysisResponse)
async def analyze_content(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Perform complete content analysis including embeddings, summaries, and quality assessment."""
    if not all([embedding_generator, text_chunker, ollama_client, qdrant_manager]):
        raise HTTPException(status_code=503, detail="Analysis services not initialized")
    
    job_id = str(uuid.uuid4())
    
    # Switch models if requested
    if request.embedding_model:
        success = embedding_generator.switch_model(request.embedding_model)
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to switch to embedding model: {request.embedding_model}"
            )
    
    # Start background processing
    background_tasks.add_task(
        process_complete_analysis_job,
        job_id,
        request.content_items,
        request.session_id,
        request.llm_model or "llama3.2:3b",
        request.generate_summary,
        request.extract_keywords,
        request.assess_quality
    )
    
    return AnalysisResponse(
        job_id=job_id,
        status="processing",
        message=f"Started complete analysis for {len(request.content_items)} items"
    )

@app.get("/llm/models/available")
async def get_available_llm_models():
    """Get all available LLM models from Ollama."""
    if not ollama_client:
        raise HTTPException(status_code=503, detail="Ollama client not initialized")
    
    return {
        "available_models": ollama_client.available_models,
        "fallback_chains": ollama_client.fallback_chains
    }

@app.post("/llm/models/ensure/{model_id}")
async def ensure_llm_model(model_id: str):
    """Ensure LLM model is available, pull if necessary."""
    if not ollama_client:
        raise HTTPException(status_code=503, detail="Ollama client not initialized")
    
    success = await ollama_client.ensure_model_available(model_id)
    
    if success:
        return {
            "status": "success",
            "message": f"Model {model_id} is available",
            "available_models": ollama_client.available_models
        }
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to ensure model availability: {model_id}"
        )

@app.get("/qdrant/collections/{collection_name}/info", response_model=QdrantConnectionInfo)
async def get_collection_info(collection_name: str):
    """Get information about a Qdrant collection."""
    if not qdrant_manager:
        raise HTTPException(status_code=503, detail="Qdrant manager not initialized")
    
    try:
        info = await qdrant_manager.get_collection_info(collection_name)
        return QdrantConnectionInfo(
            host=qdrant_manager.host,
            port=qdrant_manager.port,
            collection_name=info["collection_name"],
            vector_size=info["vector_size"],
            distance_metric=info["distance_metric"],
            points_count=info["points_count"]
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Collection not found or error: {str(e)}")

@app.get("/qdrant/collections/{collection_name}/search")
async def search_collection(
    collection_name: str,
    query_text: str,
    limit: int = 10,
    embedding_model_filter: Optional[str] = None
):
    """Search for similar content in a collection."""
    if not all([qdrant_manager, embedding_generator]):
        raise HTTPException(status_code=503, detail="Search services not initialized")
    
    try:
        # Generate embedding for query
        query_embeddings = embedding_generator.generate_embeddings([query_text])
        if not query_embeddings:
            raise HTTPException(status_code=400, detail="Failed to generate query embedding")
        
        # Search similar content
        results = await qdrant_manager.search_similar_content(
            collection_name=collection_name,
            query_vector=query_embeddings[0].tolist(),
            limit=limit,
            embedding_model_filter=embedding_model_filter
        )
        
        return {
            "query": query_text,
            "results": results,
            "embedding_model_used": embedding_generator.current_model_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Search failed: {str(e)}")

@app.get("/performance/models")
async def get_model_performance():
    """Get performance metrics for all models."""
    if not performance_monitor:
        raise HTTPException(status_code=503, detail="Performance monitor not initialized")
    
    return {
        "model_metrics": performance_monitor.get_all_metrics(),
        "system_resources": performance_monitor.get_resource_summary()
    }

@app.get("/performance/models/{model_id}")
async def get_model_performance_detail(model_id: str):
    """Get detailed performance metrics for a specific model."""
    if not performance_monitor:
        raise HTTPException(status_code=503, detail="Performance monitor not initialized")
    
    metrics = performance_monitor.get_model_metrics(model_id)
    if not metrics:
        raise HTTPException(status_code=404, detail=f"No performance data for model: {model_id}")
    
    return metrics

@app.post("/performance/record-resources")
async def record_system_resources():
    """Manually trigger system resource recording."""
    if not performance_monitor:
        raise HTTPException(status_code=503, detail="Performance monitor not initialized")
    
    performance_monitor.record_system_resources()
    return {
        "status": "success",
        "message": "System resources recorded",
        "summary": performance_monitor.get_resource_summary()
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
        # Ensure collection exists
        collection_name = f"session_{session_id}"
        model_info = embedding_generator.get_current_model_info()
        dimensions = model_info.get("dimensions", 384)
        
        await qdrant_manager.ensure_collection_exists(collection_name, dimensions)
        
        # Process each content item
        processed_items = []
        
        for item in content_items:
            # Chunk the content
            chunks = text_chunker.chunk_text(
                item.content,
                chunk_size=chunk_size,
                overlap=chunk_overlap
            )
            
            # Generate embeddings for chunks
            chunk_texts = [chunk["text"] for chunk in chunks]
            start_time = time.time()
            embeddings = embedding_generator.generate_embeddings(chunk_texts)
            embedding_time = time.time() - start_time
            
            # Record performance
            if performance_monitor:
                performance_monitor.record_model_performance(
                    model_id=embedding_generator.current_model_id,
                    model_type="embedding",
                    success=True,
                    response_time=embedding_time,
                    resource_usage={"chunks_processed": len(chunks)}
                )
            
            # Prepare items for storage
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                processed_items.append({
                    "content_id": item.id,
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                    "title": item.title,
                    "url": item.url,
                    "token_count": chunk["token_count"],
                    "embedding": embedding.tolist(),
                    "embedding_generation_time": embedding_time / len(chunks),
                    "original_metadata": item.metadata
                })
        
        # Store in Qdrant
        if processed_items:
            points_stored = await qdrant_manager.store_analyzed_content(
                collection_name=collection_name,
                content_items=processed_items,
                embedding_model=embedding_generator.current_model_id
            )
            
            logger.info("Embedding job completed", 
                       job_id=job_id, 
                       points_created=points_stored)
        
    except Exception as e:
        logger.error("Embedding job failed", job_id=job_id, error=str(e))
        if performance_monitor:
            performance_monitor.record_model_performance(
                model_id=embedding_generator.current_model_id,
                model_type="embedding",
                success=False,
                response_time=0.0
            )

async def process_complete_analysis_job(
    job_id: str,
    content_items: List[ContentItem],
    session_id: str,
    llm_model: str,
    generate_summary: bool,
    extract_keywords: bool,
    assess_quality: bool
):
    """Background task to process complete content analysis."""
    logger.info("Starting complete analysis job", 
               job_id=job_id, 
               items_count=len(content_items),
               llm_model=llm_model)
    
    try:
        # Ensure collection exists
        collection_name = f"session_{session_id}"
        model_info = embedding_generator.get_current_model_info()
        dimensions = model_info.get("dimensions", 384)
        
        await qdrant_manager.ensure_collection_exists(collection_name, dimensions)
        
        # Process each content item
        processed_items = []
        
        for item in content_items:
            logger.info("Processing content item", content_id=item.id)
            
            # Chunk the content
            chunks = text_chunker.chunk_text(item.content)
            
            # Generate embeddings for chunks
            chunk_texts = [chunk["text"] for chunk in chunks]
            embedding_start = time.time()
            embeddings = embedding_generator.generate_embeddings(chunk_texts)
            embedding_time = time.time() - embedding_start
            
            # Record embedding performance
            if performance_monitor:
                performance_monitor.record_model_performance(
                    model_id=embedding_generator.current_model_id,
                    model_type="embedding",
                    success=True,
                    response_time=embedding_time,
                    resource_usage={"chunks_processed": len(chunks)}
                )
            
            # Process each chunk with LLM analysis
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_analysis = {
                    "content_id": item.id,
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                    "title": item.title,
                    "url": item.url,
                    "token_count": chunk["token_count"],
                    "embedding": embedding.tolist(),
                    "embedding_generation_time": embedding_time / len(chunks),
                    "original_metadata": item.metadata
                }
                
                # LLM-based analysis
                llm_start_time = time.time()
                llm_success = True
                
                try:
                    # Generate summary
                    if generate_summary:
                        summary_result = await ollama_client.summarize_content(chunk["text"], llm_model)
                        if summary_result["success"]:
                            chunk_analysis["summary"] = summary_result["response"]
                        else:
                            logger.warning("Summary generation failed", 
                                         content_id=item.id, 
                                         chunk_index=i,
                                         error=summary_result.get("error"))
                    
                    # Extract keywords
                    if extract_keywords:
                        keywords_result = await ollama_client.extract_keywords(chunk["text"], llm_model)
                        if keywords_result["success"]:
                            chunk_analysis["keywords"] = keywords_result["response"]
                        else:
                            logger.warning("Keyword extraction failed", 
                                         content_id=item.id, 
                                         chunk_index=i,
                                         error=keywords_result.get("error"))
                    
                    # Assess quality
                    if assess_quality:
                        quality_result = await ollama_client.assess_content_quality(chunk["text"], llm_model)
                        if quality_result["success"]:
                            chunk_analysis["quality_assessment"] = quality_result["response"]
                            # Try to extract numeric score
                            try:
                                score_match = re.search(r'(\d+(?:\.\d+)?)/10', quality_result["response"])
                                if score_match:
                                    chunk_analysis["quality_score"] = float(score_match.group(1))
                            except Exception:
                                pass
                        else:
                            logger.warning("Quality assessment failed", 
                                         content_id=item.id, 
                                         chunk_index=i,
                                         error=quality_result.get("error"))
                    
                except Exception as e:
                    logger.error("LLM analysis failed", 
                               content_id=item.id, 
                               chunk_index=i,
                               error=str(e))
                    llm_success = False
                
                llm_time = time.time() - llm_start_time
                chunk_analysis["llm_processing_time"] = llm_time
                
                # Record LLM performance
                if performance_monitor:
                    performance_monitor.record_model_performance(
                        model_id=llm_model,
                        model_type="llm",
                        success=llm_success,
                        response_time=llm_time,
                        resource_usage={"token_count": chunk["token_count"]}
                    )
                
                processed_items.append(chunk_analysis)
        
        # Store in Qdrant
        if processed_items:
            points_stored = await qdrant_manager.store_analyzed_content(
                collection_name=collection_name,
                content_items=processed_items,
                embedding_model=embedding_generator.current_model_id,
                llm_model=llm_model
            )
            
            logger.info("Complete analysis job completed", 
                       job_id=job_id, 
                       points_created=points_stored,
                       embedding_model=embedding_generator.current_model_id,
                       llm_model=llm_model)
        
        # Record system resources
        if performance_monitor:
            performance_monitor.record_system_resources()
        
    except Exception as e:
        logger.error("Complete analysis job failed", job_id=job_id, error=str(e))
        if performance_monitor:
            performance_monitor.record_model_performance(
                model_id=llm_model,
                model_type="llm",
                success=False,
                response_time=0.0
            )