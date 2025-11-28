"""Core components for the analyzer service without heavy ML dependencies."""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import psutil
import structlog

logger = structlog.get_logger()


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
                # Import torch only when needed for GPU detection
                import torch
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
                        "recommended": False
                    },
                    "nomic-embed-text": {
                        "name": "Nomic Embed Text",
                        "size": "274MB",
                        "dimensions": 768,
                        "quality": "high",
                        "min_ram_gb": 1.0,
                        "description": "Best general purpose embedding model",
                        "model_name": "nomic-ai/nomic-embed-text-v1",
                        "max_sequence_length": 8192,
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
            import tiktoken
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