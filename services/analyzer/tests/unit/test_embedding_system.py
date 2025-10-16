"""Isolated unit tests for embedding system components without heavy ML dependencies."""

import json
import numpy as np
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Mock heavy dependencies before importing main
sys.modules['sentence_transformers'] = MagicMock()
mock_torch = MagicMock()
mock_torch_cuda = MagicMock()
mock_torch.cuda = mock_torch_cuda
sys.modules['torch'] = mock_torch
sys.modules['torch.cuda'] = mock_torch_cuda
mock_qdrant = MagicMock()
mock_qdrant.models = MagicMock()
sys.modules['qdrant_client'] = mock_qdrant
sys.modules['qdrant_client.models'] = mock_qdrant.models
sys.modules['tiktoken'] = MagicMock()

# Now we can safely import our modules
from analyzer import EmbeddingCache, HardwareDetector, ModelManager, TextChunker


class TestHardwareDetectorUnit:
    """Unit tests for hardware detection without torch dependencies."""
    
    def test_detect_hardware_basic_info(self):
        """Test basic hardware detection."""
        detector = HardwareDetector()
        
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_count') as mock_cpu, \
             patch('psutil.cpu_percent') as mock_cpu_percent:
            
            mock_memory.return_value = Mock(
                total=8 * 1024**3,
                available=4 * 1024**3,
                percent=50.0
            )
            mock_cpu.return_value = 8
            mock_cpu_percent.return_value = 25.0
            
            hardware_info = detector.detect_hardware()
            
            assert hardware_info["ram_gb"] == 8.0
            assert hardware_info["available_ram_gb"] == 4.0
            assert hardware_info["ram_usage_percent"] == 50.0
            assert hardware_info["cpu_count"] == 8
            assert hardware_info["cpu_usage_percent"] == 25.0
    
    def test_detect_hardware_with_gpu_mock(self):
        """Test hardware detection with GPU mock."""
        detector = HardwareDetector()
        
        with patch('psutil.virtual_memory') as mock_memory, \
            patch('psutil.cpu_count') as mock_cpu, \
            patch('psutil.cpu_percent') as mock_cpu_percent, \
            patch('main.torch.cuda.is_available') as mock_cuda_available, \
            patch('main.torch.cuda.get_device_properties') as mock_gpu_props, \
            patch('main.torch.cuda.get_device_name') as mock_gpu_name:
            
            mock_memory.return_value = Mock(
                total=16 * 1024**3,
                available=8 * 1024**3,
                percent=50.0
            )
            mock_cpu.return_value = 8
            mock_cpu_percent.return_value = 25.0
            
            # Mock GPU
            mock_cuda_available.return_value = True
            props_mock = Mock()
            props_mock.total_memory = 8 * 1024**3  # 8 GB in bytes
            mock_gpu_props.return_value = props_mock
            mock_gpu_name.return_value = "NVIDIA GeForce RTX 3080"
            
            hardware_info = detector.detect_hardware()
            
            assert hardware_info["has_gpu"] is True
            assert hardware_info["gpu_memory_gb"] == 8.0
            assert hardware_info["gpu_name"] == "NVIDIA GeForce RTX 3080"
            
    
    def test_detect_hardware_error_handling(self):
        """Test hardware detection error handling."""
        detector = HardwareDetector()
        
        with patch('psutil.virtual_memory', side_effect=Exception("System error")):
            hardware_info = detector.detect_hardware()
            
            # Should return safe defaults
            assert hardware_info["ram_gb"] == 8.0
            assert hardware_info["cpu_count"] == 4
            assert hardware_info["has_gpu"] is False


class TestModelManagerUnit:
    """Unit tests for model manager."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock models configuration."""
        return {
            "embedding_models": {
                "all-minilm": {
                    "name": "All-MiniLM",
                    "size": "90MB",
                    "dimensions": 384,
                    "quality": "good",
                    "min_ram_gb": 0.5,
                    "description": "Lightweight embedding model",
                    "recommended": True
                },
                "nomic-embed-text": {
                    "name": "Nomic Embed Text",
                    "size": "274MB",
                    "dimensions": 768,
                    "quality": "high",
                    "min_ram_gb": 1.0,
                    "description": "Best general purpose embedding model",
                    "recommended": False
                },
                "mxbai-embed-large": {
                    "name": "MxBai Embed Large",
                    "size": "669MB",
                    "dimensions": 1024,
                    "quality": "highest",
                    "min_ram_gb": 2.0,
                    "description": "Highest quality embeddings",
                    "recommended": False
                }
            }
        }
    
    def test_load_config_fallback(self):
        """Test config loading fallback when file doesn't exist."""
        with patch.object(Path, 'exists', return_value=False):
            manager = ModelManager()
            
            # Should have fallback config
            assert "embedding_models" in manager.models_config
            assert "all-minilm" in manager.models_config["embedding_models"]
    
    def test_get_available_models(self, mock_config):
        """Test getting available models."""
        with patch.object(ModelManager, '_load_models_config', return_value=mock_config):
            manager = ModelManager()
            models = manager.get_available_models()
            
            assert len(models) == 3
            assert "all-minilm" in models
            assert "nomic-embed-text" in models
            assert "mxbai-embed-large" in models
    
    def test_recommend_model_low_resource(self, mock_config):
        """Test model recommendation for low-resource system."""
        with patch.object(ModelManager, '_load_models_config', return_value=mock_config):
            manager = ModelManager()
            
            hardware_info = {
                "available_ram_gb": 1.0,
                "has_gpu": False
            }
            
            recommendation = manager.recommend_model(hardware_info)
            
            assert recommendation["recommended_model"] == "all-minilm"
            assert "performance_estimate" in recommendation
            assert recommendation["performance_estimate"]["embeddings_per_sec"] > 0
    
    def test_recommend_model_high_resource(self, mock_config):
        """Test model recommendation for high-resource system."""
        with patch.object(ModelManager, '_load_models_config', return_value=mock_config):
            manager = ModelManager()
            
            hardware_info = {
                "available_ram_gb": 8.0,
                "has_gpu": True
            }
            
            recommendation = manager.recommend_model(hardware_info)
            
            # Should recommend a higher-quality model
            assert recommendation["recommended_model"] in ["nomic-embed-text", "mxbai-embed-large"]
            assert len(recommendation["alternatives"]) > 0
    
    def test_model_scoring_algorithm(self, mock_config):
        """Test the model scoring algorithm."""
        with patch.object(ModelManager, '_load_models_config', return_value=mock_config):
            manager = ModelManager()
            
            config_good = mock_config["embedding_models"]["all-minilm"]
            config_high = mock_config["embedding_models"]["nomic-embed-text"]
            
            score_good = manager._calculate_model_score(config_good, has_gpu=False)
            score_high = manager._calculate_model_score(config_high, has_gpu=False)
            
            assert isinstance(score_good, float)
            assert isinstance(score_high, float)
            assert score_good > 0
            assert score_high > 0


class TestTextChunkerUnit:
    """Unit tests for text chunker."""
    
    def test_chunk_text_empty_input(self):
        """Test chunking empty or whitespace-only text."""
        chunker = TextChunker()
        
        assert chunker.chunk_text("") == []
        assert chunker.chunk_text("   ") == []
        assert chunker.chunk_text("\n\t  \n") == []
    
    def test_chunk_by_characters_basic(self):
        """Test character-based chunking."""
        chunker = TextChunker()
        
        text = "A" * 1000  # 1000 character text
        chunks = chunker._chunk_by_characters(text, chunk_size=300, overlap=50)
        
        assert len(chunks) > 1
        assert chunks[0]["text"] == "A" * 300
        assert chunks[0]["chunk_index"] == 0
        assert chunks[0]["start_char"] == 0
        assert chunks[0]["end_char"] == 300
        
        # Second chunk should start at position 250 (300 - 50 overlap)
        assert chunks[1]["start_char"] == 250
        assert chunks[1]["chunk_index"] == 1
    
    def test_chunk_by_characters_no_overlap(self):
        """Test character-based chunking without overlap."""
        chunker = TextChunker()
        
        text = "ABCDEFGHIJ" * 10  # 100 characters
        chunks = chunker._chunk_by_characters(text, chunk_size=30, overlap=0)
        
        assert len(chunks) == 4  # 100/30 = 3.33, so 4 chunks
        assert chunks[0]["start_char"] == 0
        assert chunks[1]["start_char"] == 30
        assert chunks[2]["start_char"] == 60
        assert chunks[3]["start_char"] == 90
    
    def test_chunk_text_fallback_mode(self):
        """Test chunking when tokenizer is not available."""
        chunker = TextChunker()
        chunker.tokenizer = None  # Force character-based chunking
        
        text = "This is a test text that should be chunked properly."
        chunks = chunker.chunk_text(text, chunk_size=20, overlap=5)
        
        assert len(chunks) > 0
        assert all("chunk_index" in chunk for chunk in chunks)
        assert all("text" in chunk for chunk in chunks)
        assert all("token_count" in chunk for chunk in chunks)


class TestEmbeddingCacheUnit:
    """Unit tests for embedding cache."""
    
    def test_cache_key_generation(self):
        """Test cache key generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = EmbeddingCache()
            cache.cache_dir = Path(temp_dir)
            
            key1 = cache._get_cache_key("test text", "model1")
            key2 = cache._get_cache_key("test text", "model2")
            key3 = cache._get_cache_key("different text", "model1")
            
            # Same text, different models should have different keys
            assert key1 != key2
            # Different text, same model should have different keys
            assert key1 != key3
            # Keys should include model ID
            assert "model1" in key1
            assert "model2" in key2
    
    def test_memory_cache_operations(self):
        """Test memory cache store and retrieve."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = EmbeddingCache()
            cache.cache_dir = Path(temp_dir)
            
            text = "test text"
            model_id = "test_model"
            embedding = np.array([1.0, 2.0, 3.0])
            
            # Store embedding
            cache.store_embedding(text, model_id, embedding)
            
            # Retrieve embedding
            retrieved = cache.get_embedding(text, model_id)
            
            assert retrieved is not None
            np.testing.assert_array_equal(retrieved, embedding)
    
    def test_disk_cache_operations(self):
        """Test disk cache store and retrieve."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = EmbeddingCache()
            cache.cache_dir = Path(temp_dir)
            cache.memory_cache = {}  # Clear memory cache
            
            text = "test text"
            model_id = "test_model"
            embedding = np.array([1.0, 2.0, 3.0])
            
            # Store embedding (should go to disk)
            cache.store_embedding(text, model_id, embedding)
            
            # Clear memory cache to force disk read
            cache.memory_cache = {}
            
            # Retrieve embedding (should come from disk)
            retrieved = cache.get_embedding(text, model_id)
            
            assert retrieved is not None
            np.testing.assert_array_equal(retrieved, embedding)
    
    def test_clear_model_cache(self):
        """Test clearing cache for specific model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = EmbeddingCache()
            cache.cache_dir = Path(temp_dir)
            
            # Store embeddings for different models
            embedding1 = np.array([1.0, 2.0])
            embedding2 = np.array([3.0, 4.0])
            
            cache.store_embedding("text1", "model1", embedding1)
            cache.store_embedding("text2", "model2", embedding2)
            
            # Clear model1 cache
            cache.clear_model_cache("model1")
            
            # model1 embedding should be gone
            assert cache.get_embedding("text1", "model1") is None
            # model2 embedding should still exist
            retrieved = cache.get_embedding("text2", "model2")
            np.testing.assert_array_equal(retrieved, embedding2)
    
    def test_cache_memory_limit(self):
        """Test that memory cache respects size limits."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = EmbeddingCache()
            cache.cache_dir = Path(temp_dir)
            cache.max_memory_items = 2  # Set low limit for testing
            
            # Store more embeddings than the limit
            for i in range(5):
                embedding = np.array([float(i), float(i+1)])
                cache.store_embedding(f"text{i}", "model1", embedding)
            
            # Memory cache should not exceed the limit
            assert len(cache.memory_cache) <= cache.max_memory_items


class TestSystemIntegration:
    """Integration tests for system components."""
    
    def test_hardware_to_model_recommendation_flow(self):
        """Test the flow from hardware detection to model recommendation."""
        mock_config = {
            "embedding_models": {
                "all-minilm": {
                    "name": "All-MiniLM",
                    "dimensions": 384,
                    "quality": "good",
                    "min_ram_gb": 0.5,
                    "description": "Lightweight model",
                    "recommended": True
                },
                "nomic-embed-text": {
                    "name": "Nomic Embed Text",
                    "dimensions": 768,
                    "quality": "high",
                    "min_ram_gb": 1.0,
                    "description": "General purpose model",
                    "recommended": False
                }
            }
        }
        
        with patch.object(ModelManager, '_load_models_config', return_value=mock_config), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_count') as mock_cpu, \
             patch('psutil.cpu_percent') as mock_cpu_percent:
            
            # Mock medium-resource system
            mock_memory.return_value = Mock(
                total=8 * 1024**3,
                available=4 * 1024**3,
                percent=50.0
            )
            mock_cpu.return_value = 8
            mock_cpu_percent.return_value = 25.0
            
            # Test the full workflow
            detector = HardwareDetector()
            manager = ModelManager()
            
            # Detect hardware
            hardware_info = detector.detect_hardware()
            assert hardware_info["available_ram_gb"] == 4.0
            
            # Get recommendation based on detected hardware
            recommendation = manager.recommend_model(hardware_info)
            
            # Should get a valid recommendation
            assert recommendation["recommended_model"] in ["all-minilm", "nomic-embed-text"]
            assert "performance_estimate" in recommendation
            assert recommendation["performance_estimate"]["embeddings_per_sec"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
