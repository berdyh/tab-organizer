"""Unit tests for embedding generation system."""

import json
import numpy as np
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch

# Import the classes we want to test
from main import (
    HardwareDetector,
    ModelManager,
    TextChunker,
    EmbeddingCache,
    EmbeddingGenerator,
    ContentItem
)


class TestHardwareDetector:
    """Test hardware detection functionality."""
    
    def test_detect_hardware_basic(self):
        """Test basic hardware detection."""
        detector = HardwareDetector()
        
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_count') as mock_cpu, \
             patch('psutil.cpu_percent') as mock_cpu_percent, \
             patch('torch.cuda.is_available') as mock_cuda:
            
            # Mock system info
            mock_memory.return_value = Mock(
                total=8 * 1024**3,  # 8GB
                available=4 * 1024**3,  # 4GB available
                percent=50.0
            )
            mock_cpu.return_value = 8
            mock_cpu_percent.return_value = 25.0
            mock_cuda.return_value = False
            
            hardware_info = detector.detect_hardware()
            
            assert hardware_info["ram_gb"] == 8.0
            assert hardware_info["cpu_count"] == 8
            assert hardware_info["available_ram_gb"] == 4.0
            assert hardware_info["ram_usage_percent"] == 50.0
            assert hardware_info["has_gpu"] is False
    
    def test_detect_hardware_with_gpu(self):
        """Test hardware detection with GPU."""
        detector = HardwareDetector()
        
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_count') as mock_cpu, \
             patch('psutil.cpu_percent') as mock_cpu_percent, \
             patch('torch.cuda.is_available') as mock_cuda, \
             patch('torch.cuda.get_device_properties') as mock_gpu_props, \
             patch('torch.cuda.get_device_name') as mock_gpu_name:
            
            mock_memory.return_value = Mock(
                total=16 * 1024**3,
                available=8 * 1024**3,
                percent=50.0
            )
            mock_cpu.return_value = 12
            mock_cpu_percent.return_value = 15.0
            mock_cuda.return_value = True
            mock_gpu_props.return_value = Mock(total_memory=8 * 1024**3)  # 8GB VRAM
            mock_gpu_name.return_value = "NVIDIA RTX 4080"
            
            hardware_info = detector.detect_hardware()
            
            assert hardware_info["has_gpu"] is True
            assert hardware_info["gpu_memory_gb"] == 8.0
            assert hardware_info["gpu_name"] == "NVIDIA RTX 4080"
    
    def test_detect_hardware_fallback(self):
        """Test hardware detection fallback on error."""
        detector = HardwareDetector()
        
        with patch('psutil.virtual_memory', side_effect=Exception("Mock error")):
            hardware_info = detector.detect_hardware()
            
            # Should return safe defaults
            assert hardware_info["ram_gb"] == 8.0
            assert hardware_info["cpu_count"] == 4
            assert hardware_info["has_gpu"] is False


class TestModelManager:
    """Test model management functionality."""
    
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
                    "model_name": "all-MiniLM-L6-v2",
                    "max_sequence_length": 512,
                    "recommended": True
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
                    "recommended": False
                },
                "mxbai-embed-large": {
                    "name": "MxBai Embed Large",
                    "size": "669MB",
                    "dimensions": 1024,
                    "quality": "highest",
                    "min_ram_gb": 2.0,
                    "description": "Highest quality embeddings",
                    "model_name": "mixedbread-ai/mxbai-embed-large-v1",
                    "max_sequence_length": 512,
                    "recommended": False
                }
            }
        }
    
    def test_load_config_success(self, mock_config):
        """Test successful config loading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(mock_config, f)
            config_path = f.name
        
        try:
            with patch.object(Path, 'exists', return_value=True), \
                 patch('builtins.open', mock_open_config(mock_config)):
                
                manager = ModelManager()
                assert manager.models_config == mock_config
        finally:
            Path(config_path).unlink()
    
    def test_load_config_fallback(self):
        """Test config loading fallback."""
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
            assert "low" in recommendation["reason"].lower() or "1.0" in recommendation["reason"]
    
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
    
    def test_recommend_model_no_suitable_models(self, mock_config):
        """Test model recommendation when no models fit."""
        with patch.object(ModelManager, '_load_models_config', return_value=mock_config):
            manager = ModelManager()
            
            hardware_info = {
                "available_ram_gb": 0.1,  # Very low RAM
                "has_gpu": False
            }
            
            recommendation = manager.recommend_model(hardware_info)
            
            # Should fallback to all-minilm
            assert recommendation["recommended_model"] == "all-minilm"
            assert "fallback" in recommendation["reason"].lower()


class TestTextChunker:
    """Test text chunking functionality."""
    
    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        chunker = TextChunker()
        
        # Mock tiktoken to avoid dependency issues in tests
        with patch.object(chunker, 'tokenizer') as mock_tokenizer:
            mock_tokenizer.encode.return_value = list(range(100))  # 100 tokens
            mock_tokenizer.decode.side_effect = lambda tokens: f"chunk_{len(tokens)}_tokens"
            
            text = "This is a test text that should be chunked into smaller pieces."
            chunks = chunker.chunk_text(text, chunk_size=30, overlap=5)
            
            assert len(chunks) > 1
            assert all("chunk_index" in chunk for chunk in chunks)
            assert all("text" in chunk for chunk in chunks)
            assert all("token_count" in chunk for chunk in chunks)
    
    def test_chunk_text_empty(self):
        """Test chunking empty text."""
        chunker = TextChunker()
        
        chunks = chunker.chunk_text("", chunk_size=100, overlap=10)
        assert chunks == []
        
        chunks = chunker.chunk_text("   ", chunk_size=100, overlap=10)
        assert chunks == []
    
    def test_chunk_text_fallback(self):
        """Test chunking fallback when tokenizer fails."""
        chunker = TextChunker()
        chunker.tokenizer = None  # Force character-based chunking
        
        text = "A" * 1000  # 1000 character text
        chunks = chunker._chunk_by_characters(text, chunk_size=300, overlap=50)
        
        assert len(chunks) > 1
        assert chunks[0]["text"] == "A" * 300
        assert chunks[1]["start_char"] == 250  # 300 - 50 overlap
    
    def test_chunk_text_with_overlap(self):
        """Test that overlap is preserved correctly."""
        chunker = TextChunker()
        
        with patch.object(chunker, 'tokenizer') as mock_tokenizer:
            # Create a sequence of tokens
            tokens = list(range(100))
            mock_tokenizer.encode.return_value = tokens
            mock_tokenizer.decode.side_effect = lambda t: f"tokens_{t[0]}_{t[-1]}"
            
            chunks = chunker._chunk_by_tokens("test", chunk_size=30, overlap=5)
            
            # Check that chunks have proper overlap
            assert len(chunks) >= 3
            # Second chunk should start 25 tokens after first (30 - 5 overlap)
            assert chunks[1]["start_token"] == 25


class TestEmbeddingCache:
    """Test embedding caching functionality."""
    
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


class TestEmbeddingGenerator:
    """Test embedding generation functionality."""
    
    @pytest.fixture
    def mock_model_manager(self):
        """Mock model manager."""
        manager = Mock()
        manager.get_available_models.return_value = {
            "all-minilm": {
                "name": "All-MiniLM",
                "dimensions": 384,
                "model_name": "all-MiniLM-L6-v2",
                "description": "Test model"
            }
        }
        manager.recommend_model.return_value = {
            "recommended_model": "all-minilm"
        }
        return manager
    
    @pytest.fixture
    def mock_embedding_cache(self):
        """Mock embedding cache."""
        cache = Mock()
        cache.get_embedding.return_value = None  # No cached embeddings
        return cache
    
    def test_initialization(self, mock_model_manager, mock_embedding_cache):
        """Test embedding generator initialization."""
        with patch('main.hardware_detector') as mock_detector:
            mock_detector.detect_hardware.return_value = {"available_ram_gb": 4.0}
            
            with patch('sentence_transformers.SentenceTransformer') as mock_st:
                mock_model = Mock()
                mock_st.return_value = mock_model
                
                generator = EmbeddingGenerator(mock_model_manager, mock_embedding_cache)
                
                assert generator.current_model_id == "all-minilm"
                assert generator.current_model == mock_model
    
    def test_switch_model_success(self, mock_model_manager, mock_embedding_cache):
        """Test successful model switching."""
        with patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_st.return_value = mock_model
            
            generator = EmbeddingGenerator(mock_model_manager, mock_embedding_cache)
            
            # Switch to same model (should succeed immediately)
            success = generator.switch_model("all-minilm")
            assert success is True
            
            # Add another model to config
            mock_model_manager.get_available_models.return_value["nomic-embed-text"] = {
                "name": "Nomic",
                "dimensions": 768,
                "model_name": "nomic-ai/nomic-embed-text-v1"
            }
            
            # Switch to different model
            success = generator.switch_model("nomic-embed-text")
            assert success is True
            assert generator.current_model_id == "nomic-embed-text"
    
    def test_switch_model_failure(self, mock_model_manager, mock_embedding_cache):
        """Test model switching failure."""
        with patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_st.return_value = Mock()
            
            generator = EmbeddingGenerator(mock_model_manager, mock_embedding_cache)
            
            # Try to switch to non-existent model
            success = generator.switch_model("non_existent_model")
            assert success is False
    
    def test_generate_embeddings_success(self, mock_model_manager, mock_embedding_cache):
        """Test successful embedding generation."""
        with patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.encode.return_value = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            mock_st.return_value = mock_model
            
            generator = EmbeddingGenerator(mock_model_manager, mock_embedding_cache)
            
            texts = ["text1", "text2"]
            embeddings = generator.generate_embeddings(texts)
            
            assert len(embeddings) == 2
            np.testing.assert_array_equal(embeddings[0], np.array([1.0, 2.0, 3.0]))
            np.testing.assert_array_equal(embeddings[1], np.array([4.0, 5.0, 6.0]))
    
    def test_generate_embeddings_with_cache(self, mock_model_manager, mock_embedding_cache):
        """Test embedding generation with cache hits."""
        cached_embedding = np.array([1.0, 2.0, 3.0])
        mock_embedding_cache.get_embedding.side_effect = [cached_embedding, None]
        
        with patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.encode.return_value = np.array([[4.0, 5.0, 6.0]])
            mock_st.return_value = mock_model
            
            generator = EmbeddingGenerator(mock_model_manager, mock_embedding_cache)
            
            texts = ["cached_text", "new_text"]
            embeddings = generator.generate_embeddings(texts)
            
            assert len(embeddings) == 2
            # First should be from cache
            np.testing.assert_array_equal(embeddings[0], cached_embedding)
            # Second should be newly generated
            np.testing.assert_array_equal(embeddings[1], np.array([4.0, 5.0, 6.0]))
    
    def test_generate_embeddings_empty_input(self, mock_model_manager, mock_embedding_cache):
        """Test embedding generation with empty input."""
        with patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_st.return_value = Mock()
            
            generator = EmbeddingGenerator(mock_model_manager, mock_embedding_cache)
            
            embeddings = generator.generate_embeddings([])
            assert embeddings == []
    
    def test_get_current_model_info(self, mock_model_manager, mock_embedding_cache):
        """Test getting current model information."""
        with patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_st.return_value = Mock()
            
            generator = EmbeddingGenerator(mock_model_manager, mock_embedding_cache)
            
            info = generator.get_current_model_info()
            
            assert info["model_id"] == "all-minilm"
            assert info["model_name"] == "All-MiniLM"
            assert info["dimensions"] == 384
            assert "device" in info


def mock_open_config(config_data):
    """Helper to mock file opening for config."""
    from unittest.mock import mock_open
    return mock_open(read_data=json.dumps(config_data))


# Integration tests
class TestEmbeddingSystemIntegration:
    """Integration tests for the complete embedding system."""
    
    def test_full_workflow(self):
        """Test complete embedding generation workflow."""
        # This would be a more complex integration test
        # that tests the interaction between all components
        pass


if __name__ == "__main__":
    pytest.main([__file__])