"""Unit tests for dynamic model switching functionality."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import torch
import tempfile
from pathlib import Path

from main import EmbeddingGenerator, ModelManager, EmbeddingCache


class TestDynamicModelSwitching:
    """Test dynamic model switching without service restart."""
    
    @pytest.fixture
    def mock_models_config(self):
        """Mock models configuration with multiple embedding models."""
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
                    "recommended": False
                }
            }
        }
    
    @pytest.fixture
    def model_manager(self, mock_models_config):
        """Create model manager with mock config."""
        with patch.object(ModelManager, '_load_models_config', return_value=mock_models_config):
            return ModelManager()
    
    @pytest.fixture
    def embedding_cache(self):
        """Create embedding cache with temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = EmbeddingCache()
            cache.cache_dir = Path(temp_dir)
            yield cache
    
    def test_initial_model_loading(self, model_manager, embedding_cache):
        """Test that initial model is loaded correctly."""
        with patch('main.hardware_detector') as mock_detector, \
             patch('sentence_transformers.SentenceTransformer') as mock_st:
            
            mock_detector.detect_hardware.return_value = {"available_ram_gb": 4.0}
            mock_model = Mock()
            mock_st.return_value = mock_model
            
            generator = EmbeddingGenerator(model_manager, embedding_cache)
            
            assert generator.current_model_id is not None
            assert generator.current_model == mock_model
            mock_st.assert_called_once()
    
    def test_switch_to_same_model(self, model_manager, embedding_cache):
        """Test switching to the same model (should be no-op)."""
        with patch('main.hardware_detector') as mock_detector, \
             patch('sentence_transformers.SentenceTransformer') as mock_st:
            
            mock_detector.detect_hardware.return_value = {"available_ram_gb": 4.0}
            mock_model = Mock()
            mock_st.return_value = mock_model
            
            generator = EmbeddingGenerator(model_manager, embedding_cache)
            initial_model_id = generator.current_model_id
            
            # Switch to same model
            success = generator.switch_model(initial_model_id)
            
            assert success is True
            assert generator.current_model_id == initial_model_id
            assert generator.current_model == mock_model
            # Should not create new model instance
            assert mock_st.call_count == 1
    
    def test_switch_to_different_model(self, model_manager, embedding_cache):
        """Test switching to a different model."""
        with patch('main.hardware_detector') as mock_detector, \
             patch('sentence_transformers.SentenceTransformer') as mock_st, \
             patch('torch.cuda.empty_cache') as mock_empty_cache:
            
            mock_detector.detect_hardware.return_value = {"available_ram_gb": 4.0}
            
            # Create different mock models for each call
            mock_model1 = Mock()
            mock_model2 = Mock()
            mock_st.side_effect = [mock_model1, mock_model2]
            
            generator = EmbeddingGenerator(model_manager, embedding_cache)
            initial_model_id = generator.current_model_id
            
            # Switch to different model
            target_model = "nomic-embed-text" if initial_model_id != "nomic-embed-text" else "mxbai-embed-large"
            success = generator.switch_model(target_model)
            
            assert success is True
            assert generator.current_model_id == target_model
            assert generator.current_model == mock_model2
            # Should have created two model instances
            assert mock_st.call_count == 2
    
    def test_switch_to_nonexistent_model(self, model_manager, embedding_cache):
        """Test switching to a model that doesn't exist."""
        with patch('main.hardware_detector') as mock_detector, \
             patch('sentence_transformers.SentenceTransformer') as mock_st:
            
            mock_detector.detect_hardware.return_value = {"available_ram_gb": 4.0}
            mock_st.return_value = Mock()
            
            generator = EmbeddingGenerator(model_manager, embedding_cache)
            
            # Try to switch to non-existent model
            success = generator.switch_model("nonexistent-model")
            
            assert success is False
            # Current model should remain unchanged
            assert generator.current_model_id != "nonexistent-model"
    
    def test_model_switching_with_error(self, model_manager, embedding_cache):
        """Test model switching when SentenceTransformer loading fails."""
        with patch('main.hardware_detector') as mock_detector, \
             patch('sentence_transformers.SentenceTransformer') as mock_st:
            
            mock_detector.detect_hardware.return_value = {"available_ram_gb": 4.0}
            
            # First call succeeds, second call fails
            mock_model1 = Mock()
            mock_st.side_effect = [mock_model1, Exception("Model loading failed")]
            
            generator = EmbeddingGenerator(model_manager, embedding_cache)
            initial_model_id = generator.current_model_id
            
            # Try to switch to different model (should fail)
            success = generator.switch_model("nomic-embed-text")
            
            assert success is False
            # Should keep original model
            assert generator.current_model_id == initial_model_id
            assert generator.current_model == mock_model1
    
    def test_model_name_mapping(self, model_manager, embedding_cache):
        """Test that model IDs are correctly mapped to SentenceTransformer model names."""
        with patch('main.hardware_detector') as mock_detector, \
             patch('sentence_transformers.SentenceTransformer') as mock_st:
            
            mock_detector.detect_hardware.return_value = {"available_ram_gb": 4.0}
            mock_st.return_value = Mock()
            
            generator = EmbeddingGenerator(model_manager, embedding_cache)
            
            # Test switching to each model type
            test_cases = [
                ("all-minilm", "all-MiniLM-L6-v2"),
                ("nomic-embed-text", "nomic-ai/nomic-embed-text-v1"),
                ("mxbai-embed-large", "mixedbread-ai/mxbai-embed-large-v1")
            ]
            
            for model_id, expected_name in test_cases:
                generator.switch_model(model_id)
                
                # Check that SentenceTransformer was called with correct name
                calls = mock_st.call_args_list
                last_call = calls[-1]
                assert expected_name in str(last_call)
    
    def test_memory_cleanup_on_switch(self, model_manager, embedding_cache):
        """Test that old models are properly cleaned up when switching."""
        with patch('main.hardware_detector') as mock_detector, \
             patch('sentence_transformers.SentenceTransformer') as mock_st, \
             patch('torch.cuda.is_available') as mock_cuda, \
             patch('torch.cuda.empty_cache') as mock_empty_cache:
            
            mock_detector.detect_hardware.return_value = {"available_ram_gb": 4.0}
            mock_cuda.return_value = True
            
            mock_model1 = Mock()
            mock_model2 = Mock()
            mock_st.side_effect = [mock_model1, mock_model2]
            
            generator = EmbeddingGenerator(model_manager, embedding_cache)
            
            # Switch to different model
            generator.switch_model("nomic-embed-text")
            
            # Should have called CUDA cache cleanup
            mock_empty_cache.assert_called_once()
    
    def test_embeddings_consistency_after_switch(self, model_manager, embedding_cache):
        """Test that embeddings are generated correctly after model switching."""
        with patch('main.hardware_detector') as mock_detector, \
             patch('sentence_transformers.SentenceTransformer') as mock_st:
            
            mock_detector.detect_hardware.return_value = {"available_ram_gb": 4.0}
            
            # Create mock models with different embedding dimensions
            mock_model1 = Mock()
            mock_model1.encode.return_value = np.array([[1.0, 2.0, 3.0, 4.0]])  # 4D for simplicity
            
            mock_model2 = Mock()
            mock_model2.encode.return_value = np.array([[5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])  # 6D
            
            mock_st.side_effect = [mock_model1, mock_model2]
            
            generator = EmbeddingGenerator(model_manager, embedding_cache)
            
            # Generate embeddings with first model
            embeddings1 = generator.generate_embeddings(["test text"])
            assert len(embeddings1) == 1
            np.testing.assert_array_equal(embeddings1[0], np.array([1.0, 2.0, 3.0, 4.0]))
            
            # Switch model
            generator.switch_model("nomic-embed-text")
            
            # Generate embeddings with second model
            embeddings2 = generator.generate_embeddings(["test text"])
            assert len(embeddings2) == 1
            np.testing.assert_array_equal(embeddings2[0], np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0]))
    
    def test_model_info_updates_after_switch(self, model_manager, embedding_cache):
        """Test that model info is updated correctly after switching."""
        with patch('main.hardware_detector') as mock_detector, \
             patch('sentence_transformers.SentenceTransformer') as mock_st:
            
            mock_detector.detect_hardware.return_value = {"available_ram_gb": 4.0}
            mock_st.return_value = Mock()
            
            generator = EmbeddingGenerator(model_manager, embedding_cache)
            
            # Get initial model info
            initial_info = generator.get_current_model_info()
            initial_model_id = initial_info["model_id"]
            initial_dimensions = initial_info["dimensions"]
            
            # Switch to model with different dimensions
            target_model = "mxbai-embed-large"  # 1024 dimensions
            generator.switch_model(target_model)
            
            # Get updated model info
            updated_info = generator.get_current_model_info()
            
            assert updated_info["model_id"] == target_model
            assert updated_info["dimensions"] == 1024
            assert updated_info["model_name"] == "MxBai Embed Large"
            
            # Should be different from initial
            if initial_model_id != target_model:
                assert updated_info["dimensions"] != initial_dimensions
    
    def test_cache_isolation_between_models(self, model_manager, embedding_cache):
        """Test that cache is properly isolated between different models."""
        with patch('main.hardware_detector') as mock_detector, \
             patch('sentence_transformers.SentenceTransformer') as mock_st:
            
            mock_detector.detect_hardware.return_value = {"available_ram_gb": 4.0}
            
            mock_model1 = Mock()
            mock_model1.encode.return_value = np.array([[1.0, 2.0, 3.0, 4.0]])
            
            mock_model2 = Mock()
            mock_model2.encode.return_value = np.array([[5.0, 6.0, 7.0, 8.0]])
            
            mock_st.side_effect = [mock_model1, mock_model2]
            
            generator = EmbeddingGenerator(model_manager, embedding_cache)
            initial_model = generator.current_model_id
            
            # Generate and cache embedding with first model
            text = "test text"
            embeddings1 = generator.generate_embeddings([text])
            
            # Switch model
            target_model = "nomic-embed-text" if initial_model != "nomic-embed-text" else "mxbai-embed-large"
            generator.switch_model(target_model)
            
            # Generate embedding with second model (should not use cache from first model)
            embeddings2 = generator.generate_embeddings([text])
            
            # Embeddings should be different (from different models)
            assert not np.array_equal(embeddings1[0], embeddings2[0])
            
            # Both models should have been called (no cache hit for second model)
            assert mock_model1.encode.called
            assert mock_model2.encode.called
    
    def test_concurrent_model_switching(self, model_manager, embedding_cache):
        """Test behavior when model switching happens during embedding generation."""
        # This is a more complex test that would require async testing
        # For now, we'll test the basic thread safety aspects
        
        with patch('main.hardware_detector') as mock_detector, \
             patch('sentence_transformers.SentenceTransformer') as mock_st:
            
            mock_detector.detect_hardware.return_value = {"available_ram_gb": 4.0}
            mock_st.return_value = Mock()
            
            generator = EmbeddingGenerator(model_manager, embedding_cache)
            
            # Ensure that model switching is atomic
            # (current implementation is synchronous, so this should work)
            initial_model = generator.current_model_id
            success = generator.switch_model("nomic-embed-text")
            
            if success:
                assert generator.current_model_id == "nomic-embed-text"
                # Model should be fully switched, not in intermediate state
                assert generator.current_model is not None
            else:
                # If switch failed, should remain on original model
                assert generator.current_model_id == initial_model


if __name__ == "__main__":
    pytest.main([__file__])