"""Test configuration and mocks for analyzer service tests."""

import os
import json
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
import pytest

# Test configuration
TEST_CONFIG = {
    "qdrant_host": "mock-qdrant",
    "qdrant_port": 6333,
    "ollama_base_url": "http://mock-ollama:11434",
    "test_models_config": {
        "llm_models": {
            "llama3.2:3b": {
                "name": "Llama 3.2 3B",
                "size": "2GB",
                "speed": "fast", 
                "quality": "good",
                "min_ram_gb": 4,
                "description": "Test LLM model",
                "provider": "Meta",
                "recommended": True
            },
            "llama3.2:1b": {
                "name": "Llama 3.2 1B",
                "size": "1.3GB",
                "speed": "fastest",
                "quality": "basic", 
                "min_ram_gb": 2,
                "description": "Lightweight test model",
                "provider": "Meta",
                "recommended": False
            }
        },
        "embedding_models": {
            "nomic-embed-text": {
                "name": "Nomic Embed Text",
                "size": "274MB",
                "dimensions": 768,
                "quality": "high",
                "min_ram_gb": 1,
                "description": "Test embedding model",
                "provider": "Nomic AI",
                "model_name": "nomic-ai/nomic-embed-text-v1",
                "recommended": True
            },
            "all-minilm": {
                "name": "All-MiniLM",
                "size": "90MB",
                "dimensions": 384,
                "quality": "good", 
                "min_ram_gb": 0.5,
                "description": "Lightweight embedding model",
                "provider": "Microsoft",
                "model_name": "all-MiniLM-L6-v2",
                "recommended": False
            }
        }
    }
}

def setup_test_config():
    """Setup test configuration files."""
    # Create test config directory
    config_dir = Path("/tmp/analyzer_test_config")
    config_dir.mkdir(exist_ok=True)
    
    # Write models config
    models_config_path = config_dir / "models.json"
    with open(models_config_path, "w") as f:
        json.dump(TEST_CONFIG["test_models_config"], f, indent=2)
    
    # Set environment variables
    os.environ["CONFIG_PATH"] = str(config_dir)
    os.environ["TESTING"] = "true"
    
    return config_dir

@pytest.fixture(scope="session")
def test_config():
    """Pytest fixture for test configuration."""
    config_dir = setup_test_config()
    yield TEST_CONFIG
    
    # Cleanup
    try:
        import shutil
        shutil.rmtree(config_dir)
    except Exception:
        pass

@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing."""
    mock_client = MagicMock()
    
    # Mock collections
    mock_client.get_collections.return_value.collections = []
    mock_client.create_collection = MagicMock()
    mock_client.upsert = MagicMock()
    mock_client.count.return_value.count = 0
    mock_client.scroll.return_value = ([], None)
    
    # Mock collection info
    mock_collection_info = MagicMock()
    mock_collection_info.config.params.vectors.size = 768
    mock_collection_info.config.params.vectors.distance.name = "COSINE"
    mock_collection_info.status.name = "GREEN"
    mock_client.get_collection.return_value = mock_collection_info
    
    # Mock search results
    mock_search_result = MagicMock()
    mock_search_result.id = "test_1_chunk_0"
    mock_search_result.score = 0.95
    mock_search_result.payload = {"text": "Test content", "embedding_model": "nomic-embed-text"}
    mock_client.search.return_value = [mock_search_result]
    
    return mock_client

@pytest.fixture
def mock_ollama_responses():
    """Mock Ollama API responses."""
    return {
        "tags": {
            "models": [
                {"name": "llama3.2:3b"},
                {"name": "llama3.2:1b"},
                {"name": "qwen3:1.7b"}
            ]
        },
        "generate": {
            "response": "This is a test response from the LLM model.",
            "eval_count": 25,
            "eval_duration": 1000000000  # 1 second in nanoseconds
        },
        "pull": {"status": "success"}
    }

@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer for testing."""
    mock_model = MagicMock()
    
    # Mock encode method to return dummy embeddings
    def mock_encode(texts, convert_to_numpy=True):
        import numpy as np
        # Return dummy embeddings with correct dimensions
        embeddings = []
        for text in texts:
            # Create deterministic embeddings based on text hash
            hash_val = hash(text) % 1000
            embedding = np.random.RandomState(hash_val).rand(384).astype(np.float32)
            embeddings.append(embedding)
        return np.array(embeddings)
    
    mock_model.encode = mock_encode
    return mock_model

@pytest.fixture
def mock_hardware_info():
    """Mock hardware information for testing."""
    return {
        "ram_gb": 16.0,
        "cpu_count": 8,
        "has_gpu": True,
        "gpu_memory_gb": 8.0,
        "gpu_name": "Test GPU",
        "available_ram_gb": 12.0,
        "ram_usage_percent": 25.0,
        "cpu_usage_percent": 15.0
    }

def create_mock_httpx_client(responses):
    """Create a mock httpx client with predefined responses."""
    mock_client = MagicMock()
    
    async def mock_get(url):
        mock_response = MagicMock()
        if "tags" in url:
            mock_response.status_code = 200
            mock_response.json.return_value = responses["tags"]
        else:
            mock_response.status_code = 404
        return mock_response
    
    async def mock_post(url, json=None):
        mock_response = MagicMock()
        if "generate" in url:
            mock_response.status_code = 200
            mock_response.json.return_value = responses["generate"]
        elif "pull" in url:
            mock_response.status_code = 200
            mock_response.json.return_value = responses["pull"]
        else:
            mock_response.status_code = 404
        return mock_response
    
    mock_client.get = mock_get
    mock_client.post = mock_post
    
    return mock_client

# Patch functions for easier testing
def patch_external_dependencies():
    """Context manager to patch external dependencies."""
    from unittest.mock import patch
    
    patches = [
        patch('qdrant_client.QdrantClient'),
        patch('sentence_transformers.SentenceTransformer'),
        patch('httpx.AsyncClient'),
        patch('torch.cuda.is_available', return_value=False),
        patch('psutil.virtual_memory'),
        patch('psutil.cpu_count', return_value=8),
        patch('psutil.cpu_percent', return_value=15.0)
    ]
    
    return patches