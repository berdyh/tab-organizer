"""Test configuration and fixtures for analyzer service tests."""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, Mock

import numpy as np
import pytest
from pydantic import HttpUrl

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

mock_sentence_transformers = MagicMock()
mock_torch = MagicMock()
mock_torch_cuda = MagicMock()
mock_torch.cuda = mock_torch_cuda
setattr(mock_torch, "__analyzer_placeholder__", True)
mock_qdrant = MagicMock()
mock_qdrant_models = MagicMock()
mock_qdrant.models = mock_qdrant_models
mock_tiktoken = MagicMock()

sys.modules["sentence_transformers"] = mock_sentence_transformers
sys.modules["torch"] = mock_torch
sys.modules["torch.cuda"] = mock_torch_cuda
sys.modules["qdrant_client"] = mock_qdrant
sys.modules["qdrant_client.models"] = mock_qdrant_models
sys.modules["tiktoken"] = mock_tiktoken

main_module = importlib.import_module("main")
sys.modules["main"] = main_module


@pytest.fixture
def mocks():
    """Create a namespace of all mocks for easy access and assertions."""
    return SimpleNamespace(
        qdrant=Mock(),
        embedding=Mock(),
        chunker=Mock(),
        ollama=Mock(),
        performance=Mock(),
        hardware=Mock()
    )


@pytest.fixture(autouse=True)
def wire_globals(monkeypatch, mocks):
    """Wire global variables in main.py to prevent 503 errors in API tests."""
    import main

    # Set up the mocks with proper return values
    mocks.embedding.switch_model.return_value = True
    mocks.embedding.get_current_model_info.return_value = {"dimensions": 768}
    mocks.embedding.generate_embeddings.return_value = [np.array([0.1, 0.2, 0.3])]
    mocks.embedding.current_model_id = "nomic-embed-text"
    
    mocks.qdrant.get_collection_info = AsyncMock(return_value={
        "collection_name": "test_session",
        "vector_size": 768,
        "distance_metric": "COSINE",
        "points_count": 100,
        "model_usage": {
            "embedding_models": ["nomic-embed-text"],
            "llm_models": ["llama3.2:3b"]
        }
    })
    mocks.qdrant.search_similar_content = AsyncMock(return_value=[
        {
            "id": "test_1_chunk_0",
            "score": 0.95,
            "payload": {"text": "Similar content", "title": "Test Article"}
        }
    ])
    mocks.qdrant.host = "qdrant"
    mocks.qdrant.port = 6333
    
    mocks.performance.get_all_metrics.return_value = []
    mocks.performance.get_resource_summary.return_value = {
        "average_ram_usage_percent": 65.0,
        "average_cpu_usage_percent": 25.0
    }
    
    # Wire the globals
    monkeypatch.setattr(main, "embedding_generator", mocks.embedding, raising=False)
    monkeypatch.setattr(main, "text_chunker", mocks.chunker, raising=False)
    monkeypatch.setattr(main, "ollama_client", mocks.ollama, raising=False)
    monkeypatch.setattr(main, "qdrant_manager", mocks.qdrant, raising=False)
    monkeypatch.setattr(main, "performance_monitor", mocks.performance, raising=False)
    monkeypatch.setattr(main, "hardware_detector", mocks.hardware, raising=False)


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing."""
    mock_client = MagicMock()
    
    # Mock collections
    mock_client.get_collections.return_value.collections = []
    mock_client.create_collection = MagicMock()
    mock_client.upsert = MagicMock()
    
    # Mock count method - return a mock with count attribute
    mock_count_result = MagicMock()
    mock_count_result.count = 0
    mock_client.count.return_value = mock_count_result
    
    # Mock scroll method - return tuple of (points, next_page_offset)
    mock_point = MagicMock()
    mock_point.payload = {
        "embedding_model": "nomic-embed-text",
        "llm_model": "llama3.2:3b"
    }
    mock_client.scroll.return_value = ([mock_point], None)
    
    # Mock collection info with proper nested structure
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
def sample_content_items():
    """Sample content items for testing."""
    from analyzer import ContentItem

    return [
        ContentItem(
            id="test_1",
            content="This is a test article about artificial intelligence and machine learning. It covers various aspects of AI development and applications in modern technology.",
            title="AI and ML Overview",
            url=_http("https://example.com/ai-article"),
            metadata={"category": "technology", "author": "Test Author"}
        ),
        ContentItem(
            id="test_2", 
            content="Python is a versatile programming language used for web development, data science, and automation. It has a simple syntax and powerful libraries.",
            title="Python Programming Guide",
            url=_http("https://example.com/python-guide"),
            metadata={"category": "programming", "difficulty": "beginner"}
        )
    ]


def _http(url: str) -> HttpUrl:
    """Helper for type-safe HttpUrl literals."""
    return cast(HttpUrl, url)


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
