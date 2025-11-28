"""Integration tests for multi-model analysis pipeline."""

import asyncio
import json
import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
from datetime import datetime
import os
import sys

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the helper function from conftest
from conftest import create_mock_httpx_client

from main import (
    app, OllamaClient, QdrantManager, PerformanceMonitor,
    ContentItem, AnalysisRequest
)


@pytest.fixture
def sample_content_items():
    """Sample content items for testing."""
    return [
        ContentItem(
            id="test_1",
            content="This is a test article about artificial intelligence and machine learning. It covers various aspects of AI development and applications in modern technology.",
            title="AI and ML Overview",
            url="https://example.com/ai-article",
            metadata={"category": "technology", "author": "Test Author"}
        ),
        ContentItem(
            id="test_2", 
            content="Python is a versatile programming language used for web development, data science, and automation. It has a simple syntax and powerful libraries.",
            title="Python Programming Guide",
            url="https://example.com/python-guide",
            metadata={"category": "programming", "difficulty": "beginner"}
        )
    ]


@pytest.fixture
def mock_ollama_response():
    """Mock Ollama API response."""
    return {
        "response": "This is a test summary of the content.",
        "eval_count": 50,
        "eval_duration": 1000000000  # 1 second in nanoseconds
    }


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client."""
    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = []
    mock_client.create_collection = MagicMock()
    mock_client.upsert = MagicMock()
    mock_client.count.return_value.count = 0
    mock_client.scroll.return_value = ([], None)
    return mock_client


class TestOllamaClient:
    """Test Ollama client functionality."""
    
    @pytest.mark.asyncio
    async def test_initialize_ollama_client(self, mock_ollama_responses):
        """Test Ollama client initialization."""
        client = OllamaClient("http://test-ollama:11434")
        
        with patch('httpx.AsyncClient') as mock_httpx:
            mock_client = create_mock_httpx_client(mock_ollama_responses)
            mock_httpx.return_value.__aenter__.return_value = mock_client
            
            await client.initialize()
            
            assert "llama3.2:3b" in client.available_models
            assert "llama3.2:1b" in client.available_models
            assert "qwen3:1.7b" in client.available_models
            assert len(client.fallback_chains) > 0
    
    @pytest.mark.asyncio
    async def test_generate_with_fallback_success(self, mock_ollama_responses):
        """Test successful generation with primary model."""
        client = OllamaClient()
        client.available_models = ["llama3.2:3b", "llama3.2:1b"]
        
        with patch('httpx.AsyncClient') as mock_httpx:
            mock_client = create_mock_httpx_client(mock_ollama_responses)
            mock_httpx.return_value.__aenter__.return_value = mock_client
            
            result = await client.generate_with_fallback("Test prompt", "llama3.2:3b")
            
            assert result["success"] is True
            assert result["response"] == "This is a test response from the LLM model."
            assert result["model_used"] == "llama3.2:3b"
            assert result["tokens_per_second"] > 0
    
    @pytest.mark.asyncio
    async def test_generate_with_fallback_failure_then_success(self):
        """Test fallback to smaller model when primary fails."""
        client = OllamaClient()
        client.available_models = ["llama3.2:1b"]
        client.fallback_chains = {"llama3.2:3b": ["llama3.2:1b"]}
        
        with patch('httpx.AsyncClient') as mock_httpx:
            # First call fails (primary model)
            # Second call succeeds (fallback model)
            mock_responses = [
                MagicMock(status_code=500, text="Model not available"),
                MagicMock(status_code=200)
            ]
            mock_responses[1].json.return_value = {
                "response": "Fallback response",
                "eval_count": 20
            }
            
            mock_httpx.return_value.__aenter__.return_value.post.side_effect = mock_responses
            
            result = await client.generate_with_fallback("Test prompt", "llama3.2:3b")
            
            assert result["success"] is True
            assert result["response"] == "Fallback response"
            assert result["model_used"] == "llama3.2:1b"
    
    @pytest.mark.asyncio
    async def test_summarize_content(self):
        """Test content summarization."""
        client = OllamaClient()
        client.available_models = ["llama3.2:3b"]
        
        with patch.object(client, 'generate_with_fallback') as mock_generate:
            mock_generate.return_value = {
                "success": True,
                "response": "This is a summary of the content.",
                "model_used": "llama3.2:3b"
            }
            
            result = await client.summarize_content("Long content to summarize", "llama3.2:3b")
            
            assert result["success"] is True
            assert "summary" in result["response"].lower()
            mock_generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_extract_keywords(self):
        """Test keyword extraction."""
        client = OllamaClient()
        client.available_models = ["llama3.2:3b"]
        
        with patch.object(client, 'generate_with_fallback') as mock_generate:
            mock_generate.return_value = {
                "success": True,
                "response": "artificial intelligence, machine learning, technology",
                "model_used": "llama3.2:3b"
            }
            
            result = await client.extract_keywords("Content about AI and ML", "llama3.2:3b")
            
            assert result["success"] is True
            assert "artificial intelligence" in result["response"]
            mock_generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_assess_content_quality(self):
        """Test content quality assessment."""
        client = OllamaClient()
        client.available_models = ["llama3.2:3b"]
        
        with patch.object(client, 'generate_with_fallback') as mock_generate:
            mock_generate.return_value = {
                "success": True,
                "response": "The content is well-written and informative. Quality score: 8/10",
                "model_used": "llama3.2:3b"
            }
            
            result = await client.assess_content_quality("High quality content", "llama3.2:3b")
            
            assert result["success"] is True
            assert "8/10" in result["response"]
            mock_generate.assert_called_once()


class TestQdrantManager:
    """Test Qdrant manager functionality."""
    
    @pytest.mark.asyncio
    async def test_initialize_qdrant_manager(self):
        """Test Qdrant manager initialization."""
        # Patch where QdrantClient is imported in main.py
        with patch('main.QdrantClient') as mock_qdrant_cls:
            mock_client_instance = MagicMock()
            mock_qdrant_cls.return_value = mock_client_instance
            
            manager = QdrantManager("test-host", 6333)
            await manager.initialize()
            
            assert manager.client is mock_client_instance
            mock_qdrant_cls.assert_called_once_with(host="test-host", port=6333)
    
    @pytest.mark.asyncio
    async def test_ensure_collection_exists_new(self, mock_qdrant_client):
        """Test creating new collection."""
        manager = QdrantManager()
        manager.client = mock_qdrant_client  # ✅ No parentheses
        
        await manager.ensure_collection_exists("test_collection", 384)
        
        manager.client.create_collection.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_store_analyzed_content(self, mock_qdrant_client):
        """Test storing analyzed content with model metadata."""
        manager = QdrantManager()
        manager.client = mock_qdrant_client  # ✅ No parentheses
        
        content_items = [
            {
                "content_id": "test_1",
                "chunk_index": 0,
                "text": "Test content",
                "title": "Test Title",
                "url": "https://example.com",
                "token_count": 10,
                "embedding": [0.1, 0.2, 0.3],
                "summary": "Test summary",
                "keywords": "test, content",
                "quality_score": 8.5,
                "embedding_generation_time": 0.1,
                "llm_processing_time": 0.5,
                "original_metadata": {"category": "test"}
            }
        ]
        
        points_stored = await manager.store_analyzed_content(
            collection_name="test_collection",
            content_items=content_items,
            embedding_model="nomic-embed-text",
            llm_model="llama3.2:3b"
        )
        
        assert points_stored == 1
        manager.client.upsert.assert_called_once()
        
        # Verify the point structure
        call_args = manager.client.upsert.call_args
        points = call_args[1]["points"]
        assert len(points) == 1
        
        point = points[0]
        assert point.payload["embedding_model"] == "nomic-embed-text"
        assert point.payload["llm_model"] == "llama3.2:3b"
        assert point.payload["summary"] == "Test summary"
        assert point.payload["keywords"] == "test, content"
        assert point.payload["quality_score"] == 8.5
    
    @pytest.mark.asyncio
    async def test_get_collection_info(self):
        """Test getting collection information."""
        manager = QdrantManager()
        
        # Create a properly configured mock client
        mock_client = MagicMock()
        
        # Mock the collection info with the exact structure the method expects
        mock_collection_info = MagicMock()
        mock_collection_info.config.params.vectors.size = 768
        mock_collection_info.config.params.vectors.distance.name = "COSINE"
        mock_collection_info.status.name = "GREEN"
        mock_client.get_collection.return_value = mock_collection_info
        
        # Mock the count method
        mock_count_result = MagicMock()
        mock_count_result.count = 100
        mock_client.count.return_value = mock_count_result
        
        # Mock the scroll method with sample points
        mock_point = MagicMock()
        mock_point.payload = {
            "embedding_model": "nomic-embed-text",
            "llm_model": "llama3.2:3b"
        }
        mock_client.scroll.return_value = ([mock_point], None)
        
        manager.client = mock_client
        
        info = await manager.get_collection_info("test_collection")
        
        assert info["collection_name"] == "test_collection"
        assert info["vector_size"] == 768
        assert info["distance_metric"] == "COSINE"
        assert info["points_count"] == 100
        assert "embedding_models" in info["model_usage"]
        assert "llm_models" in info["model_usage"]
        assert "nomic-embed-text" in info["model_usage"]["embedding_models"]
        assert "llama3.2:3b" in info["model_usage"]["llm_models"]
    
    @pytest.mark.asyncio
    async def test_search_similar_content(self):
        """Test searching for similar content."""
        manager = QdrantManager()
        
        # Create a properly configured mock client
        mock_client = MagicMock()
        
        # Mock search results with the exact structure the method expects
        mock_search_result = MagicMock()
        mock_search_result.id = "test_1_chunk_0"
        mock_search_result.score = 0.95
        mock_search_result.payload = {
            "text": "Test content",
            "embedding_model": "nomic-embed-text"
        }
        mock_client.search.return_value = [mock_search_result]
        
        manager.client = mock_client
        
        results = await manager.search_similar_content(
            collection_name="test_collection",
            query_vector=[0.1, 0.2, 0.3],
            limit=5,
            embedding_model_filter="nomic-embed-text"
        )
        
        assert len(results) == 1
        assert results[0]["id"] == "test_1_chunk_0"
        assert results[0]["score"] == 0.95
        assert results[0]["payload"]["text"] == "Test content"
        
        # Verify the client.search was called with correct parameters
        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args
        assert call_args[1]["collection_name"] == "test_collection"
        assert call_args[1]["query_vector"] == [0.1, 0.2, 0.3]
        assert call_args[1]["limit"] == 5


class TestPerformanceMonitor:
    """Test performance monitoring functionality."""
    
    def test_record_model_performance(self):
        """Test recording model performance metrics."""
        monitor = PerformanceMonitor()
        
        monitor.record_model_performance(
            model_id="llama3.2:3b",
            model_type="llm",
            success=True,
            response_time=1.5,
            tokens_per_second=25.0,
            resource_usage={"ram_usage": 4.2}
        )
        
        assert "llama3.2:3b" in monitor.metrics
        metrics = monitor.metrics["llama3.2:3b"]
        assert metrics["total_requests"] == 1
        assert metrics["successful_requests"] == 1
        assert metrics["failed_requests"] == 0
        assert len(metrics["response_times"]) == 1
        assert metrics["response_times"][0] == 1.5
    
    def test_get_model_metrics(self):
        """Test getting model performance metrics."""
        monitor = PerformanceMonitor()
        
        # Record some performance data
        monitor.record_model_performance("test_model", "llm", True, 1.0, 20.0)
        monitor.record_model_performance("test_model", "llm", True, 2.0, 15.0)
        monitor.record_model_performance("test_model", "llm", False, 0.0)
        
        metrics = monitor.get_model_metrics("test_model")
        
        assert metrics.model_id == "test_model"
        assert metrics.total_requests == 3
        assert metrics.successful_requests == 2
        assert metrics.failed_requests == 1
        assert metrics.average_response_time == 1.5  # (1.0 + 2.0) / 2
        assert metrics.average_tokens_per_second == 17.5  # (20.0 + 15.0) / 2
    
    def test_record_system_resources(self):
        """Test recording system resource usage."""
        monitor = PerformanceMonitor()
        
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu:
            
            mock_memory.return_value.percent = 65.0
            mock_memory.return_value.available = 4 * 1024**3  # 4GB
            mock_cpu.return_value = 25.0
            
            monitor.record_system_resources()
            
            assert len(monitor.resource_history) == 1
            resource = monitor.resource_history[0]
            assert resource["ram_usage_percent"] == 65.0
            assert resource["available_ram_gb"] == 4.0
            assert resource["cpu_usage_percent"] == 25.0
    
    def test_get_resource_summary(self):
        """Test getting resource usage summary."""
        monitor = PerformanceMonitor()
        
        # Add some mock resource history
        monitor.resource_history = [
            {
                "timestamp": "2024-01-01T10:00:00",
                "ram_usage_percent": 60.0,
                "available_ram_gb": 4.5,
                "cpu_usage_percent": 20.0
            },
            {
                "timestamp": "2024-01-01T10:01:00", 
                "ram_usage_percent": 70.0,
                "available_ram_gb": 3.5,
                "cpu_usage_percent": 30.0
            }
        ]
        
        summary = monitor.get_resource_summary()
        
        assert summary["average_ram_usage_percent"] == 65.0
        assert summary["average_cpu_usage_percent"] == 25.0
        assert summary["current_available_ram_gb"] == 3.5
        assert summary["measurements_count"] == 2


class TestCompleteAnalysisPipeline:
    """Test complete multi-model analysis pipeline."""
    
    @pytest.mark.asyncio
    async def test_complete_analysis_endpoint(self, sample_content_items, mocks):
        """Test complete analysis API endpoint."""
        request_data = AnalysisRequest(
            content_items=sample_content_items,
            session_id="test_session",
            llm_model="llama3.2:3b",
            embedding_model="nomic-embed-text",
            generate_summary=True,
            extract_keywords=True,
            assess_quality=True
        )
        
        # The globals are already wired by the wire_globals fixture
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/analysis/complete",
                json=request_data.dict()
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "processing"
            assert "job_id" in data
            
            # Verify the mock was called
            mocks.embedding.switch_model.assert_called_once_with("nomic-embed-text")
    
    @pytest.mark.asyncio
    async def test_model_performance_endpoint(self):
        """Test model performance monitoring endpoint."""
        with patch('main.performance_monitor') as mock_monitor:
            mock_metrics = [
                MagicMock(
                    model_id="llama3.2:3b",
                    model_type="llm",
                    total_requests=10,
                    successful_requests=9,
                    average_response_time=1.5
                )
            ]
            mock_monitor.get_all_metrics.return_value = mock_metrics
            mock_monitor.get_resource_summary.return_value = {
                "average_ram_usage_percent": 65.0,
                "average_cpu_usage_percent": 25.0
            }
            
            async with httpx.AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get("/performance/models")
                
                assert response.status_code == 200
                data = response.json()
                assert "model_metrics" in data
                assert "system_resources" in data
    
    @pytest.mark.asyncio
    async def test_qdrant_collection_info_endpoint(self, mocks):
        """Test Qdrant collection info endpoint."""
        # The globals are already wired by the wire_globals fixture
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/qdrant/collections/test_session/info")
            
            assert response.status_code == 200
            data = response.json()
            assert data["collection_name"] == "test_session"
            assert data["vector_size"] == 768
            assert data["points_count"] == 100
            
            # Verify the mock was called
            mocks.qdrant.get_collection_info.assert_called_once_with("test_session")
    
    @pytest.mark.asyncio
    async def test_search_collection_endpoint(self, mocks):
        """Test collection search endpoint."""
        # The globals are already wired by the wire_globals fixture
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get(
                "/qdrant/collections/test_session/search",
                params={
                    "query_text": "test query",
                    "limit": 5
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["query"] == "test query"
            assert len(data["results"]) == 1
            assert data["results"][0]["score"] == 0.95
            assert data["embedding_model_used"] == "nomic-embed-text"
            
            # Verify the mocks were called
            mocks.embedding.generate_embeddings.assert_called_once_with(["test query"])
            mocks.qdrant.search_similar_content.assert_called_once()


class TestModelFallbackStrategies:
    """Test automatic model fallback strategies."""
    
    @pytest.mark.asyncio
    async def test_llm_fallback_chain(self):
        """Test LLM model fallback when primary model fails."""
        client = OllamaClient()
        client.available_models = ["llama3.2:1b"]  # Only smaller model available
        client.fallback_chains = {"llama3.2:8b": ["llama3.2:3b", "llama3.2:1b"]}
        
        with patch('httpx.AsyncClient') as mock_httpx:
            # Mock responses: first two fail, third succeeds
            mock_responses = [
                MagicMock(status_code=404),  # llama3.2:8b not found
                MagicMock(status_code=404),  # llama3.2:3b not found  
                MagicMock(status_code=200)   # llama3.2:1b succeeds
            ]
            mock_responses[2].json.return_value = {
                "response": "Fallback model response",
                "eval_count": 15
            }
            
            mock_httpx.return_value.__aenter__.return_value.post.side_effect = mock_responses
            
            result = await client.generate_with_fallback("Test prompt", "llama3.2:8b")
            
            assert result["success"] is True
            assert result["model_used"] == "llama3.2:1b"
            assert result["response"] == "Fallback model response"
    
    @pytest.mark.asyncio
    async def test_embedding_model_fallback(self):
        """Test embedding model fallback through model manager."""
        from main import ModelManager
        
        # Use extremely low RAM to force fallback to all-minilm
        hardware_info = {
            "available_ram_gb": 0.3,  # Below all-minilm's 0.5GB requirement
            "has_gpu": False,
            "cpu_count": 2
        }
        
        model_manager = ModelManager()
        recommendation = model_manager.recommend_model(hardware_info)
        
        # With 0.3GB RAM, should fallback to the most lightweight option
        # or return the fallback recommendation
        assert recommendation["recommended_model"] in ["all-minilm", "nomic-embed-text"]
        assert "alternatives" in recommendation
        assert "reason" in recommendation
        
        # Test with slightly higher RAM that should prefer nomic-embed-text
        hardware_info_medium = {
            "available_ram_gb": 1.2,  # Above nomic-embed-text's 1GB requirement
            "has_gpu": False,
            "cpu_count": 4
        }
        
        recommendation_medium = model_manager.recommend_model(hardware_info_medium)
        # This should prefer nomic-embed-text since it's recommended and fits
        assert recommendation_medium["recommended_model"] == "nomic-embed-text"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])