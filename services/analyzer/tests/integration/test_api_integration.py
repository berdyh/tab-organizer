"""Integration tests for the FastAPI analyzer service."""

import sys
from unittest.mock import MagicMock, Mock, patch

import pytest
from analyzer import app
from fastapi.testclient import TestClient

# Mock heavy dependencies before importing main
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['torch.cuda'] = MagicMock()
sys.modules['qdrant_client'] = MagicMock()
sys.modules['qdrant_client.models'] = MagicMock()
sys.modules['tiktoken'] = MagicMock()

class TestAnalyzerAPI:
    """Test the FastAPI analyzer service endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test the health check endpoint."""
        with patch('main.embedding_generator') as mock_generator, \
             patch('main.hardware_detector') as mock_detector:
            
            mock_generator.get_current_model_info.return_value = {
                "model_id": "all-minilm",
                "dimensions": 384
            }
            mock_detector.detect_hardware.return_value = {
                "available_ram_gb": 4.0,
                "ram_usage_percent": 50.0,
                "has_gpu": False
            }
            
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["service"] == "analyzer"
            assert "timestamp" in data
            assert "current_model" in data
            assert "hardware" in data
    
    def test_root_endpoint(self, client):
        """Test the root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Content Analyzer Service"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"
        assert "features" in data
        assert len(data["features"]) > 0
    
    def test_hardware_endpoint(self, client):
        """Test the hardware information endpoint."""
        with patch('main.hardware_detector') as mock_detector:
            mock_detector.detect_hardware.return_value = {
                "ram_gb": 8.0,
                "cpu_count": 8,
                "has_gpu": False,
                "gpu_memory_gb": 0.0,
                "gpu_name": "None",
                "available_ram_gb": 4.0,
                "ram_usage_percent": 50.0,
                "cpu_usage_percent": 25.0
            }
            
            response = client.get("/hardware")
            
            assert response.status_code == 200
            data = response.json()
            assert data["ram_gb"] == 8.0
            assert data["cpu_count"] == 8
            assert data["has_gpu"] is False
            assert data["available_ram_gb"] == 4.0
    
    def test_available_models_endpoint(self, client):
        """Test the available models endpoint."""
        with patch('main.model_manager') as mock_manager:
            mock_manager.get_available_models.return_value = {
                "all-minilm": {
                    "name": "All-MiniLM",
                    "dimensions": 384,
                    "quality": "good"
                },
                "nomic-embed-text": {
                    "name": "Nomic Embed Text",
                    "dimensions": 768,
                    "quality": "high"
                }
            }
            
            response = client.get("/models/available")
            
            assert response.status_code == 200
            data = response.json()
            assert "all-minilm" in data
            assert "nomic-embed-text" in data
            assert data["all-minilm"]["dimensions"] == 384
    
    def test_current_model_endpoint(self, client):
        """Test the current model endpoint."""
        with patch('main.embedding_generator') as mock_generator:
            mock_generator.get_current_model_info.return_value = {
                "model_id": "all-minilm",
                "model_name": "All-MiniLM",
                "dimensions": 384,
                "device": "cpu",
                "description": "Lightweight embedding model"
            }
            
            response = client.get("/models/current")
            
            assert response.status_code == 200
            data = response.json()
            assert data["model_id"] == "all-minilm"
            assert data["model_name"] == "All-MiniLM"
            assert data["dimensions"] == 384
            assert data["device"] == "cpu"
    
    def test_model_recommendation_endpoint(self, client):
        """Test the model recommendation endpoint."""
        with patch('main.hardware_detector') as mock_detector, \
             patch('main.model_manager') as mock_manager:
            
            mock_detector.detect_hardware.return_value = {
                "available_ram_gb": 4.0,
                "has_gpu": False
            }
            
            mock_manager.recommend_model.return_value = {
                "recommended_model": "all-minilm",
                "reason": "Best fit for 4.0GB available RAM",
                "alternatives": ["nomic-embed-text"],
                "performance_estimate": {
                    "embeddings_per_sec": 50.0,
                    "dimensions": 384,
                    "suitable_for_batch": True
                }
            }
            
            response = client.get("/models/recommend")
            
            assert response.status_code == 200
            data = response.json()
            assert data["recommended_model"] == "all-minilm"
            assert "4.0GB" in data["reason"]
            assert len(data["alternatives"]) > 0
            assert "performance_estimate" in data
    
    def test_switch_model_endpoint(self, client):
        """Test the model switching endpoint."""
        with patch('main.embedding_generator') as mock_generator:
            mock_generator.switch_model.return_value = True
            mock_generator.get_current_model_info.return_value = {
                "model_id": "nomic-embed-text",
                "model_name": "Nomic Embed Text",
                "dimensions": 768
            }
            
            response = client.post("/models/switch", json={
                "embedding_model": "nomic-embed-text"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "nomic-embed-text" in data["message"]
            assert "current_model" in data
    
    def test_switch_model_failure(self, client):
        """Test model switching failure."""
        with patch('main.embedding_generator') as mock_generator:
            mock_generator.switch_model.return_value = False
            
            response = client.post("/models/switch", json={
                "embedding_model": "nonexistent-model"
            })
            
            assert response.status_code == 400
            assert "Failed to switch" in response.json()["detail"]
    
    def test_generate_embeddings_endpoint(self, client):
        """Test the embedding generation endpoint."""
        with patch('main.embedding_generator') as mock_generator, \
             patch('main.text_chunker') as mock_chunker:
            
            mock_generator.switch_model.return_value = True
            
            response = client.post("/embeddings/generate", json={
                "content_items": [
                    {
                        "id": "test1",
                        "content": "This is test content",
                        "title": "Test Title",
                        "url": "https://example.com",
                        "metadata": {"source": "test"}
                    }
                ],
                "session_id": "test_session",
                "embedding_model": "all-minilm",
                "chunk_size": 512,
                "chunk_overlap": 50
            })
            
            assert response.status_code == 200
            data = response.json()
            assert "job_id" in data
            assert data["status"] == "processing"
            assert "1 items" in data["message"]
    
    def test_embedding_status_endpoint(self, client):
        """Test the embedding status endpoint."""
        response = client.get("/embeddings/status/test-job-id")
        
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "test-job-id"
        assert "status" in data
        assert "message" in data
    
    def test_health_endpoint_error_handling(self, client):
        """Test health endpoint error handling."""
        with patch('main.embedding_generator') as mock_generator, \
             patch('main.hardware_detector') as mock_detector:
            
            # Make the methods raise exceptions
            mock_generator.get_current_model_info.side_effect = Exception("Generator error")
            mock_detector.detect_hardware.side_effect = Exception("Detector error")
            
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unhealthy"
            assert "error" in data
    
    def test_hardware_endpoint_not_initialized(self, client):
        """Test hardware endpoint when detector not initialized."""
        with patch('main.hardware_detector', None):
            response = client.get("/hardware")
            
            assert response.status_code == 503
            assert "Hardware detector not initialized" in response.json()["detail"]
    
    def test_models_endpoint_not_initialized(self, client):
        """Test models endpoint when manager not initialized."""
        with patch('main.model_manager', None):
            response = client.get("/models/available")
            
            assert response.status_code == 503
            assert "Model manager not initialized" in response.json()["detail"]
    
    def test_embedding_generation_not_initialized(self, client):
        """Test embedding generation when generator not initialized."""
        with patch('main.embedding_generator', None):
            response = client.post("/embeddings/generate", json={
                "content_items": [],
                "session_id": "test"
            })
            
            assert response.status_code == 503
            assert "Services not initialized" in response.json()["detail"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
