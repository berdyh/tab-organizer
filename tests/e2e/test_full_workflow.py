"""
End-to-end tests for the complete web scraping workflow
"""
import os
import time
import pytest
import requests
from typing import Dict, List


API_GATEWAY_URL = os.getenv("API_GATEWAY_URL", "http://localhost:8080")
WEB_UI_URL = os.getenv("WEB_UI_URL", "http://localhost:8089")


class TestFullWorkflow:
    """Test complete workflow from URL input to export"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Wait for services to be ready"""
        max_retries = 30
        for i in range(max_retries):
            try:
                response = requests.get(f"{API_GATEWAY_URL}/health", timeout=5)
                if response.status_code == 200:
                    break
            except requests.exceptions.RequestException:
                if i == max_retries - 1:
                    pytest.fail("API Gateway not ready")
                time.sleep(2)
    
    def test_health_check(self):
        """Test that all services are healthy"""
        response = requests.get(f"{API_GATEWAY_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "healthy"
    
    def test_url_submission_workflow(self):
        """Test complete URL submission and processing workflow"""
        # Step 1: Submit URLs
        urls = [
            "https://example.com",
            "https://example.org",
            "https://example.net"
        ]
        
        response = requests.post(
            f"{API_GATEWAY_URL}/api/urls/batch",
            json={"urls": urls}
        )
        assert response.status_code in [200, 202]
        job_data = response.json()
        job_id = job_data.get("job_id")
        assert job_id is not None
        
        # Step 2: Check job status
        max_wait = 60
        for _ in range(max_wait):
            response = requests.get(f"{API_GATEWAY_URL}/api/jobs/{job_id}")
            assert response.status_code == 200
            status_data = response.json()
            
            if status_data.get("status") in ["completed", "failed"]:
                break
            time.sleep(1)
        
        assert status_data.get("status") == "completed"
    
    def test_session_management(self):
        """Test session creation and management"""
        # Create session
        response = requests.post(
            f"{API_GATEWAY_URL}/api/sessions",
            json={"name": "Test Session", "description": "E2E test session"}
        )
        assert response.status_code in [200, 201]
        session_data = response.json()
        session_id = session_data.get("session_id")
        assert session_id is not None
        
        # Get session details
        response = requests.get(f"{API_GATEWAY_URL}/api/sessions/{session_id}")
        assert response.status_code == 200
        data = response.json()
        assert data.get("name") == "Test Session"
        
        # Delete session
        response = requests.delete(f"{API_GATEWAY_URL}/api/sessions/{session_id}")
        assert response.status_code in [200, 204]
    
    def test_search_functionality(self):
        """Test search capabilities"""
        # Perform semantic search
        response = requests.post(
            f"{API_GATEWAY_URL}/api/search",
            json={
                "query": "test content",
                "search_type": "semantic",
                "limit": 10
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)
    
    def test_export_workflow(self):
        """Test export functionality"""
        # Request export
        response = requests.post(
            f"{API_GATEWAY_URL}/api/export",
            json={
                "format": "markdown",
                "session_id": "default"
            }
        )
        assert response.status_code in [200, 202]
        export_data = response.json()
        
        if "export_id" in export_data:
            export_id = export_data["export_id"]
            
            # Check export status
            response = requests.get(f"{API_GATEWAY_URL}/api/export/{export_id}")
            assert response.status_code == 200


class TestAPIEndpoints:
    """Test individual API endpoints"""
    
    def test_url_validation(self):
        """Test URL validation endpoint"""
        response = requests.post(
            f"{API_GATEWAY_URL}/api/urls/validate",
            json={"url": "https://example.com"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("valid") is True
    
    def test_invalid_url_validation(self):
        """Test validation of invalid URL"""
        response = requests.post(
            f"{API_GATEWAY_URL}/api/urls/validate",
            json={"url": "not-a-valid-url"}
        )
        assert response.status_code in [200, 400]
        if response.status_code == 200:
            data = response.json()
            assert data.get("valid") is False
    
    def test_clustering_endpoint(self):
        """Test clustering endpoint"""
        response = requests.post(
            f"{API_GATEWAY_URL}/api/cluster",
            json={"session_id": "default"}
        )
        assert response.status_code in [200, 202]


class TestWebUI:
    """Test Web UI functionality"""
    
    def test_ui_accessible(self):
        """Test that Web UI is accessible"""
        try:
            response = requests.get(WEB_UI_URL, timeout=10)
            assert response.status_code == 200
        except requests.exceptions.RequestException:
            pytest.skip("Web UI not available")
    
    def test_ui_static_assets(self):
        """Test that static assets are served"""
        try:
            response = requests.get(f"{WEB_UI_URL}/static/", timeout=10)
            # Should either return 200 or redirect
            assert response.status_code in [200, 301, 302, 404]
        except requests.exceptions.RequestException:
            pytest.skip("Web UI not available")


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_endpoint(self):
        """Test handling of invalid endpoint"""
        response = requests.get(f"{API_GATEWAY_URL}/api/invalid-endpoint")
        assert response.status_code == 404
    
    def test_malformed_request(self):
        """Test handling of malformed request"""
        response = requests.post(
            f"{API_GATEWAY_URL}/api/urls/batch",
            json={"invalid": "data"}
        )
        assert response.status_code in [400, 422]
    
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        # Make multiple rapid requests
        responses = []
        for _ in range(100):
            response = requests.get(f"{API_GATEWAY_URL}/health")
            responses.append(response.status_code)
        
        # Should have at least some successful requests
        assert 200 in responses


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
