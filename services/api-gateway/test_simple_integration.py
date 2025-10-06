"""Simplified integration tests for API Gateway service."""

import pytest
from fastapi.testclient import TestClient

# Import and initialize the app with components
from main import app
import main
from config import Settings
from service_registry import ServiceRegistry
from health_checker import HealthChecker
from rate_limiter import RateLimiter
from auth_middleware import AuthMiddleware


def setup_module():
    """Set up test module with initialized components."""
    # Create test settings
    settings = Settings()
    
    # Initialize global components
    main.service_registry = ServiceRegistry(settings)
    main.health_checker = HealthChecker(main.service_registry)
    main.rate_limiter = RateLimiter()
    main.auth_middleware = AuthMiddleware()


class TestAPIGatewayBasic:
    """Test basic API Gateway functionality."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_root_endpoint(self):
        """Test root endpoint returns correct information."""
        response = self.client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "Web Scraping & Clustering Tool - API Gateway"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"
        assert "endpoints" in data
        assert "features" in data
    
    def test_health_simple(self):
        """Test simple health check."""
        response = self.client.get("/health/simple")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data
    
    def test_metrics_endpoint(self):
        """Test Prometheus metrics endpoint."""
        response = self.client.get("/metrics")
        assert response.status_code == 200
        assert "api_gateway_requests_total" in response.text


class TestAuthentication:
    """Test authentication functionality."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_login_success(self):
        """Test successful login."""
        response = self.client.post("/auth/login", json={
            "username": "admin",
            "password": "admin"
        })
        assert response.status_code == 200
        
        data = response.json()
        assert "token" in data
        assert "expires_in" in data
        assert "permissions" in data
        assert "admin" in data["permissions"]
    
    def test_login_failure(self):
        """Test failed login."""
        response = self.client.post("/auth/login", json={
            "username": "invalid",
            "password": "invalid"
        })
        assert response.status_code == 401
    
    def test_authenticated_endpoint(self):
        """Test accessing authenticated endpoint."""
        # First login
        login_response = self.client.post("/auth/login", json={
            "username": "admin",
            "password": "admin"
        })
        assert login_response.status_code == 200
        token = login_response.json()["token"]
        
        # Access authenticated endpoint
        response = self.client.get("/auth/me", headers={
            "Authorization": f"Bearer {token}"
        })
        assert response.status_code == 200
        
        data = response.json()
        assert data["type"] == "token"
        assert data["user_id"] == "admin"


class TestRateLimiting:
    """Test rate limiting functionality."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_rate_limit_info(self):
        """Test rate limit info endpoint."""
        response = self.client.get("/rate-limit/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "client_id" in data
        assert "client_type" in data
        assert "limits" in data
        assert "current_usage" in data
    
    def test_rate_limit_headers(self):
        """Test rate limit headers are included."""
        response = self.client.get("/rate-limit/info")
        assert response.status_code == 200
        
        # Check for rate limit headers
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers


class TestServiceDiscovery:
    """Test service discovery functionality."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_list_services_with_api_key(self):
        """Test listing services with API key."""
        response = self.client.get("/services", headers={
            "X-API-Key": "internal-service-key"
        })
        assert response.status_code == 200
        
        data = response.json()
        assert "services" in data
        assert "timestamp" in data
    
    def test_list_services_without_auth(self):
        """Test listing services without authentication."""
        response = self.client.get("/services")
        assert response.status_code == 401


class TestModelManagement:
    """Test model management functionality."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_model_config(self):
        """Test getting model configuration."""
        response = self.client.get("/models/config")
        assert response.status_code == 200
        
        data = response.json()
        assert "llm_model" in data
        assert "embedding_model" in data
        assert "ollama_url" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])