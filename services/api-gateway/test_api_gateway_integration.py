"""Integration tests for API Gateway service."""

import asyncio
import time
import pytest
import httpx
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

from main import app, rate_limiter, auth_middleware, service_registry, health_checker
from config import Settings
from service_registry import ServiceRegistry
from health_checker import HealthChecker
from rate_limiter import RateLimiter
from auth_middleware import AuthMiddleware


@pytest.fixture
async def client():
    """Test client fixture with initialized components."""
    # Initialize global components for testing
    import main
    
    # Create test settings
    settings = Settings()
    
    # Initialize components
    main.service_registry = ServiceRegistry(settings)
    main.health_checker = HealthChecker(main.service_registry)
    main.rate_limiter = RateLimiter()
    main.auth_middleware = AuthMiddleware()
    
    # Create test client
    test_client = TestClient(app)
    
    yield test_client
    
    # Cleanup
    if main.rate_limiter:
        await main.rate_limiter.close()
    if main.service_registry:
        await main.service_registry.close()


@pytest.fixture
def mock_settings():
    """Mock settings fixture."""
    settings = Settings()
    settings.services = {
        "test-service": {
            "url": "http://test-service:8080",
            "health_endpoint": "/health",
            "timeout": 10.0
        }
    }
    return settings


@pytest.fixture
async def mock_service_registry(mock_settings):
    """Mock service registry fixture."""
    registry = ServiceRegistry(mock_settings)
    yield registry
    await registry.close()


@pytest.mark.asyncio
class TestAPIGatewayBasic:
    """Test basic API Gateway functionality."""
    
    async def test_root_endpoint(self, client):
        """Test root endpoint returns correct information."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "Web Scraping & Clustering Tool - API Gateway"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"
        assert "endpoints" in data
        assert "features" in data
    
    async def test_health_simple(self, client):
        """Test simple health check."""
        response = client.get("/health/simple")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data
    
    async def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "api_gateway_requests_total" in response.text


@pytest.mark.asyncio
class TestAuthentication:
    """Test authentication functionality."""
    
    async def test_login_success(self, client):
        """Test successful login."""
        response = client.post("/auth/login", json={
            "username": "admin",
            "password": "admin"
        })
        assert response.status_code == 200
        
        data = response.json()
        assert "token" in data
        assert "expires_in" in data
        assert "permissions" in data
        assert "admin" in data["permissions"]
    
    async def test_login_failure(self, client):
        """Test failed login."""
        response = client.post("/auth/login", json={
            "username": "invalid",
            "password": "invalid"
        })
        assert response.status_code == 401
    
    async def test_authenticated_endpoint(self, client):
        """Test accessing authenticated endpoint."""
        # First login
        login_response = client.post("/auth/login", json={
            "username": "admin",
            "password": "admin"
        })
        token = login_response.json()["token"]
        
        # Access authenticated endpoint
        response = client.get("/auth/me", headers={
            "Authorization": f"Bearer {token}"
        })
        assert response.status_code == 200
        
        data = response.json()
        assert data["type"] == "token"
        assert data["user_id"] == "admin"
    
    async def test_unauthorized_access(self, client):
        """Test unauthorized access to protected endpoint."""
        response = client.get("/admin/auth/stats")
        assert response.status_code == 401
    
    async def test_logout(self, client):
        """Test logout functionality."""
        # First login
        login_response = client.post("/auth/login", json={
            "username": "admin",
            "password": "admin"
        })
        token = login_response.json()["token"]
        
        # Logout
        response = client.post("/auth/logout", headers={
            "Authorization": f"Bearer {token}"
        })
        assert response.status_code == 200
        
        # Try to use token after logout
        response = client.get("/auth/me", headers={
            "Authorization": f"Bearer {token}"
        })
        assert response.status_code == 401


@pytest.mark.asyncio
class TestRateLimiting:
    """Test rate limiting functionality."""
    
    async def test_rate_limit_info(self, client):
        """Test rate limit info endpoint."""
        response = client.get("/rate-limit/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "client_id" in data
        assert "client_type" in data
        assert "limits" in data
        assert "current_usage" in data
    
    async def test_rate_limiting_enforcement(self, client):
        """Test that rate limiting is enforced."""
        # Make many requests quickly to trigger rate limiting
        # Use a different endpoint to avoid caching issues
        responses = []
        for i in range(70):  # Exceed default limit of 60/minute
            response = client.get("/rate-limit/info")  # Use an endpoint that should be rate limited
            responses.append(response.status_code)
        
        # Should have some 429 responses
        assert 429 in responses
    
    async def test_rate_limit_headers(self, client):
        """Test rate limit headers are included."""
        response = client.get("/rate-limit/info")  # Use endpoint that gets rate limited
        assert response.status_code == 200
        
        # Check for rate limit headers
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers


@pytest.mark.asyncio
class TestServiceDiscovery:
    """Test service discovery functionality."""
    
    @patch('main.service_registry')
    async def test_list_services(self, mock_registry, client):
        """Test listing services."""
        mock_registry.get_all_services.return_value = {
            "test-service": {
                "url": "http://test-service:8080",
                "healthy": True,
                "type": "internal"
            }
        }
        
        response = client.get("/services", headers={
            "X-API-Key": "internal-service-key"
        })
        assert response.status_code == 200
        
        data = response.json()
        assert "services" in data
        assert "timestamp" in data
    
    @patch('main.service_registry')
    async def test_get_service_info(self, mock_registry, client):
        """Test getting specific service info."""
        mock_registry.get_service.return_value = {
            "url": "http://test-service:8080",
            "healthy": True,
            "type": "internal"
        }
        
        response = client.get("/services/test-service", headers={
            "X-API-Key": "internal-service-key"
        })
        assert response.status_code == 200
        
        data = response.json()
        assert "service" in data
    
    @patch('main.service_registry')
    async def test_service_not_found(self, mock_registry, client):
        """Test getting info for non-existent service."""
        mock_registry.get_service.return_value = None
        
        response = client.get("/services/nonexistent", headers={
            "X-API-Key": "internal-service-key"
        })
        assert response.status_code == 404


@pytest.mark.asyncio
class TestServiceProxy:
    """Test service proxy functionality."""
    
    @patch('main.service_registry')
    @patch('httpx.AsyncClient')
    async def test_proxy_success(self, mock_client_class, mock_registry, client):
        """Test successful service proxy."""
        # Mock service registry
        mock_registry.get_service.return_value = {
            "url": "http://test-service:8080",
            "healthy": True
        }
        
        # Mock HTTP client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}
        mock_response.headers = {"content-type": "application/json"}
        
        mock_client = AsyncMock()
        mock_client.request.return_value = mock_response
        mock_client_class.return_value.__aenter__.return_value = mock_client
        
        # Make request
        response = client.get("/api/test-service/test-endpoint")
        assert response.status_code == 200
    
    @patch('main.service_registry')
    async def test_proxy_service_not_found(self, mock_registry, client):
        """Test proxy to non-existent service."""
        mock_registry.get_service.return_value = None
        
        response = client.get("/api/nonexistent/test")
        assert response.status_code == 404
    
    @patch('main.service_registry')
    async def test_proxy_unhealthy_service(self, mock_registry, client):
        """Test proxy to unhealthy service."""
        mock_registry.get_service.return_value = {
            "url": "http://test-service:8080",
            "healthy": False
        }
        
        response = client.get("/api/test-service/test")
        assert response.status_code == 503


@pytest.mark.asyncio
class TestHealthMonitoring:
    """Test health monitoring functionality."""
    
    @patch('main.health_checker')
    async def test_comprehensive_health(self, mock_health_checker, client):
        """Test comprehensive health check."""
        # Create an async mock
        async_mock = AsyncMock(return_value={
            "status": "healthy",
            "uptime": 3600,
            "services": {
                "test-service": {
                    "status": "healthy",
                    "response_time": 0.1
                }
            }
        })
        mock_health_checker.get_comprehensive_health = async_mock
        
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "services" in data
        assert "uptime" in data
    
    @patch('main.health_checker')
    async def test_services_health(self, mock_health_checker, client):
        """Test services health endpoint."""
        # Create an async mock
        async_mock = AsyncMock(return_value={
            "test-service": {
                "status": "healthy",
                "response_time": 0.1
            }
        })
        mock_health_checker.check_all_services = async_mock
        
        response = client.get("/health/services")
        assert response.status_code == 200
        
        data = response.json()
        assert "services" in data
        assert "timestamp" in data


@pytest.mark.asyncio
class TestModelManagement:
    """Test model management functionality."""
    
    @patch('main.service_registry')
    @patch('httpx.AsyncClient')
    async def test_list_models(self, mock_client_class, mock_registry, client):
        """Test listing Ollama models."""
        # Mock service registry
        mock_registry.get_service.return_value = {
            "url": "http://ollama:11434",
            "healthy": True
        }
        
        # Mock HTTP client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.2:3b", "size": 2000000000}
            ]
        }
        
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__aenter__.return_value = mock_client
        
        # Make request
        response = client.get("/models")
        assert response.status_code == 200
        
        data = response.json()
        assert "installed_models" in data
    
    async def test_model_config(self, client):
        """Test getting model configuration."""
        response = client.get("/models/config")
        assert response.status_code == 200
        
        data = response.json()
        assert "llm_model" in data
        assert "embedding_model" in data
        assert "ollama_url" in data


@pytest.mark.asyncio
class TestAsyncComponents:
    """Test async components of the API Gateway."""
    
    async def test_rate_limiter_initialization(self):
        """Test rate limiter initialization."""
        rate_limiter = RateLimiter()
        assert rate_limiter is not None
        assert rate_limiter.buckets is not None
        assert rate_limiter.rules is not None
        
        await rate_limiter.close()
    
    async def test_auth_middleware_initialization(self):
        """Test auth middleware initialization."""
        auth_middleware = AuthMiddleware()
        assert auth_middleware is not None
        assert auth_middleware.tokens is not None
        assert auth_middleware.api_keys is not None
        assert auth_middleware.permissions is not None
    
    async def test_service_registry_initialization(self, mock_settings):
        """Test service registry initialization."""
        registry = ServiceRegistry(mock_settings)
        assert registry is not None
        assert registry.services is not None
        
        await registry.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])