"""
API Gateway Service - Central orchestration layer for the web scraping system.
Handles routing, health checks, service coordination, rate limiting, and authentication.
"""

import os
import time
import asyncio
import hashlib
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
from collections import defaultdict, deque
from datetime import datetime, timedelta

import httpx
import structlog
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from config import Settings
from logging_config import setup_logging
from health_checker import HealthChecker
from service_registry import ServiceRegistry
from rate_limiter import RateLimiter
from auth_middleware import AuthMiddleware

# Initialize logging
setup_logging()
logger = structlog.get_logger()

# Metrics
REQUEST_COUNT = Counter('api_gateway_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('api_gateway_request_duration_seconds', 'Request duration')

# Global variables
health_checker: Optional[HealthChecker] = None
service_registry: Optional[ServiceRegistry] = None
rate_limiter: Optional[RateLimiter] = None
auth_middleware: Optional[AuthMiddleware] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global health_checker, service_registry, rate_limiter, auth_middleware
    
    logger.info("Starting API Gateway...")
    
    # Initialize settings
    settings = Settings()
    
    # Initialize service registry
    service_registry = ServiceRegistry(settings)
    
    # Initialize health checker
    health_checker = HealthChecker(service_registry)
    
    # Initialize rate limiter
    rate_limiter = RateLimiter()
    
    # Initialize authentication middleware
    auth_middleware = AuthMiddleware()
    
    # Start background health monitoring
    health_task = asyncio.create_task(health_checker.start_monitoring())
    
    # Start background cleanup tasks
    cleanup_task = asyncio.create_task(_background_cleanup())
    
    logger.info("API Gateway started successfully")
    
    yield
    
    # Cleanup
    logger.info("Shutting down API Gateway...")
    health_task.cancel()
    cleanup_task.cancel()
    
    try:
        await health_task
        await cleanup_task
    except asyncio.CancelledError:
        pass
    
    if rate_limiter:
        await rate_limiter.close()
    
    await service_registry.close()
    logger.info("API Gateway shutdown complete")


async def _background_cleanup():
    """Background task for periodic cleanup."""
    while True:
        try:
            if auth_middleware:
                auth_middleware.cleanup_expired_tokens()
            await asyncio.sleep(300)  # Cleanup every 5 minutes
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("Background cleanup error", error=str(e))
            await asyncio.sleep(60)


# Create FastAPI app
app = FastAPI(
    title="Web Scraping & Clustering Tool - API Gateway",
    description="Central orchestration layer for web scraping, analysis, and clustering services",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class HealthResponse(BaseModel):
    status: str
    timestamp: float
    services: Dict[str, Dict[str, Any]]
    uptime: float


class ServiceStatus(BaseModel):
    name: str
    status: str
    url: str
    last_check: float
    response_time: Optional[float] = None
    error: Optional[str] = None


class AuthRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    token: str
    expires_in: int
    permissions: List[str]


class RateLimitInfo(BaseModel):
    client_id: str
    client_type: str
    limits: Dict[str, int]
    current_usage: Dict[str, int]
    blocked_until: Optional[float] = None


# Dependencies
def get_settings() -> Settings:
    return Settings()


async def check_rate_limit(request: Request):
    """Rate limiting dependency."""
    if rate_limiter:
        await rate_limiter.check_rate_limit(request)
    return True


async def authenticate(request: Request) -> Dict[str, Any]:
    """Authentication dependency."""
    if auth_middleware:
        auth_info = await auth_middleware.authenticate_request(request)
        await auth_middleware.authorize_request(request, auth_info)
        return auth_info
    return {"type": "disabled", "permissions": ["*"]}


# Middleware for request logging, metrics, and rate limiting
@app.middleware("http")
async def process_request(request, call_next):
    start_time = time.time()
    
    try:
        # Apply rate limiting (except for health checks)
        if not request.url.path.startswith("/health") and rate_limiter:
            await rate_limiter.check_rate_limit(request)
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Update metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        REQUEST_DURATION.observe(duration)
        
        # Add rate limit headers
        if rate_limiter and not request.url.path.startswith("/health"):
            try:
                rate_info = rate_limiter.get_rate_limit_info(request)
                response.headers["X-RateLimit-Limit"] = str(rate_info["limits"]["requests_per_minute"])
                response.headers["X-RateLimit-Remaining"] = str(
                    max(0, rate_info["limits"]["requests_per_minute"] - rate_info["current_usage"]["requests_last_minute"])
                )
                response.headers["X-RateLimit-Reset"] = str(int(time.time() + 60))
            except Exception:
                pass  # Don't fail request if rate limit info fails
        
        # Log request
        logger.info(
            "Request processed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration=duration,
            client_ip=request.client.host if request.client else None
        )
        
        return response
        
    except HTTPException as e:
        # Handle rate limiting and auth errors
        duration = time.time() - start_time
        
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=e.status_code
        ).inc()
        REQUEST_DURATION.observe(duration)
        
        logger.warning(
            "Request rejected",
            method=request.method,
            path=request.url.path,
            status_code=e.status_code,
            detail=e.detail,
            duration=duration,
            client_ip=request.client.host if request.client else None
        )
        
        raise e


# Health check endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check for the API Gateway and all services.
    Returns detailed status information for monitoring and debugging.
    """
    if not health_checker:
        raise HTTPException(status_code=503, detail="Health checker not initialized")
    
    try:
        health_status = await health_checker.get_comprehensive_health()
        
        return HealthResponse(
            status=health_status["status"],
            timestamp=time.time(),
            services=health_status["services"],
            uptime=health_status["uptime"]
        )
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")


@app.get("/health/simple")
async def simple_health_check():
    """Simple health check that returns OK if the gateway is running."""
    return {"status": "ok", "timestamp": time.time()}


@app.get("/health/services")
async def services_health():
    """Get health status of all registered services."""
    if not health_checker:
        raise HTTPException(status_code=503, detail="Health checker not initialized")
    
    try:
        services_status = await health_checker.check_all_services()
        return {"services": services_status, "timestamp": time.time()}
    except Exception as e:
        logger.error("Services health check failed", error=str(e))
        raise HTTPException(status_code=503, detail=f"Services health check failed: {str(e)}")


# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()


# Service discovery endpoints
@app.get("/services")
async def list_services(auth_info: Dict[str, Any] = Depends(authenticate)):
    """List all registered services and their status."""
    if not service_registry:
        raise HTTPException(status_code=503, detail="Service registry not initialized")
    
    services = service_registry.get_all_services()
    return {"services": services, "timestamp": time.time()}


@app.get("/services/{service_name}")
async def get_service_info(service_name: str, auth_info: Dict[str, Any] = Depends(authenticate)):
    """Get detailed information about a specific service."""
    if not service_registry:
        raise HTTPException(status_code=503, detail="Service registry not initialized")
    
    service = service_registry.get_service(service_name)
    if not service:
        raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found")
    
    return {"service": service, "timestamp": time.time()}


# Proxy endpoints for service communication
@app.api_route("/api/{service_name}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_to_service(service_name: str, path: str, request: Request):
    """
    Proxy requests to appropriate microservices.
    Handles routing, load balancing, and error handling.
    """
    if not service_registry:
        raise HTTPException(status_code=503, detail="Service registry not initialized")
    
    # Get service URL
    service = service_registry.get_service(service_name)
    if not service:
        raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found")
    
    if not service.get("healthy", False):
        raise HTTPException(status_code=503, detail=f"Service '{service_name}' is unhealthy")
    
    base_url = service["url"].rstrip("/")
    path_prefix = str(service.get("base_path", "")).strip("/")
    request_path = path.lstrip("/")
    combined_path_parts = [part for part in (path_prefix, request_path) if part]
    combined_path = "/".join(combined_path_parts)
    target_url = f"{base_url}/{combined_path}" if combined_path else base_url
    
    # Get request data
    body = await request.body()
    headers = dict(request.headers)
    
    # Remove hop-by-hop headers
    hop_by_hop_headers = {
        'connection', 'keep-alive', 'proxy-authenticate',
        'proxy-authorization', 'te', 'trailers', 'transfer-encoding', 'upgrade'
    }
    headers = {k: v for k, v in headers.items() if k.lower() not in hop_by_hop_headers}
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(
                method=request.method,
                url=target_url,
                content=body,
                headers=headers,
                params=request.query_params
            )
            
            # Return response
            return JSONResponse(
                content=response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
                status_code=response.status_code,
                headers=dict(response.headers)
            )
            
    except httpx.TimeoutException:
        logger.error("Service request timeout", service=service_name, path=path)
        raise HTTPException(status_code=504, detail=f"Service '{service_name}' request timeout")
    except httpx.ConnectError:
        logger.error("Service connection error", service=service_name, path=path)
        raise HTTPException(status_code=503, detail=f"Cannot connect to service '{service_name}'")
    except Exception as e:
        logger.error("Service proxy error", service=service_name, path=path, error=str(e))
        raise HTTPException(status_code=502, detail=f"Service '{service_name}' error: {str(e)}")


# Model management endpoints
@app.get("/models")
async def list_models():
    """List available and installed Ollama models."""
    if not service_registry:
        raise HTTPException(status_code=503, detail="Service registry not initialized")
    
    ollama_service = service_registry.get_service("ollama")
    if not ollama_service or not ollama_service.get("healthy", False):
        raise HTTPException(status_code=503, detail="Ollama service is not available")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{ollama_service['url']}/api/tags")
            if response.status_code == 200:
                models_data = response.json()
                return {
                    "installed_models": models_data.get("models", []),
                    "timestamp": time.time()
                }
            else:
                raise HTTPException(status_code=502, detail="Failed to fetch models from Ollama")
    except Exception as e:
        logger.error("Failed to list models", error=str(e))
        raise HTTPException(status_code=502, detail=f"Model listing error: {str(e)}")


@app.get("/models/config")
async def get_model_config(settings: Settings = Depends(get_settings)):
    """Get current model configuration."""
    return {
        "llm_model": settings.ollama_model,
        "embedding_model": settings.ollama_embedding_model,
        "ollama_url": settings.ollama_url,
        "timestamp": time.time()
    }


# Authentication endpoints
@app.post("/auth/login", response_model=AuthResponse)
async def login(auth_request: AuthRequest):
    """Authenticate user and return access token."""
    if not auth_middleware:
        raise HTTPException(status_code=503, detail="Authentication service not available")
    
    # Simple authentication (in production, verify against database)
    if auth_request.username == "admin" and auth_request.password == "admin":
        permissions = ["admin", "*"]
    elif auth_request.username == "user" and auth_request.password == "user":
        permissions = ["read", "write", "scrape", "analyze", "cluster", "export"]
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = auth_middleware.generate_token(auth_request.username, permissions, expires_in=3600)
    
    return AuthResponse(
        token=token,
        expires_in=3600,
        permissions=permissions
    )


@app.post("/auth/logout")
async def logout(auth_info: Dict[str, Any] = Depends(authenticate)):
    """Logout user and invalidate token."""
    if not auth_middleware:
        raise HTTPException(status_code=503, detail="Authentication service not available")
    
    if auth_info.get("type") == "token" and "token" in auth_info:
        token_obj = auth_info["token"]
        if token_obj.token in auth_middleware.tokens:
            del auth_middleware.tokens[token_obj.token]
            logger.info("User logged out", user_id=token_obj.user_id)
    
    return {"message": "Logged out successfully"}


@app.get("/auth/me")
async def get_current_user(auth_info: Dict[str, Any] = Depends(authenticate)):
    """Get current user information."""
    return {
        "type": auth_info.get("type"),
        "user_id": auth_info.get("user_id"),
        "permissions": auth_info.get("permissions", []),
        "timestamp": time.time()
    }


# Rate limiting endpoints
@app.get("/rate-limit/info", response_model=RateLimitInfo)
async def get_rate_limit_info(request: Request):
    """Get current rate limit status."""
    if not rate_limiter:
        raise HTTPException(status_code=503, detail="Rate limiter not available")
    
    rate_info = rate_limiter.get_rate_limit_info(request)
    
    return RateLimitInfo(
        client_id=rate_info["client_id"],
        client_type=rate_info["client_type"],
        limits=rate_info["limits"],
        current_usage=rate_info["current_usage"],
        blocked_until=rate_info["blocked_until"]
    )


@app.get("/admin/auth/stats")
async def get_auth_stats(auth_info: Dict[str, Any] = Depends(authenticate)):
    """Get authentication statistics (admin only)."""
    if not auth_middleware:
        raise HTTPException(status_code=503, detail="Authentication service not available")
    
    # Check admin permission
    if not auth_middleware.check_permission(auth_info.get("permissions", []), "admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return auth_middleware.get_auth_stats()


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Web Scraping & Clustering Tool - API Gateway",
        "version": "1.0.0",
        "status": "running",
        "timestamp": time.time(),
        "endpoints": {
            "health": "/health",
            "services": "/services",
            "models": "/models",
            "metrics": "/metrics",
            "auth": "/auth/login",
            "rate_limit": "/rate-limit/info",
            "api_proxy": "/api/{service_name}/{path}"
        },
        "features": {
            "rate_limiting": rate_limiter is not None,
            "authentication": auth_middleware is not None,
            "service_discovery": service_registry is not None,
            "health_monitoring": health_checker is not None
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
