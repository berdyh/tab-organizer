"""
API Gateway Service - Central orchestration layer for the web scraping system.
Handles routing, health checks, and service coordination.
"""

import os
import time
import asyncio
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

import httpx
import structlog
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from config import Settings
from logging_config import setup_logging
from health_checker import HealthChecker
from service_registry import ServiceRegistry

# Initialize logging
setup_logging()
logger = structlog.get_logger()

# Metrics
REQUEST_COUNT = Counter('api_gateway_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('api_gateway_request_duration_seconds', 'Request duration')

# Global variables
health_checker: Optional[HealthChecker] = None
service_registry: Optional[ServiceRegistry] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global health_checker, service_registry
    
    logger.info("Starting API Gateway...")
    
    # Initialize settings
    settings = Settings()
    
    # Initialize service registry
    service_registry = ServiceRegistry(settings)
    
    # Initialize health checker
    health_checker = HealthChecker(service_registry)
    
    # Start background health monitoring
    health_task = asyncio.create_task(health_checker.start_monitoring())
    
    logger.info("API Gateway started successfully")
    
    yield
    
    # Cleanup
    logger.info("Shutting down API Gateway...")
    health_task.cancel()
    try:
        await health_task
    except asyncio.CancelledError:
        pass
    
    await service_registry.close()
    logger.info("API Gateway shutdown complete")


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


# Dependency to get settings
def get_settings() -> Settings:
    return Settings()


# Middleware for request logging and metrics
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    
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
async def list_services():
    """List all registered services and their status."""
    if not service_registry:
        raise HTTPException(status_code=503, detail="Service registry not initialized")
    
    services = service_registry.get_all_services()
    return {"services": services, "timestamp": time.time()}


@app.get("/services/{service_name}")
async def get_service_info(service_name: str):
    """Get detailed information about a specific service."""
    if not service_registry:
        raise HTTPException(status_code=503, detail="Service registry not initialized")
    
    service = service_registry.get_service(service_name)
    if not service:
        raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found")
    
    return {"service": service, "timestamp": time.time()}


# Proxy endpoints for service communication
@app.api_route("/api/{service_name}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_to_service(service_name: str, path: str, request):
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
    
    # Build target URL
    target_url = f"{service['url']}/{path}"
    
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
            "api_proxy": "/api/{service_name}/{path}"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)