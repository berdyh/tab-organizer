"""Health checking system for monitoring service availability."""

import time
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

import httpx
import structlog

logger = structlog.get_logger()


class HealthChecker:
    """Monitors health of all services and external dependencies."""
    
    def __init__(self, service_registry):
        self.service_registry = service_registry
        self.start_time = time.time()
        self.health_cache: Dict[str, Dict[str, Any]] = {}
        self.monitoring_task: Optional[asyncio.Task] = None
        
    async def start_monitoring(self):
        """Start background health monitoring."""
        logger.info("Starting health monitoring")
        
        while True:
            try:
                await self.check_all_services()
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                logger.info("Health monitoring cancelled")
                break
            except Exception as e:
                logger.error("Health monitoring error", error=str(e))
                await asyncio.sleep(5)  # Short delay on error
    
    async def check_service_health(self, service_name: str, service_config: Dict[str, Any]) -> Dict[str, Any]:
        """Check health of a single service."""
        start_time = time.time()
        
        try:
            url = service_config["url"]
            health_endpoint = service_config.get("health_endpoint", "/health")
            timeout = service_config.get("timeout", 10.0)
            
            health_url = f"{url}{health_endpoint}"
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(health_url)
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    status = "healthy"
                    error = None
                else:
                    status = "unhealthy"
                    error = f"HTTP {response.status_code}"
                    
        except httpx.TimeoutException:
            response_time = time.time() - start_time
            status = "timeout"
            error = "Request timeout"
        except httpx.ConnectError:
            response_time = time.time() - start_time
            status = "unreachable"
            error = "Connection failed"
        except Exception as e:
            response_time = time.time() - start_time
            status = "error"
            error = str(e)
        
        health_info = {
            "status": status,
            "response_time": response_time,
            "last_check": time.time(),
            "error": error,
            "url": service_config["url"]
        }
        
        # Update cache
        self.health_cache[service_name] = health_info
        
        # Update service registry
        self.service_registry.update_service_health(service_name, status == "healthy")
        
        logger.debug(
            "Service health check completed",
            service=service_name,
            status=status,
            response_time=response_time,
            error=error
        )
        
        return health_info
    
    async def check_all_services(self) -> Dict[str, Dict[str, Any]]:
        """Check health of all registered services."""
        all_services = {
            **self.service_registry.settings.services,
            **self.service_registry.settings.external_services
        }
        
        # Create tasks for concurrent health checks
        tasks = []
        for service_name, service_config in all_services.items():
            task = asyncio.create_task(
                self.check_service_health(service_name, service_config),
                name=f"health_check_{service_name}"
            )
            tasks.append((service_name, task))
        
        # Wait for all health checks to complete
        results = {}
        for service_name, task in tasks:
            try:
                results[service_name] = await task
            except Exception as e:
                logger.error("Health check task failed", service=service_name, error=str(e))
                results[service_name] = {
                    "status": "error",
                    "response_time": None,
                    "last_check": time.time(),
                    "error": str(e),
                    "url": all_services[service_name]["url"]
                }
        
        return results
    
    async def get_comprehensive_health(self) -> Dict[str, Any]:
        """Get comprehensive health status including system metrics."""
        services_health = await self.check_all_services()
        
        # Calculate overall status
        healthy_services = sum(1 for s in services_health.values() if s["status"] == "healthy")
        total_services = len(services_health)
        
        if healthy_services == total_services:
            overall_status = "healthy"
        elif healthy_services > total_services // 2:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
        
        return {
            "status": overall_status,
            "uptime": time.time() - self.start_time,
            "services": services_health,
            "summary": {
                "total_services": total_services,
                "healthy_services": healthy_services,
                "unhealthy_services": total_services - healthy_services
            },
            "timestamp": time.time()
        }
    
    def get_cached_health(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get cached health information for a service."""
        return self.health_cache.get(service_name)