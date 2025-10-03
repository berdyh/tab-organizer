"""Service registry for managing microservice discovery and routing."""

import time
from typing import Dict, Any, Optional, List

import httpx
import structlog

from config import Settings

logger = structlog.get_logger()


class ServiceRegistry:
    """Manages service discovery and health status for microservices."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.services: Dict[str, Dict[str, Any]] = {}
        self.http_client: Optional[httpx.AsyncClient] = None
        
        # Initialize services from configuration
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize service registry from configuration."""
        # Add internal services
        for service_name, config in self.settings.services.items():
            self.services[service_name] = {
                **config,
                "type": "internal",
                "healthy": False,
                "last_health_check": None,
                "consecutive_failures": 0,
                "registered_at": time.time()
            }
        
        # Add external services
        for service_name, config in self.settings.external_services.items():
            self.services[service_name] = {
                **config,
                "type": "external",
                "healthy": False,
                "last_health_check": None,
                "consecutive_failures": 0,
                "registered_at": time.time()
            }
        
        logger.info("Service registry initialized", service_count=len(self.services))
    
    def get_service(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get service configuration by name."""
        return self.services.get(service_name)
    
    def get_all_services(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered services."""
        return self.services.copy()
    
    def get_healthy_services(self) -> Dict[str, Dict[str, Any]]:
        """Get only healthy services."""
        return {
            name: service for name, service in self.services.items()
            if service.get("healthy", False)
        }
    
    def get_services_by_type(self, service_type: str) -> Dict[str, Dict[str, Any]]:
        """Get services by type (internal/external)."""
        return {
            name: service for name, service in self.services.items()
            if service.get("type") == service_type
        }
    
    def update_service_health(self, service_name: str, is_healthy: bool):
        """Update health status of a service."""
        if service_name not in self.services:
            logger.warning("Attempted to update health for unknown service", service=service_name)
            return
        
        service = self.services[service_name]
        previous_health = service.get("healthy", False)
        
        service["healthy"] = is_healthy
        service["last_health_check"] = time.time()
        
        if is_healthy:
            service["consecutive_failures"] = 0
            if not previous_health:
                logger.info("Service recovered", service=service_name)
        else:
            service["consecutive_failures"] = service.get("consecutive_failures", 0) + 1
            if previous_health:
                logger.warning("Service became unhealthy", service=service_name)
    
    def register_service(self, service_name: str, config: Dict[str, Any]):
        """Register a new service dynamically."""
        self.services[service_name] = {
            **config,
            "type": config.get("type", "internal"),
            "healthy": False,
            "last_health_check": None,
            "consecutive_failures": 0,
            "registered_at": time.time()
        }
        
        logger.info("Service registered", service=service_name, url=config.get("url"))
    
    def unregister_service(self, service_name: str):
        """Unregister a service."""
        if service_name in self.services:
            del self.services[service_name]
            logger.info("Service unregistered", service=service_name)
    
    def get_service_url(self, service_name: str) -> Optional[str]:
        """Get the URL for a service."""
        service = self.get_service(service_name)
        return service.get("url") if service else None
    
    def is_service_healthy(self, service_name: str) -> bool:
        """Check if a service is healthy."""
        service = self.get_service(service_name)
        return service.get("healthy", False) if service else False
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get statistics about registered services."""
        total_services = len(self.services)
        healthy_services = len(self.get_healthy_services())
        internal_services = len(self.get_services_by_type("internal"))
        external_services = len(self.get_services_by_type("external"))
        
        return {
            "total_services": total_services,
            "healthy_services": healthy_services,
            "unhealthy_services": total_services - healthy_services,
            "internal_services": internal_services,
            "external_services": external_services,
            "health_percentage": (healthy_services / total_services * 100) if total_services > 0 else 0
        }
    
    async def close(self):
        """Clean up resources."""
        if self.http_client:
            await self.http_client.aclose()
            logger.info("Service registry HTTP client closed")