"""
Health Monitor - Monitors the health of all containerized services.
Provides comprehensive health checking with Docker health checks integration.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

import httpx

from ..config import MonitoringSettings
from ..logging import get_logger
from ..clients.docker import init_docker_client

logger = get_logger("health_monitor")


class HealthMonitor:
    """Monitors health of all services and containers."""
    
    def __init__(self, settings: MonitoringSettings):
        self.settings = settings
        self.check_interval = settings.health_check_interval
        self.check_timeout = settings.health_check_timeout
        self.check_retries = settings.health_check_retries
        
        self.docker_client = None
        self.health_cache = {}
        self.last_check_time = 0
        self.service_history = {}  # Track health history for trend analysis
        self._refresh_task: Optional[asyncio.Task] = None
        
        # Initialize Docker client
        try:
            self.docker_client = init_docker_client(settings.docker_socket)
            logger.info(
                "Docker client initialized for health monitoring",
                base_url=self.docker_client.api.base_url
            )
        except Exception as e:
            self.docker_client = None
            logger.error("Failed to initialize Docker client", error=str(e))
    
    async def start_monitoring(self):
        """Start the health monitoring loop."""
        logger.info("Starting health monitoring", interval=self.check_interval)
        
        while True:
            try:
                start_time = time.time()
                
                # Perform health checks
                await self._perform_health_checks()
                
                check_duration = time.time() - start_time
                logger.log_performance("health_check_cycle", check_duration)
                
                # Wait for next check interval
                await asyncio.sleep(max(0, self.check_interval - check_duration))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health monitoring error", error=str(e))
                await asyncio.sleep(self.check_interval)
    
    async def _perform_health_checks(self):
        """Perform health checks on all services and containers."""
        check_time = time.time()
        
        # Check service health
        service_health = await self._check_service_health()
        
        # Check container health
        container_health = await self._check_container_health()
        
        # Update health cache
        overall_status = self._calculate_overall_status(service_health, container_health)
        self.health_cache = {
            "timestamp": check_time,
            "services": service_health,
            "containers": container_health,
            "overall_status": overall_status,
            # Keep a dedicated status key so existing consumers that expect
            # "status" continue to work without additional mapping.
            "status": overall_status
        }
        
        self.last_check_time = check_time
        
        # Update service history
        self._update_service_history(service_health)
        
        # Log health summary
        healthy_services = sum(1 for s in service_health.values() if s.get("healthy", False))
        healthy_containers = sum(1 for c in container_health.values() if c.get("healthy", False))
        
        logger.info("Health check completed",
                   healthy_services=healthy_services,
                   total_services=len(service_health),
                   healthy_containers=healthy_containers,
                   total_containers=len(container_health),
                   overall_status=self.health_cache["overall_status"])
    
    async def _check_service_health(self) -> Dict[str, Any]:
        """Check health of all registered services."""
        service_health = {}
        
        async with httpx.AsyncClient(
            timeout=self.check_timeout,
            headers={"User-Agent": "internal-service-monitor"}
        ) as client:
            # Create tasks for concurrent health checks
            tasks = []
            for service_name, service_url in self.settings.services.items():
                task = asyncio.create_task(
                    self._check_single_service_health(client, service_name, service_url)
                )
                tasks.append(task)
            
            # Wait for all health checks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                service_name = list(self.settings.services.keys())[i]
                if isinstance(result, Exception):
                    service_health[service_name] = {
                        "healthy": False,
                        "status": "error",
                        "error": str(result),
                        "last_check": time.time()
                    }
                else:
                    service_health[service_name] = result
        
        return service_health
    
    async def _check_single_service_health(self, client: httpx.AsyncClient, 
                                         service_name: str, service_url: str) -> Dict[str, Any]:
        """Check health of a single service with retries."""
        health_endpoints = ["/health/simple", "/health", "/"]
        
        for attempt in range(self.check_retries):
            for endpoint in health_endpoints:
                try:
                    health_url = f"{service_url}{endpoint}"
                    
                    start_time = time.time()
                    response = await client.get(health_url)
                    response_time = (time.time() - start_time) * 1000  # Convert to ms
                    
                    if response.status_code == 200:
                        # Try to parse JSON response
                        try:
                            health_data = response.json()
                        except:
                            health_data = {"status": "ok"}
                        
                        result = {
                            "healthy": True,
                            "status": "healthy",
                            "response_time_ms": response_time,
                            "endpoint": endpoint,
                            "data": health_data,
                            "last_check": time.time(),
                            "attempt": attempt + 1
                        }
                        
                        logger.log_health_check(service_name, "healthy", response_time)
                        return result
                    
                    elif response.status_code in [404, 405]:
                        # Try next endpoint
                        continue
                    
                    else:
                        # Service responded but with error status
                        result = {
                            "healthy": False,
                            "status": "unhealthy",
                            "response_time_ms": response_time,
                            "endpoint": endpoint,
                            "error": f"HTTP {response.status_code}",
                            "last_check": time.time(),
                            "attempt": attempt + 1
                        }
                        
                        if attempt == self.check_retries - 1:  # Last attempt
                            logger.log_health_check(service_name, "unhealthy", response_time, 
                                                  error=result["error"])
                            return result
                
                except httpx.TimeoutException:
                    if attempt == self.check_retries - 1 and endpoint == health_endpoints[-1]:
                        result = {
                            "healthy": False,
                            "status": "timeout",
                            "error": "Request timeout",
                            "last_check": time.time(),
                            "attempt": attempt + 1
                        }
                        logger.log_health_check(service_name, "timeout")
                        return result
                
                except httpx.ConnectError:
                    if attempt == self.check_retries - 1 and endpoint == health_endpoints[-1]:
                        result = {
                            "healthy": False,
                            "status": "unreachable",
                            "error": "Connection failed",
                            "last_check": time.time(),
                            "attempt": attempt + 1
                        }
                        logger.log_health_check(service_name, "unreachable")
                        return result
                
                except Exception as e:
                    if attempt == self.check_retries - 1 and endpoint == health_endpoints[-1]:
                        result = {
                            "healthy": False,
                            "status": "error",
                            "error": str(e),
                            "last_check": time.time(),
                            "attempt": attempt + 1
                        }
                        logger.log_health_check(service_name, "error", error=str(e))
                        return result
            
            # Wait before retry
            if attempt < self.check_retries - 1:
                await asyncio.sleep(1)
        
        # Should not reach here, but just in case
        return {
            "healthy": False,
            "status": "unknown",
            "error": "All health check attempts failed",
            "last_check": time.time()
        }
    
    async def _check_container_health(self) -> Dict[str, Any]:
        """Check health of all Docker containers."""
        if not self.docker_client:
            return {}
        
        container_health = {}
        
        try:
            containers = await asyncio.to_thread(self.docker_client.containers.list, all=True)
            
            for container in containers:
                try:
                    # Get container status
                    container.reload()
                    
                    # Check Docker health check status
                    health_status = container.attrs.get('State', {}).get('Health', {})
                    docker_health = health_status.get('Status', 'none')
                    
                    # Determine overall health
                    is_running = container.status == 'running'
                    is_healthy = docker_health in ['healthy', 'none'] and is_running
                    
                    # Get resource usage if container is running
                    cpu_percent = 0
                    memory_usage = 0
                    if is_running:
                        try:
                            stats = await asyncio.to_thread(container.stats, stream=False)
                            cpu_percent = self._calculate_cpu_percent(stats)
                            memory_usage = stats['memory_stats'].get('usage', 0)
                        except:
                            pass  # Stats might not be available
                    
                    container_health[container.name] = {
                        "healthy": is_healthy,
                        "status": container.status,
                        "docker_health": docker_health,
                        "image": container.image.tags[0] if container.image.tags else "unknown",
                        "created": container.attrs['Created'],
                        "started": container.attrs['State'].get('StartedAt'),
                        "finished": container.attrs['State'].get('FinishedAt'),
                        "exit_code": container.attrs['State'].get('ExitCode'),
                        "cpu_percent": cpu_percent,
                        "memory_usage": memory_usage,
                        "last_check": time.time()
                    }
                    
                    # Add health check logs if available
                    if health_status.get('Log'):
                        container_health[container.name]["health_logs"] = health_status['Log'][-3:]  # Last 3 entries
                
                except Exception as e:
                    logger.error("Failed to check container health",
                               container_name=container.name,
                               error=str(e))
                    container_health[container.name] = {
                        "healthy": False,
                        "status": "error",
                        "error": str(e),
                        "last_check": time.time()
                    }
            
            logger.info("Container health check completed", container_count=len(container_health))
            return container_health
            
        except Exception as e:
            logger.error("Failed to check container health", error=str(e))
            return {}
    
    def _calculate_cpu_percent(self, stats: Dict[str, Any]) -> float:
        """Calculate CPU percentage from Docker stats."""
        try:
            cpu_stats = stats['cpu_stats']
            precpu_stats = stats['precpu_stats']
            
            cpu_usage = cpu_stats['cpu_usage']['total_usage']
            precpu_usage = precpu_stats['cpu_usage']['total_usage']
            
            system_usage = cpu_stats['system_cpu_usage']
            presystem_usage = precpu_stats['system_cpu_usage']
            
            cpu_count = cpu_stats['online_cpus']
            
            cpu_delta = cpu_usage - precpu_usage
            system_delta = system_usage - presystem_usage
            
            if system_delta > 0 and cpu_delta > 0:
                return (cpu_delta / system_delta) * cpu_count * 100.0
            
            return 0.0
        except (KeyError, ZeroDivisionError):
            return 0.0
    
    def _calculate_overall_status(self, service_health: Dict[str, Any], 
                                container_health: Dict[str, Any]) -> str:
        """Calculate overall system health status."""
        # Count healthy services and containers
        healthy_services = sum(1 for s in service_health.values() if s.get("healthy", False))
        total_services = len(service_health)
        
        healthy_containers = sum(1 for c in container_health.values() if c.get("healthy", False))
        total_containers = len(container_health)
        
        # Calculate health percentages
        service_health_percent = (healthy_services / total_services * 100) if total_services > 0 else 100
        container_health_percent = (healthy_containers / total_containers * 100) if total_containers > 0 else 100
        
        # Determine overall status
        if service_health_percent >= 90 and container_health_percent >= 90:
            return "healthy"
        elif service_health_percent >= 70 and container_health_percent >= 70:
            return "degraded"
        else:
            return "unhealthy"
    
    def _update_service_history(self, service_health: Dict[str, Any]):
        """Update service health history for trend analysis."""
        current_time = time.time()
        
        for service_name, health_data in service_health.items():
            if service_name not in self.service_history:
                self.service_history[service_name] = []
            
            # Add current health status to history
            self.service_history[service_name].append({
                "timestamp": current_time,
                "healthy": health_data.get("healthy", False),
                "status": health_data.get("status", "unknown"),
                "response_time_ms": health_data.get("response_time_ms")
            })
            
            # Keep only last 24 hours of history
            cutoff_time = current_time - (24 * 60 * 60)
            self.service_history[service_name] = [
                entry for entry in self.service_history[service_name]
                if entry["timestamp"] > cutoff_time
            ]
    
    async def get_comprehensive_health(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        current_time = time.time()
        cache_empty = not self.health_cache
        cache_stale = (current_time - self.last_check_time) > self.check_interval

        if cache_empty:
            await self._perform_health_checks()
        elif cache_stale and (not self._refresh_task or self._refresh_task.done()):
            self._refresh_task = asyncio.create_task(self._perform_health_checks())
        
        # Add uptime information
        uptime = current_time - self.last_check_time if self.last_check_time > 0 else 0
        
        result = self.health_cache.copy()
        result["uptime"] = uptime
        result["check_interval"] = self.check_interval
        result.setdefault("status", result.get("overall_status", "unknown"))
        
        return result
    
    async def check_all_services(self) -> Dict[str, Any]:
        """Get health status of all services."""
        health_data = await self.get_comprehensive_health()
        return health_data.get("services", {})
    
    async def check_service_health(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get health status of a specific service."""
        services_health = await self.check_all_services()
        return services_health.get(service_name)
    
    async def get_service_history(self, service_name: str) -> List[Dict[str, Any]]:
        """Get health history for a specific service."""
        return self.service_history.get(service_name, [])
    
    async def get_health_trends(self) -> Dict[str, Any]:
        """Get health trends and statistics."""
        trends = {}
        
        for service_name, history in self.service_history.items():
            if not history:
                continue
            
            # Calculate uptime percentage
            total_checks = len(history)
            healthy_checks = sum(1 for entry in history if entry["healthy"])
            uptime_percent = (healthy_checks / total_checks * 100) if total_checks > 0 else 0
            
            # Calculate average response time
            response_times = [entry["response_time_ms"] for entry in history 
                            if entry["response_time_ms"] is not None]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            
            # Find recent incidents
            incidents = []
            for i, entry in enumerate(history):
                if not entry["healthy"] and (i == 0 or history[i-1]["healthy"]):
                    # Start of an incident
                    incident_start = entry["timestamp"]
                    incident_end = incident_start
                    
                    # Find end of incident
                    for j in range(i+1, len(history)):
                        if history[j]["healthy"]:
                            incident_end = history[j]["timestamp"]
                            break
                        incident_end = history[j]["timestamp"]
                    
                    incidents.append({
                        "start": incident_start,
                        "end": incident_end,
                        "duration_minutes": (incident_end - incident_start) / 60,
                        "status": entry["status"]
                    })
            
            trends[service_name] = {
                "uptime_percent": uptime_percent,
                "avg_response_time_ms": avg_response_time,
                "total_checks": total_checks,
                "healthy_checks": healthy_checks,
                "incidents": incidents[-5:],  # Last 5 incidents
                "last_check": history[-1]["timestamp"] if history else None
            }
        
        return trends
    
    async def reload_config(self):
        """Reload configuration."""
        self.settings = MonitoringSettings()
        self.check_interval = self.settings.health_check_interval
        self.check_timeout = self.settings.health_check_timeout
        self.check_retries = self.settings.health_check_retries
        logger.info("Health monitor configuration reloaded")
    
    async def close(self):
        """Clean up resources."""
        if self.docker_client:
            self.docker_client.close()
        logger.info("Health monitor closed")
