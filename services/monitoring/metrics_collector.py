"""
Metrics Collector - Collects performance metrics from all containerized services.
Provides comprehensive metrics collection including system metrics, container metrics,
and application-specific metrics.
"""

import asyncio
import time
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

import httpx
import docker
import psutil
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram

from config import MonitoringSettings
from logging_config import get_logger

logger = get_logger("metrics_collector")


class MetricsCollector:
    """Collects and aggregates metrics from all services and containers."""
    
    def __init__(self, settings: MonitoringSettings):
        self.settings = settings
        self.collection_interval = settings.collection_interval
        self.docker_client = None
        self.metrics_cache = {}
        self.last_collection_time = 0
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
            logger.info("Docker client initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize Docker client", error=str(e))
    
    async def start_collection(self):
        """Start the metrics collection loop."""
        logger.info("Starting metrics collection", interval=self.collection_interval)
        
        while True:
            try:
                start_time = time.time()
                
                # Collect all metrics
                await self._collect_all_metrics()
                
                collection_duration = time.time() - start_time
                logger.log_performance("metrics_collection", collection_duration)
                
                # Wait for next collection interval
                await asyncio.sleep(max(0, self.collection_interval - collection_duration))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Metrics collection error", error=str(e))
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_all_metrics(self):
        """Collect all types of metrics."""
        collection_time = time.time()
        
        # Collect system metrics
        system_metrics = await self._collect_system_metrics()
        
        # Collect container metrics
        container_metrics = await self._collect_container_metrics()
        
        # Collect service metrics
        service_metrics = await self._collect_service_metrics()
        
        # Store in cache
        self.metrics_cache = {
            "timestamp": collection_time,
            "system": system_metrics,
            "containers": container_metrics,
            "services": service_metrics
        }
        
        self.last_collection_time = collection_time
        
        logger.info("Metrics collection completed",
                   system_metrics_count=len(system_metrics),
                   container_count=len(container_metrics),
                   service_count=len(service_metrics))
    
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-level metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_count_logical = psutil.cpu_count(logical=True)
            cpu_freq = psutil.cpu_freq()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            # Network metrics
            network_io = psutil.net_io_counters()
            network_connections = len(psutil.net_connections())
            
            # Process metrics
            process_count = len(psutil.pids())
            
            # Load average (Unix only)
            load_avg = None
            try:
                load_avg = psutil.getloadavg()
            except AttributeError:
                pass  # Windows doesn't have load average
            
            return {
                "cpu": {
                    "percent": cpu_percent,
                    "count_physical": cpu_count,
                    "count_logical": cpu_count_logical,
                    "frequency_mhz": cpu_freq.current if cpu_freq else None,
                    "load_average": load_avg
                },
                "memory": {
                    "total_bytes": memory.total,
                    "available_bytes": memory.available,
                    "used_bytes": memory.used,
                    "free_bytes": memory.free,
                    "percent": memory.percent,
                    "cached_bytes": getattr(memory, 'cached', 0),
                    "buffers_bytes": getattr(memory, 'buffers', 0)
                },
                "swap": {
                    "total_bytes": swap.total,
                    "used_bytes": swap.used,
                    "free_bytes": swap.free,
                    "percent": swap.percent
                },
                "disk": {
                    "total_bytes": disk_usage.total,
                    "used_bytes": disk_usage.used,
                    "free_bytes": disk_usage.free,
                    "percent": (disk_usage.used / disk_usage.total) * 100,
                    "read_bytes": disk_io.read_bytes if disk_io else 0,
                    "write_bytes": disk_io.write_bytes if disk_io else 0,
                    "read_count": disk_io.read_count if disk_io else 0,
                    "write_count": disk_io.write_count if disk_io else 0
                },
                "network": {
                    "bytes_sent": network_io.bytes_sent,
                    "bytes_recv": network_io.bytes_recv,
                    "packets_sent": network_io.packets_sent,
                    "packets_recv": network_io.packets_recv,
                    "errors_in": network_io.errin,
                    "errors_out": network_io.errout,
                    "drops_in": network_io.dropin,
                    "drops_out": network_io.dropout,
                    "connections": network_connections
                },
                "processes": {
                    "count": process_count
                }
            }
        except Exception as e:
            logger.error("Failed to collect system metrics", error=str(e))
            return {}
    
    async def collect_container_metrics(self) -> Dict[str, Any]:
        """Collect Docker container metrics."""
        return await self._collect_container_metrics()
    
    async def _collect_container_metrics(self) -> Dict[str, Any]:
        """Collect Docker container metrics."""
        if not self.docker_client:
            return {}
        
        container_metrics = {}
        
        try:
            containers = self.docker_client.containers.list()
            
            for container in containers:
                try:
                    # Get container stats
                    stats = container.stats(stream=False)
                    
                    # Calculate CPU percentage
                    cpu_percent = self._calculate_cpu_percent(stats)
                    
                    # Calculate memory usage
                    memory_usage = stats['memory_stats'].get('usage', 0)
                    memory_limit = stats['memory_stats'].get('limit', 0)
                    memory_percent = (memory_usage / memory_limit * 100) if memory_limit > 0 else 0
                    
                    # Network I/O
                    network_rx = 0
                    network_tx = 0
                    if 'networks' in stats:
                        for interface in stats['networks'].values():
                            network_rx += interface.get('rx_bytes', 0)
                            network_tx += interface.get('tx_bytes', 0)
                    
                    # Block I/O
                    block_read = 0
                    block_write = 0
                    if 'blkio_stats' in stats and 'io_service_bytes_recursive' in stats['blkio_stats']:
                        for entry in stats['blkio_stats']['io_service_bytes_recursive']:
                            if entry['op'] == 'Read':
                                block_read += entry['value']
                            elif entry['op'] == 'Write':
                                block_write += entry['value']
                    
                    container_metrics[container.name] = {
                        "id": container.id[:12],
                        "name": container.name,
                        "status": container.status,
                        "image": container.image.tags[0] if container.image.tags else "unknown",
                        "cpu_percent": cpu_percent,
                        "memory_usage": memory_usage,
                        "memory_limit": memory_limit,
                        "memory_percent": memory_percent,
                        "network_rx_bytes": network_rx,
                        "network_tx_bytes": network_tx,
                        "block_read_bytes": block_read,
                        "block_write_bytes": block_write,
                        "created": container.attrs['Created'],
                        "started": container.attrs['State']['StartedAt']
                    }
                    
                except Exception as e:
                    logger.error("Failed to collect metrics for container",
                               container_name=container.name,
                               error=str(e))
                    continue
            
            logger.info("Container metrics collected", container_count=len(container_metrics))
            return container_metrics
            
        except Exception as e:
            logger.error("Failed to collect container metrics", error=str(e))
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
    
    async def _collect_service_metrics(self) -> Dict[str, Any]:
        """Collect metrics from all registered services."""
        service_metrics = {}
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for service_name, service_url in self.settings.services.items():
                try:
                    # Try to get metrics from service
                    metrics_url = f"{service_url}/metrics"
                    
                    start_time = time.time()
                    response = await client.get(metrics_url)
                    response_time = (time.time() - start_time) * 1000  # Convert to ms
                    
                    if response.status_code == 200:
                        # Parse Prometheus metrics if available
                        metrics_text = response.text
                        parsed_metrics = self._parse_prometheus_metrics(metrics_text)
                        
                        service_metrics[service_name] = {
                            "status": "healthy",
                            "response_time_ms": response_time,
                            "metrics": parsed_metrics,
                            "last_check": time.time()
                        }
                    else:
                        service_metrics[service_name] = {
                            "status": "unhealthy",
                            "response_time_ms": response_time,
                            "error": f"HTTP {response.status_code}",
                            "last_check": time.time()
                        }
                
                except httpx.TimeoutException:
                    service_metrics[service_name] = {
                        "status": "timeout",
                        "error": "Request timeout",
                        "last_check": time.time()
                    }
                except httpx.ConnectError:
                    service_metrics[service_name] = {
                        "status": "unreachable",
                        "error": "Connection failed",
                        "last_check": time.time()
                    }
                except Exception as e:
                    service_metrics[service_name] = {
                        "status": "error",
                        "error": str(e),
                        "last_check": time.time()
                    }
        
        logger.info("Service metrics collected", service_count=len(service_metrics))
        return service_metrics
    
    def _parse_prometheus_metrics(self, metrics_text: str) -> Dict[str, Any]:
        """Parse Prometheus metrics format."""
        parsed_metrics = {}
        
        try:
            lines = metrics_text.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Simple parsing - just extract metric name and value
                if ' ' in line:
                    parts = line.split(' ')
                    if len(parts) >= 2:
                        metric_name = parts[0]
                        try:
                            metric_value = float(parts[1])
                            parsed_metrics[metric_name] = metric_value
                        except ValueError:
                            continue
            
        except Exception as e:
            logger.error("Failed to parse Prometheus metrics", error=str(e))
        
        return parsed_metrics
    
    async def collect_all_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        if not self.metrics_cache or (time.time() - self.last_collection_time) > self.collection_interval:
            await self._collect_all_metrics()
        
        return self.metrics_cache.copy()
    
    async def get_service_metrics(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific service."""
        metrics = await self.collect_all_metrics()
        return metrics.get("services", {}).get(service_name)
    
    async def get_container_metrics(self, container_name: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific container."""
        metrics = await self.collect_all_metrics()
        return metrics.get("containers", {}).get(container_name)
    
    async def reload_config(self):
        """Reload configuration."""
        self.settings = MonitoringSettings()
        self.collection_interval = self.settings.collection_interval
        logger.info("Metrics collector configuration reloaded")
    
    async def close(self):
        """Clean up resources."""
        if self.docker_client:
            self.docker_client.close()
        logger.info("Metrics collector closed")