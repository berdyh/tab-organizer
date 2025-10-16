"""
Performance Tracker - Tracks and analyzes performance metrics with optimization recommendations.
Provides performance benchmarks and resource usage monitoring for containers and AI models.
"""

import asyncio
import time
import statistics
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

import httpx
import psutil

from ..config import MonitoringSettings
from ..logging import get_logger

logger = get_logger("performance_tracker")


class PerformanceTracker:
    """Tracks performance metrics and provides optimization recommendations."""
    
    def __init__(self, settings: MonitoringSettings):
        self.settings = settings
        self.tracking_enabled = settings.performance_tracking_enabled
        self.benchmark_interval = settings.benchmark_interval_minutes * 60
        
        self.performance_data = {}
        self.benchmark_results = {}
        self.last_benchmark_time = 0
        self.performance_history = {}
    
    async def start_tracking(self):
        """Start performance tracking loop."""
        if not self.tracking_enabled:
            logger.info("Performance tracking disabled")
            return
        
        logger.info("Starting performance tracking", 
                   benchmark_interval_minutes=self.settings.benchmark_interval_minutes)
        
        while True:
            try:
                # Run performance analysis
                await self._analyze_performance()
                
                # Run benchmarks if due
                if (time.time() - self.last_benchmark_time) >= self.benchmark_interval:
                    await self._run_performance_benchmarks()
                    self.last_benchmark_time = time.time()
                
                # Wait before next analysis
                await asyncio.sleep(60)  # Analyze every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Performance tracking error", error=str(e))
                await asyncio.sleep(60)
    
    async def _analyze_performance(self):
        """Analyze current performance metrics."""
        analysis_time = time.time()
        
        # Collect system performance
        system_perf = await self._collect_system_performance()
        
        # Collect service performance
        service_perf = await self._collect_service_performance()
        
        # Store performance data
        self.performance_data = {
            "timestamp": analysis_time,
            "system": system_perf,
            "services": service_perf
        }
        
        # Update performance history
        self._update_performance_history(system_perf, service_perf)
        
        logger.log_performance("performance_analysis", time.time() - analysis_time)
    
    async def _collect_system_performance(self) -> Dict[str, Any]:
        """Collect system performance metrics."""
        try:
            # CPU performance
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_freq = psutil.cpu_freq()
            load_avg = None
            try:
                load_avg = psutil.getloadavg()
            except AttributeError:
                pass
            
            # Memory performance
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk performance
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            # Network performance
            network_io = psutil.net_io_counters()
            
            # Calculate performance scores
            cpu_score = max(0, 100 - cpu_percent)  # Lower CPU usage = higher score
            memory_score = max(0, 100 - memory.percent)
            disk_score = max(0, 100 - (disk_usage.used / disk_usage.total * 100))
            
            overall_score = (cpu_score + memory_score + disk_score) / 3
            
            return {
                "cpu": {
                    "percent": cpu_percent,
                    "frequency_mhz": cpu_freq.current if cpu_freq else None,
                    "load_average": load_avg,
                    "score": cpu_score
                },
                "memory": {
                    "percent": memory.percent,
                    "available_gb": memory.available / (1024**3),
                    "score": memory_score
                },
                "disk": {
                    "percent": (disk_usage.used / disk_usage.total * 100),
                    "free_gb": disk_usage.free / (1024**3),
                    "io_read_mb_s": (disk_io.read_bytes / (1024**2)) if disk_io else 0,
                    "io_write_mb_s": (disk_io.write_bytes / (1024**2)) if disk_io else 0,
                    "score": disk_score
                },
                "network": {
                    "bytes_sent_mb": network_io.bytes_sent / (1024**2),
                    "bytes_recv_mb": network_io.bytes_recv / (1024**2),
                    "packets_sent": network_io.packets_sent,
                    "packets_recv": network_io.packets_recv
                },
                "overall_score": overall_score,
                "performance_grade": self._calculate_performance_grade(overall_score)
            }
        
        except Exception as e:
            logger.error("Failed to collect system performance", error=str(e))
            return {}
    
    async def _collect_service_performance(self) -> Dict[str, Any]:
        """Collect service performance metrics."""
        service_performance = {}
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for service_name, service_url in self.settings.services.items():
                try:
                    # Measure response time
                    start_time = time.time()
                    
                    # Try health endpoint first
                    health_url = f"{service_url}/health"
                    response = await client.get(health_url)
                    
                    response_time = (time.time() - start_time) * 1000  # Convert to ms
                    
                    # Calculate performance metrics
                    performance_score = self._calculate_service_performance_score(
                        response_time, response.status_code
                    )
                    
                    service_performance[service_name] = {
                        "response_time_ms": response_time,
                        "status_code": response.status_code,
                        "available": response.status_code == 200,
                        "performance_score": performance_score,
                        "performance_grade": self._calculate_performance_grade(performance_score),
                        "last_check": time.time()
                    }
                    
                    # Try to get service-specific metrics
                    try:
                        metrics_url = f"{service_url}/metrics"
                        metrics_response = await client.get(metrics_url)
                        if metrics_response.status_code == 200:
                            service_performance[service_name]["has_metrics"] = True
                    except:
                        service_performance[service_name]["has_metrics"] = False
                
                except httpx.TimeoutException:
                    service_performance[service_name] = {
                        "response_time_ms": None,
                        "status_code": None,
                        "available": False,
                        "performance_score": 0,
                        "performance_grade": "F",
                        "error": "timeout",
                        "last_check": time.time()
                    }
                
                except Exception as e:
                    service_performance[service_name] = {
                        "response_time_ms": None,
                        "status_code": None,
                        "available": False,
                        "performance_score": 0,
                        "performance_grade": "F",
                        "error": str(e),
                        "last_check": time.time()
                    }
        
        return service_performance
    
    def _calculate_service_performance_score(self, response_time_ms: float, status_code: int) -> float:
        """Calculate performance score for a service."""
        if status_code != 200:
            return 0
        
        # Score based on response time
        if response_time_ms <= 100:
            return 100
        elif response_time_ms <= 500:
            return 90 - ((response_time_ms - 100) / 400 * 40)  # 90-50
        elif response_time_ms <= 1000:
            return 50 - ((response_time_ms - 500) / 500 * 30)  # 50-20
        elif response_time_ms <= 5000:
            return 20 - ((response_time_ms - 1000) / 4000 * 20)  # 20-0
        else:
            return 0
    
    def _calculate_performance_grade(self, score: float) -> str:
        """Calculate performance grade from score."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F" 
   
    def _update_performance_history(self, system_perf: Dict[str, Any], service_perf: Dict[str, Any]):
        """Update performance history for trend analysis."""
        current_time = time.time()
        
        # Update system performance history
        if "system" not in self.performance_history:
            self.performance_history["system"] = []
        
        self.performance_history["system"].append({
            "timestamp": current_time,
            "cpu_percent": system_perf.get("cpu", {}).get("percent", 0),
            "memory_percent": system_perf.get("memory", {}).get("percent", 0),
            "disk_percent": system_perf.get("disk", {}).get("percent", 0),
            "overall_score": system_perf.get("overall_score", 0)
        })
        
        # Update service performance history
        for service_name, perf_data in service_perf.items():
            if service_name not in self.performance_history:
                self.performance_history[service_name] = []
            
            self.performance_history[service_name].append({
                "timestamp": current_time,
                "response_time_ms": perf_data.get("response_time_ms"),
                "available": perf_data.get("available", False),
                "performance_score": perf_data.get("performance_score", 0)
            })
        
        # Keep only last 24 hours of history
        cutoff_time = current_time - (24 * 60 * 60)
        for key in self.performance_history:
            self.performance_history[key] = [
                entry for entry in self.performance_history[key]
                if entry["timestamp"] > cutoff_time
            ]
    
    async def _run_performance_benchmarks(self):
        """Run comprehensive performance benchmarks."""
        logger.info("Starting performance benchmarks")
        benchmark_start = time.time()
        
        benchmarks = {}
        
        # System benchmarks
        benchmarks["system"] = await self._run_system_benchmarks()
        
        # Service benchmarks
        benchmarks["services"] = await self._run_service_benchmarks()
        
        # AI model benchmarks (if applicable)
        benchmarks["ai_models"] = await self._run_ai_model_benchmarks()
        
        benchmark_duration = time.time() - benchmark_start
        benchmarks["benchmark_duration"] = benchmark_duration
        benchmarks["timestamp"] = time.time()
        
        self.benchmark_results = benchmarks
        
        logger.info("Performance benchmarks completed", 
                   duration=benchmark_duration,
                   system_score=benchmarks["system"].get("overall_score", 0))
    
    async def _run_system_benchmarks(self) -> Dict[str, Any]:
        """Run system-level performance benchmarks."""
        try:
            # CPU benchmark - simple computation test
            cpu_start = time.time()
            result = sum(i * i for i in range(100000))  # Simple CPU-intensive task
            cpu_benchmark_time = time.time() - cpu_start
            
            # Memory benchmark - allocation test
            memory_start = time.time()
            test_data = [i for i in range(100000)]  # Memory allocation
            memory_benchmark_time = time.time() - memory_start
            del test_data
            
            # Disk benchmark - write/read test
            disk_start = time.time()
            test_file = "/tmp/benchmark_test.txt"
            with open(test_file, "w") as f:
                f.write("x" * 10000)  # Write 10KB
            with open(test_file, "r") as f:
                content = f.read()
            disk_benchmark_time = time.time() - disk_start
            
            # Clean up
            try:
                import os
                os.remove(test_file)
            except:
                pass
            
            # Calculate benchmark scores
            cpu_score = max(0, 100 - (cpu_benchmark_time * 1000))  # Lower time = higher score
            memory_score = max(0, 100 - (memory_benchmark_time * 1000))
            disk_score = max(0, 100 - (disk_benchmark_time * 100))
            
            overall_score = (cpu_score + memory_score + disk_score) / 3
            
            return {
                "cpu_benchmark_ms": cpu_benchmark_time * 1000,
                "memory_benchmark_ms": memory_benchmark_time * 1000,
                "disk_benchmark_ms": disk_benchmark_time * 1000,
                "cpu_score": cpu_score,
                "memory_score": memory_score,
                "disk_score": disk_score,
                "overall_score": overall_score,
                "grade": self._calculate_performance_grade(overall_score)
            }
        
        except Exception as e:
            logger.error("System benchmark failed", error=str(e))
            return {"error": str(e), "overall_score": 0, "grade": "F"}
    
    async def _run_service_benchmarks(self) -> Dict[str, Any]:
        """Run service performance benchmarks."""
        service_benchmarks = {}
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            for service_name, service_url in self.settings.services.items():
                try:
                    # Run multiple requests to get average performance
                    response_times = []
                    success_count = 0
                    
                    for i in range(5):  # 5 test requests
                        try:
                            start_time = time.time()
                            response = await client.get(f"{service_url}/health")
                            response_time = (time.time() - start_time) * 1000
                            
                            response_times.append(response_time)
                            if response.status_code == 200:
                                success_count += 1
                        
                        except Exception:
                            response_times.append(None)
                    
                    # Calculate statistics
                    valid_times = [t for t in response_times if t is not None]
                    if valid_times:
                        avg_response_time = statistics.mean(valid_times)
                        min_response_time = min(valid_times)
                        max_response_time = max(valid_times)
                        std_dev = statistics.stdev(valid_times) if len(valid_times) > 1 else 0
                    else:
                        avg_response_time = None
                        min_response_time = None
                        max_response_time = None
                        std_dev = None
                    
                    success_rate = (success_count / 5) * 100
                    
                    # Calculate performance score
                    if avg_response_time:
                        perf_score = self._calculate_service_performance_score(avg_response_time, 200)
                        perf_score *= (success_rate / 100)  # Adjust for success rate
                    else:
                        perf_score = 0
                    
                    service_benchmarks[service_name] = {
                        "avg_response_time_ms": avg_response_time,
                        "min_response_time_ms": min_response_time,
                        "max_response_time_ms": max_response_time,
                        "std_dev_ms": std_dev,
                        "success_rate_percent": success_rate,
                        "performance_score": perf_score,
                        "grade": self._calculate_performance_grade(perf_score)
                    }
                
                except Exception as e:
                    service_benchmarks[service_name] = {
                        "error": str(e),
                        "performance_score": 0,
                        "grade": "F"
                    }
        
        return service_benchmarks
    
    async def _run_ai_model_benchmarks(self) -> Dict[str, Any]:
        """Run AI model performance benchmarks."""
        ai_benchmarks = {}
        
        # Test Ollama service if available
        ollama_url = self.settings.services.get("ollama")
        if ollama_url:
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    # Test model loading time
                    start_time = time.time()
                    response = await client.get(f"{ollama_url}/api/tags")
                    model_list_time = (time.time() - start_time) * 1000
                    
                    if response.status_code == 200:
                        models = response.json().get("models", [])
                        
                        # Test inference time with a simple prompt (if models available)
                        inference_times = []
                        if models:
                            model_name = models[0]["name"]
                            
                            for i in range(3):  # 3 test inferences
                                try:
                                    start_time = time.time()
                                    inference_response = await client.post(
                                        f"{ollama_url}/api/generate",
                                        json={
                                            "model": model_name,
                                            "prompt": "Hello",
                                            "stream": False
                                        }
                                    )
                                    inference_time = (time.time() - start_time) * 1000
                                    
                                    if inference_response.status_code == 200:
                                        inference_times.append(inference_time)
                                
                                except Exception:
                                    continue
                        
                        # Calculate AI performance metrics
                        avg_inference_time = statistics.mean(inference_times) if inference_times else None
                        
                        # Score based on inference time
                        if avg_inference_time:
                            if avg_inference_time <= 1000:  # 1 second
                                ai_score = 100
                            elif avg_inference_time <= 5000:  # 5 seconds
                                ai_score = 80
                            elif avg_inference_time <= 10000:  # 10 seconds
                                ai_score = 60
                            elif avg_inference_time <= 30000:  # 30 seconds
                                ai_score = 40
                            else:
                                ai_score = 20
                        else:
                            ai_score = 0
                        
                        ai_benchmarks["ollama"] = {
                            "model_list_time_ms": model_list_time,
                            "available_models": len(models),
                            "avg_inference_time_ms": avg_inference_time,
                            "inference_tests": len(inference_times),
                            "performance_score": ai_score,
                            "grade": self._calculate_performance_grade(ai_score)
                        }
                    
                    else:
                        ai_benchmarks["ollama"] = {
                            "error": f"HTTP {response.status_code}",
                            "performance_score": 0,
                            "grade": "F"
                        }
            
            except Exception as e:
                ai_benchmarks["ollama"] = {
                    "error": str(e),
                    "performance_score": 0,
                    "grade": "F"
                }
        
        return ai_benchmarks
    
    async def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report with recommendations."""
        if not self.performance_data:
            await self._analyze_performance()
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        # Calculate trends
        trends = self._calculate_performance_trends()
        
        return {
            "current_performance": self.performance_data,
            "benchmark_results": self.benchmark_results,
            "performance_trends": trends,
            "recommendations": recommendations,
            "report_timestamp": time.time()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        if not self.performance_data:
            return recommendations
        
        system_perf = self.performance_data.get("system", {})
        service_perf = self.performance_data.get("services", {})
        
        # System recommendations
        cpu_percent = system_perf.get("cpu", {}).get("percent", 0)
        memory_percent = system_perf.get("memory", {}).get("percent", 0)
        disk_percent = system_perf.get("disk", {}).get("percent", 0)
        
        if cpu_percent > 80:
            recommendations.append("High CPU usage detected. Consider scaling horizontally or optimizing CPU-intensive processes.")
        
        if memory_percent > 85:
            recommendations.append("High memory usage detected. Consider increasing memory allocation or optimizing memory usage.")
        
        if disk_percent > 90:
            recommendations.append("Disk space is critically low. Clean up old files or increase disk capacity.")
        
        # Service recommendations
        slow_services = []
        unavailable_services = []
        
        for service_name, perf_data in service_perf.items():
            response_time = perf_data.get("response_time_ms")
            available = perf_data.get("available", False)
            
            if not available:
                unavailable_services.append(service_name)
            elif response_time and response_time > 2000:
                slow_services.append((service_name, response_time))
        
        if unavailable_services:
            recommendations.append(f"Services unavailable: {', '.join(unavailable_services)}. Check service health and restart if necessary.")
        
        if slow_services:
            slow_list = [f"{name} ({time:.0f}ms)" for name, time in slow_services]
            recommendations.append(f"Slow response times detected: {', '.join(slow_list)}. Consider performance optimization.")
        
        # AI model recommendations
        if self.benchmark_results.get("ai_models", {}).get("ollama"):
            ollama_perf = self.benchmark_results["ai_models"]["ollama"]
            inference_time = ollama_perf.get("avg_inference_time_ms")
            
            if inference_time and inference_time > 10000:
                recommendations.append("AI model inference is slow. Consider using GPU acceleration or smaller models.")
        
        # General recommendations
        overall_score = system_perf.get("overall_score", 100)
        if overall_score < 70:
            recommendations.append("Overall system performance is below optimal. Consider resource optimization or scaling.")
        
        return recommendations
    
    def _calculate_performance_trends(self) -> Dict[str, Any]:
        """Calculate performance trends from historical data."""
        trends = {}
        
        for component, history in self.performance_history.items():
            if len(history) < 2:
                continue
            
            if component == "system":
                # System trends
                cpu_values = [entry["cpu_percent"] for entry in history]
                memory_values = [entry["memory_percent"] for entry in history]
                scores = [entry["overall_score"] for entry in history]
                
                trends[component] = {
                    "cpu_trend": self._calculate_trend(cpu_values),
                    "memory_trend": self._calculate_trend(memory_values),
                    "score_trend": self._calculate_trend(scores),
                    "data_points": len(history)
                }
            
            else:
                # Service trends
                response_times = [entry["response_time_ms"] for entry in history 
                                if entry["response_time_ms"] is not None]
                scores = [entry["performance_score"] for entry in history]
                availability = [entry["available"] for entry in history]
                
                if response_times:
                    trends[component] = {
                        "response_time_trend": self._calculate_trend(response_times),
                        "score_trend": self._calculate_trend(scores),
                        "availability_percent": (sum(availability) / len(availability)) * 100,
                        "data_points": len(history)
                    }
        
        return trends
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values."""
        if len(values) < 2:
            return "stable"
        
        # Simple linear trend calculation
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))
        
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        
        if abs(slope) < 0.1:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"
    
    async def run_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks on demand."""
        await self._run_performance_benchmarks()
        return self.benchmark_results
    
    async def get_performance_history(self, component: str = None) -> Dict[str, Any]:
        """Get performance history for a specific component or all components."""
        if component:
            return {component: self.performance_history.get(component, [])}
        return self.performance_history
    
    async def reload_config(self):
        """Reload configuration."""
        self.settings = MonitoringSettings()
        self.tracking_enabled = self.settings.performance_tracking_enabled
        self.benchmark_interval = self.settings.benchmark_interval_minutes * 60
        logger.info("Performance tracker configuration reloaded")
    
    async def close(self):
        """Clean up resources."""
        logger.info("Performance tracker closed")
