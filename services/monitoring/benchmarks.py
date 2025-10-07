"""
Performance Benchmarks - Containerized monitoring tests and performance benchmarks.
Provides comprehensive benchmarking for all services and system components.
"""

import asyncio
import time
import statistics
import json
from typing import Dict, Any, List
from datetime import datetime

import httpx
import psutil

from config import MonitoringSettings
from logging_config import get_logger

logger = get_logger("benchmarks")


class PerformanceBenchmarks:
    """Comprehensive performance benchmarking suite."""
    
    def __init__(self, settings: MonitoringSettings = None):
        self.settings = settings or MonitoringSettings()
        self.results = {}
    
    async def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all performance benchmarks."""
        logger.info("Starting comprehensive performance benchmarks")
        start_time = time.time()
        
        # System benchmarks
        logger.info("Running system benchmarks...")
        system_results = await self.run_system_benchmarks()
        
        # Container benchmarks
        logger.info("Running container benchmarks...")
        container_results = await self.run_container_benchmarks()
        
        # Service benchmarks
        logger.info("Running service benchmarks...")
        service_results = await self.run_service_benchmarks()
        
        # AI model benchmarks
        logger.info("Running AI model benchmarks...")
        ai_results = await self.run_ai_model_benchmarks()
        
        # Network benchmarks
        logger.info("Running network benchmarks...")
        network_results = await self.run_network_benchmarks()
        
        total_duration = time.time() - start_time
        
        self.results = {
            "timestamp": time.time(),
            "duration_seconds": total_duration,
            "system": system_results,
            "containers": container_results,
            "services": service_results,
            "ai_models": ai_results,
            "network": network_results,
            "summary": self._generate_summary()
        }
        
        logger.info("Performance benchmarks completed", duration=total_duration)
        return self.results
    
    async def run_system_benchmarks(self) -> Dict[str, Any]:
        """Run comprehensive system performance benchmarks."""
        results = {}
        
        # CPU benchmarks
        results["cpu"] = await self._benchmark_cpu()
        
        # Memory benchmarks
        results["memory"] = await self._benchmark_memory()
        
        # Disk I/O benchmarks
        results["disk"] = await self._benchmark_disk()
        
        # System load benchmarks
        results["load"] = await self._benchmark_system_load()
        
        return results
    
    async def _benchmark_cpu(self) -> Dict[str, Any]:
        """Benchmark CPU performance."""
        logger.info("Benchmarking CPU performance...")
        
        # Single-threaded CPU test
        start_time = time.time()
        result = sum(i * i for i in range(1000000))  # 1M iterations
        single_thread_time = time.time() - start_time
        
        # Multi-threaded CPU test
        async def cpu_task():
            return sum(i * i for i in range(500000))
        
        start_time = time.time()
        tasks = [cpu_task() for _ in range(psutil.cpu_count())]
        await asyncio.gather(*tasks)
        multi_thread_time = time.time() - start_time
        
        # CPU utilization test
        cpu_before = psutil.cpu_percent(interval=1)
        
        # Stress test
        start_time = time.time()
        stress_result = sum(i ** 2 for i in range(100000))
        stress_time = time.time() - start_time
        
        cpu_after = psutil.cpu_percent(interval=1)
        
        return {
            "single_thread_ms": single_thread_time * 1000,
            "multi_thread_ms": multi_thread_time * 1000,
            "stress_test_ms": stress_time * 1000,
            "cpu_before_percent": cpu_before,
            "cpu_after_percent": cpu_after,
            "cpu_cores": psutil.cpu_count(),
            "cpu_logical": psutil.cpu_count(logical=True),
            "performance_score": self._calculate_cpu_score(single_thread_time, multi_thread_time)
        }
    
    async def _benchmark_memory(self) -> Dict[str, Any]:
        """Benchmark memory performance."""
        logger.info("Benchmarking memory performance...")
        
        memory_before = psutil.virtual_memory()
        
        # Memory allocation test
        start_time = time.time()
        test_data = []
        for i in range(100):
            test_data.append([j for j in range(10000)])  # 1M integers total
        allocation_time = time.time() - start_time
        
        # Memory access test
        start_time = time.time()
        total = sum(sum(chunk) for chunk in test_data)
        access_time = time.time() - start_time
        
        # Memory cleanup
        start_time = time.time()
        del test_data
        cleanup_time = time.time() - start_time
        
        memory_after = psutil.virtual_memory()
        
        return {
            "allocation_ms": allocation_time * 1000,
            "access_ms": access_time * 1000,
            "cleanup_ms": cleanup_time * 1000,
            "memory_before_mb": memory_before.used / (1024 * 1024),
            "memory_after_mb": memory_after.used / (1024 * 1024),
            "memory_diff_mb": (memory_after.used - memory_before.used) / (1024 * 1024),
            "performance_score": self._calculate_memory_score(allocation_time, access_time)
        }
    
    async def _benchmark_disk(self) -> Dict[str, Any]:
        """Benchmark disk I/O performance."""
        logger.info("Benchmarking disk I/O performance...")
        
        test_file = "/tmp/benchmark_test.dat"
        test_data = b"x" * (1024 * 1024)  # 1MB of data
        
        # Write test
        start_time = time.time()
        with open(test_file, "wb") as f:
            for _ in range(10):  # Write 10MB
                f.write(test_data)
        write_time = time.time() - start_time
        
        # Read test
        start_time = time.time()
        with open(test_file, "rb") as f:
            data = f.read()
        read_time = time.time() - start_time
        
        # Random access test
        start_time = time.time()
        with open(test_file, "rb") as f:
            for _ in range(100):
                f.seek(0)
                f.read(1024)
        random_access_time = time.time() - start_time
        
        # Cleanup
        try:
            import os
            os.remove(test_file)
        except:
            pass
        
        # Get disk usage
        disk_usage = psutil.disk_usage('/')
        
        return {
            "write_mb_per_sec": 10 / write_time,
            "read_mb_per_sec": len(data) / (1024 * 1024) / read_time,
            "random_access_ms": random_access_time * 1000,
            "disk_total_gb": disk_usage.total / (1024 ** 3),
            "disk_free_gb": disk_usage.free / (1024 ** 3),
            "disk_used_percent": (disk_usage.used / disk_usage.total) * 100,
            "performance_score": self._calculate_disk_score(write_time, read_time)
        }
    
    async def _benchmark_system_load(self) -> Dict[str, Any]:
        """Benchmark system load characteristics."""
        logger.info("Benchmarking system load...")
        
        # Get load averages (Unix only)
        load_avg = None
        try:
            load_avg = psutil.getloadavg()
        except AttributeError:
            load_avg = [0, 0, 0]  # Windows fallback
        
        # Process count
        process_count = len(psutil.pids())
        
        # Network connections
        try:
            network_connections = len(psutil.net_connections())
        except:
            network_connections = 0
        
        # Boot time
        boot_time = psutil.boot_time()
        uptime = time.time() - boot_time
        
        return {
            "load_1min": load_avg[0],
            "load_5min": load_avg[1],
            "load_15min": load_avg[2],
            "process_count": process_count,
            "network_connections": network_connections,
            "uptime_hours": uptime / 3600,
            "performance_score": self._calculate_load_score(load_avg[0], process_count)
        }
    
    async def run_container_benchmarks(self) -> Dict[str, Any]:
        """Run Docker container performance benchmarks."""
        try:
            import docker
            client = docker.from_env()
            containers = client.containers.list()
            
            container_results = {}
            
            for container in containers:
                try:
                    stats = container.stats(stream=False)
                    
                    # Calculate metrics
                    cpu_percent = self._calculate_container_cpu_percent(stats)
                    memory_usage = stats['memory_stats'].get('usage', 0)
                    memory_limit = stats['memory_stats'].get('limit', 0)
                    memory_percent = (memory_usage / memory_limit * 100) if memory_limit > 0 else 0
                    
                    container_results[container.name] = {
                        "cpu_percent": cpu_percent,
                        "memory_usage_mb": memory_usage / (1024 * 1024),
                        "memory_percent": memory_percent,
                        "status": container.status,
                        "performance_score": self._calculate_container_score(cpu_percent, memory_percent)
                    }
                
                except Exception as e:
                    container_results[container.name] = {
                        "error": str(e),
                        "performance_score": 0
                    }
            
            return container_results
        
        except Exception as e:
            logger.error("Container benchmarks failed", error=str(e))
            return {"error": str(e)}
    
    async def run_service_benchmarks(self) -> Dict[str, Any]:
        """Run service performance benchmarks."""
        service_results = {}
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            for service_name, service_url in self.settings.services.items():
                try:
                    # Health check benchmark
                    health_times = []
                    for _ in range(5):
                        start_time = time.time()
                        response = await client.get(f"{service_url}/health")
                        response_time = (time.time() - start_time) * 1000
                        health_times.append(response_time)
                        
                        if response.status_code != 200:
                            break
                    
                    # Load test (concurrent requests)
                    concurrent_times = []
                    if health_times and all(t < 5000 for t in health_times):  # Only if healthy
                        tasks = []
                        start_time = time.time()
                        
                        for _ in range(10):  # 10 concurrent requests
                            task = client.get(f"{service_url}/health")
                            tasks.append(task)
                        
                        responses = await asyncio.gather(*tasks, return_exceptions=True)
                        total_time = time.time() - start_time
                        
                        successful_responses = [r for r in responses if not isinstance(r, Exception)]
                        success_rate = len(successful_responses) / len(responses) * 100
                    else:
                        success_rate = 0
                        total_time = 0
                    
                    # Calculate performance metrics
                    if health_times:
                        avg_response_time = statistics.mean(health_times)
                        min_response_time = min(health_times)
                        max_response_time = max(health_times)
                        std_dev = statistics.stdev(health_times) if len(health_times) > 1 else 0
                    else:
                        avg_response_time = None
                        min_response_time = None
                        max_response_time = None
                        std_dev = None
                    
                    service_results[service_name] = {
                        "avg_response_time_ms": avg_response_time,
                        "min_response_time_ms": min_response_time,
                        "max_response_time_ms": max_response_time,
                        "std_dev_ms": std_dev,
                        "concurrent_load_time_ms": total_time * 1000,
                        "success_rate_percent": success_rate,
                        "performance_score": self._calculate_service_score(avg_response_time, success_rate)
                    }
                
                except Exception as e:
                    service_results[service_name] = {
                        "error": str(e),
                        "performance_score": 0
                    }
        
        return service_results
    
    async def run_ai_model_benchmarks(self) -> Dict[str, Any]:
        """Run AI model performance benchmarks."""
        ai_results = {}
        
        # Test Ollama if available
        ollama_url = self.settings.services.get("ollama")
        if ollama_url:
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    # Model listing benchmark
                    start_time = time.time()
                    response = await client.get(f"{ollama_url}/api/tags")
                    model_list_time = (time.time() - start_time) * 1000
                    
                    if response.status_code == 200:
                        models = response.json().get("models", [])
                        
                        if models:
                            model_name = models[0]["name"]
                            
                            # Inference benchmarks
                            inference_times = []
                            test_prompts = [
                                "Hello",
                                "What is 2+2?",
                                "Explain machine learning in one sentence."
                            ]
                            
                            for prompt in test_prompts:
                                try:
                                    start_time = time.time()
                                    inference_response = await client.post(
                                        f"{ollama_url}/api/generate",
                                        json={
                                            "model": model_name,
                                            "prompt": prompt,
                                            "stream": False
                                        }
                                    )
                                    inference_time = (time.time() - start_time) * 1000
                                    
                                    if inference_response.status_code == 200:
                                        inference_times.append(inference_time)
                                
                                except Exception as e:
                                    logger.warning("AI inference benchmark failed", prompt=prompt, error=str(e))
                            
                            # Calculate AI performance metrics
                            if inference_times:
                                avg_inference_time = statistics.mean(inference_times)
                                min_inference_time = min(inference_times)
                                max_inference_time = max(inference_times)
                            else:
                                avg_inference_time = None
                                min_inference_time = None
                                max_inference_time = None
                            
                            ai_results["ollama"] = {
                                "model_list_time_ms": model_list_time,
                                "available_models": len(models),
                                "test_model": model_name,
                                "avg_inference_time_ms": avg_inference_time,
                                "min_inference_time_ms": min_inference_time,
                                "max_inference_time_ms": max_inference_time,
                                "successful_inferences": len(inference_times),
                                "performance_score": self._calculate_ai_score(avg_inference_time)
                            }
                        else:
                            ai_results["ollama"] = {
                                "model_list_time_ms": model_list_time,
                                "available_models": 0,
                                "error": "No models available",
                                "performance_score": 0
                            }
                    else:
                        ai_results["ollama"] = {
                            "error": f"HTTP {response.status_code}",
                            "performance_score": 0
                        }
            
            except Exception as e:
                ai_results["ollama"] = {
                    "error": str(e),
                    "performance_score": 0
                }
        
        return ai_results
    
    async def run_network_benchmarks(self) -> Dict[str, Any]:
        """Run network performance benchmarks."""
        network_results = {}
        
        # Test internal service connectivity
        connectivity_results = {}
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for service_name, service_url in self.settings.services.items():
                try:
                    start_time = time.time()
                    response = await client.get(f"{service_url}/health")
                    response_time = (time.time() - start_time) * 1000
                    
                    connectivity_results[service_name] = {
                        "response_time_ms": response_time,
                        "status_code": response.status_code,
                        "reachable": response.status_code == 200
                    }
                
                except Exception as e:
                    connectivity_results[service_name] = {
                        "error": str(e),
                        "reachable": False
                    }
        
        # Network statistics
        network_io = psutil.net_io_counters()
        
        network_results = {
            "connectivity": connectivity_results,
            "network_io": {
                "bytes_sent_mb": network_io.bytes_sent / (1024 * 1024),
                "bytes_recv_mb": network_io.bytes_recv / (1024 * 1024),
                "packets_sent": network_io.packets_sent,
                "packets_recv": network_io.packets_recv,
                "errors_in": network_io.errin,
                "errors_out": network_io.errout
            },
            "performance_score": self._calculate_network_score(connectivity_results)
        }
        
        return network_results
    
    def _calculate_cpu_score(self, single_time: float, multi_time: float) -> float:
        """Calculate CPU performance score."""
        # Lower times = better performance
        single_score = max(0, 100 - (single_time * 100))
        multi_score = max(0, 100 - (multi_time * 50))
        return (single_score + multi_score) / 2
    
    def _calculate_memory_score(self, alloc_time: float, access_time: float) -> float:
        """Calculate memory performance score."""
        alloc_score = max(0, 100 - (alloc_time * 1000))
        access_score = max(0, 100 - (access_time * 100))
        return (alloc_score + access_score) / 2
    
    def _calculate_disk_score(self, write_time: float, read_time: float) -> float:
        """Calculate disk performance score."""
        write_score = max(0, 100 - (write_time * 10))
        read_score = max(0, 100 - (read_time * 100))
        return (write_score + read_score) / 2
    
    def _calculate_load_score(self, load_avg: float, process_count: int) -> float:
        """Calculate system load score."""
        load_score = max(0, 100 - (load_avg * 25))
        process_score = max(0, 100 - (process_count / 10))
        return (load_score + process_score) / 2
    
    def _calculate_container_score(self, cpu_percent: float, memory_percent: float) -> float:
        """Calculate container performance score."""
        cpu_score = max(0, 100 - cpu_percent)
        memory_score = max(0, 100 - memory_percent)
        return (cpu_score + memory_score) / 2
    
    def _calculate_service_score(self, response_time: float, success_rate: float) -> float:
        """Calculate service performance score."""
        if response_time is None:
            return 0
        
        time_score = max(0, 100 - (response_time / 50))  # 5000ms = 0 score
        success_score = success_rate
        return (time_score + success_score) / 2
    
    def _calculate_ai_score(self, inference_time: float) -> float:
        """Calculate AI model performance score."""
        if inference_time is None:
            return 0
        
        if inference_time <= 1000:  # 1 second
            return 100
        elif inference_time <= 5000:  # 5 seconds
            return 80
        elif inference_time <= 10000:  # 10 seconds
            return 60
        elif inference_time <= 30000:  # 30 seconds
            return 40
        else:
            return 20
    
    def _calculate_network_score(self, connectivity_results: Dict[str, Any]) -> float:
        """Calculate network performance score."""
        if not connectivity_results:
            return 0
        
        reachable_services = sum(1 for result in connectivity_results.values() 
                               if result.get("reachable", False))
        total_services = len(connectivity_results)
        
        connectivity_score = (reachable_services / total_services) * 100
        
        # Factor in response times
        response_times = [result.get("response_time_ms", 5000) 
                         for result in connectivity_results.values() 
                         if result.get("reachable", False)]
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            time_score = max(0, 100 - (avg_response_time / 50))
            return (connectivity_score + time_score) / 2
        
        return connectivity_score
    
    def _calculate_container_cpu_percent(self, stats: Dict[str, Any]) -> float:
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
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate benchmark summary."""
        if not self.results:
            return {}
        
        # Calculate overall scores
        system_score = 0
        if "system" in self.results:
            system_scores = []
            for component, data in self.results["system"].items():
                if "performance_score" in data:
                    system_scores.append(data["performance_score"])
            system_score = statistics.mean(system_scores) if system_scores else 0
        
        service_score = 0
        if "services" in self.results:
            service_scores = []
            for service, data in self.results["services"].items():
                if "performance_score" in data:
                    service_scores.append(data["performance_score"])
            service_score = statistics.mean(service_scores) if service_scores else 0
        
        overall_score = (system_score + service_score) / 2
        
        return {
            "overall_score": overall_score,
            "system_score": system_score,
            "service_score": service_score,
            "grade": self._calculate_grade(overall_score),
            "recommendations": self._generate_recommendations()
        }
    
    def _calculate_grade(self, score: float) -> str:
        """Calculate performance grade."""
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
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        if not self.results:
            return recommendations
        
        # System recommendations
        if "system" in self.results:
            cpu_data = self.results["system"].get("cpu", {})
            memory_data = self.results["system"].get("memory", {})
            disk_data = self.results["system"].get("disk", {})
            
            if cpu_data.get("performance_score", 100) < 70:
                recommendations.append("CPU performance is below optimal. Consider upgrading CPU or optimizing CPU-intensive processes.")
            
            if memory_data.get("performance_score", 100) < 70:
                recommendations.append("Memory performance is suboptimal. Consider adding more RAM or optimizing memory usage.")
            
            if disk_data.get("performance_score", 100) < 70:
                recommendations.append("Disk I/O performance is slow. Consider using SSD storage or optimizing disk operations.")
        
        # Service recommendations
        if "services" in self.results:
            slow_services = []
            for service_name, data in self.results["services"].items():
                if data.get("performance_score", 100) < 60:
                    slow_services.append(service_name)
            
            if slow_services:
                recommendations.append(f"Services with poor performance: {', '.join(slow_services)}. Consider optimization or scaling.")
        
        # AI model recommendations
        if "ai_models" in self.results:
            ollama_data = self.results["ai_models"].get("ollama", {})
            if ollama_data.get("performance_score", 100) < 60:
                recommendations.append("AI model inference is slow. Consider GPU acceleration or model optimization.")
        
        return recommendations
    
    def save_results(self, filename: str = None):
        """Save benchmark results to file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)
        
        logger.info("Benchmark results saved", filename=filename)


async def main():
    """Run performance benchmarks."""
    benchmarks = PerformanceBenchmarks()
    results = await benchmarks.run_all_benchmarks()
    
    # Print summary
    summary = results.get("summary", {})
    print(f"\n🏆 Performance Benchmark Results")
    print(f"Overall Score: {summary.get('overall_score', 0):.1f}/100 (Grade: {summary.get('grade', 'F')})")
    print(f"System Score: {summary.get('system_score', 0):.1f}/100")
    print(f"Service Score: {summary.get('service_score', 0):.1f}/100")
    
    recommendations = summary.get("recommendations", [])
    if recommendations:
        print(f"\n📋 Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    
    # Save results
    benchmarks.save_results()


if __name__ == "__main__":
    asyncio.run(main())