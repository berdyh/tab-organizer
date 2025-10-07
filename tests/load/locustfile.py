"""
Load testing configuration for the Web Scraping Tool
"""
import random
from locust import HttpUser, task, between, events
from locust.exception import StopUser


class WebScrapingUser(HttpUser):
    """Simulated user for load testing"""
    
    wait_time = between(1, 3)
    
    def on_start(self):
        """Initialize user session"""
        # Create a session for this user
        response = self.client.post("/api/sessions", json={
            "name": f"Load Test Session {self.environment.runner.user_count}",
            "description": "Automated load test session"
        })
        
        if response.status_code in [200, 201]:
            self.session_id = response.json().get("session_id", "default")
        else:
            self.session_id = "default"
    
    @task(3)
    def health_check(self):
        """Check system health"""
        self.client.get("/health")
    
    @task(5)
    def validate_url(self):
        """Validate a URL"""
        urls = [
            "https://example.com",
            "https://example.org",
            "https://example.net",
            "https://wikipedia.org",
            "https://github.com"
        ]
        
        self.client.post("/api/urls/validate", json={
            "url": random.choice(urls)
        })
    
    @task(2)
    def submit_urls(self):
        """Submit URLs for scraping"""
        urls = [
            "https://example.com",
            "https://example.org"
        ]
        
        response = self.client.post("/api/urls/batch", json={
            "urls": urls,
            "session_id": self.session_id
        })
        
        if response.status_code in [200, 202]:
            job_id = response.json().get("job_id")
            if job_id:
                # Check job status
                self.client.get(f"/api/jobs/{job_id}")
    
    @task(2)
    def search_content(self):
        """Perform content search"""
        queries = [
            "machine learning",
            "web scraping",
            "data analysis",
            "artificial intelligence",
            "python programming"
        ]
        
        self.client.post("/api/search", json={
            "query": random.choice(queries),
            "search_type": random.choice(["semantic", "keyword"]),
            "limit": 10
        })
    
    @task(1)
    def get_session_info(self):
        """Get session information"""
        self.client.get(f"/api/sessions/{self.session_id}")
    
    @task(1)
    def list_sessions(self):
        """List all sessions"""
        self.client.get("/api/sessions")
    
    @task(1)
    def trigger_clustering(self):
        """Trigger clustering operation"""
        self.client.post("/api/cluster", json={
            "session_id": self.session_id
        })
    
    @task(1)
    def export_data(self):
        """Export session data"""
        self.client.post("/api/export", json={
            "format": random.choice(["markdown", "json"]),
            "session_id": self.session_id
        })


class AdminUser(HttpUser):
    """Simulated admin user with different behavior"""
    
    wait_time = between(5, 10)
    weight = 1  # Less frequent than regular users
    
    @task
    def monitor_system(self):
        """Monitor system health and metrics"""
        self.client.get("/health")
        self.client.get("/api/metrics")
        self.client.get("/api/sessions")


class BurstUser(HttpUser):
    """User that creates burst traffic"""
    
    wait_time = between(0.1, 0.5)
    weight = 2
    
    @task(10)
    def rapid_health_checks(self):
        """Rapid health check requests"""
        self.client.get("/health")
    
    @task(5)
    def rapid_validations(self):
        """Rapid URL validations"""
        self.client.post("/api/urls/validate", json={
            "url": "https://example.com"
        })


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when load test starts"""
    print("Load test starting...")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when load test stops"""
    print("Load test completed!")
    print(f"Total requests: {environment.stats.total.num_requests}")
    print(f"Total failures: {environment.stats.total.num_failures}")
    print(f"Average response time: {environment.stats.total.avg_response_time:.2f}ms")
    print(f"RPS: {environment.stats.total.total_rps:.2f}")


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Called for each request"""
    if exception:
        print(f"Request failed: {name} - {exception}")
