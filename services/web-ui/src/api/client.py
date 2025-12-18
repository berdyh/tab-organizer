"""API client for backend services."""

import os
from typing import Optional

import httpx


class APIClient:
    """Client for communicating with backend services."""
    
    def __init__(
        self,
        backend_url: Optional[str] = None,
        ai_url: Optional[str] = None,
        browser_url: Optional[str] = None,
    ):
        self.backend_url = backend_url or os.getenv(
            "BACKEND_URL", "http://localhost:8080"
        )
        self.ai_url = ai_url or os.getenv(
            "AI_ENGINE_URL", "http://localhost:8090"
        )
        self.browser_url = browser_url or os.getenv(
            "BROWSER_ENGINE_URL", "http://localhost:8083"
        )
        self.timeout = 60.0
    
    # Session endpoints
    async def create_session(self, name: Optional[str] = None) -> dict:
        """Create a new session."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.backend_url}/api/v1/sessions",
                json={"name": name},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
    
    async def list_sessions(self) -> list[dict]:
        """List all sessions."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.backend_url}/api/v1/sessions",
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
    
    async def get_session(self, session_id: str) -> dict:
        """Get session details."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.backend_url}/api/v1/sessions/{session_id}",
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
    
    async def delete_session(self, session_id: str) -> dict:
        """Delete a session."""
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{self.backend_url}/api/v1/sessions/{session_id}",
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
    
    # URL endpoints
    async def add_urls(
        self, urls: list[str], session_id: Optional[str] = None
    ) -> dict:
        """Add URLs to a session."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.backend_url}/api/v1/urls",
                json={"urls": urls, "session_id": session_id},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
    
    async def get_urls(
        self, session_id: str, status: Optional[str] = None
    ) -> list[dict]:
        """Get URLs for a session."""
        params = {}
        if status:
            params["status"] = status
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.backend_url}/api/v1/urls/{session_id}",
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
    
    # Scraping endpoints
    async def start_scraping(
        self, session_id: str, urls: Optional[list[str]] = None
    ) -> dict:
        """Start scraping URLs."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.backend_url}/api/v1/scrape",
                json={"session_id": session_id, "urls": urls},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
    
    async def get_scrape_status(self, session_id: str) -> dict:
        """Get scraping status."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.browser_url}/scrape/status/{session_id}",
                timeout=self.timeout,
            )
            if response.status_code == 404:
                return {"status": "not_started"}
            response.raise_for_status()
            return response.json()
    
    # Auth endpoints
    async def get_pending_auth(self) -> dict:
        """Get pending authentication requests."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.backend_url}/api/v1/auth/pending",
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
    
    async def submit_credentials(self, domain: str, credentials: dict) -> dict:
        """Submit credentials for a domain."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.backend_url}/api/v1/auth/credentials",
                params={"domain": domain},
                json=credentials,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
    
    # Clustering endpoints
    async def start_clustering(self, session_id: str) -> dict:
        """Start clustering for a session."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.backend_url}/api/v1/cluster",
                json={"session_id": session_id},
                timeout=120.0,
            )
            response.raise_for_status()
            return response.json()
    
    async def get_clusters(self, session_id: str) -> dict:
        """Get clusters for a session."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.backend_url}/api/v1/clusters/{session_id}",
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
    
    # Chat endpoints
    async def chat(self, query: str, session_id: Optional[str] = None) -> dict:
        """Chat with the AI about scraped content."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.ai_url}/chat",
                json={"query": query, "session_id": session_id},
                timeout=120.0,
            )
            response.raise_for_status()
            return response.json()
    
    async def search(self, query: str, session_id: Optional[str] = None) -> dict:
        """Search scraped content."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.ai_url}/search",
                json={"query": query, "session_id": session_id},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
    
    async def summarize_session(self, session_id: str) -> dict:
        """Get summary of session content."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.ai_url}/summarize/{session_id}",
                timeout=120.0,
            )
            response.raise_for_status()
            return response.json()
    
    # Export endpoints
    async def export_session(self, session_id: str, format: str = "markdown") -> dict:
        """Export session to specified format."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.backend_url}/api/v1/export",
                json={"session_id": session_id, "format": format},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
    
    # Provider endpoints
    async def get_providers(self) -> dict:
        """Get current AI provider configuration."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.ai_url}/providers",
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
    
    async def switch_provider(
        self,
        llm_provider: Optional[str] = None,
        embedding_provider: Optional[str] = None,
    ) -> dict:
        """Switch AI providers."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.ai_url}/providers/switch",
                json={
                    "llm_provider": llm_provider,
                    "embedding_provider": embedding_provider,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
    
    # Health checks
    async def check_health(self) -> dict:
        """Check health of all services."""
        results = {}
        
        services = [
            ("backend", self.backend_url),
            ("ai_engine", self.ai_url),
            ("browser_engine", self.browser_url),
        ]
        
        async with httpx.AsyncClient() as client:
            for name, url in services:
                try:
                    response = await client.get(
                        f"{url}/health",
                        timeout=5.0,
                    )
                    results[name] = response.status_code == 200
                except Exception:
                    results[name] = False
        
        return results


# Synchronous wrapper for Streamlit
class SyncAPIClient:
    """Synchronous wrapper for API client."""
    
    def __init__(self):
        self._async_client = APIClient()
    
    def _run(self, coro):
        """Run async coroutine synchronously."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    
    def create_session(self, name=None):
        return self._run(self._async_client.create_session(name))
    
    def list_sessions(self):
        return self._run(self._async_client.list_sessions())
    
    def get_session(self, session_id):
        return self._run(self._async_client.get_session(session_id))
    
    def delete_session(self, session_id):
        return self._run(self._async_client.delete_session(session_id))
    
    def add_urls(self, urls, session_id=None):
        return self._run(self._async_client.add_urls(urls, session_id))
    
    def get_urls(self, session_id, status=None):
        return self._run(self._async_client.get_urls(session_id, status))
    
    def start_scraping(self, session_id, urls=None):
        return self._run(self._async_client.start_scraping(session_id, urls))
    
    def get_scrape_status(self, session_id):
        return self._run(self._async_client.get_scrape_status(session_id))
    
    def get_pending_auth(self):
        return self._run(self._async_client.get_pending_auth())
    
    def submit_credentials(self, domain, credentials):
        return self._run(self._async_client.submit_credentials(domain, credentials))
    
    def start_clustering(self, session_id):
        return self._run(self._async_client.start_clustering(session_id))
    
    def get_clusters(self, session_id):
        return self._run(self._async_client.get_clusters(session_id))
    
    def chat(self, query, session_id=None):
        return self._run(self._async_client.chat(query, session_id))
    
    def search(self, query, session_id=None):
        return self._run(self._async_client.search(query, session_id))
    
    def summarize_session(self, session_id):
        return self._run(self._async_client.summarize_session(session_id))
    
    def export_session(self, session_id, format="markdown"):
        return self._run(self._async_client.export_session(session_id, format))
    
    def get_providers(self):
        return self._run(self._async_client.get_providers())
    
    def switch_provider(self, llm_provider=None, embedding_provider=None):
        return self._run(self._async_client.switch_provider(llm_provider, embedding_provider))
    
    def check_health(self):
        return self._run(self._async_client.check_health())
