"""Synchronous HTTP client used by Streamlit pages."""

import os
from typing import Optional

import requests


class SyncAPIClient:
    """Simple blocking API client for backend, AI, and browser services."""

    def __init__(self):
        backend_base = os.getenv("BACKEND_URL", "http://backend-core:8080")
        self.backend_url = f"{backend_base.rstrip('/')}/api/v1"
        self.ai_url = os.getenv("AI_ENGINE_URL", "http://ai-engine:8090").rstrip("/")
        self.browser_url = os.getenv(
            "BROWSER_ENGINE_URL", "http://browser-engine:8083"
        ).rstrip("/")
        self.timeout = float(os.getenv("UI_API_TIMEOUT", "30"))

    def _request(self, method: str, url: str, **kwargs):
        response = requests.request(method, url, timeout=self.timeout, **kwargs)
        response.raise_for_status()
        if response.content:
            return response.json()
        return {}

    # Sessions / URL management
    def create_session(self, name: str) -> dict:
        return self._request(
            "POST", f"{self.backend_url}/sessions", json={"name": name}
        )

    def list_sessions(self) -> list[dict]:
        return self._request("GET", f"{self.backend_url}/sessions")

    def get_session(self, session_id: str) -> dict:
        return self._request("GET", f"{self.backend_url}/sessions/{session_id}")

    def delete_session(self, session_id: str) -> dict:
        return self._request("DELETE", f"{self.backend_url}/sessions/{session_id}")

    def add_urls(self, urls: list[str], session_id: Optional[str] = None) -> dict:
        payload = {"urls": urls, "session_id": session_id}
        return self._request("POST", f"{self.backend_url}/urls", json=payload)

    def get_urls(self, session_id: str, status: Optional[str] = None) -> list[dict]:
        params = {"status": status} if status else None
        return self._request(
            "GET", f"{self.backend_url}/urls/{session_id}", params=params
        )

    # Scraping / Auth
    def start_scraping(self, session_id: str) -> dict:
        return self._request(
            "POST", f"{self.backend_url}/scrape", json={"session_id": session_id}
        )

    def get_scrape_status(self, session_id: str) -> dict:
        try:
            return self._request(
                "GET", f"{self.backend_url}/scrape/status/{session_id}"
            )
        except requests.HTTPError:
            stats = self.get_session(session_id)
            counts = stats.get("status_counts", {})
            total = stats.get("total_urls", 0)
            done = (
                counts.get("scraped", 0)
                + counts.get("failed", 0)
                + counts.get("auth_required", 0)
            )
            return {
                "session_id": session_id,
                "status": "completed" if total and done >= total else "not_started",
                "total": total,
                "completed": done,
                "success": counts.get("scraped", 0),
                "failed": counts.get("failed", 0),
                "auth_required": counts.get("auth_required", 0),
            }

    def get_pending_auth(self) -> dict:
        return self._request("GET", f"{self.backend_url}/auth/pending")

    def submit_credentials(self, domain: str, credentials: dict) -> dict:
        return self._request(
            "POST",
            f"{self.backend_url}/auth/credentials",
            params={"domain": domain},
            json=credentials,
        )

    # AI features
    def start_clustering(self, session_id: str) -> dict:
        return self._request(
            "POST", f"{self.backend_url}/cluster", json={"session_id": session_id}
        )

    def get_clusters(self, session_id: str) -> dict:
        return self._request("GET", f"{self.backend_url}/clusters/{session_id}")

    def chat(self, query: str, session_id: Optional[str] = None) -> dict:
        return self._request(
            "POST",
            f"{self.ai_url}/chat",
            json={"query": query, "session_id": session_id},
        )

    def search(self, query: str, session_id: Optional[str] = None) -> dict:
        return self._request(
            "POST",
            f"{self.ai_url}/search",
            json={"query": query, "session_id": session_id},
        )

    def summarize_session(self, session_id: str) -> dict:
        return self._request("GET", f"{self.ai_url}/summarize/{session_id}")

    # Settings / health / providers
    def get_providers(self) -> dict:
        return self._request("GET", f"{self.ai_url}/providers")

    def switch_provider(
        self,
        llm_provider: Optional[str] = None,
        embedding_provider: Optional[str] = None,
    ) -> dict:
        return self._request(
            "POST",
            f"{self.ai_url}/providers/switch",
            json={
                "llm_provider": llm_provider,
                "embedding_provider": embedding_provider,
            },
        )

    def export_session(self, session_id: str, export_format: str) -> dict:
        return self._request(
            "POST",
            f"{self.backend_url}/export",
            json={"session_id": session_id, "format": export_format},
        )

    def check_health(self) -> dict:
        health = {"backend": False, "ai_engine": False, "browser_engine": False}
        try:
            self._request("GET", f"{self.backend_url}/health")
            health["backend"] = True
        except Exception:
            pass

        try:
            self._request("GET", f"{self.ai_url}/health")
            health["ai_engine"] = True
        except Exception:
            pass

        try:
            self._request("GET", f"{self.browser_url}/health")
            health["browser_engine"] = True
        except Exception:
            pass

        return health
