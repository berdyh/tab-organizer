"""Synchronous HTTP client used by Streamlit pages."""

import os
from typing import Any, Optional

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
        self.ai_engine_token = os.getenv("AI_ENGINE_API_TOKEN", "").strip()
        self.timeout = float(os.getenv("UI_API_TIMEOUT", "30"))

    def _request(self, method: str, url: str, **kwargs):
        response = requests.request(method, url, timeout=self.timeout, **kwargs)
        response.raise_for_status()
        if response.content:
            return response.json()
        return {}

    def _platform_headers(self, token: Optional[str] = None) -> dict[str, str]:
        if not token:
            return {}
        return {"Authorization": f"Bearer {token}"}

    def _ai_headers(self) -> dict[str, str]:
        if not self.ai_engine_token:
            return {}
        return {"Authorization": f"Bearer {self.ai_engine_token}"}

    def _clean_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in payload.items() if value is not None}

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
    def start_scraping(self, session_id: str, use_browser: bool = False) -> dict:
        return self._request(
            "POST",
            f"{self.backend_url}/scrape",
            json={"session_id": session_id, "use_browser": use_browser},
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
            headers=self._ai_headers(),
        )

    def search(self, query: str, session_id: Optional[str] = None) -> dict:
        return self._request(
            "POST",
            f"{self.ai_url}/search",
            json={"query": query, "session_id": session_id},
            headers=self._ai_headers(),
        )

    def summarize_session(self, session_id: str) -> dict:
        return self._request(
            "GET",
            f"{self.ai_url}/summarize/{session_id}",
            headers=self._ai_headers(),
        )

    # Settings / health / providers
    def get_providers(self) -> dict:
        return self._request(
            "GET", f"{self.ai_url}/providers", headers=self._ai_headers()
        )

    def switch_provider(
        self,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> dict:
        return self._request(
            "POST",
            f"{self.ai_url}/providers/switch",
            json={
                **{
                    "llm_provider": llm_provider,
                    "embedding_provider": embedding_provider,
                },
                **({"llm_model": llm_model} if llm_model is not None else {}),
                **(
                    {"embedding_model": embedding_model}
                    if embedding_model is not None
                    else {}
                ),
            },
            headers=self._ai_headers(),
        )

    def update_ai_config(
        self,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
        api_keys: Optional[dict[str, str]] = None,
    ) -> dict:
        return self._request(
            "POST",
            f"{self.ai_url}/config",
            json={
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "embedding_provider": embedding_provider,
                "embedding_model": embedding_model,
                "api_keys": api_keys or {},
            },
            headers=self._ai_headers(),
        )

    def export_session(self, session_id: str, export_format: str) -> dict:
        return self._request(
            "POST",
            f"{self.backend_url}/export",
            json={"session_id": session_id, "format": export_format},
        )

    def check_health(self) -> dict:
        health = {
            "backend": False,
            "ai_engine": False,
            "browser_engine": False,
            "backend_detail": None,
            "ai_engine_detail": None,
            "browser_engine_detail": None,
        }
        try:
            detail = self._request("GET", f"{self.backend_url}/health")
            health["backend_detail"] = detail
            health["backend"] = detail.get("status") == "healthy"
        except Exception:
            pass

        try:
            detail = self._request(
                "GET", f"{self.ai_url}/health", headers=self._ai_headers()
            )
            health["ai_engine_detail"] = detail
            health["ai_engine"] = detail.get("status") == "healthy"
        except Exception:
            pass

        try:
            detail = self._request("GET", f"{self.browser_url}/health")
            health["browser_engine_detail"] = detail
            health["browser_engine"] = detail.get("status") == "healthy"
        except Exception:
            pass

        return health

    # Platform auth / company discovery / B2B / maintainer features
    def platform_signup(
        self,
        email: str,
        password: str,
        name: str,
        account_type: str,
        company_name: Optional[str] = None,
        role: Optional[str] = None,
        maintainer_code: Optional[str] = None,
    ) -> dict:
        return self._request(
            "POST",
            f"{self.backend_url}/platform/auth/signup",
            json=self._clean_payload(
                {
                    "email": email,
                    "password": password,
                    "name": name,
                    "account_type": account_type,
                    "company_name": company_name,
                    "role": role,
                    "maintainer_code": maintainer_code,
                }
            ),
        )

    def platform_login(self, email: str, password: str) -> dict:
        return self._request(
            "POST",
            f"{self.backend_url}/platform/auth/login",
            json={"email": email, "password": password},
        )

    def platform_me(self, token: Optional[str] = None) -> dict:
        return self._request(
            "GET",
            f"{self.backend_url}/platform/me",
            headers=self._platform_headers(token),
        )

    def search_companies(
        self,
        query: str,
        limit: int = 10,
        token: Optional[str] = None,
    ) -> dict | list[dict]:
        return self._request(
            "GET",
            f"{self.backend_url}/platform/companies/search",
            params={"q": query, "limit": limit},
            headers=self._platform_headers(token),
        )

    def get_company(
        self,
        company_id: str,
        token: Optional[str] = None,
    ) -> dict:
        return self._request(
            "GET",
            f"{self.backend_url}/platform/companies/{company_id}",
            headers=self._platform_headers(token),
        )

    def create_b2b_token(
        self,
        name: str,
        scopes: Optional[list[str]] = None,
        token: Optional[str] = None,
    ) -> dict:
        return self._request(
            "POST",
            f"{self.backend_url}/platform/b2b/tokens",
            json=self._clean_payload({"name": name, "scopes": scopes}),
            headers=self._platform_headers(token),
        )

    def list_b2b_tokens(self, token: Optional[str] = None) -> dict | list[dict]:
        return self._request(
            "GET",
            f"{self.backend_url}/platform/b2b/tokens",
            headers=self._platform_headers(token),
        )

    def revoke_b2b_token(self, token_id: str, token: Optional[str] = None) -> dict:
        return self._request(
            "DELETE",
            f"{self.backend_url}/platform/b2b/tokens/{token_id}",
            headers=self._platform_headers(token),
        )

    def get_b2b_first_call(self, token: Optional[str] = None) -> dict:
        return self._request(
            "GET",
            f"{self.backend_url}/platform/b2b/first-call",
            headers=self._platform_headers(token),
        )

    def search_companies_with_api_token(
        self,
        query: str,
        api_token: str,
        limit: int = 10,
    ) -> dict | list[dict]:
        return self._request(
            "GET",
            f"{self.backend_url}/platform/v1/companies/search",
            params={"query": query, "limit": limit},
            headers=self._platform_headers(api_token),
        )

    def get_platform_dashboard(self, token: Optional[str] = None) -> dict:
        return self._request(
            "GET",
            f"{self.backend_url}/platform/dashboard",
            headers=self._platform_headers(token),
        )

    def get_maintainer_issues(
        self,
        status: Optional[str] = None,
        token: Optional[str] = None,
    ) -> dict | list[dict]:
        params = {"status": status} if status else None
        return self._request(
            "GET",
            f"{self.backend_url}/platform/maintainer/issues",
            params=params,
            headers=self._platform_headers(token),
        )
