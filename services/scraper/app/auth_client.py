"""Client for interacting with the authentication service."""

from __future__ import annotations

from typing import Any, Dict, Optional

import aiohttp

from .config import Settings
from .logging import get_logger
from .models import AuthSession

logger = get_logger()


class AuthenticationServiceClient:
    """Client for communicating with the Authentication Service."""

    def __init__(
        self,
        auth_service_url: Optional[str] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialise client with optional explicit URL or settings."""
        if auth_service_url and settings:
            raise ValueError("Provide either auth_service_url or settings, not both.")

        self.settings = settings or Settings()
        if auth_service_url:
            self.auth_service_url = auth_service_url.rstrip("/")
        else:
            self.auth_service_url = str(self.settings.auth_service_url).rstrip("/")
        self.session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session

    async def get_session_for_domain(
        self, domain: str, correlation_id: str
    ) -> Optional[AuthSession]:
        """Get active authentication session for a domain."""
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.auth_service_url}/sessions/{domain}",
                headers={"X-Correlation-ID": correlation_id},
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return AuthSession(**data)
                if response.status == 404:
                    logger.info("No active session found for domain", domain=domain)
                    return None

                logger.warning(
                    "Failed to get session", domain=domain, status=response.status
                )
                return None
        except Exception as exc:
            logger.error(
                "Error getting session from auth service",
                domain=domain,
                error=str(exc),
            )
            return None

    async def check_auth_required(
        self,
        url: str,
        response_content: str,
        status_code: int,
        headers: Dict[str, str],
        correlation_id: str,
    ) -> Dict[str, Any]:
        """Check if URL requires authentication."""
        payload = {
            "url": url,
            "response_content": response_content,
            "status_code": status_code,
            "headers": headers,
        }

        try:
            session = await self._get_session()
            async with session.post(
                f"{self.auth_service_url}/detect-auth",
                json=payload,
                headers={"X-Correlation-ID": correlation_id},
            ) as response:
                if response.status == 200:
                    return await response.json()

                logger.warning(
                    "Auth detection failed", url=url, status=response.status
                )
                return {"requires_auth": False, "confidence": 0.0}
        except Exception as exc:
            logger.error(
                "Error checking auth requirements", url=url, error=str(exc)
            )
            return {"requires_auth": False, "confidence": 0.0}

    async def close(self) -> None:
        """Close the HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
