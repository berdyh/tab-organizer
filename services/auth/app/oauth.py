"""OAuth 2.0 flow handling."""

import secrets
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import aiohttp
import structlog
from fastapi import HTTPException

from .models import OAuthConfig


logger = structlog.get_logger()


class OAuthFlowHandler:
    """Handles OAuth 2.0 authentication flows."""

    def __init__(self) -> None:
        self.oauth_configs: Dict[str, OAuthConfig] = {}
        self.active_flows: Dict[str, Dict[str, Any]] = {}

    def register_oauth_provider(self, provider: str, config: OAuthConfig) -> None:
        """Register an OAuth provider configuration."""
        self.oauth_configs[provider] = config
        logger.info("OAuth provider registered", provider=provider)

    def initiate_oauth_flow(self, provider: str, state: Optional[str] = None) -> Dict[str, Any]:
        """Initiate OAuth 2.0 authorization flow."""
        if provider not in self.oauth_configs:
            raise HTTPException(status_code=400, detail=f"OAuth provider {provider} not configured")

        config = self.oauth_configs[provider]

        if not state:
            state = secrets.token_urlsafe(32)

        auth_params = {
            "client_id": config.client_id,
            "redirect_uri": config.redirect_uri,
            "scope": " ".join(config.scope) if config.scope else "",
            "state": state,
            "response_type": "code",
        }
        auth_params.update(config.additional_params)

        auth_url = f"{config.authorization_url}?" + "&".join([f"{k}={v}" for k, v in auth_params.items() if v])

        flow_id = str(uuid.uuid4())
        self.active_flows[flow_id] = {
            "provider": provider,
            "state": state,
            "created_at": datetime.now(),
            "config": config,
        }

        return {"flow_id": flow_id, "authorization_url": auth_url, "state": state}

    async def complete_oauth_flow(self, flow_id: str, authorization_code: str, state: str) -> Dict[str, Any]:
        """Complete OAuth 2.0 flow by exchanging code for tokens."""
        if flow_id not in self.active_flows:
            raise HTTPException(status_code=400, detail="Invalid or expired OAuth flow")

        flow = self.active_flows[flow_id]

        if flow["state"] != state:
            raise HTTPException(status_code=400, detail="Invalid state parameter")

        config = flow["config"]
        token_data = {
            "grant_type": "authorization_code",
            "client_id": config.client_id,
            "client_secret": config.client_secret,
            "code": authorization_code,
            "redirect_uri": config.redirect_uri,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(config.token_url, data=token_data) as response:
                if response.status == 200:
                    tokens = await response.json()
                    del self.active_flows[flow_id]
                    return {"success": True, "tokens": tokens, "provider": flow["provider"]}

                error_text = await response.text()
                logger.error("OAuth token exchange failed", status=response.status, error=error_text)
                raise HTTPException(status_code=400, detail="Token exchange failed")


__all__ = ["OAuthFlowHandler"]
