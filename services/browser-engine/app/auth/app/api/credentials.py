"""Credential management endpoints."""

import structlog
from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_state
from ..models import CredentialStoreRequest
from ..state import AuthServiceState


logger = structlog.get_logger()

router = APIRouter(prefix="/api/auth", tags=["credentials"])


@router.post("/store-credentials")
async def store_domain_credentials(
    request: CredentialStoreRequest, state: AuthServiceState = Depends(get_state)
) -> dict:
    """Store encrypted credentials for a domain."""
    try:
        success = state.credential_store.store_credentials(request.domain, request.credentials)

        if success:
            mapping = state.domain_mapper.get_domain_auth_info(request.domain)
            if mapping:
                mapping.auth_method = request.auth_method
                mapping.login_url = request.login_url
                mapping.requires_auth = True

            return {"success": True, "message": "Credentials stored securely"}
        raise HTTPException(status_code=500, detail="Failed to store credentials")
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - routing layer
        logger.error("Credential storage failed", domain=request.domain, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Storage failed: {str(exc)}") from exc


@router.get("/credentials/{domain}")
async def get_domain_credentials(domain: str, state: AuthServiceState = Depends(get_state)) -> dict:
    """Retrieve stored credentials metadata for a domain."""
    try:
        has_credentials = domain in state.credential_store.list_stored_domains()
        domain_info = state.domain_mapper.get_domain_auth_info(domain)

        return {
            "domain": domain,
            "has_stored_credentials": has_credentials,
            "auth_method": domain_info.auth_method if domain_info else None,
            "requires_auth": domain_info.requires_auth if domain_info else False,
            "last_verified": domain_info.last_verified.isoformat() if domain_info and domain_info.last_verified else None,
        }
    except Exception as exc:  # pragma: no cover - routing layer
        logger.error("Credential retrieval failed", domain=domain, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(exc)}") from exc


@router.delete("/credentials/{domain}")
async def delete_domain_credentials(domain: str, state: AuthServiceState = Depends(get_state)) -> dict:
    """Delete stored credentials for a domain."""
    try:
        success = state.credential_store.delete_credentials(domain)

        if success:
            return {"success": True, "message": f"Credentials deleted for {domain}"}
        return {"success": False, "message": f"No credentials found for {domain}"}
    except Exception as exc:  # pragma: no cover - routing layer
        logger.error("Credential deletion failed", domain=domain, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(exc)}") from exc


__all__ = ["router"]
