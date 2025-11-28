"""Domain-level authentication mapping endpoints."""

from datetime import datetime
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..dependencies import get_state
from ..models import AuthenticationRequirement
from ..state import AuthServiceState


logger = structlog.get_logger()

router = APIRouter(prefix="/api/auth", tags=["domains"])


class ManualDomainLearningRequest(BaseModel):
    domain: str
    auth_method: str
    requires_auth: bool
    login_url: Optional[str] = None


@router.get("/domains")
async def list_domains_with_auth(state: AuthServiceState = Depends(get_state)) -> dict:
    """List all domains with authentication requirements or stored credentials."""
    try:
        stored_domains = set(state.credential_store.list_stored_domains())
        mapped_domains = set(state.domain_mapper.get_all_mappings().keys())
        all_domains = stored_domains.union(mapped_domains)

        domain_list = []
        for domain in sorted(all_domains):
            domain_info = state.domain_mapper.get_domain_auth_info(domain)
            domain_list.append(
                {
                    "domain": domain,
                    "has_credentials": domain in stored_domains,
                    "requires_auth": domain_info.requires_auth if domain_info else False,
                    "auth_method": domain_info.auth_method if domain_info else "unknown",
                    "success_count": domain_info.success_count if domain_info else 0,
                    "failure_count": domain_info.failure_count if domain_info else 0,
                    "last_verified": domain_info.last_verified.isoformat() if domain_info and domain_info.last_verified else None,
                    "login_url": domain_info.login_url if domain_info else None,
                }
            )

        return {"domains": domain_list, "total_count": len(domain_list)}
    except Exception as exc:  # pragma: no cover - routing layer
        logger.error("Domain listing failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Listing failed: {str(exc)}") from exc


@router.post("/learn-domain")
async def learn_domain_authentication(
    request: ManualDomainLearningRequest, state: AuthServiceState = Depends(get_state)
) -> dict:
    """Manually teach the system about a domain's authentication requirements."""
    try:
        auth_req = AuthenticationRequirement(
            url=f"https://{request.domain}",
            domain=request.domain,
            detected_method=request.auth_method,
            auth_indicators=["Manual learning"],
            detection_confidence=1.0 if request.requires_auth else 0.0,
        )

        state.domain_mapper.learn_domain_auth(request.domain, auth_req)
        mapping = state.domain_mapper.get_domain_auth_info(request.domain)
        if mapping and request.login_url:
            mapping.login_url = request.login_url
            mapping.last_verified = datetime.now()

        return {
            "success": True,
            "message": f"Domain {request.domain} learned with auth method: {request.auth_method}",
            "requires_auth": request.requires_auth,
        }
    except Exception as exc:  # pragma: no cover - routing layer
        logger.error("Domain learning failed", domain=request.domain, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Learning failed: {str(exc)}") from exc


@router.post("/mark-success/{domain}")
async def mark_authentication_success(domain: str, state: AuthServiceState = Depends(get_state)) -> dict:
    """Mark successful authentication for a domain."""
    try:
        state.domain_mapper.mark_auth_success(domain)
        return {"success": True, "message": f"Authentication success recorded for {domain}"}
    except Exception as exc:  # pragma: no cover - routing layer
        logger.error("Failed to mark auth success", domain=domain, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Failed to mark success: {str(exc)}") from exc


@router.post("/mark-failure/{domain}")
async def mark_authentication_failure(domain: str, state: AuthServiceState = Depends(get_state)) -> dict:
    """Mark failed authentication for a domain."""
    try:
        state.domain_mapper.mark_auth_failure(domain)
        return {"success": True, "message": f"Authentication failure recorded for {domain}"}
    except Exception as exc:  # pragma: no cover - routing layer
        logger.error("Failed to mark auth failure", domain=domain, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Failed to mark failure: {str(exc)}") from exc


@router.get("/domain-mapping/{domain}")
async def get_domain_mapping(domain: str, state: AuthServiceState = Depends(get_state)) -> dict:
    """Get detailed authentication mapping for a specific domain."""
    try:
        mapping = state.domain_mapper.get_domain_auth_info(domain)

        if not mapping:
            raise HTTPException(status_code=404, detail=f"No authentication mapping found for {domain}")

        return {
            "domain": mapping.domain,
            "auth_method": mapping.auth_method,
            "requires_auth": mapping.requires_auth,
            "login_url": mapping.login_url,
            "form_selectors": mapping.form_selectors,
            "oauth_config": mapping.oauth_config,
            "last_verified": mapping.last_verified.isoformat() if mapping.last_verified else None,
            "success_count": mapping.success_count,
            "failure_count": mapping.failure_count,
        }
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - routing layer
        logger.error("Domain mapping retrieval failed", domain=domain, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Mapping retrieval failed: {str(exc)}") from exc


__all__ = ["router"]
