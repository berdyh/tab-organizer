"""Service discovery endpoints."""

import time
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import authenticate, get_service_registry
from ..routing import resolve_service_alias

router = APIRouter()


@router.get("/services")
async def list_services(
    auth_info: Dict[str, Any] = Depends(authenticate),
    service_registry=Depends(get_service_registry),
):
    """List all registered services and their status."""
    services = service_registry.get_all_services()
    return {"services": services, "timestamp": time.time()}


@router.get("/services/{service_name}")
async def get_service_info(
    service_name: str,
    auth_info: Dict[str, Any] = Depends(authenticate),
    service_registry=Depends(get_service_registry),
):
    """Get detailed information about a specific service."""
    resolved_name, alias_path = resolve_service_alias(service_name)
    service = service_registry.get_service(resolved_name)
    if not service:
        raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found")

    response: Dict[str, Any] = {"service": service, "timestamp": time.time()}
    if resolved_name != service_name:
        response["resolved_name"] = resolved_name
        if alias_path:
            response["alias_path_prefix"] = alias_path
    return response

