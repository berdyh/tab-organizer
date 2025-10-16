"""Service proxy endpoint for forwarding API calls to downstream services."""

from typing import Dict, Set

import httpx
import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from ..dependencies import get_service_registry
from ..routing import resolve_service_alias

logger = structlog.get_logger()
router = APIRouter()

HOP_BY_HOP_HEADERS: Set[str] = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


@router.api_route("/api/{service_name}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_to_service(
    service_name: str,
    path: str,
    request: Request,
    service_registry=Depends(get_service_registry),
):
    """
    Proxy requests to appropriate microservices.
    Handles routing, load balancing, and error handling.
    """
    resolved_name, alias_path = resolve_service_alias(service_name)
    service = service_registry.get_service(resolved_name)
    if not service:
        raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found")

    if not service.get("healthy", False):
        raise HTTPException(status_code=503, detail=f"Service '{service_name}' is unhealthy")

    base_url = service["url"].rstrip("/")
    path_prefix = str(service.get("base_path", "")).strip("/")
    alias_prefix = alias_path.strip("/")
    request_path = path.lstrip("/")
    combined_path_parts = [part for part in (path_prefix, alias_prefix, request_path) if part]
    combined_path = "/".join(combined_path_parts)
    target_url = f"{base_url}/{combined_path}" if combined_path else base_url

    body = await request.body()
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(
                method=request.method,
                url=target_url,
                content=body,
                headers=headers,
                params=request.query_params,
            )

            content_type = response.headers.get("content-type", "")
            content = (
                response.json()
                if content_type.startswith("application/json")
                else response.text
            )

            # Include upstream headers but avoid hop-by-hop ones
            filtered_headers: Dict[str, str] = {
                key: value
                for key, value in response.headers.items()
                if key.lower() not in HOP_BY_HOP_HEADERS
            }

            return JSONResponse(
                content=content,
                status_code=response.status_code,
                headers=filtered_headers,
            )

    except httpx.TimeoutException:
        logger.error("Service request timeout", service=service_name, path=path)
        raise HTTPException(
            status_code=504, detail=f"Service '{service_name}' request timeout"
        )
    except httpx.ConnectError:
        logger.error("Service connection error", service=service_name, path=path)
        raise HTTPException(
            status_code=503, detail=f"Cannot connect to service '{service_name}'"
        )
    except Exception as exc:
        logger.error(
            "Service proxy error", service=service_name, path=path, error=str(exc)
        )
        raise HTTPException(
            status_code=502, detail=f"Service '{service_name}' error: {exc}"
        ) from exc

