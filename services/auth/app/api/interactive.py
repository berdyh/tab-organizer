"""Interactive authentication endpoints."""

import structlog
from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_state
from ..models import AuthTaskResponse, InteractiveAuthRequest
from ..state import AuthServiceState


logger = structlog.get_logger()

router = APIRouter(prefix="/api/auth", tags=["interactive"])


@router.post("/interactive", response_model=AuthTaskResponse)
async def authenticate_interactively(
    request: InteractiveAuthRequest, state: AuthServiceState = Depends(get_state)
) -> AuthTaskResponse:
    """Queue an interactive authentication task using browser automation."""
    try:
        task_id = state.auth_queue.queue_authentication(
            domain=request.domain,
            auth_method=request.auth_method,
            credentials=request.credentials,
            login_url=request.login_url,
            priority=1,
        )

        return AuthTaskResponse(
            task_id=task_id,
            status="queued",
            message=f"Authentication task queued for {request.domain}",
        )
    except Exception as exc:  # pragma: no cover - routing layer
        logger.error("Interactive authentication request failed", domain=request.domain, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Authentication request failed: {str(exc)}") from exc


@router.get("/task/{task_id}")
async def get_authentication_task_status(task_id: str, state: AuthServiceState = Depends(get_state)) -> dict:
    """Get the status of an authentication task."""
    try:
        task = state.auth_queue.get_task_status(task_id)

        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        response_data = {
            "task_id": task.task_id,
            "domain": task.domain,
            "auth_method": task.auth_method,
            "status": task.status,
            "created_at": task.created_at.isoformat(),
            "error_message": task.error_message,
        }

        if task.result:
            response_data["result"] = task.result
            if task.result.get("session_id"):
                response_data["session_id"] = task.result["session_id"]

        return response_data
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - routing layer
        logger.error("Task status retrieval failed", task_id=task_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Task status failed: {str(exc)}") from exc


__all__ = ["router"]
