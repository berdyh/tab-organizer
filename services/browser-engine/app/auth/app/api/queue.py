"""Authentication queue status endpoints."""

import structlog
from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_state
from ..state import AuthServiceState


logger = structlog.get_logger()

router = APIRouter(prefix="/api/auth", tags=["queue"])


@router.get("/queue/status")
async def get_queue_status(state: AuthServiceState = Depends(get_state)) -> dict:
    """Get authentication queue status."""
    try:
        return {
            "active_tasks": len(state.auth_queue.active_tasks),
            "completed_tasks": len(state.auth_queue.completed_tasks),
            "queue_size": state.auth_queue.task_queue.qsize(),
            "max_workers": state.auth_queue.max_workers,
            "is_running": state.auth_queue._running,
        }
    except Exception as exc:  # pragma: no cover - routing layer
        logger.error("Queue status retrieval failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Queue status failed: {str(exc)}") from exc


__all__ = ["router"]
