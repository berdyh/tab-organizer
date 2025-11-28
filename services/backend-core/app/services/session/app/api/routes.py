"""FastAPI routes for the Session service."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from .. import session_service
from ..models import (
    CreateSessionRequest,
    MergeSessionsRequest,
    RetentionPolicy,
    SessionExportData,
    SessionModel,
    SessionStatus,
    ShareSessionRequest,
    SplitSessionRequest,
    UpdateSessionRequest,
)


def get_session_service():
    return session_service


router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    from time import time as _now  # local import to avoid module-level time dependency

    return {"status": "healthy", "service": "session", "timestamp": _now()}


@router.get("/")
async def root() -> Dict[str, str]:
    return {"service": "Session Management Service", "version": "1.0.0", "status": "running"}


@router.post("/sessions", response_model=SessionModel)
async def create_session(
    request: CreateSessionRequest,
    service = Depends(get_session_service),
):
    return service.create_session(request)


@router.get("/sessions", response_model=List[SessionModel])
async def list_sessions(
    status_filter: Optional[SessionStatus] = Query(default=None, alias="status"),
    owner_id: Optional[str] = None,
    tags: Optional[str] = None,
    service = Depends(get_session_service),
):
    tag_list = [tag.strip() for tag in tags.split(",")] if tags else None
    return service.list_sessions(status=status_filter, owner_id=owner_id, tags=tag_list)


@router.get("/sessions/compare")
async def compare_sessions(session_ids: str, service = Depends(get_session_service)):
    try:
        ids = [sid.strip() for sid in session_ids.split(",") if sid.strip()]
        return service.compare_sessions(ids)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc


@router.get("/sessions/{session_id}", response_model=SessionModel)
async def get_session(session_id: str, service = Depends(get_session_service)):
    try:
        return service.get_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc


@router.put("/sessions/{session_id}", response_model=SessionModel)
async def update_session(
    session_id: str,
    request: UpdateSessionRequest,
    service = Depends(get_session_service),
):
    try:
        return service.update_session(session_id, request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    permanent: bool = False,
    service = Depends(get_session_service),
):
    try:
        return service.delete_session(session_id, permanent=permanent)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc


@router.post("/sessions/{session_id}/archive")
async def archive_session(session_id: str, service = Depends(get_session_service)):
    try:
        return service.archive_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc


@router.post("/sessions/{session_id}/restore")
async def restore_session(session_id: str, service = Depends(get_session_service)):
    try:
        return service.restore_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc


@router.post("/sessions/{session_id}/share", response_model=SessionModel)
async def share_session(
    session_id: str,
    request: ShareSessionRequest,
    service = Depends(get_session_service),
):
    try:
        return service.share_session(session_id, request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc


@router.delete("/sessions/{session_id}/share/{user_id}")
async def unshare_session(
    session_id: str,
    user_id: str,
    service = Depends(get_session_service),
):
    try:
        return service.unshare_session(session_id, user_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc


@router.get("/sessions/{session_id}/collaborators")
async def get_collaborators(session_id: str, service = Depends(get_session_service)):
    try:
        return service.get_collaborators(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc


@router.post("/sessions/{session_id}/incremental-clustering")
async def trigger_incremental_clustering(
    session_id: str,
    new_content_ids: List[str],
    service = Depends(get_session_service),
):
    try:
        return service.trigger_incremental_clustering(session_id, new_content_ids)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc


@router.post("/sessions/cleanup")
async def cleanup_sessions(
    policy: RetentionPolicy,
    service = Depends(get_session_service),
):
    return service.cleanup_sessions(policy)


@router.get("/sessions/retention-policy")
async def get_retention_policy(service = Depends(get_session_service)):
    return service.get_retention_policy()


@router.put("/sessions/retention-policy")
async def update_retention_policy(
    policy: RetentionPolicy,
    service = Depends(get_session_service),
):
    return service.update_retention_policy(policy)


@router.get("/sessions/{session_id}/export", response_model=SessionExportData)
async def export_session(
    session_id: str,
    include_data: bool = Query(True, description="Include Qdrant vectors in export"),
    service = Depends(get_session_service),
):
    try:
        return service.export_session(session_id, include_data=include_data)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc


@router.post("/sessions/import", response_model=SessionModel)
async def import_session(
    export_data: SessionExportData,
    new_name: Optional[str] = Query(None, description="Override session name when importing"),
    service = Depends(get_session_service),
):
    return service.import_session(export_data, new_name=new_name)


@router.post("/sessions/merge", response_model=SessionModel)
async def merge_sessions(request: MergeSessionsRequest, service = Depends(get_session_service)):
    try:
        return service.merge_sessions(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="One or more sessions not found") from exc


@router.post("/sessions/{session_id}/split")
async def split_session(session_id: str, request: SplitSessionRequest, service = Depends(get_session_service)):
    try:
        return service.split_session(session_id, request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc


@router.get("/sessions/{session_id}/stats")
async def get_session_stats(session_id: str, service = Depends(get_session_service)):
    try:
        return service.get_session_stats(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc


@router.put("/sessions/{session_id}/stats")
async def update_session_stats(
    session_id: str,
    stats_update: Dict[str, Any],
    service = Depends(get_session_service),
):
    try:
        return service.update_session_stats(session_id, stats_update)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc


@router.put("/sessions/{session_id}/model-usage")
async def update_model_usage(
    session_id: str,
    model_type: str,
    model_name: str,
    service = Depends(get_session_service),
):
    try:
        return service.update_model_usage(session_id, model_type, model_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


__all__ = ["router"]
