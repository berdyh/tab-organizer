"""Authentication session management endpoints."""

from datetime import datetime, timedelta

import structlog
from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_state
from ..state import AuthServiceState


logger = structlog.get_logger()

router = APIRouter(prefix="/api/auth", tags=["sessions"])


@router.get("/session/{session_id}")
async def get_session_info(session_id: str, state: AuthServiceState = Depends(get_state)) -> dict:
    """Retrieve details for a specific authentication session."""
    try:
        session = state.interactive_auth.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found or expired")

        return {
            "session_id": session.session_id,
            "domain": session.domain,
            "auth_method": session.auth_method,
            "created_at": session.created_at.isoformat(),
            "expires_at": session.expires_at.isoformat() if session.expires_at else None,
            "last_used": session.last_used.isoformat(),
            "is_active": session.is_active,
            "has_cookies": len(session.cookies) > 0,
            "has_tokens": len(session.tokens) > 0,
        }
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - routing layer
        logger.error("Session info retrieval failed", session_id=session_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Session retrieval failed: {str(exc)}") from exc


@router.post("/session/{session_id}/renew")
async def renew_session(session_id: str, state: AuthServiceState = Depends(get_state)) -> dict:
    """Renew an authentication session."""
    try:
        session = state.interactive_auth.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found or expired")

        session.expires_at = datetime.now() + timedelta(hours=24)
        session.last_used = datetime.now()

        return {
            "success": True,
            "session_id": session_id,
            "new_expires_at": session.expires_at.isoformat() if session.expires_at else None,
            "message": "Session renewed successfully",
        }
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - routing layer
        logger.error("Session renewal failed", session_id=session_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Session renewal failed: {str(exc)}") from exc


@router.delete("/session/{session_id}")
async def invalidate_session(session_id: str, state: AuthServiceState = Depends(get_state)) -> dict:
    """Invalidate an authentication session."""
    try:
        success = state.interactive_auth.invalidate_session(session_id)

        if success:
            return {"success": True, "message": f"Session {session_id} invalidated successfully"}
        raise HTTPException(status_code=404, detail="Session not found")
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - routing layer
        logger.error("Session invalidation failed", session_id=session_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Session invalidation failed: {str(exc)}") from exc


@router.get("/sessions")
async def list_active_sessions(state: AuthServiceState = Depends(get_state)) -> dict:
    """List all active authentication sessions."""
    try:
        sessions = []
        for session_id, session in state.interactive_auth.active_sessions.items():
            if session.is_active:
                sessions.append(
                    {
                        "session_id": session_id,
                        "domain": session.domain,
                        "auth_method": session.auth_method,
                        "created_at": session.created_at.isoformat(),
                        "expires_at": session.expires_at.isoformat() if session.expires_at else None,
                        "last_used": session.last_used.isoformat(),
                    }
                )

        return {"sessions": sessions, "total_count": len(sessions)}
    except Exception as exc:  # pragma: no cover - routing layer
        logger.error("Session listing failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Session listing failed: {str(exc)}") from exc


@router.post("/sessions/cleanup")
async def cleanup_expired_sessions(state: AuthServiceState = Depends(get_state)) -> dict:
    """Clean up expired authentication sessions."""
    try:
        initial_count = len(state.interactive_auth.active_sessions)
        state.interactive_auth.cleanup_expired_sessions()
        final_count = len(state.interactive_auth.active_sessions)
        cleaned_count = initial_count - final_count

        return {
            "success": True,
            "cleaned_sessions": cleaned_count,
            "remaining_sessions": final_count,
            "message": f"Cleaned up {cleaned_count} expired sessions",
        }
    except Exception as exc:  # pragma: no cover - routing layer
        logger.error("Session cleanup failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Session cleanup failed: {str(exc)}") from exc


__all__ = ["router"]
