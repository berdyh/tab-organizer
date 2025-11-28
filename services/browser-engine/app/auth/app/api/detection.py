"""Detection-related endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_state
from ..models import AuthDetectionResponse, URLAnalysisRequest
from ..state import AuthServiceState


router = APIRouter(prefix="/api/auth", tags=["detection"])


@router.post("/detect", response_model=AuthDetectionResponse)
async def detect_authentication_requirement(
    request: URLAnalysisRequest, state: AuthServiceState = Depends(get_state)
) -> AuthDetectionResponse:
    """Detect if a URL requires authentication based on response analysis."""
    try:
        auth_req = await state.detector.detect_auth_required(
            url=str(request.url),
            response_content=request.response_content,
            status_code=request.status_code,
            headers=request.headers,
        )

        state.domain_mapper.learn_domain_auth(auth_req.domain, auth_req)

        if auth_req.detection_confidence > 0.7:
            recommended_action = "setup_authentication"
        elif auth_req.detection_confidence > 0.4:
            recommended_action = "manual_verification_needed"
        else:
            recommended_action = "no_authentication_required"

        return AuthDetectionResponse(
            requires_auth=auth_req.detection_confidence > 0.5,
            detected_method=auth_req.detected_method,
            confidence=auth_req.detection_confidence,
            indicators=auth_req.auth_indicators,
            recommended_action=recommended_action,
        )
    except Exception as exc:  # pragma: no cover - routing layer
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(exc)}") from exc


__all__ = ["router"]
