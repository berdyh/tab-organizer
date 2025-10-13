"""API routers for the authentication service."""

from fastapi import APIRouter

from . import credentials, detection, domains, interactive, queue, sessions, status


router = APIRouter()

router.include_router(status.router)
router.include_router(detection.router)
router.include_router(credentials.router)
router.include_router(domains.router)
router.include_router(interactive.router)
router.include_router(sessions.router)
router.include_router(queue.router)


__all__ = ["router"]
