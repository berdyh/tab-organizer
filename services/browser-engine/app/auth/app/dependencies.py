"""FastAPI dependency helpers."""

from functools import lru_cache

from .state import AuthServiceState, build_state


@lru_cache(maxsize=1)
def get_state() -> AuthServiceState:
    """Return the singleton service state."""
    return build_state()


__all__ = ["get_state"]
