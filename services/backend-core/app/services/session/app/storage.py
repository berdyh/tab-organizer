"""In-memory persistence layer for sessions."""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, Optional

from .models import SessionModel


class SessionRepository:
    """Simple repository abstraction around an in-memory dictionary."""

    def __init__(self) -> None:
        self._sessions: Dict[str, SessionModel] = {}

    @property
    def data(self) -> Dict[str, SessionModel]:
        """Direct access to the underlying storage (compatibility)."""
        return self._sessions

    def all(self) -> Iterable[SessionModel]:
        return self._sessions.values()

    def items(self) -> Iterator[tuple[str, SessionModel]]:
        return iter(self._sessions.items())

    def items_for_session(self, session_id: str) -> Iterator[tuple[str, SessionModel]]:
        for key, session in self._sessions.items():
            if session.id == session_id:
                yield key, session

    def get(self, session_id: str) -> Optional[SessionModel]:
        return self._sessions.get(session_id)

    def upsert(self, session: SessionModel) -> None:
        self._sessions[session.id] = session

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def clear(self) -> None:
        self._sessions.clear()

    def exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def __len__(self) -> int:  # pragma: no cover - convenience
        return len(self._sessions)


session_repository = SessionRepository()


__all__ = ["SessionRepository", "session_repository"]
