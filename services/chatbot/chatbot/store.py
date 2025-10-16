"""Conversation state management."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Dict, Iterable, Iterator, List, Sequence

from .models import ConversationEntry


class ConversationStore(MutableMapping[str, List[ConversationEntry]]):
    """In-memory storage for chat transcripts."""

    def __init__(self) -> None:
        self._conversations: Dict[str, List[ConversationEntry]] = {}

    def append(self, session_id: str, entry: ConversationEntry) -> None:
        """Add a new conversation entry for the session."""
        history = self._conversations.setdefault(session_id, [])
        history.append(entry)

    def get(self, session_id: str) -> List[ConversationEntry]:
        """Return the history for a session."""
        return list(self._conversations.get(session_id, []))

    def clear(self, session_id: str) -> None:
        """Remove a session's history."""
        self._conversations.pop(session_id, None)

    def sessions(self) -> Iterable[str]:
        """Iterate over known session identifiers."""
        return tuple(self._conversations.keys())

    # MutableMapping interface for legacy compatibility -----------------

    def __getitem__(self, key: str) -> List[ConversationEntry]:
        return self._conversations[key]

    def __setitem__(self, key: str, value: Sequence[ConversationEntry]) -> None:
        entries: List[ConversationEntry] = []
        for item in value:
            if isinstance(item, ConversationEntry):
                entries.append(item)
            else:
                entries.append(ConversationEntry(**item))
        self._conversations[key] = entries

    def __delitem__(self, key: str) -> None:
        del self._conversations[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._conversations)

    def __len__(self) -> int:
        return len(self._conversations)

