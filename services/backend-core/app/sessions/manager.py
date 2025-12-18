"""Session management for organizing URLs into collections."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from ..url_input.store import URLStore, URLRecord


@dataclass
class Session:
    """A session represents a collection of URLs being organized."""
    id: str
    name: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    url_store: URLStore = field(default_factory=URLStore)
    clusters: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    status: str = "active"  # active, archived, deleted


class SessionManager:
    """Manage multiple sessions."""
    
    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._current_session_id: Optional[str] = None
    
    def create_session(self, name: Optional[str] = None) -> Session:
        """Create a new session."""
        session_id = str(uuid.uuid4())
        session_name = name or f"Session {len(self._sessions) + 1}"
        
        session = Session(id=session_id, name=session_name)
        self._sessions[session_id] = session
        
        if self._current_session_id is None:
            self._current_session_id = session_id
        
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        return self._sessions.get(session_id)
    
    def get_current_session(self) -> Optional[Session]:
        """Get the current active session."""
        if self._current_session_id:
            return self._sessions.get(self._current_session_id)
        return None
    
    def set_current_session(self, session_id: str) -> bool:
        """Set the current active session."""
        if session_id in self._sessions:
            self._current_session_id = session_id
            return True
        return False
    
    def list_sessions(self, include_archived: bool = False) -> list[Session]:
        """List all sessions."""
        sessions = list(self._sessions.values())
        if not include_archived:
            sessions = [s for s in sessions if s.status != "archived"]
        return sorted(sessions, key=lambda s: s.updated_at, reverse=True)
    
    def update_session(
        self, 
        session_id: str, 
        name: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> Optional[Session]:
        """Update session properties."""
        session = self._sessions.get(session_id)
        if not session:
            return None
        
        if name:
            session.name = name
        if metadata:
            session.metadata.update(metadata)
        session.updated_at = datetime.utcnow()
        
        return session
    
    def archive_session(self, session_id: str) -> bool:
        """Archive a session."""
        session = self._sessions.get(session_id)
        if not session:
            return False
        
        session.status = "archived"
        session.updated_at = datetime.utcnow()
        
        if self._current_session_id == session_id:
            self._current_session_id = None
        
        return True
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session permanently."""
        if session_id not in self._sessions:
            return False
        
        del self._sessions[session_id]
        
        if self._current_session_id == session_id:
            self._current_session_id = None
        
        return True
    
    def add_urls_to_session(
        self, 
        session_id: str, 
        urls: list[str]
    ) -> tuple[int, int, list[URLRecord]]:
        """
        Add URLs to a session.
        
        Returns:
            Tuple of (added_count, duplicate_count, new_records)
        """
        session = self._sessions.get(session_id)
        if not session:
            return 0, 0, []
        
        added, duplicates, records = session.url_store.add_batch(urls)
        session.updated_at = datetime.utcnow()
        
        return added, duplicates, records
    
    def get_session_stats(self, session_id: str) -> Optional[dict]:
        """Get statistics for a session."""
        session = self._sessions.get(session_id)
        if not session:
            return None
        
        status_counts = session.url_store.count_by_status()
        
        return {
            "session_id": session.id,
            "name": session.name,
            "total_urls": session.url_store.count(),
            "status_counts": status_counts,
            "cluster_count": len(session.clusters),
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
        }
    
    def set_session_clusters(
        self, 
        session_id: str, 
        clusters: list[dict]
    ) -> bool:
        """Set clusters for a session."""
        session = self._sessions.get(session_id)
        if not session:
            return False
        
        session.clusters = clusters
        session.updated_at = datetime.utcnow()
        return True
    
    def get_or_create_current_session(self) -> Session:
        """Get current session or create one if none exists."""
        session = self.get_current_session()
        if not session:
            session = self.create_session()
        return session
