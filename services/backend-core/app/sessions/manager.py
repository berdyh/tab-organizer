"""Session management for organizing URLs into collections."""

import json
import os
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..url_input.store import URLRecord, URLStore


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


@dataclass
class TabImportJob:
    """Durable state for a live-browser tab import."""

    id: str
    session_id: str
    cdp_url: str
    status: str = "queued"
    total: int = 0
    imported: int = 0
    indexed: int = 0
    failed: int = 0
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


ACTIVE_TAB_IMPORT_STATUSES = {"queued", "running"}
TERMINAL_TAB_IMPORT_STATUSES = {
    "completed",
    "completed_with_errors",
    "failed",
    "cancelled",
}


class SessionManager:
    """Manage multiple sessions."""

    def __init__(self, db_path: Optional[str] = None):
        self._lock = threading.RLock()
        self._sessions: dict[str, Session] = {}
        self._tab_import_jobs: dict[str, TabImportJob] = {}
        self._current_session_id: Optional[str] = None
        self._db_path = db_path if db_path is not None else os.getenv("BACKEND_DB_PATH")
        if self._db_path:
            self._init_db()
            self._load_from_db()

    @property
    def persistence_enabled(self) -> bool:
        """Return whether session data is backed by SQLite."""
        return bool(self._db_path)

    @property
    def db_path(self) -> Optional[str]:
        """Return the SQLite path when persistence is enabled."""
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        if not self._db_path:
            raise RuntimeError("Session persistence is not enabled")
        conn = sqlite3.connect(self._db_path, timeout=30)
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        assert self._db_path is not None
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    clusters TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS url_records (
                    session_id TEXT NOT NULL,
                    normalized TEXT NOT NULL,
                    original TEXT NOT NULL,
                    content_hash TEXT,
                    embedding_id TEXT,
                    scraped_at TEXT,
                    status TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (session_id, normalized),
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS session_state (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );

                CREATE TABLE IF NOT EXISTS tab_import_jobs (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    cdp_url TEXT NOT NULL,
                    status TEXT NOT NULL,
                    total INTEGER NOT NULL,
                    imported INTEGER NOT NULL,
                    indexed INTEGER NOT NULL,
                    failed INTEGER NOT NULL,
                    error TEXT,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS tab_search_fts USING fts5(
                    session_id UNINDEXED,
                    normalized UNINDEXED,
                    url UNINDEXED,
                    domain UNINDEXED,
                    title,
                    content
                );
                """
            )

    def _load_from_db(self) -> None:
        with self._connect() as conn:
            session_rows = conn.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
            url_rows = conn.execute(
                "SELECT * FROM url_records ORDER BY created_at ASC"
            ).fetchall()
            state_row = conn.execute(
                "SELECT value FROM session_state WHERE key = 'current_session_id'"
            ).fetchone()
            job_rows = conn.execute(
                "SELECT * FROM tab_import_jobs ORDER BY updated_at DESC"
            ).fetchall()

        urls_by_session: dict[str, list[URLRecord]] = {}
        for row in url_rows:
            urls_by_session.setdefault(row["session_id"], []).append(
                URLRecord(
                    original=row["original"],
                    normalized=row["normalized"],
                    content_hash=row["content_hash"],
                    embedding_id=row["embedding_id"],
                    scraped_at=self._parse_datetime(row["scraped_at"]),
                    status=row["status"],
                    metadata=self._loads_json(row["metadata"], {}),
                    created_at=self._parse_datetime(row["created_at"])
                    or datetime.utcnow(),
                )
            )

        for row in session_rows:
            store = URLStore()
            store.replace_records(urls_by_session.get(row["id"], []))
            self._sessions[row["id"]] = Session(
                id=row["id"],
                name=row["name"],
                status=row["status"],
                metadata=self._loads_json(row["metadata"], {}),
                clusters=self._loads_json(row["clusters"], []),
                created_at=self._parse_datetime(row["created_at"]) or datetime.utcnow(),
                updated_at=self._parse_datetime(row["updated_at"]) or datetime.utcnow(),
                url_store=store,
            )

        for row in job_rows:
            self._tab_import_jobs[row["id"]] = TabImportJob(
                id=row["id"],
                session_id=row["session_id"],
                cdp_url=row["cdp_url"],
                status=row["status"],
                total=row["total"],
                imported=row["imported"],
                indexed=row["indexed"],
                failed=row["failed"],
                error=row["error"],
                metadata=self._loads_json(row["metadata"], {}),
                created_at=self._parse_datetime(row["created_at"]) or datetime.utcnow(),
                updated_at=self._parse_datetime(row["updated_at"]) or datetime.utcnow(),
            )

        if state_row:
            current_id = state_row["value"]
            if (
                current_id
                and current_id in self._sessions
                and self._sessions[current_id].status == "active"
            ):
                self._current_session_id = current_id
            return

        for session in self._sessions.values():
            if session.status == "active":
                self._current_session_id = session.id
                break

    def _save_session(self, session: Session) -> None:
        if not self._db_path:
            return

        with self._connect() as conn:
            self._save_session_row(conn, session)
            conn.execute("DELETE FROM url_records WHERE session_id = ?", (session.id,))
            for record in session.url_store.get_all():
                self._save_url_record(conn, session.id, record)
            self._save_state(conn)

    def _save_session_row(self, conn: sqlite3.Connection, session: Session) -> None:
        conn.execute(
            """
            INSERT INTO sessions (
                id, name, status, metadata, clusters, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                status = excluded.status,
                metadata = excluded.metadata,
                clusters = excluded.clusters,
                created_at = excluded.created_at,
                updated_at = excluded.updated_at
            """,
            (
                session.id,
                session.name,
                session.status,
                self._dumps_json(session.metadata),
                self._dumps_json(session.clusters),
                session.created_at.isoformat(),
                session.updated_at.isoformat(),
            ),
        )

    def _upsert_search_record(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        record: URLRecord,
    ) -> None:
        """Update the SQLite FTS row for a scraped URL record."""
        conn.execute(
            "DELETE FROM tab_search_fts WHERE session_id = ? AND normalized = ?",
            (session_id, record.normalized),
        )
        if record.status != "scraped":
            return

        title = str(record.metadata.get("title") or "")
        content = str(record.metadata.get("content") or "")
        if not title and not content:
            return

        domain = self._domain_for_url(record.normalized)
        conn.execute(
            """
            INSERT INTO tab_search_fts (
                session_id, normalized, url, domain, title, content
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, record.normalized, record.original, domain, title, content),
        )

    def _save_tab_import_job(self, job: TabImportJob) -> None:
        if not self._db_path:
            return

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tab_import_jobs (
                    id, session_id, cdp_url, status, total, imported, indexed, failed,
                    error, metadata, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    session_id = excluded.session_id,
                    cdp_url = excluded.cdp_url,
                    status = excluded.status,
                    total = excluded.total,
                    imported = excluded.imported,
                    indexed = excluded.indexed,
                    failed = excluded.failed,
                    error = excluded.error,
                    metadata = excluded.metadata,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at
                """,
                (
                    job.id,
                    job.session_id,
                    job.cdp_url,
                    job.status,
                    job.total,
                    job.imported,
                    job.indexed,
                    job.failed,
                    job.error,
                    self._dumps_json(job.metadata),
                    job.created_at.isoformat(),
                    job.updated_at.isoformat(),
                ),
            )

    def _save_url_record(
        self, conn: sqlite3.Connection, session_id: str, record: URLRecord
    ) -> None:
        conn.execute(
            """
            INSERT INTO url_records (
                session_id, normalized, original, content_hash, embedding_id,
                scraped_at, status, metadata, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, normalized) DO UPDATE SET
                original = excluded.original,
                content_hash = excluded.content_hash,
                embedding_id = excluded.embedding_id,
                scraped_at = excluded.scraped_at,
                status = excluded.status,
                metadata = excluded.metadata,
                created_at = excluded.created_at
            """,
            (
                session_id,
                record.normalized,
                record.original,
                record.content_hash,
                record.embedding_id,
                record.scraped_at.isoformat() if record.scraped_at else None,
                record.status,
                self._dumps_json(record.metadata),
                record.created_at.isoformat(),
            ),
        )
        self._upsert_search_record(conn, session_id, record)

    def _delete_session_from_db(self, session_id: str) -> None:
        if not self._db_path:
            return
        with self._connect() as conn:
            conn.execute("DELETE FROM url_records WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM tab_search_fts WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM tab_import_jobs WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            self._save_state(conn)

    def _save_state(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            INSERT INTO session_state (key, value)
            VALUES ('current_session_id', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (self._current_session_id,),
        )

    @staticmethod
    def _dumps_json(value) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _loads_json(raw: Optional[str], fallback):
        if not raw:
            return fallback
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return fallback

    @staticmethod
    def _parse_datetime(raw: Optional[str]) -> Optional[datetime]:
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None

    def create_session(self, name: Optional[str] = None) -> Session:
        """Create a new session."""
        with self._lock:
            session_id = str(uuid.uuid4())
            session_name = name or f"Session {len(self._sessions) + 1}"

            session = Session(id=session_id, name=session_name)
            self._sessions[session_id] = session

            if self._current_session_id is None:
                self._current_session_id = session_id

            self._save_session(session)
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
        with self._lock:
            if session_id in self._sessions:
                self._current_session_id = session_id
                if self._db_path:
                    with self._connect() as conn:
                        self._save_state(conn)
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
        metadata: Optional[dict] = None,
    ) -> Optional[Session]:
        """Update session properties."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None

            if name:
                session.name = name
            if metadata:
                session.metadata.update(metadata)
            session.updated_at = datetime.utcnow()
            self._save_session(session)

            return session

    def archive_session(self, session_id: str) -> bool:
        """Archive a session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False

            session.status = "archived"
            session.updated_at = datetime.utcnow()

            if self._current_session_id == session_id:
                self._current_session_id = None

            self._save_session(session)
            return True

    def delete_session(self, session_id: str) -> bool:
        """Delete a session permanently."""
        with self._lock:
            if session_id not in self._sessions:
                return False

            del self._sessions[session_id]

            if self._current_session_id == session_id:
                self._current_session_id = None

            self._delete_session_from_db(session_id)
            return True

    def add_urls_to_session(
        self, session_id: str, urls: list[str]
    ) -> tuple[int, int, list[URLRecord]]:
        """
        Add URLs to a session.

        Returns:
            Tuple of (added_count, duplicate_count, new_records)
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return 0, 0, []

            added, duplicates, records = session.url_store.add_batch(urls)
            session.updated_at = datetime.utcnow()
            self._save_session(session)

            return added, duplicates, records

    def update_url_status(self, session_id: str, url: str, status: str, **kwargs) -> bool:
        """Update a URL status and persist the containing session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False

            updated = session.url_store.update_status(url, status, **kwargs)
            if updated:
                session.updated_at = datetime.utcnow()
                record = session.url_store.get(url)
                if self._db_path and record is not None:
                    with self._connect() as conn:
                        self._save_session_row(conn, session)
                        self._save_url_record(conn, session.id, record)
                        self._save_state(conn)
                else:
                    self._save_session(session)
            return updated

    def create_tab_import_job(self, session_id: str, cdp_url: str) -> TabImportJob:
        """Create a queued import job unless an equivalent active job exists."""
        with self._lock:
            if session_id not in self._sessions:
                raise ValueError("Session not found")

            for job in self._tab_import_jobs.values():
                if (
                    job.session_id == session_id
                    and job.cdp_url == cdp_url
                    and job.status in ACTIVE_TAB_IMPORT_STATUSES
                ):
                    raise ValueError("An active import already exists for this session")

            job = TabImportJob(
                id=str(uuid.uuid4()),
                session_id=session_id,
                cdp_url=cdp_url,
            )
            self._tab_import_jobs[job.id] = job
            self._save_tab_import_job(job)
            return job

    def update_tab_import_job(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        total: Optional[int] = None,
        imported: Optional[int] = None,
        indexed: Optional[int] = None,
        failed: Optional[int] = None,
        error: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Optional[TabImportJob]:
        """Update durable tab import job state."""
        with self._lock:
            job = self._tab_import_jobs.get(job_id)
            if not job:
                return None
            if status is not None:
                allowed = ACTIVE_TAB_IMPORT_STATUSES | TERMINAL_TAB_IMPORT_STATUSES
                if status not in allowed:
                    raise ValueError(f"Unsupported tab import status: {status}")
                job.status = status
            if total is not None:
                job.total = max(0, int(total))
            if imported is not None:
                job.imported = max(0, int(imported))
            if indexed is not None:
                job.indexed = max(0, int(indexed))
            if failed is not None:
                job.failed = max(0, int(failed))
            if error is not None:
                job.error = error
            if metadata:
                job.metadata = {**job.metadata, **metadata}
            job.updated_at = datetime.utcnow()
            self._save_tab_import_job(job)
            return job

    def get_tab_import_job(self, job_id: str) -> Optional[TabImportJob]:
        """Return a tab import job by id."""
        return self._tab_import_jobs.get(job_id)

    def search_indexed_tabs(
        self,
        session_id: Optional[str],
        query: str,
        limit: int = 10,
    ) -> list[dict]:
        """Search persisted tab title/content with SQLite FTS or memory fallback."""
        cleaned_query = self._fts_query(query)
        if not cleaned_query:
            return []
        limit = min(max(int(limit), 1), 50)

        if not self._db_path:
            return self._search_indexed_tabs_in_memory(session_id, query, limit)

        with self._connect() as conn:
            if session_id:
                rows = conn.execute(
                    """
                    SELECT session_id, url, title, content, domain,
                           bm25(tab_search_fts) AS rank
                    FROM tab_search_fts
                    WHERE session_id = ? AND tab_search_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (session_id, cleaned_query, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT session_id, url, title, content, domain,
                           bm25(tab_search_fts) AS rank
                    FROM tab_search_fts
                    WHERE tab_search_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (cleaned_query, limit),
                ).fetchall()

        return [
            {
                "session_id": row["session_id"],
                "url": row["url"],
                "title": row["title"],
                "content": row["content"],
                "domain": row["domain"],
                "score": 1.0 / float(index + 1),
                "source": "keyword",
            }
            for index, row in enumerate(rows)
        ]

    def _search_indexed_tabs_in_memory(
        self,
        session_id: Optional[str],
        query: str,
        limit: int,
    ) -> list[dict]:
        if session_id:
            session = self._sessions.get(session_id)
            sessions = [session] if session else []
        else:
            sessions = list(self._sessions.values())
        terms = [term.lower() for term in re.findall(r"\w+", query or "")]
        results = []
        for session in sessions:
            for record in session.url_store.get_all():
                haystack = " ".join(
                    [
                        record.original,
                        str(record.metadata.get("title") or ""),
                        str(record.metadata.get("content") or ""),
                    ]
                ).lower()
                if all(term in haystack for term in terms):
                    results.append(
                        {
                            "session_id": session.id,
                            "url": record.original,
                            "title": str(
                                record.metadata.get("title") or record.original
                            ),
                            "content": str(record.metadata.get("content") or ""),
                            "domain": self._domain_for_url(record.normalized),
                            "score": 1.0 / float(len(results) + 1),
                            "source": "keyword",
                        }
                    )
                if len(results) >= limit:
                    break
            if len(results) >= limit:
                break
        return results

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

    def set_session_clusters(self, session_id: str, clusters: list[dict]) -> bool:
        """Set clusters for a session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False

            session.clusters = clusters
            session.updated_at = datetime.utcnow()
            self._save_session(session)
            return True

    def get_or_create_current_session(self) -> Session:
        """Get current session or create one if none exists."""
        session = self.get_current_session()
        if not session:
            session = self.create_session()
        return session

    @staticmethod
    def _fts_query(query: str) -> str:
        """Convert user text into a safe FTS query."""
        terms = re.findall(r"\w+", query or "")
        escaped_terms = [f'"{term}"' for term in terms if term.strip()]
        return " ".join(escaped_terms)

    @staticmethod
    def _domain_for_url(url: str) -> str:
        from urllib.parse import urlparse

        return urlparse(url).netloc.lower()
