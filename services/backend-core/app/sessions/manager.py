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

# Canonical browser-engine scrape status -> URL record status map. Lives here
# (the persistence layer) so the ingest writer and the legacy callback adapter
# in routes.py share one source of truth.
SCRAPE_STATUS_TO_URL_STATUS = {
    "success": "scraped",
    "failed": "failed",
    "auth_required": "auth_required",
    "timeout": "failed",
    "blocked": "failed",
}


@dataclass
class IngestCapture:
    """One fetch-attempt result offered to the idempotent ingest writer.

    ``attempt`` is the browser-engine per-(session,url) in-process counter
    (legacy-mapped writes pass ``0`` so they lose every tie against a real v1
    write). ``fetched_at`` is an ISO-8601 UTC string assigned by the single
    browser-engine writer at fetch completion; the ordering rule assumes all
    ``fetched_at`` values are mutually comparable because they come from one
    writer's clock (see the sessions MODULE card).
    """

    capture_id: str
    attempt: int
    session_id: str
    url: str
    status: str
    content: Optional[str]
    metadata: dict
    auth_used: bool
    fetched_at: str


@dataclass
class IngestOutcome:
    """Result of ``ingest_scrape_result`` for the route/adapter to act on."""

    outcome: str  # applied | duplicate | stale | session_not_found |
    #               url_not_registered
    capture_id: str
    normalized: Optional[str] = None
    index_state: Optional[str] = None  # skipped|pending|indexed|failed|superseded
    should_forward: bool = False
    forward_document: Optional[dict] = None
    session_id: Optional[str] = None


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
        # In-memory ingest ledger (used only when persistence is disabled; the
        # SQLite path queries the ingest_captures table directly). Kept append/
        # update-only, never load-all-into-dicts + reinsert.
        self._captures: dict[str, dict] = {}
        self._latest_applied: dict[tuple[str, str], tuple[str, int, str]] = {}
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
            conn.executescript("""
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

                CREATE TABLE IF NOT EXISTS ingest_captures (
                    capture_id  TEXT PRIMARY KEY,
                    session_id  TEXT NOT NULL,
                    normalized  TEXT NOT NULL,
                    attempt     INTEGER NOT NULL,
                    status      TEXT NOT NULL,
                    auth_used   INTEGER NOT NULL DEFAULT 0,
                    fetched_at  TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    outcome     TEXT NOT NULL,
                    index_state TEXT NOT NULL,
                    index_error TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_ingest_captures_key
                    ON ingest_captures(session_id, normalized);
                """)

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
            conn.execute(
                "DELETE FROM tab_search_fts WHERE session_id = ?", (session_id,)
            )
            conn.execute(
                "DELETE FROM tab_import_jobs WHERE session_id = ?", (session_id,)
            )
            conn.execute(
                "DELETE FROM ingest_captures WHERE session_id = ?", (session_id,)
            )
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

            # Purge in-memory ledger rows for the session (SQLite rows are
            # dropped in _delete_session_from_db).
            self._captures = {
                cid: cap
                for cid, cap in self._captures.items()
                if cap["session_id"] != session_id
            }
            self._latest_applied = {
                key: value
                for key, value in self._latest_applied.items()
                if key[0] != session_id
            }

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

    def update_url_status(
        self, session_id: str, url: str, status: str, **kwargs
    ) -> bool:
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

    # ------------------------------------------------------------------
    # Idempotent ingest ledger (single versioned ingest endpoint)
    # ------------------------------------------------------------------

    def ingest_scrape_result(self, capture: IngestCapture) -> IngestOutcome:
        """Apply one capture under the replay/ordering rule, atomically.

        Decision procedure (entirely under the lock; the SQLite writes for an
        applied capture commit the ledger row + url_record + FTS row together):

        1. Duplicate ``capture_id`` already in the ledger -> no state change; a
           stored applied row still awaiting/failed indexing re-schedules the
           forward (retry-heals).
        2. Stale: incoming ``(fetched_at, attempt)`` older than the max over
           *applied* captures for ``(session_id, normalized)`` -> record a stale
           ledger row, leave the url_record untouched.
        3. Apply: ledger row + url_record + FTS in one transaction.
        """
        with self._lock:
            session = self._sessions.get(capture.session_id)
            if not session:
                return IngestOutcome("session_not_found", capture.capture_id)
            try:
                normalized = session.url_store.normalize(capture.url)
            except ValueError:
                return IngestOutcome("url_not_registered", capture.capture_id)
            record = session.url_store.get(capture.url)
            if record is None:
                return IngestOutcome(
                    "url_not_registered", capture.capture_id, normalized=normalized
                )

            received_at = datetime.utcnow().isoformat()
            if self._db_path:
                with self._connect() as conn:
                    return self._ingest_apply(
                        conn, session, capture, normalized, received_at
                    )
            return self._ingest_apply(None, session, capture, normalized, received_at)

    def _ingest_apply(
        self,
        conn: Optional[sqlite3.Connection],
        session: Session,
        capture: IngestCapture,
        normalized: str,
        received_at: str,
    ) -> IngestOutcome:
        existing = self._capture_lookup(conn, capture.capture_id)
        if existing is not None:
            return self._duplicate_outcome(session, capture, normalized, existing)

        latest = self._latest_applied_lookup(conn, capture.session_id, normalized)
        if latest is not None and (capture.fetched_at, capture.attempt) < (
            latest[0],
            latest[1],
        ):
            self._insert_capture(
                conn, capture, normalized, received_at, "stale", "skipped"
            )
            return IngestOutcome("stale", capture.capture_id, normalized=normalized)

        url_status = SCRAPE_STATUS_TO_URL_STATUS.get(capture.status, "failed")
        should_forward = capture.status == "success" and bool(capture.content)
        index_state = "pending" if should_forward else "skipped"

        if capture.content:
            metadata = {**capture.metadata, "content": capture.content}
        else:
            metadata = dict(capture.metadata)
        session.url_store.update_status(capture.url, url_status, metadata=metadata)
        session.updated_at = datetime.utcnow()
        record = session.url_store.get(capture.url)

        self._insert_capture(
            conn, capture, normalized, received_at, "applied", index_state
        )
        self._set_latest_applied(conn, capture, normalized)

        if conn is not None and record is not None:
            self._save_session_row(conn, session)
            self._save_url_record(conn, session.id, record)
            self._save_state(conn)

        forward_document = (
            self._forward_document(capture, normalized, record)
            if should_forward
            else None
        )
        return IngestOutcome(
            "applied",
            capture.capture_id,
            normalized=normalized,
            index_state=index_state,
            should_forward=should_forward,
            forward_document=forward_document,
            session_id=capture.session_id,
        )

    def _duplicate_outcome(
        self,
        session: Session,
        capture: IngestCapture,
        normalized: str,
        existing: dict,
    ) -> IngestOutcome:
        """Replay of a known capture_id: no state change, maybe re-forward."""
        should_forward = False
        forward_document = None
        record = session.url_store.get(capture.url)
        if (
            existing["outcome"] == "applied"
            and existing["index_state"] in ("pending", "failed")
            and existing["status"] == "success"
            and record is not None
            and record.metadata.get("content")
        ):
            should_forward = True
            forward_document = self._forward_document(capture, normalized, record)
        return IngestOutcome(
            "duplicate",
            capture.capture_id,
            normalized=normalized,
            index_state=existing["index_state"],
            should_forward=should_forward,
            forward_document=forward_document,
            session_id=capture.session_id,
        )

    @staticmethod
    def _forward_document(
        capture: IngestCapture,
        normalized: str,
        record: Optional[URLRecord],
    ) -> dict:
        """Build the ai-engine /index document for an applied/replayed capture.

        Row id is the normalized url so ai-engine's per-(session,url) upsert
        (delete-then-insert on the same document id) dedupes replays and newer
        attempts. auth_used + capture_id ride in metadata (finding 37 hook).
        """
        source_metadata = dict(record.metadata) if record else {}
        content = source_metadata.pop("content", None)
        if capture.content is not None:
            content = capture.content
        title = str(source_metadata.get("title") or "")
        source_metadata["auth_used"] = capture.auth_used
        source_metadata["capture_id"] = capture.capture_id
        return {
            "id": normalized,
            "url": record.original if record else capture.url,
            "title": title,
            "content": content or "",
            "metadata": source_metadata,
        }

    def _capture_lookup(
        self, conn: Optional[sqlite3.Connection], capture_id: str
    ) -> Optional[dict]:
        if conn is not None:
            row = conn.execute(
                "SELECT * FROM ingest_captures WHERE capture_id = ?", (capture_id,)
            ).fetchone()
            return dict(row) if row else None
        return self._captures.get(capture_id)

    def _latest_applied_lookup(
        self,
        conn: Optional[sqlite3.Connection],
        session_id: str,
        normalized: str,
    ) -> Optional[tuple[str, int, str]]:
        if conn is not None:
            row = conn.execute(
                """
                SELECT fetched_at, attempt, capture_id FROM ingest_captures
                WHERE session_id = ? AND normalized = ? AND outcome = 'applied'
                ORDER BY fetched_at DESC, attempt DESC
                LIMIT 1
                """,
                (session_id, normalized),
            ).fetchone()
            if row is None:
                return None
            return (row["fetched_at"], row["attempt"], row["capture_id"])
        return self._latest_applied.get((session_id, normalized))

    def _insert_capture(
        self,
        conn: Optional[sqlite3.Connection],
        capture: IngestCapture,
        normalized: str,
        received_at: str,
        outcome: str,
        index_state: str,
    ) -> None:
        if conn is not None:
            conn.execute(
                """
                INSERT INTO ingest_captures (
                    capture_id, session_id, normalized, attempt, status,
                    auth_used, fetched_at, received_at, outcome, index_state,
                    index_error
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    capture.capture_id,
                    capture.session_id,
                    normalized,
                    capture.attempt,
                    capture.status,
                    1 if capture.auth_used else 0,
                    capture.fetched_at,
                    received_at,
                    outcome,
                    index_state,
                ),
            )
            return
        self._captures[capture.capture_id] = {
            "capture_id": capture.capture_id,
            "session_id": capture.session_id,
            "normalized": normalized,
            "attempt": capture.attempt,
            "status": capture.status,
            "auth_used": 1 if capture.auth_used else 0,
            "fetched_at": capture.fetched_at,
            "received_at": received_at,
            "outcome": outcome,
            "index_state": index_state,
            "index_error": None,
        }

    def _set_latest_applied(
        self,
        conn: Optional[sqlite3.Connection],
        capture: IngestCapture,
        normalized: str,
    ) -> None:
        # SQLite path reads latest from the table directly; only the in-memory
        # ledger keeps a per-key pointer.
        if conn is not None:
            return
        self._latest_applied[(capture.session_id, normalized)] = (
            capture.fetched_at,
            capture.attempt,
            capture.capture_id,
        )

    def is_latest_applied(
        self, session_id: str, normalized: str, capture_id: str
    ) -> bool:
        """Whether ``capture_id`` is still the newest applied capture for a key."""
        with self._lock:
            if self._db_path:
                with self._connect() as conn:
                    latest = self._latest_applied_lookup(conn, session_id, normalized)
            else:
                latest = self._latest_applied_lookup(None, session_id, normalized)
            return latest is not None and latest[2] == capture_id

    def update_capture_index_state(
        self, capture_id: str, index_state: str, error: Optional[str] = None
    ) -> None:
        """Record the outcome of forwarding a capture to ai-engine /index."""
        with self._lock:
            if self._db_path:
                with self._connect() as conn:
                    conn.execute(
                        """
                        UPDATE ingest_captures
                        SET index_state = ?, index_error = ?
                        WHERE capture_id = ?
                        """,
                        (index_state, error, capture_id),
                    )
                return
            capture = self._captures.get(capture_id)
            if capture is not None:
                capture["index_state"] = index_state
                capture["index_error"] = error

    def capture_index_counts(self, session_id: str) -> dict:
        """Ledger aggregates overlaid onto /scrape/status (B5 per-document).

        Returns per-document ``ai_index_failed``/``ai_index_pending`` counts and
        ``downstream_errors`` entries for failed forwards. Forwarding is per
        capture, so these are inherently per-document (no batch under-report).
        """
        if self._db_path:
            with self._connect() as conn:
                rows = [
                    dict(row)
                    for row in conn.execute(
                        """
                        SELECT normalized, index_state, index_error
                        FROM ingest_captures
                        WHERE session_id = ? AND outcome = 'applied'
                        """,
                        (session_id,),
                    ).fetchall()
                ]
        else:
            rows = [
                capture
                for capture in self._captures.values()
                if capture["session_id"] == session_id
                and capture["outcome"] == "applied"
            ]

        failed = 0
        pending = 0
        downstream_errors: list[dict] = []
        for row in rows:
            state = row["index_state"]
            if state == "failed":
                failed += 1
                downstream_errors.append(
                    {
                        "source": "ai_index",
                        "url": row["normalized"],
                        "message": row.get("index_error") or "index failed",
                    }
                )
            elif state == "pending":
                pending += 1
        return {
            "ai_index_failed": failed,
            "ai_index_pending": pending,
            "downstream_errors": downstream_errors,
        }

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
