"""Session management for organizing URLs into collections."""

import copy
import json
import logging
import os
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from services.observability import log_event

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


def _parse_datetime_utc(value: str) -> datetime:
    """Parse an ISO-8601 timestamp to a UTC-naive datetime.

    tz-aware inputs are converted to UTC then made naive so every parsed value
    is mutually comparable; unparseable inputs return ``datetime.min`` so any
    pre-existing garbage ``fetched_at`` row ranks below every real capture
    (inverting the "zzz pins forever" poison into "zzz loses forever").
    """
    try:
        parsed = datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return datetime.min
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def capture_order_key(fetched_at: str, attempt: int) -> tuple[int, datetime, int]:
    """Total order over captures; legacy (attempt 0) always loses to any v1.

    ``source_rank`` is derived from ``attempt`` (the legacy shim writes
    ``attempt=0``, v1 enforces ``attempt>=1``), so a legacy capture can never
    outrank a v1 capture regardless of receipt-clock skew during rolling
    deploy. Within a tier, newest ``fetched_at`` then highest attempt wins.
    Comparison is always parse-based, never lexicographic.
    """
    source_rank = 1 if attempt >= 1 else 0
    return (source_rank, _parse_datetime_utc(fetched_at), attempt)


@dataclass
class IngestCapture:
    """One fetch-attempt result offered to the idempotent ingest writer.

    ``attempt`` is the browser-engine per-(session,url) in-process counter
    (legacy-mapped writes pass ``0`` → ``source_rank=0`` in ``capture_order_key``
    so they lose to every real v1 write). ``fetched_at`` is an ISO-8601 string;
    v1 callers have it validated + normalized to canonical UTC-naive microsecond
    ISO at the boundary, and ordering is always via ``capture_order_key`` (parse
    based, never lexicographic), so cross-writer/legacy rows still compare
    correctly (see the sessions MODULE card).
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
    # Tabs deliberately NOT imported, with per-tab reasons in `metadata`
    # ("skipped_tabs"). Distinct from `failed`: a skip is a decision the
    # harvester made and can explain, a failure is something that went wrong.
    # Counted separately so neither hides inside the other.
    skipped: int = 0
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


# Migration identifiers are interpolated into DDL because SQLite cannot bind
# them. These patterns are the guard that keeps that interpolation safe.
_SQL_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SQL_DEFINITION = re.compile(r"^[A-Za-z0-9_ ]+$")


class SessionManager:
    """Manage multiple sessions."""

    def __init__(self, db_path: Optional[str] = None):
        self._lock = threading.RLock()
        self._sessions: dict[str, Session] = {}
        self._tab_import_jobs: dict[str, TabImportJob] = {}
        # In-memory mirror of domain_index_consent; the file-backed and
        # in-memory paths must stay behaviourally equivalent.
        self._domain_consent: dict[str, str] = {}
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
                    skipped INTEGER NOT NULL DEFAULT 0,
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

                CREATE TABLE IF NOT EXISTS domain_index_consent (
                    domain     TEXT PRIMARY KEY,
                    decision   TEXT NOT NULL,
                    reason     TEXT,
                    updated_at TEXT NOT NULL
                );
                """)
            self._migrate_schema(conn)

    @staticmethod
    def _migrate_schema(conn) -> None:
        """Additive column migrations for databases created by earlier builds.

        The schema above is `CREATE TABLE IF NOT EXISTS`, which is a no-op
        against an existing file -- so a column added to that block never
        reaches a database that already exists. Every deployment that has ever
        run this service therefore needs the column added explicitly.

        Additive and idempotent only: a new column with a DEFAULT, guarded by
        `PRAGMA table_info`. No drops, no renames, no type changes -- those need
        a real migration story rather than a startup side effect.
        """
        migrations = (("tab_import_jobs", "skipped", "INTEGER NOT NULL DEFAULT 0"),)
        for table, column, definition in migrations:
            # SQLite cannot parameterise identifiers or DDL, so these must be
            # interpolated. Guard the CLASS rather than trusting that the tuple
            # above stays hardcoded: if anything ever threads a caller-supplied
            # name in here, it fails loudly instead of becoming injection.
            if not all(
                _SQL_IDENTIFIER.match(part) for part in (table, column)
            ) or not _SQL_DEFINITION.match(definition):
                raise ValueError(
                    f"refusing unsafe migration identifier: {table}.{column}"
                )
            try:
                existing = {
                    row[1] for row in conn.execute(f"PRAGMA table_info({table})")
                }
            except Exception:
                continue
            if not existing or column in existing:
                continue
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

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
                skipped=self._row_value(row, "skipped", 0),
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
                    id, session_id, cdp_url, status, total, imported, indexed,
                    skipped, failed, error, metadata, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    session_id = excluded.session_id,
                    cdp_url = excluded.cdp_url,
                    status = excluded.status,
                    total = excluded.total,
                    imported = excluded.imported,
                    indexed = excluded.indexed,
                    skipped = excluded.skipped,
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
                    job.skipped,
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

    def list_sessions(self, include_archived: bool = False) -> list[Session]:
        """List all sessions."""
        sessions = list(self._sessions.values())
        if not include_archived:
            sessions = [s for s in sessions if s.status != "archived"]
        return sorted(sessions, key=lambda s: s.updated_at, reverse=True)

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
                    # Acquire the SQLite write lock BEFORE the duplicate/ordering
                    # reads so the whole check-then-write is one serialized unit
                    # even across processes / two SessionManager instances on one
                    # DB (bounded by busy_timeout; contention -> OperationalError
                    # -> 500 -> browser-engine retry). The RLock only guards
                    # in-process memory state, not cross-process ordering.
                    conn.execute("BEGIN IMMEDIATE")
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
            return self._duplicate_outcome(conn, session, capture, normalized, existing)

        latest = self._latest_applied_lookup(conn, capture.session_id, normalized)
        if latest is not None and capture_order_key(
            capture.fetched_at, capture.attempt
        ) < capture_order_key(latest[0], latest[1]):
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

        # Snapshot in-memory state so a failed SQLite write leaves memory AND DB
        # at the pre-call state (the SQLite side rolls back via the connection
        # context manager; without this, memory would report "scraped" while
        # SQLite still says "pending"). All under the RLock, so no reader
        # observes the intermediate.
        snapshot = copy.deepcopy(session.url_store.get(capture.url))
        prev_updated_at = session.updated_at
        try:
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
        except Exception:
            if snapshot is not None:
                self._restore_url_record(session, capture.url, snapshot)
            session.updated_at = prev_updated_at
            raise

        forward_document = (
            self._forward_document(
                record, normalized, capture.capture_id, capture.auth_used
            )
            if should_forward and record is not None
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

    @staticmethod
    def _restore_url_record(session: Session, url: str, snapshot: URLRecord) -> None:
        """Copy a pre-mutation snapshot back onto the live record in place."""
        live = session.url_store.get(url)
        if live is None:
            return
        live.status = snapshot.status
        live.metadata = snapshot.metadata
        live.content_hash = snapshot.content_hash
        live.embedding_id = snapshot.embedding_id
        live.scraped_at = snapshot.scraped_at
        live.original = snapshot.original
        live.created_at = snapshot.created_at

    def _duplicate_outcome(
        self,
        conn: Optional[sqlite3.Connection],
        session: Session,
        capture: IngestCapture,
        normalized: str,
        existing: dict,
    ) -> IngestOutcome:
        """Replay of a known capture_id: no state change, maybe re-forward.

        ``capture_id`` is the ONLY field read from the incoming request; every
        other value (session, url key, auth_used, content) comes from the stored
        ledger row + record so a duplicate delivery carrying mutated content can
        never overwrite or forward the wrong body.
        """
        stored_session_id = existing["session_id"]
        stored_normalized = existing["normalized"]
        if capture.session_id != stored_session_id or normalized != stored_normalized:
            log_event(
                "ingest.duplicate_key_mismatch",
                level=logging.WARNING,
                capture_id=capture.capture_id,
                stored_session_id=stored_session_id,
            )

        forward_document = None
        should_forward = False
        if (
            existing["outcome"] == "applied"
            and existing["index_state"] in ("pending", "failed")
            and existing["status"] == "success"
        ):
            forward_document = self._build_forward_document(
                conn, stored_session_id, stored_normalized, existing["capture_id"]
            )
            should_forward = forward_document is not None
        return IngestOutcome(
            "duplicate",
            existing["capture_id"],
            normalized=stored_normalized,
            index_state=existing["index_state"],
            should_forward=should_forward,
            forward_document=forward_document,
            session_id=stored_session_id,
        )

    def _build_forward_document(
        self,
        conn: Optional[sqlite3.Connection],
        session_id: str,
        normalized: str,
        capture_id: str,
    ) -> Optional[dict]:
        """Build the /index doc for a stored capture, or None if it is no longer
        the latest applied capture for its key / its record is gone."""
        latest = self._latest_applied_lookup(conn, session_id, normalized)
        if latest is None or latest[2] != capture_id:
            return None
        stored = self._capture_lookup(conn, capture_id)
        if stored is None or stored["outcome"] != "applied":
            return None
        session = self._sessions.get(session_id)
        if session is None:
            return None
        record = session.url_store.get(normalized)
        if record is None:
            return None
        return self._forward_document(
            record, normalized, capture_id, bool(stored["auth_used"])
        )

    def build_forward_document(
        self, session_id: str, normalized: str, capture_id: str
    ) -> Optional[dict]:
        """Public wrapper (own lock/connection) for the outbox reconcile sweep."""
        with self._lock:
            if self._db_path:
                with self._connect() as conn:
                    return self._build_forward_document(
                        conn, session_id, normalized, capture_id
                    )
            return self._build_forward_document(
                None, session_id, normalized, capture_id
            )

    @staticmethod
    def _forward_document(
        record: URLRecord,
        normalized: str,
        capture_id: str,
        auth_used: bool,
    ) -> dict:
        """Build the ai-engine /index document from STORED record state only.

        Row id is the normalized url so ai-engine's per-(session,url) upsert
        (delete-then-insert on the same document id) dedupes replays and newer
        attempts. auth_used + capture_id ride in metadata (finding 37 hook).
        Never reads content/title/auth_used from an incoming replay request.
        """
        source_metadata = dict(record.metadata)
        content = source_metadata.pop("content", None)
        title = str(source_metadata.get("title") or "")
        source_metadata["auth_used"] = auth_used
        source_metadata["capture_id"] = capture_id
        return {
            "id": normalized,
            "url": record.original,
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
            rows = conn.execute(
                """
                SELECT fetched_at, attempt, capture_id FROM ingest_captures
                WHERE session_id = ? AND normalized = ? AND outcome = 'applied'
                """,
                (session_id, normalized),
            ).fetchall()
            if not rows:
                return None
            # Pick the max via the parse-based order key in Python (the applied
            # set per url is small) so no lexicographic ORDER BY survives.
            best = max(
                rows, key=lambda r: capture_order_key(r["fetched_at"], r["attempt"])
            )
            return (best["fetched_at"], best["attempt"], best["capture_id"])
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

    def get_capture_index_state(self, capture_id: str) -> Optional[str]:
        """Public read of a ledger row's index_state (post-lock forward guard)."""
        with self._lock:
            if self._db_path:
                with self._connect() as conn:
                    row = self._capture_lookup(conn, capture_id)
            else:
                row = self._capture_lookup(None, capture_id)
            return row["index_state"] if row else None

    def list_unforwarded_captures(
        self,
        session_id: Optional[str],
        states: tuple[str, ...],
        limit: int,
    ) -> list[dict]:
        """Outbox reader: applied captures whose index_state is un-settled.

        Rows that are no longer the latest applied capture for their key are
        marked ``superseded`` in the same pass (so they stop being reselected)
        and excluded from the return. Oldest-received first, bounded by limit.
        """
        if not states:
            return []
        with self._lock:
            if self._db_path:
                with self._connect() as conn:
                    placeholders = ",".join("?" for _ in states)
                    params: list = list(states)
                    sql = (
                        "SELECT capture_id, session_id, normalized, auth_used "
                        "FROM ingest_captures "
                        "WHERE outcome = 'applied' "
                        f"AND index_state IN ({placeholders})"
                    )
                    if session_id is not None:
                        sql += " AND session_id = ?"
                        params.append(session_id)
                    sql += " ORDER BY received_at ASC LIMIT ?"
                    params.append(limit)
                    rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
                    result = []
                    for row in rows:
                        latest = self._latest_applied_lookup(
                            conn, row["session_id"], row["normalized"]
                        )
                        if latest is None or latest[2] != row["capture_id"]:
                            conn.execute(
                                "UPDATE ingest_captures SET index_state = "
                                "'superseded' WHERE capture_id = ?",
                                (row["capture_id"],),
                            )
                            continue
                        result.append(row)
                    return result

            candidates = sorted(
                (
                    capture
                    for capture in self._captures.values()
                    if capture["outcome"] == "applied"
                    and capture["index_state"] in states
                    and (session_id is None or capture["session_id"] == session_id)
                ),
                key=lambda c: c["received_at"],
            )[: max(0, limit)]
            result = []
            for capture in candidates:
                latest = self._latest_applied_lookup(
                    None, capture["session_id"], capture["normalized"]
                )
                if latest is None or latest[2] != capture["capture_id"]:
                    capture["index_state"] = "superseded"
                    continue
                result.append(
                    {
                        "capture_id": capture["capture_id"],
                        "session_id": capture["session_id"],
                        "normalized": capture["normalized"],
                        "auth_used": capture["auth_used"],
                    }
                )
            return result

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
                        SELECT normalized, index_state, index_error,
                               fetched_at, attempt
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

        # Count only the LATEST applied capture per url: a doc that failed on
        # attempt 1 and succeeded on attempt 2 must not keep reporting a stale
        # failure. Group by normalized, keep each group's max by order key.
        latest_by_norm: dict[str, dict] = {}
        for row in rows:
            key = row["normalized"]
            current = latest_by_norm.get(key)
            if current is None or capture_order_key(
                row["fetched_at"], row["attempt"]
            ) > capture_order_key(current["fetched_at"], current["attempt"]):
                latest_by_norm[key] = row

        failed = 0
        pending = 0
        downstream_errors: list[dict] = []
        for row in latest_by_norm.values():
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

    @staticmethod
    def _row_value(row, column: str, default):
        """Read a column that may be absent from an older database file.

        There is no migration framework here; the schema is `CREATE TABLE IF
        NOT EXISTS`, so a table created before a column existed keeps its old
        shape. `_migrate_schema` adds the column, but this stays defensive
        because a read must never crash on a database written by an older build.
        """
        try:
            value = row[column]
        except (IndexError, KeyError):
            return default
        return default if value is None else value

    def update_tab_import_job(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        total: Optional[int] = None,
        imported: Optional[int] = None,
        indexed: Optional[int] = None,
        skipped: Optional[int] = None,
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
            if skipped is not None:
                job.skipped = max(0, int(skipped))
            if failed is not None:
                job.failed = max(0, int(failed))
            if error is not None:
                job.error = error
            if metadata:
                job.metadata = {**job.metadata, **metadata}
            job.updated_at = datetime.utcnow()
            self._save_tab_import_job(job)
            return job

    # ------------------------------------------------------------------
    # Per-domain indexing consent (plan decision 37's "explicit opt-in")
    # ------------------------------------------------------------------

    def get_domain_consent(self, domain: str) -> Optional[str]:
        """Return ``"allow"``, ``"deny"``, or None when nobody has decided yet.

        None is the important value: it means the question has not been put to
        a human, and a document from that domain is HELD rather than embedded.
        Absence of a decision is never read as permission.
        """
        key = (domain or "").strip().lower()
        if not key:
            return None
        if self._db_path:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT decision FROM domain_index_consent WHERE domain = ?",
                    (key,),
                ).fetchone()
                return row["decision"] if row else None
        return self._domain_consent.get(key)

    def set_domain_consent(self, domain: str, decision: str, reason: str = "") -> None:
        """Record a human's answer for one domain. ``allow`` or ``deny`` only."""
        key = (domain or "").strip().lower()
        if not key:
            raise ValueError("domain is required")
        if decision not in {"allow", "deny"}:
            raise ValueError(f"decision must be 'allow' or 'deny', got {decision!r}")
        with self._lock:
            self._domain_consent[key] = decision
            if self._db_path:
                with self._connect() as conn:
                    conn.execute(
                        """
                        INSERT INTO domain_index_consent
                            (domain, decision, reason, updated_at)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(domain) DO UPDATE SET
                            decision = excluded.decision,
                            reason = excluded.reason,
                            updated_at = excluded.updated_at
                        """,
                        (key, decision, reason, datetime.utcnow().isoformat()),
                    )

    def forget_domain_consent(self, domain: str) -> bool:
        """Remove a decision, returning it to undecided. True if one existed.

        Deliberately does NOT purge anything already embedded under a previous
        `allow`: forgetting the answer and deleting the vectors it authorised
        are different operations, and quietly doing the second inside the first
        would make a settings toggle destroy data. Documents from this domain
        are simply HELD again on the next run.
        """
        key = (domain or "").strip().lower()
        if not key:
            return False
        with self._lock:
            existed = self._domain_consent.pop(key, None) is not None
            if self._db_path:
                with self._connect() as conn:
                    cursor = conn.execute(
                        "DELETE FROM domain_index_consent WHERE domain = ?", (key,)
                    )
                    existed = existed or cursor.rowcount > 0
            return existed

    def list_domain_consent(self) -> list[dict]:
        """Every recorded decision, for the settings list the TS UI will show."""
        if self._db_path:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT domain, decision, reason, updated_at "
                    "FROM domain_index_consent ORDER BY domain"
                ).fetchall()
                return [dict(row) for row in rows]
        return [
            {"domain": d, "decision": v, "reason": "", "updated_at": ""}
            for d, v in sorted(self._domain_consent.items())
        ]

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
        results: list[dict] = []
        for session in sessions:
            for record in session.url_store.get_all():
                # Match the SQLite FTS insertion condition EXACTLY so keyword
                # hits are identical across backends: only scraped rows with a
                # non-empty title/content, and the haystack is title+content
                # only (the FTS url/domain columns are UNINDEXED, so URL-only
                # term hits must not match in memory either).
                if record.status != "scraped":
                    continue
                title = str(record.metadata.get("title") or "")
                content = str(record.metadata.get("content") or "")
                if not title and not content:
                    continue
                haystack = f"{title} {content}".lower()
                if all(term in haystack for term in terms):
                    results.append(
                        {
                            "session_id": session.id,
                            "url": record.original,
                            "title": str(
                                record.metadata.get("title") or record.original
                            ),
                            "content": content,
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
