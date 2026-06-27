"""SQLite-backed platform store for local product flows."""

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class PlatformError(RuntimeError):
    """Base platform-domain error."""


class PlatformValidationError(PlatformError):
    """Raised for invalid platform input."""


class AuthenticationError(PlatformError):
    """Raised when credentials or bearer tokens are invalid."""


class PermissionDeniedError(PlatformError):
    """Raised when a user lacks required permissions."""


class NotFoundError(PlatformError):
    """Raised when a requested platform object does not exist."""


class ConflictError(PlatformError):
    """Raised when a unique platform object already exists."""


class PlatformStore:
    """Local platform data store for users, companies, tokens, and issues."""

    API_TOKEN_SCOPES = {"companies:read"}

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = (
            db_path
            if db_path is not None
            else os.getenv("PLATFORM_DB_PATH") or os.getenv("BACKEND_DB_PATH")
        )
        self._memory_conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        if self._db_path:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self._db_path, timeout=30)
        else:
            if self._memory_conn is None:
                self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
            conn = self._memory_conn
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA foreign_keys = ON")
        if self._db_path:
            conn.execute("PRAGMA journal_mode = WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        close_after = bool(self._db_path)
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS platform_users (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    company_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS platform_sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    token_hash TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    last_used_at TEXT,
                    FOREIGN KEY (user_id) REFERENCES platform_users(id)
                );

                CREATE TABLE IF NOT EXISTS platform_companies (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    domain TEXT,
                    website TEXT,
                    industry TEXT,
                    size TEXT,
                    status TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS api_tokens (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    token_hash TEXT NOT NULL UNIQUE,
                    prefix TEXT NOT NULL,
                    scopes TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    revoked_at TEXT,
                    last_used_at TEXT,
                    FOREIGN KEY (user_id) REFERENCES platform_users(id)
                );

                CREATE TABLE IF NOT EXISTS api_requests (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    token_id TEXT NOT NULL,
                    path TEXT NOT NULL,
                    query TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES platform_users(id),
                    FOREIGN KEY (token_id) REFERENCES api_tokens(id)
                );

                CREATE TABLE IF NOT EXISTS dashboard_events (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES platform_users(id)
                );

                CREATE TABLE IF NOT EXISTS platform_issues (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES platform_users(id)
                );
                """)
            self._seed_companies(conn)
            conn.commit()
        finally:
            if close_after:
                conn.close()

    def _seed_companies(self, conn: sqlite3.Connection) -> None:
        count = conn.execute("SELECT COUNT(*) FROM platform_companies").fetchone()[0]
        if count:
            return

        now = self._now()
        companies = [
            {
                "id": "co_acme",
                "name": "Acme Corporation",
                "domain": "acme.example",
                "website": "https://acme.example",
                "industry": "Manufacturing",
                "size": "1001-5000",
                "status": "active",
                "description": "Demo company for local company discovery.",
            },
            {
                "id": "co_globex",
                "name": "Globex",
                "domain": "globex.example",
                "website": "https://globex.example",
                "industry": "Technology",
                "size": "501-1000",
                "status": "active",
                "description": "Seed B2B account target for integration testing.",
            },
            {
                "id": "co_initech",
                "name": "Initech",
                "domain": "initech.example",
                "website": "https://initech.example",
                "industry": "Software",
                "size": "51-200",
                "status": "active",
                "description": "Seed company used for search and dashboard demos.",
            },
        ]
        conn.executemany(
            """
            INSERT INTO platform_companies (
                id, name, domain, website, industry, size, status, description,
                created_at, updated_at
            )
            VALUES (
                :id, :name, :domain, :website, :industry, :size, :status,
                :description, :created_at, :updated_at
            )
            """,
            [
                {
                    **company,
                    "created_at": now,
                    "updated_at": now,
                }
                for company in companies
            ],
        )

    def create_user(
        self,
        email: str,
        password: str,
        name: Optional[str] = None,
        role: str = "user",
        company_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create a local platform user."""
        normalized_email = email.strip().lower()
        if not normalized_email or "@" not in normalized_email:
            raise PlatformValidationError("Valid email is required")
        if len(password) < 8:
            raise PlatformValidationError("Password must be at least 8 characters")

        role = self._normalize_role(role, company_name)
        now = self._now()
        user_id = f"user_{uuid.uuid4().hex}"

        conn = self._connect()
        close_after = bool(self._db_path)
        try:
            company_id = None
            if company_name:
                company_id = self._ensure_company(conn, company_name.strip(), now)

            conn.execute(
                """
                INSERT INTO platform_users (
                    id, email, password_hash, name, role, company_id,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    normalized_email,
                    self._hash_password(password),
                    name or normalized_email.split("@", 1)[0],
                    role,
                    company_id,
                    now,
                    now,
                ),
            )
            self._record_event_conn(conn, user_id, "user.signup", {"role": role}, now)
            conn.commit()
            return self._public_user(
                conn.execute(
                    "SELECT * FROM platform_users WHERE id = ?", (user_id,)
                ).fetchone()
            )
        except sqlite3.IntegrityError as exc:
            raise ConflictError("A user with this email already exists") from exc
        finally:
            if close_after:
                conn.close()

    def login_user(self, email: str, password: str) -> dict[str, Any]:
        """Authenticate a user by email and password."""
        conn = self._connect()
        close_after = bool(self._db_path)
        try:
            row = conn.execute(
                "SELECT * FROM platform_users WHERE email = ?",
                (email.strip().lower(),),
            ).fetchone()
            if not row or not self._verify_password(password, row["password_hash"]):
                raise AuthenticationError("Invalid email or password")
            self._record_event_conn(conn, row["id"], "user.login", {}, self._now())
            conn.commit()
            return self._public_user(row)
        finally:
            if close_after:
                conn.close()

    def create_session(self, user_id: str) -> dict[str, Any]:
        """Create a bearer session for a platform user."""
        token = f"tbs_{secrets.token_urlsafe(32)}"
        now = self._now()
        session_id = f"sess_{uuid.uuid4().hex}"
        conn = self._connect()
        close_after = bool(self._db_path)
        try:
            self._get_user_row(conn, user_id)
            conn.execute(
                """
                INSERT INTO platform_sessions (
                    id, user_id, token_hash, created_at, last_used_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, user_id, self._hash_token(token), now, now),
            )
            conn.commit()
            return {"session_id": session_id, "session_token": token}
        finally:
            if close_after:
                conn.close()

    def authenticate_session(self, session_token: str) -> dict[str, Any]:
        """Resolve a bearer session token into a platform user."""
        token_hash = self._hash_token(session_token)
        now = self._now()
        conn = self._connect()
        close_after = bool(self._db_path)
        try:
            row = conn.execute(
                """
                SELECT u.*
                FROM platform_sessions s
                JOIN platform_users u ON u.id = s.user_id
                WHERE s.token_hash = ?
                """,
                (token_hash,),
            ).fetchone()
            if not row:
                raise AuthenticationError("Invalid session token")
            conn.execute(
                "UPDATE platform_sessions SET last_used_at = ? WHERE token_hash = ?",
                (now, token_hash),
            )
            conn.commit()
            return self._public_user(row)
        finally:
            if close_after:
                conn.close()

    def search_companies(
        self,
        query: Optional[str] = None,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        """Search local company records."""
        limit = max(1, min(int(limit or 25), 100))
        conn = self._connect()
        close_after = bool(self._db_path)
        try:
            if query:
                term = f"%{query.strip().lower()}%"
                rows = conn.execute(
                    """
                    SELECT * FROM platform_companies
                    WHERE lower(name) LIKE ?
                       OR lower(coalesce(domain, '')) LIKE ?
                       OR lower(coalesce(industry, '')) LIKE ?
                       OR lower(coalesce(description, '')) LIKE ?
                    ORDER BY name ASC
                    LIMIT ?
                    """,
                    (term, term, term, term, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM platform_companies ORDER BY name ASC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [self._company(row) for row in rows]
        finally:
            if close_after:
                conn.close()

    def get_company(self, company_id: str) -> dict[str, Any]:
        """Return a company by id."""
        conn = self._connect()
        close_after = bool(self._db_path)
        try:
            row = conn.execute(
                "SELECT * FROM platform_companies WHERE id = ?", (company_id,)
            ).fetchone()
            if not row:
                raise NotFoundError("Company not found")
            return self._company(row)
        finally:
            if close_after:
                conn.close()

    def create_api_token(
        self,
        user_id: str,
        name: str,
        scopes: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Create a B2B API token. Raw token is returned once."""
        if not name.strip():
            raise PlatformValidationError("Token name is required")
        normalized_scopes = self._normalize_api_token_scopes(scopes)
        token = f"tbo_{secrets.token_urlsafe(32)}"
        now = self._now()
        token_id = f"tok_{uuid.uuid4().hex}"
        conn = self._connect()
        close_after = bool(self._db_path)
        try:
            user = self._get_user_row(conn, user_id)
            if user["role"] not in {"b2b", "maintainer"}:
                raise PermissionDeniedError("Business access is required")
            row_values = (
                token_id,
                user_id,
                name.strip(),
                self._hash_token(token),
                token[:12],
                self._dumps(normalized_scopes),
                "active",
                now,
                None,
                None,
            )
            conn.execute(
                """
                INSERT INTO api_tokens (
                    id, user_id, name, token_hash, prefix, scopes, status,
                    created_at, revoked_at, last_used_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row_values,
            )
            self._record_event_conn(
                conn,
                user_id,
                "api_token.created",
                {"token_id": token_id, "name": name.strip()},
                now,
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM api_tokens WHERE id = ?", (token_id,)
            ).fetchone()
            return {"api_token": self._api_token(row), "token": token}
        finally:
            if close_after:
                conn.close()

    def list_api_tokens(self, user_id: str) -> list[dict[str, Any]]:
        """List API tokens without revealing raw token values."""
        conn = self._connect()
        close_after = bool(self._db_path)
        try:
            self._get_user_row(conn, user_id)
            rows = conn.execute(
                """
                SELECT * FROM api_tokens
                WHERE user_id = ?
                ORDER BY created_at DESC
                """,
                (user_id,),
            ).fetchall()
            return [self._api_token(row) for row in rows]
        finally:
            if close_after:
                conn.close()

    def revoke_api_token(self, user_id: str, token_id: str) -> dict[str, Any]:
        """Revoke an API token owned by a user."""
        now = self._now()
        conn = self._connect()
        close_after = bool(self._db_path)
        try:
            row = conn.execute(
                "SELECT * FROM api_tokens WHERE id = ? AND user_id = ?",
                (token_id, user_id),
            ).fetchone()
            if not row:
                raise NotFoundError("API token not found")
            conn.execute(
                """
                UPDATE api_tokens
                SET status = 'revoked', revoked_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (now, token_id, user_id),
            )
            self._record_event_conn(
                conn, user_id, "api_token.revoked", {"token_id": token_id}, now
            )
            conn.commit()
            return self._api_token(
                conn.execute(
                    "SELECT * FROM api_tokens WHERE id = ?", (token_id,)
                ).fetchone()
            )
        finally:
            if close_after:
                conn.close()

    def authenticate_api_token(self, token: str) -> dict[str, Any]:
        """Authenticate an active B2B API token."""
        conn = self._connect()
        close_after = bool(self._db_path)
        try:
            row = conn.execute(
                """
                SELECT * FROM api_tokens
                WHERE token_hash = ? AND status = 'active'
                """,
                (self._hash_token(token),),
            ).fetchone()
            if not row:
                raise AuthenticationError("Invalid API token")
            return self._api_token(row)
        finally:
            if close_after:
                conn.close()

    def search_companies_with_api_token(
        self,
        token: str,
        query: Optional[str] = None,
        limit: int = 25,
        path: str = "/api/v1/platform/v1/companies/search",
    ) -> dict[str, Any]:
        """Serve a token-authenticated company search and record usage."""
        api_token = self.authenticate_api_token(token)
        self._require_scope(api_token, "companies:read")
        companies = self.search_companies(query=query, limit=limit)
        now = self._now()
        conn = self._connect()
        close_after = bool(self._db_path)
        try:
            conn.execute(
                "UPDATE api_tokens SET last_used_at = ? WHERE id = ?",
                (now, api_token["id"]),
            )
            conn.execute(
                """
                INSERT INTO api_requests (
                    id, user_id, token_id, path, query, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    f"req_{uuid.uuid4().hex}",
                    api_token["user_id"],
                    api_token["id"],
                    path,
                    query or "",
                    now,
                ),
            )
            event = self._record_event_conn(
                conn,
                api_token["user_id"],
                "api.request",
                {
                    "token_id": api_token["id"],
                    "path": path,
                    "query": query or "",
                    "result_count": len(companies),
                },
                now,
            )
            conn.commit()
            return {
                "companies": companies,
                "request": {
                    "event_id": event["id"],
                    "api_token_id": api_token["id"],
                    "result_count": len(companies),
                },
            }
        finally:
            if close_after:
                conn.close()

    def get_dashboard(self, user_id: str) -> dict[str, Any]:
        """Return dashboard counters and recent events for a user."""
        conn = self._connect()
        close_after = bool(self._db_path)
        try:
            self._get_user_row(conn, user_id)
            counters = conn.execute(
                """
                SELECT
                    (
                        SELECT COUNT(*) FROM api_tokens WHERE user_id = ?
                    ) AS api_tokens_total,
                    (
                        SELECT COUNT(*) FROM api_tokens
                        WHERE user_id = ? AND status = 'active'
                    ) AS api_tokens_active,
                    (
                        SELECT COUNT(*) FROM api_requests WHERE user_id = ?
                    ) AS api_requests_total,
                    (
                        SELECT COUNT(*) FROM platform_issues WHERE user_id = ?
                    ) AS issues_created,
                    (
                        SELECT COUNT(*) FROM platform_companies
                    ) AS companies_available,
                    (
                        SELECT MIN(created_at) FROM api_requests WHERE user_id = ?
                    ) AS first_request_at,
                    (
                        SELECT MAX(created_at) FROM api_requests WHERE user_id = ?
                    ) AS last_request_at
                """,
                (user_id, user_id, user_id, user_id, user_id, user_id),
            ).fetchone()
            events = conn.execute(
                """
                SELECT * FROM dashboard_events
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT 20
                """,
                (user_id,),
            ).fetchall()
            return {
                "counters": dict(counters),
                "recent_events": [self._event(row) for row in events],
            }
        finally:
            if close_after:
                conn.close()

    def record_event(
        self,
        user_id: str,
        event_type: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Create a dashboard event for a user."""
        conn = self._connect()
        close_after = bool(self._db_path)
        try:
            self._get_user_row(conn, user_id)
            event = self._record_event_conn(
                conn, user_id, event_type, metadata or {}, self._now()
            )
            conn.commit()
            return event
        finally:
            if close_after:
                conn.close()

    def create_issue(
        self,
        user_id: str,
        title: str,
        description: str,
        severity: str = "medium",
    ) -> dict[str, Any]:
        """Create an issue visible to maintainers."""
        title = title.strip()
        description = description.strip()
        severity = severity.strip().lower()
        if not title:
            raise PlatformValidationError("Issue title is required")
        if not description:
            raise PlatformValidationError("Issue description is required")
        if severity not in {"low", "medium", "high", "critical"}:
            raise PlatformValidationError("Invalid issue severity")

        now = self._now()
        issue_id = f"issue_{uuid.uuid4().hex}"
        conn = self._connect()
        close_after = bool(self._db_path)
        try:
            self._get_user_row(conn, user_id)
            conn.execute(
                """
                INSERT INTO platform_issues (
                    id, user_id, title, description, severity, status, source,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    issue_id,
                    user_id,
                    title,
                    description,
                    severity,
                    "open",
                    "user",
                    now,
                    now,
                ),
            )
            self._record_event_conn(
                conn, user_id, "issue.created", {"issue_id": issue_id}, now
            )
            conn.commit()
            return self._issue(
                conn.execute(
                    "SELECT * FROM platform_issues WHERE id = ?", (issue_id,)
                ).fetchone()
            )
        finally:
            if close_after:
                conn.close()

    def list_issues(
        self, maintainer_user_id: str, status: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """List issues for maintainers."""
        conn = self._connect()
        close_after = bool(self._db_path)
        try:
            maintainer = self._get_user_row(conn, maintainer_user_id)
            if maintainer["role"] != "maintainer":
                raise PermissionDeniedError("Only maintainers can list issues")
            params: list[Any] = []
            where = ""
            if status:
                where = "WHERE i.status = ?"
                params.append(status)
            rows = conn.execute(
                f"""
                SELECT i.*, u.email AS creator_email
                FROM platform_issues i
                JOIN platform_users u ON u.id = i.user_id
                {where}
                ORDER BY i.created_at DESC
                """,
                params,
            ).fetchall()
            return [self._issue(row) for row in rows]
        finally:
            if close_after:
                conn.close()

    def first_api_call(self, user_id: str) -> dict[str, Any]:
        """Return a docs-ready first API request example for B2B users."""
        self._require_business_user(user_id)
        endpoint = "/api/v1/platform/v1/companies/search?query=acme"
        base_url = os.getenv("BACKEND_PUBLIC_URL", "http://localhost:8080").rstrip("/")
        url = f"{base_url}{endpoint}"
        return {
            "description": "Use a B2B API token to search local company records.",
            "endpoint": endpoint,
            "base_url": base_url,
            "url": url,
            "curl": f"curl -H 'Authorization: Bearer <api-token>' '{url}'",
        }

    def _normalize_role(self, role: str, company_name: Optional[str]) -> str:
        value = (role or ("b2b" if company_name else "user")).strip().lower()
        if value in {"business", "team", "enterprise"}:
            value = "b2b"
        if company_name and value == "user":
            value = "b2b"
        if value not in {"user", "b2b", "maintainer"}:
            raise PlatformValidationError("Invalid user role")
        return value

    def _ensure_company(
        self, conn: sqlite3.Connection, company_name: str, now: str
    ) -> str:
        row = conn.execute(
            "SELECT id FROM platform_companies WHERE lower(name) = lower(?)",
            (company_name,),
        ).fetchone()
        if row:
            return row["id"]
        company_id = f"co_{uuid.uuid4().hex}"
        conn.execute(
            """
            INSERT INTO platform_companies (
                id, name, domain, website, industry, size, status, description,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                company_id,
                company_name,
                None,
                None,
                "Unknown",
                "Unknown",
                "active",
                "Local business account.",
                now,
                now,
            ),
        )
        return company_id

    def _get_user_row(self, conn: sqlite3.Connection, user_id: str) -> sqlite3.Row:
        row = conn.execute(
            "SELECT * FROM platform_users WHERE id = ?", (user_id,)
        ).fetchone()
        if not row:
            raise NotFoundError("User not found")
        return row

    def _require_business_user(self, user_id: str) -> sqlite3.Row:
        conn = self._connect()
        close_after = bool(self._db_path)
        try:
            user = self._get_user_row(conn, user_id)
            if user["role"] not in {"b2b", "maintainer"}:
                raise PermissionDeniedError("Business access is required")
            return user
        finally:
            if close_after:
                conn.close()

    @staticmethod
    def _normalize_scope_value(scope: Any) -> str:
        return str(scope).strip().lower()

    @classmethod
    def _normalize_api_token_scopes(cls, scopes: Optional[list[str]]) -> list[str]:
        if scopes is None:
            return ["companies:read"]

        normalized: list[str] = []
        for scope in scopes:
            value = cls._normalize_scope_value(scope)
            if not value:
                continue
            if value not in cls.API_TOKEN_SCOPES:
                raise PlatformValidationError(f"Unsupported API token scope: {value}")
            if value not in normalized:
                normalized.append(value)
        return normalized

    @classmethod
    def _require_scope(cls, api_token: dict[str, Any], required_scope: str) -> None:
        scopes = {str(scope).strip().lower() for scope in api_token.get("scopes", [])}
        if required_scope not in scopes or required_scope not in cls.API_TOKEN_SCOPES:
            raise PermissionDeniedError(
                f"API token requires the {required_scope} scope"
            )

    def _record_event_conn(
        self,
        conn: sqlite3.Connection,
        user_id: str,
        event_type: str,
        metadata: dict[str, Any],
        created_at: str,
    ) -> dict[str, Any]:
        event_type = event_type.strip()
        if not event_type:
            raise PlatformValidationError("Event type is required")
        event = {
            "id": f"evt_{uuid.uuid4().hex}",
            "user_id": user_id,
            "event_type": event_type,
            "metadata": metadata,
            "created_at": created_at,
        }
        conn.execute(
            """
            INSERT INTO dashboard_events (
                id, user_id, event_type, metadata, created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                event["id"],
                event["user_id"],
                event["event_type"],
                self._dumps(event["metadata"]),
                event["created_at"],
            ),
        )
        return event

    @staticmethod
    def _public_user(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "email": row["email"],
            "name": row["name"],
            "role": row["role"],
            "roles": [row["role"]],
            "account_type": "business" if row["role"] == "b2b" else row["role"],
            "company_id": row["company_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "is_maintainer": row["role"] == "maintainer",
        }

    @staticmethod
    def _company(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "name": row["name"],
            "domain": row["domain"],
            "website": row["website"],
            "industry": row["industry"],
            "size": row["size"],
            "status": row["status"],
            "description": row["description"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _api_token(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "user_id": row["user_id"],
            "name": row["name"],
            "prefix": row["prefix"],
            "scopes": self._loads(row["scopes"], []),
            "status": row["status"],
            "created_at": row["created_at"],
            "revoked_at": row["revoked_at"],
            "last_used_at": row["last_used_at"],
        }

    def _event(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "event_type": row["event_type"],
            "metadata": self._loads(row["metadata"], {}),
            "created_at": row["created_at"],
        }

    @staticmethod
    def _issue(row: sqlite3.Row) -> dict[str, Any]:
        issue = {
            "id": row["id"],
            "user_id": row["user_id"],
            "title": row["title"],
            "description": row["description"],
            "severity": row["severity"],
            "status": row["status"],
            "source": row["source"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        if "creator_email" in row.keys():
            issue["creator_email"] = row["creator_email"]
        return issue

    @staticmethod
    def _hash_password(password: str) -> str:
        salt = secrets.token_bytes(16)
        iterations = 260_000
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, iterations
        )
        return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"

    @staticmethod
    def _verify_password(password: str, stored_hash: str) -> bool:
        try:
            algorithm, iterations, salt_hex, digest_hex = stored_hash.split("$", 3)
            if algorithm != "pbkdf2_sha256":
                return False
            digest = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                bytes.fromhex(salt_hex),
                int(iterations),
            )
            return hmac.compare_digest(digest.hex(), digest_hex)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _hash_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def _dumps(value: Any) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _loads(raw: Optional[str], fallback: Any) -> Any:
        if not raw:
            return fallback
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return fallback

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
