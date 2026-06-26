"""Platform backend domain and API regressions."""

import sqlite3
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.backend_core.app.api import routes
from services.backend_core.app.platform.store import PlatformStore


def _platform_client(tmp_path, monkeypatch):
    store = PlatformStore(db_path=str(tmp_path / "platform.sqlite3"))
    monkeypatch.setattr(routes, "platform_store", store)
    monkeypatch.setenv("PLATFORM_MAINTAINER_SIGNUP_CODE", "local-maintainer")

    app = FastAPI()
    app.include_router(routes.router, prefix="/api/v1")
    return TestClient(app)


def test_platform_routes_cover_auth_company_api_token_dashboard_and_issues(
    tmp_path, monkeypatch
):
    client = _platform_client(tmp_path, monkeypatch)

    signup = client.post(
        "/api/v1/platform/auth/signup",
        json={
            "email": "builder@example.com",
            "password": "local-secret",
            "name": "Builder",
            "role": "b2b",
            "company_name": "Builder Co",
        },
    )
    assert signup.status_code == 200
    user_token = signup.json()["session_token"]
    user_auth = {"Authorization": f"Bearer {user_token}"}

    login = client.post(
        "/api/v1/platform/auth/login",
        json={"email": "builder@example.com", "password": "local-secret"},
    )
    assert login.status_code == 200
    assert login.json()["user"]["role"] == "b2b"

    session = client.get("/api/v1/platform/auth/session", headers=user_auth)
    assert session.status_code == 200
    assert session.json()["user"]["email"] == "builder@example.com"

    unauthenticated_companies = client.get(
        "/api/v1/platform/companies",
        params={"query": "acme"},
    )
    assert unauthenticated_companies.status_code == 401

    companies = client.get(
        "/api/v1/platform/companies",
        params={"query": "acme"},
        headers=user_auth,
    )
    assert companies.status_code == 200
    assert companies.json()["companies"]
    company_id = companies.json()["companies"][0]["id"]

    company = client.get(f"/api/v1/platform/companies/{company_id}", headers=user_auth)
    assert company.status_code == 200
    assert company.json()["company"]["id"] == company_id

    first_call_docs = client.get("/api/v1/platform/b2b/first-call", headers=user_auth)
    assert first_call_docs.status_code == 200
    assert first_call_docs.json()["endpoint"].endswith(
        "/platform/v1/companies/search?query=acme"
    )

    created_token = client.post(
        "/api/v1/platform/api-tokens",
        headers=user_auth,
        json={"name": "Docs quickstart"},
    )
    assert created_token.status_code == 200
    raw_api_token = created_token.json()["token"]
    assert raw_api_token.startswith("tbo_")

    token_list = client.get("/api/v1/platform/api-tokens", headers=user_auth)
    assert token_list.status_code == 200
    assert token_list.json()["tokens"][0]["prefix"] == raw_api_token[:12]
    assert "token" not in token_list.json()["tokens"][0]

    api_auth = {"Authorization": f"Bearer {raw_api_token}"}
    first_request = client.get(
        "/api/v1/platform/v1/companies/search",
        headers=api_auth,
        params={"query": "acme"},
    )
    assert first_request.status_code == 200
    assert first_request.json()["companies"][0]["id"] == company_id

    dashboard = client.get("/api/v1/platform/dashboard", headers=user_auth)
    assert dashboard.status_code == 200
    assert dashboard.json()["counters"]["api_requests_total"] == 1
    assert dashboard.json()["counters"]["api_tokens_active"] == 1

    limited_token = client.post(
        "/api/v1/platform/api-tokens",
        headers=user_auth,
        json={"name": "Dashboard only", "scopes": ["dashboard:read"]},
    )
    assert limited_token.status_code == 400
    assert "Unsupported API token scope" in limited_token.json()["detail"]

    wildcard_token = client.post(
        "/api/v1/platform/api-tokens",
        headers=user_auth,
        json={"name": "Wildcard", "scopes": ["*"]},
    )
    assert wildcard_token.status_code == 400
    assert "Unsupported API token scope" in wildcard_token.json()["detail"]

    empty_scope_token = client.post(
        "/api/v1/platform/api-tokens",
        headers=user_auth,
        json={"name": "No scopes", "scopes": []},
    )
    assert empty_scope_token.status_code == 200
    empty_scope_request = client.get(
        "/api/v1/platform/v1/companies/search",
        headers={"Authorization": f"Bearer {empty_scope_token.json()['token']}"},
        params={"query": "acme"},
    )
    assert empty_scope_request.status_code == 403

    issue = client.post(
        "/api/v1/platform/issues",
        headers=user_auth,
        json={
            "title": "API docs quickstart is confusing",
            "description": "The first request example needs a company search.",
            "severity": "medium",
        },
    )
    assert issue.status_code == 200

    maintainer_signup = client.post(
        "/api/v1/platform/auth/signup",
        json={
            "email": "maintainer@example.com",
            "password": "maintainer-secret",
            "role": "maintainer",
        },
    )
    assert maintainer_signup.status_code == 403

    maintainer_signup = client.post(
        "/api/v1/platform/auth/signup",
        json={
            "email": "maintainer@example.com",
            "password": "maintainer-secret",
            "role": "maintainer",
            "maintainer_code": "local-maintainer",
        },
    )
    assert maintainer_signup.status_code == 200
    maintainer_auth = {
        "Authorization": f"Bearer {maintainer_signup.json()['session_token']}"
    }

    issues = client.get("/api/v1/platform/maintainer/issues", headers=maintainer_auth)
    assert issues.status_code == 200
    assert issues.json()["issues"][0]["title"] == "API docs quickstart is confusing"

    revoked = client.delete(
        f"/api/v1/platform/api-tokens/{created_token.json()['api_token']['id']}",
        headers=user_auth,
    )
    assert revoked.status_code == 200

    revoked_request = client.get(
        "/api/v1/platform/v1/companies/search",
        headers=api_auth,
        params={"query": "acme"},
    )
    assert revoked_request.status_code == 401


def test_platform_store_persists_across_reload_and_hashes_tokens(tmp_path):
    db_path = tmp_path / "platform.sqlite3"
    store = PlatformStore(db_path=str(db_path))
    user = store.create_user(
        email="buyer@example.com",
        **{"password": "not-a-real-secret"},
        role="b2b",
        company_name="Buyer Co",
    )
    session = store.create_session(user["id"])
    api_token = store.create_api_token(user["id"], "Docs token")
    api_response = store.search_companies_with_api_token(
        api_token["token"], query="acme", path="/api/v1/platform/v1/companies/search"
    )
    issue = store.create_issue(
        user["id"],
        title="Search request failed in docs",
        description="Quickstart should show the Authorization header.",
    )
    maintainer = store.create_user(
        email="ops@example.com",
        **{"password": "not-a-real-secret"},
        role="maintainer",
    )

    reloaded = PlatformStore(db_path=str(db_path))

    assert reloaded.authenticate_session(session["session_token"])["email"] == (
        "buyer@example.com"
    )
    assert reloaded.authenticate_api_token(api_token["token"])["user_id"] == user["id"]
    assert api_response["companies"]
    assert reloaded.get_dashboard(user["id"])["counters"]["api_requests_total"] == 1
    assert reloaded.list_issues(maintainer["id"])[0]["id"] == issue["id"]

    listed_tokens = reloaded.list_api_tokens(user["id"])
    assert listed_tokens[0]["prefix"] == api_token["token"][:12]
    assert "token" not in listed_tokens[0]

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT token_hash, prefix FROM api_tokens WHERE id = ?",
            (api_token["api_token"]["id"],),
        ).fetchone()

    assert row is not None
    assert row[0] != api_token["token"]
    assert row[1] == api_token["token"][:12]


def test_platform_store_prefers_platform_db_path_override(tmp_path, monkeypatch):
    backend_path = tmp_path / "backend.sqlite3"
    platform_path = tmp_path / "platform.sqlite3"
    monkeypatch.setenv("BACKEND_DB_PATH", str(backend_path))
    monkeypatch.setenv("PLATFORM_DB_PATH", str(platform_path))

    store = PlatformStore()

    assert store._db_path == str(platform_path)
    assert platform_path.exists()
    assert not backend_path.exists()


def test_platform_store_uses_sqlite_pragmas_for_file_db(tmp_path):
    db_path = tmp_path / "platform.sqlite3"
    store = PlatformStore(db_path=str(db_path))

    with store._connect() as conn:
        busy_timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        foreign_keys = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]

    assert busy_timeout == 5000
    assert foreign_keys == 1
    assert journal_mode == "wal"


def test_first_api_call_uses_configurable_public_base_url(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKEND_PUBLIC_URL", "https://tab-organizer.example")
    store = PlatformStore(db_path=str(tmp_path / "platform.sqlite3"))
    user = store.create_user(
        email="buyer-public-url@example.com",
        **{"password": "not-a-real-secret"},
        role="b2b",
    )

    first_call = store.first_api_call(user["id"])

    assert first_call["base_url"] == "https://tab-organizer.example"
    assert (
        first_call["url"]
        == "https://tab-organizer.example/api/v1/platform/v1/companies/search?query=acme"
    )
    assert first_call["curl"].startswith(
        "curl -H 'Authorization: Bearer <api-token>' "
        "'https://tab-organizer.example/api/v1/platform/v1/companies/search"
    )


def test_maintainer_signup_accepts_account_type_with_valid_code(tmp_path, monkeypatch):
    client = _platform_client(tmp_path, monkeypatch)

    response = client.post(
        "/api/v1/platform/auth/signup",
        json={
            "email": "account-type-maintainer@example.com",
            "password": "maintainer-secret",
            "account_type": "maintainer",
            "maintainer_code": "local-maintainer",
        },
    )

    assert response.status_code == 200
    assert response.json()["user"]["role"] == "maintainer"


def test_maintainer_signup_rejects_account_type_without_valid_code(
    tmp_path, monkeypatch
):
    client = _platform_client(tmp_path, monkeypatch)

    response = client.post(
        "/api/v1/platform/auth/signup",
        json={
            "email": "account-type-maintainer-invalid@example.com",
            "password": "maintainer-secret",
            "account_type": "maintainer",
            "maintainer_code": "wrong-code",
        },
    )

    assert response.status_code == 403
    assert "Maintainer signup requires" in response.json()["detail"]


def test_docker_compose_passes_backend_public_url_to_backend_service():
    compose_path = Path(__file__).resolve().parents[2] / "docker-compose.yml"

    assert (
        "- BACKEND_PUBLIC_URL=${BACKEND_PUBLIC_URL:-http://localhost:8080}"
        in compose_path.read_text()
    )
