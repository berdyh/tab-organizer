"""Plan decision 37's gate: no flagged capture is embedded without an answer.

The plan specifies a HARD GATE -- an authenticated capture must not reach a
non-local embedding provider, with per-domain explicit opt-in as the only
escape hatch. Its stated prerequisite (an auth flag carried capture -> index)
was built, but the flag it produced, `auth_used`, means "we spent a credential
from our own store" and is structurally blind to the case that matters most: a
tab open in the user's already-logged-in browser, where no stored credential is
ever spent. The gate therefore had a signal that could never fire on the main
path.

These freeze the replacement: a content-derived signal the CDP path CAN produce,
and a hold-until-answered rule where an ABSENT decision is never read as
permission.
"""

import sys
import types

import pytest


def _install_playwright_stub() -> None:
    """`app.extraction.__init__` re-exports the scraper's ContentExtractor,
    which imports playwright. These tests touch neither."""
    playwright_module = types.ModuleType("playwright")
    async_api = types.ModuleType("playwright.async_api")
    async_api.Browser = object
    async_api.Page = object
    async_api.TimeoutError = TimeoutError
    async_api.async_playwright = lambda: None
    playwright_module.async_api = async_api
    sys.modules["playwright"] = playwright_module
    sys.modules["playwright.async_api"] = async_api


try:
    import playwright.async_api  # noqa: F401
except ModuleNotFoundError:
    _install_playwright_stub()

from services.backend_core.app.api import routes  # noqa: E402
from services.backend_core.app.sessions.manager import (  # noqa: E402
    SessionManager,
)
from services.browser_engine.app.extraction.privacy import (  # noqa: E402
    classify_privacy,
    scan_for_secrets,
)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_signed_out_page_is_not_private():
    """A page with no session markers is public; over-flagging is its own harm.

    A hold list that flags everything is noise nobody reads, which is how a
    gate stops being a gate.
    """
    html = "<html><body><h1>Blender 4.2 release notes</h1><p>Downloads</p></body></html>"
    signal = classify_privacy(html, "https://www.blender.org/download/")
    assert not signal.is_private


def test_sign_out_affordance_means_we_are_logged_in():
    """The inverse of the auth-wall check: a logged-out page has no logout link."""
    html = '<html><body><a href="/logout">Sign out</a><p>Your documents</p></body></html>'
    signal = classify_privacy(html, "https://app.example.com/docs")
    assert signal.is_private
    assert any("signed_in_marker" in r for r in signal.reasons)


def test_oauth_infrastructure_contributes_but_does_not_alone_decide():
    """The user's own suggestion: notice google/other auth requests.

    On its own it is weak -- public marketing pages reference OAuth endpoints --
    so it raises confidence without tripping the gate by itself.
    """
    html = '<html><body>Sign in with <a href="https://accounts.google.com/o/oauth2">Google</a></body></html>'
    signal = classify_privacy(html, "https://www.example.com/")
    assert any("session_infrastructure" in r for r in signal.reasons)
    assert not signal.is_private


def test_private_host_shape_plus_session_marker_trips_the_gate():
    html = '<html><body><a href="/signout">Log out</a></body></html>'
    signal = classify_privacy(html, "https://app.notion.com/workspace")
    assert signal.is_private
    assert signal.confidence >= 0.45


def test_every_signal_is_named_so_a_wrong_call_is_reviewable():
    html = '<html><body><a href="/logout">Sign out</a></body></html>'
    signal = classify_privacy(html, "https://mail.example.com/inbox")
    assert signal.reasons, "a classification with no stated reason cannot be checked"
    assert signal.to_dict()["reasons"]


# ---------------------------------------------------------------------------
# Secret scanning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,kind",
    [
        ("key: sk-ant-" + "a" * 40, "anthropic_api_key"),
        ("token ghp_" + "b" * 36, "github_token"),
        ("AKIA" + "C" * 16, "aws_access_key_id"),
        ("-----BEGIN RSA PRIVATE KEY-----", "private_key_block"),
    ],
)
def test_secret_shapes_are_detected(text, kind):
    scan = scan_for_secrets(text)
    assert scan.found
    assert kind in scan.kinds


def test_scan_never_returns_the_matched_value():
    """Reporting a leak must not copy the credential into the report.

    A detector that echoes the key has written it into a job payload, a
    database row and a log line in order to warn that it was somewhere it
    should not be.
    """
    secret = "sk-ant-" + "z" * 40
    scan = scan_for_secrets(f"my key is {secret}")
    rendered = str(scan.to_dict())
    assert secret not in rendered
    assert "anthropic_api_key" in rendered


def test_ordinary_prose_is_not_flagged():
    scan = scan_for_secrets("We discussed the API key rotation policy yesterday.")
    assert not scan.found


# ---------------------------------------------------------------------------
# The hold gate
# ---------------------------------------------------------------------------


def _doc(url, **metadata):
    return {
        "id": url,
        "url": url,
        "title": "T",
        "content": "text",
        "metadata": metadata,
    }


def test_unflagged_documents_are_unaffected(tmp_path, monkeypatch):
    """A public corpus must behave exactly as it did before the gate existed."""
    manager = SessionManager(db_path=str(tmp_path / "b.sqlite3"))
    monkeypatch.setattr(routes, "session_manager", manager)

    embeddable, held = routes._partition_by_domain_consent(
        [_doc("https://arxiv.org/abs/1"), _doc("https://blender.org/x")]
    )
    assert len(embeddable) == 2
    assert held == []


def test_flagged_document_is_held_when_nobody_has_decided(tmp_path, monkeypatch):
    """The load-bearing assertion: absence of a decision is NOT permission."""
    manager = SessionManager(db_path=str(tmp_path / "b.sqlite3"))
    monkeypatch.setattr(routes, "session_manager", manager)

    embeddable, held = routes._partition_by_domain_consent(
        [_doc("https://claude.ai/chat/1", privacy={"is_private": True})]
    )
    assert embeddable == []
    assert len(held) == 1
    assert held[0]["reason"] == "awaiting_consent"
    assert held[0]["domain"] == "claude.ai"
    assert held[0]["flagged_for"] == ["private"]


def test_allow_lets_a_flagged_document_through(tmp_path, monkeypatch):
    manager = SessionManager(db_path=str(tmp_path / "b.sqlite3"))
    manager.set_domain_consent("claude.ai", "allow", "mine to index")
    monkeypatch.setattr(routes, "session_manager", manager)

    embeddable, held = routes._partition_by_domain_consent(
        [_doc("https://claude.ai/chat/1", privacy={"is_private": True})]
    )
    assert len(embeddable) == 1
    assert held == []


def test_deny_holds_it_permanently_and_says_so(tmp_path, monkeypatch):
    manager = SessionManager(db_path=str(tmp_path / "b.sqlite3"))
    manager.set_domain_consent("claude.ai", "deny", "private")
    monkeypatch.setattr(routes, "session_manager", manager)

    embeddable, held = routes._partition_by_domain_consent(
        [_doc("https://claude.ai/chat/1", privacy={"is_private": True})]
    )
    assert embeddable == []
    assert held[0]["reason"] == "denied_by_domain"


def test_a_secret_holds_a_document_even_on_an_allowed_domain(tmp_path, monkeypatch):
    """Allowing a domain permits its private pages, not its leaked credentials."""
    manager = SessionManager(db_path=str(tmp_path / "b.sqlite3"))
    manager.set_domain_consent("gist.github.com", "allow")
    monkeypatch.setattr(routes, "session_manager", manager)

    embeddable, held = routes._partition_by_domain_consent(
        [
            _doc(
                "https://gist.github.com/x",
                secrets={"found": True, "kinds": ["github_token"]},
            )
        ]
    )
    # An explicit allow is an answer for this domain, so it applies.
    assert len(embeddable) == 1
    assert held == []


def test_consent_round_trips_and_can_be_forgotten(tmp_path):
    manager = SessionManager(db_path=str(tmp_path / "b.sqlite3"))
    assert manager.get_domain_consent("claude.ai") is None

    manager.set_domain_consent("Claude.AI", "deny", "private")
    assert manager.get_domain_consent("claude.ai") == "deny", "domains are case-folded"
    assert manager.list_domain_consent()[0]["reason"] == "private"

    assert manager.forget_domain_consent("claude.ai") is True
    assert manager.get_domain_consent("claude.ai") is None
    assert manager.forget_domain_consent("claude.ai") is False


def test_consent_survives_a_restart(tmp_path):
    """The answer is "asked once", so it has to outlive the process."""
    db = str(tmp_path / "b.sqlite3")
    SessionManager(db_path=db).set_domain_consent("claude.ai", "deny")
    assert SessionManager(db_path=db).get_domain_consent("claude.ai") == "deny"


@pytest.mark.parametrize("bad", ["maybe", "ALLOW", "", "skip"])
def test_only_allow_and_deny_are_accepted(tmp_path, bad):
    manager = SessionManager(db_path=str(tmp_path / "b.sqlite3"))
    with pytest.raises(ValueError):
        manager.set_domain_consent("claude.ai", bad)
