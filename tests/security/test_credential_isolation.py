"""SEC-24..27, SEC-42: credential isolation (premise 4 in executable form)."""

import json
import re

import pytest

from tests.security.conftest import (
    FIXTURES_DIR,
    TOKEN_ENVS,
    ai_generate,
    switch_llm_provider,
)

pytestmark = [pytest.mark.security]


def _agent_env_allowlist_fixture() -> dict:
    path = FIXTURES_DIR / "agent_env_allowlist.json"
    return json.loads(path.read_text(encoding="utf-8"))


BROWSER_AUTH_ENDPOINTS = [
    ("GET", "/auth/pending"),
    ("GET", "/auth/pending/sess-123"),
    ("POST", "/auth/credentials"),
    ("DELETE", "/auth/pending/example.com"),
    ("POST", "/auth/expire"),
]


@pytest.mark.parametrize("method,path", BROWSER_AUTH_ENDPOINTS)
def test_agent_principal_cannot_reach_credential_surfaces(method, path, browser):
    """SEC-24: the agent token opens none of the browser auth endpoints."""
    import os

    agent_token = os.environ[TOKEN_ENVS["agent"]]
    body = {"domain": "example.com", "credentials": {"u": "x"}} if method == "POST" else None
    response = browser.request(method, path, token=agent_token, json=body)
    assert response.status_code == 401, (
        f"{method} {path} accepted the agent token (expected 401)"
    )


@pytest.mark.sec_managed
def test_agent_subprocess_env_contains_no_secrets(ai, agent_cli_recorder, monkeypatch):
    """SEC-25: the spawned agent CLI env is allowlisted and secret-free.

    Black-box: the observed subprocess env is compared against the frozen
    ``fixtures/agent_env_allowlist.json`` contract, not against the Python
    provider's ``ENV_ALLOWLIST`` constant (that comparison is SEC-42, a
    separate ``sec_seam`` drift check). No longer a seam exception -- see
    tests/security/README.md's "Seam exceptions" section.
    """
    import os

    allowlist = set(_agent_env_allowlist_fixture()["allowlist"])

    sentinels = {
        "OPENROUTER_API_KEY": "sentinel-openrouter-" + "a" * 12,
        "OPENAI_API_KEY": "sentinel-openai-" + "b" * 12,
        "ANTHROPIC_API_KEY": "sentinel-anthropic-" + "c" * 12,
        "CREDENTIAL_ENCRYPTION_KEY": "sentinel-credkey-" + "d" * 12,
    }
    for key, value in sentinels.items():
        monkeypatch.setenv(key, value)
    # The configured service tokens are themselves secret material.
    for env_name in TOKEN_ENVS.values():
        token_value = os.environ.get(env_name, "")
        if token_value:
            sentinels[env_name] = token_value

    monkeypatch.setenv("CLAUDE_CODE_COMMAND", str(agent_cli_recorder.command))
    switch_llm_provider(ai, "claude_code")
    ai_generate(ai, "Summarise the retrieved context.")

    dumps = agent_cli_recorder.dumps()
    assert dumps, "recorder captured no agent invocation"
    recorded_env = dumps[-1]["env"]
    assert set(recorded_env) <= allowlist, (
        f"agent env leaked non-allowlisted keys: "
        f"{set(recorded_env) - allowlist}"
    )
    flattened = "\n".join(f"{k}={v}" for k, v in recorded_env.items())
    for name, value in sentinels.items():
        assert value not in flattened, f"secret {name} leaked into agent env"


@pytest.mark.sec_seam
def test_sec42_env_allowlist_matches_frozen_fixture():
    """SEC-42: ``AgentCLILLMProvider.ENV_ALLOWLIST`` must equal the frozen fixture.

    Seam exception: imports the Python constant to pin it against
    ``fixtures/agent_env_allowlist.json``, the contract SEC-25 checks
    black-box observed behaviour against. If the Python allowlist drifts
    from the fixture -- widens or narrows without a deliberate fixture
    update -- this fails loudly instead of silently changing what SEC-25
    permits. TS porting rule: once the TS Agent SDK adapter lands, pin its
    equivalent allowlist constant/config against the same fixture file.
    """
    from services.ai_engine.app.providers.agent_cli import AgentCLILLMProvider

    fixture_allowlist = set(_agent_env_allowlist_fixture()["allowlist"])
    assert AgentCLILLMProvider.ENV_ALLOWLIST == fixture_allowlist, (
        "AgentCLILLMProvider.ENV_ALLOWLIST has drifted from the frozen "
        "fixtures/agent_env_allowlist.json contract -- if this widening/narrowing "
        "is deliberate, update the fixture (with a docs/MODULE_INDEX.md ledger "
        "row + SECSUITE_VERSION bump) in the same change; symmetric diff: "
        f"{AgentCLILLMProvider.ENV_ALLOWLIST ^ fixture_allowlist}"
    )


@pytest.mark.sec_seam
def test_mcp_tool_surface_has_no_credential_verbs():
    """SEC-26: the agent corpus-tool registry exposes no credential verbs.

    TS porting rule: run the TS MCP server over stdio, issue tools/list, and
    apply the same allowlist + forbidden-pattern assertions to the returned
    tool names.
    """
    from scripts.mcp import tabs

    tool_names = set(tabs.TOOL_FUNCTIONS)
    allowed = {
        "tab_import_from_browser",
        "tab_import_status",
        "tab_search",
        "tab_cluster",
        "tab_open",
        "tab_export",
    }
    assert tool_names <= allowed, f"unexpected tab tools: {tool_names - allowed}"

    forbidden = re.compile(r"(auth|credential|password|secret|token(?!_env))", re.I)
    for name in tool_names:
        assert not forbidden.search(name), f"tool name exposes a credential verb: {name}"


@pytest.mark.sec_managed
def test_agent_diagnostics_redact_planted_secret(ai, agent_cli_recorder, monkeypatch, caplog):
    """SEC-27: a planted secret never surfaces in responses or diagnostics."""
    import logging

    secret = "sk-ant-sentinel" + "e" * 16
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    monkeypatch.setenv("CLAUDE_CODE_COMMAND", str(agent_cli_recorder.command))
    agent_cli_recorder.set_control(exit_code=1, leak=secret)

    switch_llm_provider(ai, "claude_code")
    with caplog.at_level(logging.WARNING):
        response = ai_generate(ai, "Summarise the retrieved context.")

    assert response.status_code >= 500  # provider failure surfaces as 5xx
    assert secret not in response.text, "planted secret leaked into HTTP response"
    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert secret not in log_text, "planted secret leaked into service diagnostics"
