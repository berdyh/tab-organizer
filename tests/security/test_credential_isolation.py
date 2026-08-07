"""SEC-24..27, SEC-42: credential isolation (premise 4 in executable form).

The two ``sec_managed`` probes here (SEC-25, SEC-27) take their planted
secrets, their staging and their expected refusals from
``fixtures/credential_isolation.json`` (plan decision 44) — they auto-skip in
attached mode, so a language port that only runs attached mode never exercises
them, and the planned ``SEC_BOOT_*_CMD`` runner needs the contract as data.
The env-name allowlist itself stays where it already was, in
``fixtures/agent_env_allowlist.json``.
"""

import json
import re

import pytest

from tests.security import contracts
from tests.security.conftest import (
    FIXTURES_DIR,
    TOKEN_ENVS,
    ai_generate,
    switch_llm_provider,
)

pytestmark = [pytest.mark.security]

FIXTURE = "credential_isolation"


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

    spec = contracts.probe_spec(FIXTURE, "SEC-25")
    expect = contracts.expectations(spec, "SEC-25")
    contracts.assert_expect_keys_consumed(
        expect,
        [
            "subprocess_spawned",
            "env_key_allowlist_fixture",
            "env_keys_subset_of_allowlist",
            "sentinel_values_absent_from_env",
        ],
        label="SEC-25",
    )
    staging = spec["staging"]

    allowlist_file = expect["env_key_allowlist_fixture"]
    assert allowlist_file == "agent_env_allowlist", (
        f"SEC-25 names an unknown allowlist fixture: {allowlist_file!r}"
    )
    allowlist = set(_agent_env_allowlist_fixture()["allowlist"])

    sentinels = dict(staging["planted_secrets"])
    for key, value in sentinels.items():
        monkeypatch.setenv(key, value)
    # The configured service tokens are themselves secret material.
    if staging["additional_sentinels"]["configured_service_tokens"]:
        for env_name in TOKEN_ENVS.values():
            token_value = os.environ.get(env_name, "")
            if token_value:
                sentinels[env_name] = token_value

    monkeypatch.setenv(staging["command_env"], str(agent_cli_recorder.command))
    switch_llm_provider(ai, staging["provider"])
    ai_generate(ai, spec["input"]["generate_prompt"])

    assert expect["subprocess_spawned"] is True
    dumps = agent_cli_recorder.dumps()
    assert dumps, "recorder captured no agent invocation"
    recorded_env = dumps[-1]["env"]
    assert expect["env_keys_subset_of_allowlist"] is True
    assert set(recorded_env) <= allowlist, (
        f"agent env leaked non-allowlisted keys: "
        f"{set(recorded_env) - allowlist}"
    )
    assert expect["sentinel_values_absent_from_env"] is True
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

    spec = contracts.probe_spec(FIXTURE, "SEC-27")
    expect = contracts.expectations(spec, "SEC-27")
    contracts.assert_expect_keys_consumed(
        expect,
        ["response", "secret_absent_from", "service_log_level"],
        label="SEC-27",
    )
    staging = spec["staging"]

    leak_var = staging["recorder_leaks_secret_on_stderr"]
    secret = staging["planted_secrets"][leak_var]
    for name, value in staging["planted_secrets"].items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv(staging["command_env"], str(agent_cli_recorder.command))
    agent_cli_recorder.set_control(
        exit_code=staging["recorder_exit_code"], leak=secret
    )

    switch_llm_provider(ai, staging["provider"])
    with caplog.at_level(getattr(logging, expect["service_log_level"])):
        response = ai_generate(ai, spec["input"]["generate_prompt"])

    # Provider failure surfaces as 5xx.
    contracts.assert_response(response, expect["response"], label="SEC-27")

    surfaces = set(expect["secret_absent_from"])
    unknown = surfaces - {"http_response_body", "service_logs"}
    assert not unknown, f"SEC-27 names unknown leak surfaces: {sorted(unknown)}"
    assert "http_response_body" in surfaces and "service_logs" in surfaces, (
        "SEC-27 must check both the HTTP response body and service diagnostics"
    )
    assert secret not in response.text, "planted secret leaked into HTTP response"
    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert secret not in log_text, "planted secret leaked into service diagnostics"
