"""SEC-28..33, SEC-46..47: agent subprocess hardening (all sec_managed, recorder-based).

These freeze the agent_cli.py invocation properties that Addendum (b) says must
survive the SDK migration. The agent-backed generation is exercised through the
tool-free ``/generate`` seam; the frozen observable is the argv / stdin / env /
cwd of the spawned CLI, which the recorder stub captures.

Every input and every expected refusal in this file lives in
``fixtures/agent_subprocess_hardening.json`` (plan decision 44), not inline: all
eight probes are ``sec_managed`` and therefore auto-skip in attached mode, so
without a data contract a language port could go green having never exercised
any of them, and the planned ``SEC_BOOT_*_CMD`` runner would have nothing to
run over. The probe bodies below are staging + observation only; what must hold
is in the fixture, and ``contracts.assert_expect_keys_consumed`` fails if a
fixture expectation stops being checked here.
"""

import time

import pytest

from tests.security import contracts
from tests.security.conftest import ai_generate, switch_llm_provider

pytestmark = [pytest.mark.security, pytest.mark.sec_managed]

FIXTURE = "agent_subprocess_hardening"


def _spec(probe_id: str) -> dict:
    return contracts.probe_spec(FIXTURE, probe_id)


def _stage(spec, ai, recorder, monkeypatch, tmp_path) -> dict:
    """Apply a probe's ``staging`` block; return runtime substitutions.

    Mirrors, in order, what the probes used to do inline: plant/remove env
    vars, point the provider's command env at the recording stub, stage a HOME
    when the adapter needs credentials to be selectable at all, then select the
    provider.
    """
    staging = spec.get("staging") or {}
    known = {
        "provider",
        "command_env",
        "recorder_format",
        "recorder_sleep_seconds",
        "set_env",
        "unset_env",
        "home_files",
        "provider_switch_expect_status",
        "workdir_env",
        "workdir_basename",
    }
    unknown = set(staging) - known
    if unknown:
        raise contracts.FixtureContractError(
            f"{FIXTURE}: unknown staging keys {sorted(unknown)}"
        )

    subs: dict = {}

    for name in staging.get("unset_env", []):
        monkeypatch.delenv(name, raising=False)
    for name, value in (staging.get("set_env") or {}).items():
        monkeypatch.setenv(name, value)

    if "workdir_env" in staging:
        workdir = tmp_path / staging["workdir_basename"]
        monkeypatch.setenv(staging["workdir_env"], str(workdir))
        subs["workdir"] = str(workdir)

    home_files = staging.get("home_files") or {}
    if home_files:
        home = tmp_path / "staged-home"
        for relative, content in home_files.items():
            target = home / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        monkeypatch.setenv("HOME", str(home))

    monkeypatch.setenv(staging["command_env"], str(recorder.command))

    control: dict = {}
    if "recorder_format" in staging:
        control["fmt"] = staging["recorder_format"]
    if "recorder_sleep_seconds" in staging:
        control["sleep"] = staging["recorder_sleep_seconds"]
    if control:
        recorder.set_control(**control)

    provider = staging["provider"]
    if "provider_switch_expect_status" in staging:
        # Asserted, never skipped on: an adapter that reports unavailable would
        # otherwise turn this probe into a silent no-op.
        expected = staging["provider_switch_expect_status"]
        response = ai.post(
            "/providers/switch", token=ai.token, json={"llm_provider": provider}
        )
        assert response.status_code == expected, (
            f"{provider} was not selectable, so this probe would assert nothing: "
            f"{response.status_code} {response.text[:300]}"
        )
    else:
        switch_llm_provider(ai, provider)

    return subs


def _last_dump(recorder, expect, probe_id):
    """Return the last recorded invocation, honouring ``expect.subprocess_spawned``."""
    assert expect["subprocess_spawned"] is True, (
        f"{probe_id}: fixture must declare subprocess_spawned:true for this probe "
        "to have an invocation to inspect"
    )
    dumps = recorder.dumps()
    assert dumps, f"{probe_id}: recorder captured no invocation"
    return dumps[-1]


def test_sec28_claude_tools_disabled_by_default(
    ai, agent_cli_recorder, monkeypatch, tmp_path
):
    spec = _spec("SEC-28")
    expect = contracts.expectations(spec, "SEC-28")
    contracts.assert_expect_keys_consumed(
        expect, ["subprocess_spawned", "argv"], label="SEC-28"
    )

    _stage(spec, ai, agent_cli_recorder, monkeypatch, tmp_path)
    ai_generate(ai, spec["input"]["generate_prompt"])

    dump = _last_dump(agent_cli_recorder, expect, "SEC-28")
    contracts.assert_argv(dump["argv"], expect["argv"], label="SEC-28 argv")


def test_sec29_codex_sandboxed_and_ephemeral(
    ai, agent_cli_recorder, monkeypatch, tmp_path
):
    spec = _spec("SEC-29")
    expect = contracts.expectations(spec, "SEC-29")
    contracts.assert_expect_keys_consumed(
        expect, ["subprocess_spawned", "argv"], label="SEC-29"
    )

    _stage(spec, ai, agent_cli_recorder, monkeypatch, tmp_path)
    ai_generate(ai, spec["input"]["generate_prompt"])

    dump = _last_dump(agent_cli_recorder, expect, "SEC-29")
    contracts.assert_argv(dump["argv"], expect["argv"], label="SEC-29 argv")


def test_sec30_codex_refuses_untrusted_context_by_default(
    ai, agent_cli_recorder, monkeypatch, tmp_path
):
    spec = _spec("SEC-30")
    expect = contracts.expectations(spec, "SEC-30")
    contracts.assert_expect_keys_consumed(
        expect, ["response", "subprocess_invocation_delta"], label="SEC-30"
    )

    _stage(spec, ai, agent_cli_recorder, monkeypatch, tmp_path)
    before = agent_cli_recorder.invocation_count

    response = ai_generate(ai, spec["input"]["generate_prompt"])

    contracts.assert_response(response, expect["response"], label="SEC-30")
    assert agent_cli_recorder.invocation_count - before == (
        expect["subprocess_invocation_delta"]
    ), "codex was invoked on untrusted input"


def test_sec31_guardrail_preamble_precedes_untrusted_content(
    ai, agent_cli_recorder, monkeypatch, tmp_path
):
    spec = _spec("SEC-31")
    expect = contracts.expectations(spec, "SEC-31")
    contracts.assert_expect_keys_consumed(
        expect, ["subprocess_spawned", "stdin"], label="SEC-31"
    )

    _stage(spec, ai, agent_cli_recorder, monkeypatch, tmp_path)
    ai_generate(ai, spec["input"]["generate_prompt"])

    dump = _last_dump(agent_cli_recorder, expect, "SEC-31")
    contracts.assert_text(dump["stdin"], expect["stdin"], label="SEC-31 stdin")


@pytest.mark.slow
def test_sec32_runaway_agent_bounded(ai, agent_cli_recorder, monkeypatch, tmp_path):
    spec = _spec("SEC-32")
    expect = contracts.expectations(spec, "SEC-32")
    contracts.assert_expect_keys_consumed(
        expect, ["max_elapsed_seconds", "response"], label="SEC-32"
    )

    _stage(spec, ai, agent_cli_recorder, monkeypatch, tmp_path)

    start = time.monotonic()
    response = ai_generate(ai, spec["input"]["generate_prompt"])
    elapsed = time.monotonic() - start

    assert elapsed < expect["max_elapsed_seconds"], (
        f"runaway agent not bounded (took {elapsed:.1f}s, contract "
        f"requires < {expect['max_elapsed_seconds']}s)"
    )
    contracts.assert_response(response, expect["response"], label="SEC-32")


def test_sec33_agent_cwd_confined_to_workdir(
    ai, agent_cli_recorder, monkeypatch, tmp_path
):
    spec = _spec("SEC-33")
    expect = contracts.expectations(spec, "SEC-33")
    contracts.assert_expect_keys_consumed(
        expect, ["subprocess_spawned", "cwd"], label="SEC-33"
    )

    subs = _stage(spec, ai, agent_cli_recorder, monkeypatch, tmp_path)
    ai_generate(ai, spec["input"]["generate_prompt"])

    dump = _last_dump(agent_cli_recorder, expect, "SEC-33")
    assert dump["cwd"] == expect["cwd"].format(**subs), (
        "agent cwd escaped the scratch workdir"
    )


# --------------------------------------------------------------------------
# SEC-46..47 (SECSUITE 1.5.0): the gemini_cli adapter.
#
# Added because the frozen suite did NOT cover it. SEC-28/31/32/33 drive
# claude_code and SEC-29/30 drive codex_cli; every one of them names its
# provider, so a third subscription CLI inherits only the base-class
# guarantees that happen to be shared (env allowlist via SEC-25/42, workdir
# and timeout via the base class -- and those two are still only EXERCISED
# through claude). The two properties that are gemini-specific are the two
# below, and neither had a probe.
#
# Both stage `~/.gemini/oauth_creds.json` into a temp HOME and ASSERT the
# provider switch returned 200 (fixture key `provider_switch_expect_status`)
# rather than skipping on it: the adapter refuses to report available without
# credentials, so an unstaged probe would skip silently and assert nothing.
# --------------------------------------------------------------------------


def test_sec46_gemini_headless_read_only_and_never_interactive(
    ai, agent_cli_recorder, monkeypatch, tmp_path
):
    """SEC-46: the gemini subprocess is headless, read-only, and stdin-free.

    Three things must hold together, and all three are one-flag regressions:
    `-p` (headless -- without it the CLI starts an interactive session and
    never returns), `--approval-mode plan` (read-only -- `yolo` would hand a
    model driven by page text the write and shell tools), and stdin closed
    (the CLI's login prompt ignores EOF, so nothing must resemble an answer
    to it). This adapter puts the guardrail envelope in argv, not stdin, so
    SEC-31's ordering invariant is re-checked on the `-p` argument.
    """
    spec = _spec("SEC-46")
    expect = contracts.expectations(spec, "SEC-46")
    contracts.assert_expect_keys_consumed(
        expect,
        [
            "subprocess_spawned",
            "argv",
            "stdin",
            "argv_value_after",
            "argv_value_text",
        ],
        label="SEC-46",
    )

    _stage(spec, ai, agent_cli_recorder, monkeypatch, tmp_path)
    ai_generate(ai, spec["input"]["generate_prompt"])

    dump = _last_dump(agent_cli_recorder, expect, "SEC-46")
    argv = dump["argv"]
    contracts.assert_argv(argv, expect["argv"], label="SEC-46 argv")
    contracts.assert_text(dump["stdin"], expect["stdin"], label="SEC-46 stdin")

    flag = expect["argv_value_after"]
    assert flag in argv, f"SEC-46: {flag!r} absent from argv: {argv}"
    prompt_arg = argv[argv.index(flag) + 1]
    contracts.assert_text(
        prompt_arg, expect["argv_value_text"], label=f"SEC-46 argv value after {flag}"
    )


def test_sec47_gemini_refuses_untrusted_context_by_default(
    ai, agent_cli_recorder, monkeypatch, tmp_path
):
    """SEC-47: scraped content never reaches a CLI with no tool-free mode.

    Same reasoning as SEC-30 for codex: gemini's most restrictive documented
    approval mode still allows `read_file`, `google_web_search` and
    `web_fetch` (read from the CLI's own bundled `policies/read-only.toml`),
    each an exfiltration channel for instructions injected into a captured
    page. claude_code is the flow for untrusted content because `--tools ""`
    removes the channel entirely.
    """
    spec = _spec("SEC-47")
    expect = contracts.expectations(spec, "SEC-47")
    contracts.assert_expect_keys_consumed(
        expect, ["response", "subprocess_invocation_delta"], label="SEC-47"
    )

    _stage(spec, ai, agent_cli_recorder, monkeypatch, tmp_path)
    before = agent_cli_recorder.invocation_count

    response = ai_generate(ai, spec["input"]["generate_prompt"])

    contracts.assert_response(response, expect["response"], label="SEC-47")
    assert agent_cli_recorder.invocation_count - before == (
        expect["subprocess_invocation_delta"]
    ), "gemini was invoked on untrusted scraped input"
