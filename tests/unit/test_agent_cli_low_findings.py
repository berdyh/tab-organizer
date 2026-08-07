"""The two LOW findings from the independent security review.

Both are the same class as the MEDIUM ones already fixed -- something the
environment controls reaching a subprocess -- with a smaller reach. Fixed
together because "smaller reach" is a reason to rank them lower, not a reason
to leave them.
"""

import asyncio

import pytest

from services.ai_engine.app.core.llm_client import LLMConfig
from services.ai_engine.app.providers.agent_cli import (
    AgentCLIError,
    CodexAcpLLMProvider,
)


def _provider() -> CodexAcpLLMProvider:
    return CodexAcpLLMProvider(LLMConfig(provider="codex_acp", model=""))


@pytest.mark.parametrize(
    "hostile",
    [
        "--approve-all",
        "-y",
        "name with spaces",
        "name;rm -rf /",
        "x" * 129,
    ],
)
def test_a_session_name_that_could_be_read_as_a_flag_is_refused(monkeypatch, hostile):
    """The value is emitted as the VALUE of --name/--session and as a bare
    positional on `sessions close`, so one starting with '-' is parsed by acpx
    as a flag rather than as the session name."""
    monkeypatch.setenv("CODEX_ACP_SESSION_NAME", hostile)

    with pytest.raises(AgentCLIError) as exc:
        _provider()._session_name()

    assert "agent_cli_session_name_rejected" in str(exc.value)
    # The refusal must not echo the value: an operator could paste anything.
    assert hostile not in str(exc.value)


def test_a_well_shaped_session_name_still_works(monkeypatch):
    """Non-vacuity: the rule must not refuse every configured name, which
    would silently disable session reuse rather than secure it."""
    monkeypatch.setenv("CODEX_ACP_SESSION_NAME", "tab-organizer.review_1")

    assert _provider()._session_name() == "tab-organizer.review_1"


def test_the_generated_name_satisfies_its_own_rule(monkeypatch):
    """The shape check and the generator must agree, or an unconfigured run
    would start failing the moment the check is applied to it."""
    monkeypatch.delenv("CODEX_ACP_SESSION_NAME", raising=False)
    generated = _provider()._session_name()

    assert CodexAcpLLMProvider.SESSION_NAME_RE.match(generated)


@pytest.mark.asyncio
async def test_an_outer_cancellation_kills_the_child_process(monkeypatch):
    """`verify_provider_live` wraps the adapter call in its own wait_for(20).

    That CancelledError propagates through `communicate()` without reaching
    `_run`'s own TimeoutError branch, and the child is spawned with
    `start_new_session=True` so it never sees the parent's signals. Every probe
    of a hung CLI therefore leaked a detached process -- concretely, a
    logged-out `gemini -p` blocks forever on a browser-login prompt.
    """
    killed = []

    class _HangingProcess:
        pid = 4242
        returncode = None

        async def communicate(self, input=None):
            await asyncio.sleep(3600)

    async def _fake_exec(*args, **kwargs):
        return _HangingProcess()

    monkeypatch.setattr(
        "services.ai_engine.app.providers.agent_cli.asyncio.create_subprocess_exec",
        _fake_exec,
    )

    provider = _provider()
    monkeypatch.setattr(
        provider,
        "_terminate_process",
        lambda proc: killed.append(getattr(proc, "pid", None)) or asyncio.sleep(0),
    )

    task = asyncio.create_task(provider._run(["acpx", "--version"], None))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert killed == [4242], "the child survived an outer cancellation"
