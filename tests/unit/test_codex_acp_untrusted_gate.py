"""`approve-all` and scraped page content must never meet.

`codex_acp` is safe by default (`--deny-all`) and had no untrusted-content
gate for exactly that reason. `CODEX_ACP_PERMISSION_MODE=approve-all` removes
the property that absence relied on: every tool call the model makes is
auto-approved. And this is the adapter the codex_cli refusal explicitly routes
scraped content TO -- "use codex_acp or claude_code for those flows" -- so the
dangerous combination is not hypothetical, it is the documented path.

The knob stays. It is a real consent control (docs/AI_CONFIG.md, the web-ui
settings page). Only the combination is refused.
"""

import pytest

from services.ai_engine.app.core.llm_client import LLMConfig
from services.ai_engine.app.providers.agent_cli import (
    AgentCLIError,
    CodexAcpLLMProvider,
)

# Any of the markers the envelope stamps onto scraped context.
UNTRUSTED_PROMPT = (
    "Summarise these tabs.\n<untrusted_web_content>\nBuy now\n</untrusted_web_content>"
)
CLEAN_PROMPT = "Summarise the following three bullet points I wrote myself."


def _provider() -> CodexAcpLLMProvider:
    return CodexAcpLLMProvider(LLMConfig(provider="codex_acp", model=""))


@pytest.mark.asyncio
async def test_approve_all_refuses_a_prompt_carrying_scraped_content(monkeypatch):
    monkeypatch.setenv("CODEX_ACP_PERMISSION_MODE", "approve-all")

    with pytest.raises(AgentCLIError) as exc:
        await _provider().generate(UNTRUSTED_PROMPT, None)

    assert "agent_cli_untrusted_context_refused" in str(exc.value)
    assert "approve-reads" in str(exc.value)


@pytest.mark.parametrize("mode", ["deny-all", "approve-reads", ""])
def test_the_safe_modes_are_what_they_claim(monkeypatch, mode):
    """Non-vacuity: the refusal must be about approve-all, not about the gate
    firing for every mode. If this passed vacuously the adapter would be
    unusable rather than safe."""
    if mode:
        monkeypatch.setenv("CODEX_ACP_PERMISSION_MODE", mode)
    else:
        monkeypatch.delenv("CODEX_ACP_PERMISSION_MODE", raising=False)

    resolved = _provider()._permission_mode()

    assert resolved != "approve-all"
    assert "--approve-all" not in _provider()._permission_args()


def test_an_unrecognised_mode_falls_back_to_the_narrower_one(monkeypatch):
    """A typo must not be read as approve-all, and must not crash."""
    monkeypatch.setenv("CODEX_ACP_PERMISSION_MODE", "APPROVE_EVERYTHING")

    assert _provider()._permission_mode() == "approve-reads"


def test_approve_all_still_works_for_a_prompt_with_no_scraped_content(monkeypatch):
    """The knob is a consent control, not a bug. An operator's own prompt is
    still allowed to run with full approval -- refusing that would be removing
    the feature rather than closing the combination."""
    monkeypatch.setenv("CODEX_ACP_PERMISSION_MODE", "approve-all")
    provider = _provider()

    assert provider._permission_mode() == "approve-all"
    assert provider._permission_args() == ["--approve-all"]
    assert not provider._has_untrusted_context_marker(CLEAN_PROMPT, None)
