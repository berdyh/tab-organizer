"""What `configure-provider` is allowed to call "verified".

`scripts/cli.py configure-provider` captioned its menus "(only
verified-available options shown)" and wrote the result to .env as the user's
verified choice. For a `type: cloud` provider the entire basis for that word
was, in `LLMClient._raw_provider_runtime_state`:

    configured = bool(api_key_env and os.getenv(api_key_env, "").strip())
    state["available"] = configured

A key revoked an hour ago, a key with a typo, a key for the wrong account and a
key that works are all identical to that expression. It is the same shape of
error as the one this branch's headline fix corrected -- `openrouter.supports.
embeddings: false`, annotated "VERIFIED", inferred from a `GET /v1/models`
listing that cannot see `POST /v1/embeddings`. CLAUDE.md's rule after that
correction: a capability claim comes from calling the endpoint, never from
reading a listing; if you cannot call it, record the claim as unverified.

These tests pin the three states apart at the `LLMClient` layer.
`tests/unit/test_cli_configure_provider.py` covers what the CLI does with them.
"""

import asyncio

import httpx
import pytest

from services.ai_engine.app.core.llm_client import (
    EmbeddingConfig,
    LLMClient,
    LLMConfig,
)


def _client():
    """An LLMClient with both roles pinned, so nothing here depends on the
    ambient AI_PROVIDER/EMBEDDING_PROVIDER (which have no default anywhere)."""
    return LLMClient(
        LLMConfig(provider="ollama", model="qwen3"),
        EmbeddingConfig(provider="ollama", model="nomic-embed-text"),
    )


def _http_status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request(
        "POST", "https://generativelanguage.googleapis.com/v1beta/x?key=AIzaSuperSecret"
    )
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError(
        f"Client error '{status}' for url '{request.url}'",
        request=request,
        response=response,
    )


# --------------------------------------------------------------------------
# A set API key is "configured". It is not "verified".
# --------------------------------------------------------------------------


def test_a_set_api_key_is_configured_but_explicitly_not_verified(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-revoked-yesterday-but-still-a-string")

    state = _client().get_provider_runtime_state("openai", "llm")

    # `available` still means configured -- flipping it to False offline would
    # make the tool unusable on a plane, which is not the fix.
    assert state["available"] is True
    assert state["api_key_configured"] is True
    # ...but nothing here is entitled to the word "verified".
    assert state["verification"] == LLMClient.VERIFICATION_UNVERIFIED
    assert "not a working credential" in state["verification_reason"]


def test_a_local_provider_is_also_only_unverified_until_called():
    """The conflation was not cloud-only: `state["available"] = True` at the end
    of the function is just as unproven."""
    state = _client().get_provider_runtime_state("ollama", "embeddings")

    assert state["verification"] == LLMClient.VERIFICATION_UNVERIFIED


# --------------------------------------------------------------------------
# Only a real call moves the verdict, and it moves it in three directions.
# --------------------------------------------------------------------------


def test_a_successful_real_call_is_the_only_thing_that_verifies(monkeypatch):
    class _Adapter:
        async def generate(self, prompt, system=None):
            return "pong"

    monkeypatch.setattr(
        LLMClient, "_create_llm_provider", lambda self, config=None: _Adapter()
    )

    result = asyncio.run(_client().verify_provider_live("openai", "llm", "gpt-5.6-luna"))

    assert result["verification"] == LLMClient.VERIFICATION_VERIFIED


def test_an_empty_response_is_not_a_verification(monkeypatch):
    """An adapter that answers but serves nothing is a broken path reporting
    success -- the WI0-B1 shape."""

    class _Adapter:
        async def generate(self, prompt, system=None):
            return "   "

    monkeypatch.setattr(
        LLMClient, "_create_llm_provider", lambda self, config=None: _Adapter()
    )

    result = asyncio.run(_client().verify_provider_live("openai", "llm", "gpt-5.6-luna"))

    assert result["verification"] == LLMClient.VERIFICATION_REFUTED


@pytest.mark.parametrize("status", [401, 403, 402, 400, 404])
def test_the_provider_rejecting_the_call_refutes_the_route(status):
    """A 401 is not "we could not tell" -- the provider looked at the credential
    and said no. Positive proof, so it must not be recoverable as merely
    unknown."""
    result = LLMClient.classify_live_probe_error(_http_status_error(status))

    assert result["verification"] == LLMClient.VERIFICATION_REFUTED
    assert result["status_code"] == status


@pytest.mark.parametrize(
    "exc",
    [
        asyncio.TimeoutError(),
        httpx.ConnectError("All connection attempts failed"),
        httpx.ReadTimeout("timed out"),
    ],
)
def test_being_unable_to_call_never_refutes_and_never_verifies(exc):
    """Offline is not proof of anything. Refuting hides a provider from the
    operator, so it takes a verdict from the provider, not a dead socket."""
    result = LLMClient.classify_live_probe_error(exc)

    assert result["verification"] == LLMClient.VERIFICATION_UNVERIFIED


def test_a_provider_side_5xx_does_not_refute():
    result = LLMClient.classify_live_probe_error(_http_status_error(503))

    assert result["verification"] == LLMClient.VERIFICATION_UNVERIFIED
    assert result["status_code"] == 503


def test_a_probe_timeout_is_reported_rather_than_raised(monkeypatch):
    class _Adapter:
        async def generate(self, prompt, system=None):
            await asyncio.sleep(5)
            return "pong"

    monkeypatch.setattr(
        LLMClient, "_create_llm_provider", lambda self, config=None: _Adapter()
    )

    result = asyncio.run(
        _client().verify_provider_live("openai", "llm", "gpt-5.6-luna", timeout=0.05)
    )

    assert result["verification"] == LLMClient.VERIFICATION_UNVERIFIED
    assert "timed out" in result["verification_reason"]


# --------------------------------------------------------------------------
# The probe path handles provider-authored strings, so it redacts harder.
# --------------------------------------------------------------------------


def test_the_http_branch_reports_the_status_and_nothing_the_provider_wrote():
    """First line of defence: the HTTP verdict is built from the status code
    alone.

    An earlier version of this test asserted only that the key was absent from
    the result -- which it could not fail, because on this branch the key never
    enters the string in the first place. Asserting the reason EXACTLY pins the
    property that actually holds the line: httpx puts the whole request URL,
    query string included, into the text of every error it raises, and the
    Gemini adapter authenticates with `params={"key": self.api_key}`. So
    `f"...: {exc}"` here would be printing the API key.
    """
    result = LLMClient.classify_live_probe_error(_http_status_error(401))

    assert result["verification_reason"] == (
        "the provider rejected the call with HTTP 401"
    )


def test_a_url_query_credential_is_redacted(monkeypatch):
    """Second line of defence, for the branches that DO carry provider text.

    `redact_url_userinfo` covers `scheme://user:pass@host` and nothing else --
    a `?key=` credential walks straight through it, which is the same
    enumerate-what-you-remember failure that let a credentialed `OLLAMA_HOST`
    onto `/health`.
    """
    result = LLMClient.classify_live_probe_error(
        RuntimeError(
            "failed calling "
            "https://generativelanguage.googleapis.com/v1beta/x?key=AIzaSuperSecret&n=1"
        )
    )

    assert "AIzaSuperSecret" not in result["verification_reason"]
    assert "key=***" in result["verification_reason"]


def test_a_non_http_exception_message_is_redacted_too(monkeypatch):
    """The catch-all branch interpolates the exception text, which is the one
    place a provider gets to choose what this tool prints."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-averylongsecretvalue")

    result = LLMClient.classify_live_probe_error(
        RuntimeError("upstream said: Bearer sk-or-v1-averylongsecretvalue")
    )

    assert "sk-or-v1-averylongsecretvalue" not in repr(result)


def test_url_userinfo_is_still_stripped_on_this_path(monkeypatch):
    """The original leak (`OLLAMA_HOST=http://admin:s3cret@host:11434`) must not
    come back through the new path."""
    result = LLMClient.classify_live_probe_error(
        RuntimeError("could not reach http://admin:s3cret@ollama.internal:11434")
    )

    assert "s3cret" not in result["verification_reason"]
    assert "***@" in result["verification_reason"]
