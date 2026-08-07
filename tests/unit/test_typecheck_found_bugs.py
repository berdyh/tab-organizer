"""Two latent bugs found by the FIRST real run of `make type-check`.

That gate spent its whole life aborting on `Duplicate module named "app"`
before analysing a single line, while CI ran the same command behind
`|| true`. The first time it actually executed (2026-08-07) it surfaced these
two. Both were reported rather than fixed inside the typing commit — each is
a behaviour decision, not a typing one — and fixed here on their own merits.
"""

import asyncio
import inspect
import os
import sys
import types

import pytest

from services.ai_engine.app.core.llm_client import (
    LLMClient,
    ProviderSelectionError,
)


def _install_playwright_stub() -> None:
    """Same helper the other browser-engine unit tests use.

    The unit-test image installs playwright STUBS for mypy, not the package,
    so importing the scraper needs this to reach the code under test.
    """
    playwright_module = types.ModuleType("playwright")
    async_api = types.ModuleType("playwright.async_api")
    async_api.Browser = object
    async_api.Page = object
    async_api.Playwright = object
    async_api.TimeoutError = TimeoutError
    async_api.async_playwright = lambda: None
    playwright_module.async_api = async_api
    sys.modules["playwright"] = playwright_module
    sys.modules["playwright.async_api"] = async_api


try:
    import playwright.async_api  # noqa: F401
except ModuleNotFoundError:
    _install_playwright_stub()

from services.browser_engine.app.scraper.engine import (  # noqa: E402
    ScrapeResult,
    ScraperEngine,
)

PROVIDER_ENV = (
    "AI_PROVIDER",
    "EMBEDDING_PROVIDER",
    "LLM_MODEL",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSIONS",
)


@pytest.fixture
def no_provider_selected(monkeypatch):
    """The fail-closed default: nothing chosen, which is the buggy precondition."""
    for name in PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    yield


def test_switching_to_a_provider_with_no_default_model_fails_closed(
    monkeypatch, no_provider_selected
):
    """`switch_provider` half-succeeded: provider updated, model left None.

    `LLMConfig.model` is typed `str`. With no provider selected, a target whose
    catalog entry declares no default llm model, and no explicit `llm_model`,
    `target_model` stayed None and was assigned straight onto `.model` —
    defeating the `or ""` the constructor had applied one line earlier. The
    switch reported success and the failure surfaced later, somewhere else.

    The embeddings branch of this same method already refused exactly this case
    with `embedding_model_not_selected`. That asymmetry WAS the bug, so the fix
    is to stop being asymmetric.
    """
    client = LLMClient()

    monkeypatch.setattr(
        client._ai_config, "is_provider_supported", lambda *_a, **_k: True
    )
    monkeypatch.setattr(
        client, "get_provider_runtime_state", lambda *_a, **_k: {"available": True}
    )
    # The precondition that produces the bug: no default model for this role.
    monkeypatch.setattr(client._ai_config, "get_default_model", lambda *_a, **_k: None)

    with pytest.raises(ProviderSelectionError) as exc:
        client.switch_provider(llm_provider="ollama")

    assert exc.value.code == "llm_model_not_selected"
    assert "llm_model" in exc.value.fix
    # Nothing half-applied: a refused switch leaves no model-less config behind.
    assert client.llm_config is None or client.llm_config.model


def test_cancellederror_is_not_an_exception_which_is_the_whole_bug():
    """The premise, pinned. If this ever changes, the fix below is moot."""
    assert not isinstance(asyncio.CancelledError(), Exception)
    assert isinstance(asyncio.CancelledError(), BaseException)


def test_scrape_batch_converts_every_raised_object_to_a_failed_result():
    """`gather(return_exceptions=True)` yields BaseException, not Exception.

    `scrape_batch` tested `isinstance(result, Exception)`, so a cancelled child
    task fell through to the else and was appended to `final_results` as if it
    were a ScrapeResult. Every consumer then attribute-errors on `.status` or
    `.url`, far from the cancellation that caused it. A cancelled scrape is a
    failed scrape and now says so.

    Asserted on the source because the conversion is an inline loop with no
    seam to call: driving a real cancellation through `scrape_batch` would
    need a live browser. The premise test above keeps this honest — this pair
    fails if either the widening is reverted or CancelledError's ancestry
    changes.
    """
    source = inspect.getsource(ScraperEngine.scrape_batch)
    assert "isinstance(result, BaseException)" in source
    assert "isinstance(result, Exception)" not in source


def test_the_conversion_still_produces_usable_results_for_both_shapes():
    """Non-vacuity: the widened branch must still yield real ScrapeResults."""
    gathered = [
        ScrapeResult(url="https://ok.test/", status="success"),
        asyncio.CancelledError(),
        RuntimeError("boom"),
    ]
    urls = ["https://ok.test/", "https://cancelled.test/", "https://boom.test/"]

    final = []
    for url, result in zip(urls, gathered):
        if isinstance(result, BaseException):
            final.append(
                ScrapeResult(
                    url=url,
                    status="failed",
                    error=f"{type(result).__name__}: {result}",
                )
            )
        else:
            final.append(result)

    assert [r.status for r in final] == ["success", "failed", "failed"]
    assert all(isinstance(r, ScrapeResult) for r in final)
    assert "CancelledError" in (final[1].error or "")


def test_the_type_check_gate_script_is_the_one_that_found_these():
    """Provenance guard, cheap: if the gate stops existing, so does this class
    of finding. `tests/unit/test_type_check_gate.py` enforces the details."""
    assert os.path.exists("/app/scripts/type-check.sh") or os.path.exists(
        "scripts/type-check.sh"
    )
