"""Provider switching regressions for the AI Engine app."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from services.ai_engine.app import main


def test_ai_engine_auth_dependency_requires_configured_token(monkeypatch):
    monkeypatch.setenv("AI_ENGINE_API_TOKEN", "local-secret")

    with pytest.raises(HTTPException) as missing:
        main._require_ai_engine_auth(None)
    assert missing.value.status_code == 401

    with pytest.raises(HTTPException) as wrong:
        main._require_ai_engine_auth("Bearer wrong")
    assert wrong.value.status_code == 401

    assert main._require_ai_engine_auth("Bearer local-secret") is None

    monkeypatch.delenv("AI_ENGINE_API_TOKEN", raising=False)
    with pytest.raises(HTTPException) as unconfigured:
        main._require_ai_engine_auth(None)
    assert unconfigured.value.status_code == 401

    monkeypatch.setenv("AI_ENGINE_ALLOW_UNAUTHENTICATED", "true")
    assert main._require_ai_engine_auth(None) is None


@pytest.mark.asyncio
async def test_search_route_honors_requested_top_k(monkeypatch):
    class FakeChatbot:
        async def search(self, query, session_id=None, top_k=5):
            return [
                {
                    "query": query,
                    "session_id": session_id,
                    "top_k": top_k,
                }
            ]

    monkeypatch.setattr(main, "chatbot", FakeChatbot())

    response = await main.search(
        main.ChatRequest(query="browser tabs", session_id="session", top_k=17)
    )

    assert response == {
        "results": [
            {
                "query": "browser tabs",
                "session_id": "session",
                "top_k": 17,
            }
        ]
    }


@pytest.mark.asyncio
async def test_ai_health_reports_degraded_when_runtime_config_is_unavailable(
    monkeypatch,
):
    class FakeChatbot:
        db_uri = "/tmp/lancedb"
        TABLE_NAME = "tab_organizer_docs"
        table = object()

    class FakeLLMClient:
        def get_runtime_health(self):
            return {
                "ready": False,
                "llm": {
                    "provider": "openrouter",
                    "model": "openai/gpt-4o-mini",
                    "available": False,
                    "reason": "OPENROUTER_API_KEY is not configured",
                },
                "embeddings": {
                    "provider": "openrouter",
                    "model": "embed",
                    "available": False,
                    "reason": "OPENROUTER_API_KEY is not configured",
                },
            }

        def get_active_providers(self):
            # `/health` announces the active provider per role (R4) alongside
            # the runtime diagnostics; an unusable provider is still named.
            return {
                "llm": {
                    "provider": "openrouter",
                    "model": "openai/gpt-4o-mini",
                    "cost_model": "metered",
                },
                "embedding": {
                    "provider": "openrouter",
                    "model": "embed",
                    "cost_model": "metered",
                },
            }

    monkeypatch.setattr(main, "chatbot", FakeChatbot())
    monkeypatch.setattr(main, "llm_client", FakeLLMClient())

    response = await main.health()

    assert response["status"] == "degraded"
    assert response["runtime"]["ready"] is False
    assert "OPENROUTER_API_KEY" in response["runtime"]["llm"]["reason"]


@pytest.mark.asyncio
async def test_embedding_provider_switch_rejected_when_documents_are_indexed(
    monkeypatch,
):
    class FakeChatbot:
        def has_indexed_documents(self):
            return True

        def reconfigure_embeddings(self, embedding_dim):
            raise AssertionError("reconfigure should not run after rejection")

    class FakeLLMClient:
        embedding_config = SimpleNamespace(provider="ollama", dimensions=768)

        def switch_provider(self, **kwargs):
            raise AssertionError("switch_provider should not run after rejection")

        def get_provider_info(self):
            return {}

    monkeypatch.setattr(main, "chatbot", FakeChatbot())
    monkeypatch.setattr(main, "llm_client", FakeLLMClient())

    with pytest.raises(HTTPException) as exc_info:
        await main.switch_provider(
            main.ProviderSwitchRequest(embedding_provider="openrouter")
        )

    assert exc_info.value.status_code == 400
    assert "clear and reindex" in exc_info.value.detail


@pytest.mark.asyncio
async def test_embedding_model_switch_rejected_when_documents_are_indexed(monkeypatch):
    class FakeChatbot:
        def has_indexed_documents(self):
            return True

        def reconfigure_embeddings(self, embedding_dim):
            raise AssertionError("reconfigure should not run after rejection")

    class FakeLLMClient:
        embedding_config = SimpleNamespace(
            provider="ollama",
            model="nomic-embed-text",
            dimensions=768,
        )

        def switch_provider(self, **kwargs):
            raise AssertionError("switch_provider should not run after rejection")

        def get_provider_info(self):
            return {}

    monkeypatch.setattr(main, "chatbot", FakeChatbot())
    monkeypatch.setattr(main, "llm_client", FakeLLMClient())

    with pytest.raises(HTTPException) as exc_info:
        await main.switch_provider(
            main.ProviderSwitchRequest(embedding_model="text-embedding-3-small")
        )

    assert exc_info.value.status_code == 400
    assert "clear and reindex" in exc_info.value.detail


@pytest.mark.asyncio
async def test_embedding_provider_switch_reconfigures_empty_vector_store(monkeypatch):
    class FakeChatbot:
        reconfigured_dimension = None

        def has_indexed_documents(self):
            return False

        def reconfigure_embeddings(self, embedding_dim):
            self.reconfigured_dimension = embedding_dim

    class FakeLLMClient:
        embedding_config = SimpleNamespace(provider="ollama", dimensions=768)

        def switch_provider(self, **kwargs):
            self.embedding_config.provider = kwargs["embedding_provider"]
            self.embedding_config.dimensions = 2048

        def get_provider_info(self):
            return {"embeddings": {"provider": self.embedding_config.provider}}

    chatbot = FakeChatbot()
    monkeypatch.setattr(main, "chatbot", chatbot)
    monkeypatch.setattr(main, "llm_client", FakeLLMClient())

    response = await main.switch_provider(
        main.ProviderSwitchRequest(embedding_provider="openrouter")
    )

    assert response["status"] == "switched"
    assert response["providers"]["embeddings"]["provider"] == "openrouter"
    assert chatbot.reconfigured_dimension == 2048


@pytest.mark.asyncio
async def test_runtime_config_updates_api_keys_models_and_reconfigures_embeddings(
    monkeypatch,
):
    class FakeChatbot:
        reconfigured_dimension = None

        def has_indexed_documents(self):
            return False

        def reconfigure_embeddings(self, embedding_dim):
            self.reconfigured_dimension = embedding_dim

    class FakeLLMClient:
        llm_config = SimpleNamespace(provider="openrouter", model="openai/gpt-4o-mini")
        embedding_config = SimpleNamespace(
            provider="openrouter",
            model="nvidia/llama-nemotron-embed-vl-1b-v2:free",
            dimensions=1024,
        )
        refreshed = False

        def switch_provider(self, **kwargs):
            self.switch_kwargs = kwargs
            self.llm_config.provider = kwargs["llm_provider"]
            self.llm_config.model = kwargs["llm_model"]
            self.embedding_config.provider = kwargs["embedding_provider"]
            self.embedding_config.model = kwargs["embedding_model"]
            self.embedding_config.dimensions = 768

        def refresh_runtime_credentials(self):
            self.refreshed = True

        def get_provider_info(self):
            return {"llm": {"provider": self.llm_config.provider}}

        def get_runtime_health(self):
            return {"ready": True}

    chatbot = FakeChatbot()
    llm_client = FakeLLMClient()
    # `setenv` first, then `delenv`: this test asserts that the endpoint WRITES
    # the key into the real `os.environ`, and that write outlives the test
    # unless monkeypatch has something recorded to restore. `delenv(...,
    # raising=False)` on an already-absent name records nothing, so on its own
    # it leaks `OPENROUTER_API_KEY=sk-or-local` into every later test in the
    # session -- which made the live-endpoint probes in
    # `test_provider_routing.py` skip-guard on a present-but-bogus key and fail
    # with 401 under `make test-ai`, where this file runs first. The setenv
    # gives monkeypatch a prior value to roll back to.
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(main, "chatbot", chatbot)
    monkeypatch.setattr(main, "llm_client", llm_client)

    response = await main.update_runtime_config(
        main.AIConfigRequest(
            llm_provider="openrouter",
            llm_model="openrouter/auto",
            embedding_provider="ollama",
            embedding_model="nomic-embed-text",
            api_keys={"OPENROUTER_API_KEY": "sk-or-local"},
        )
    )

    assert response["status"] == "updated"
    assert llm_client.switch_kwargs == {
        "llm_provider": "openrouter",
        "llm_model": "openrouter/auto",
        "embedding_provider": "ollama",
        "embedding_model": "nomic-embed-text",
    }
    assert llm_client.refreshed is True
    assert chatbot.reconfigured_dimension == 768
    assert main.os.environ["OPENROUTER_API_KEY"] == "sk-or-local"


# --------------------------------------------------------------------------- #
# An UNSELECTED embedding provider must not mean an EMPTY vector table.
#
# With no embedding provider (now the correct fail-closed default), the schema
# width was bootstrapped from an arbitrary 768. On an upgrade the volume can
# already hold rows at the older default (1024/1536), and then:
#   * `GET /health` opens the table and 503s on the mismatch, and
#   * `/providers/switch` calls `has_indexed_documents()` -- which opens the
#     same table -- BEFORE reconfiguring anything,
# so the user cannot select the provider that would fix it. The stack is dead
# after an upgrade, with no path out through the UI.
# --------------------------------------------------------------------------- #
def _seed_indexed_table(db_path, dimensions):
    """Write one real row at `dimensions`, as an older default would have left."""
    from services.ai_engine.app.chatbot.rag import RAGChatbot

    seeded = RAGChatbot(db_uri=str(db_path), embedding_dim=dimensions)
    seeded._append_rows(
        seeded.table,
        [
            {
                "id": "row-1",
                "session_id": "s1",
                "url": "https://example.test/p",
                "title": "t",
                "content": "c",
                "embedding": [0.125] * dimensions,
                "metadata": {},
            }
        ],
    )
    return seeded


def test_an_upgraded_volume_at_the_old_width_is_a_real_dead_end(tmp_path):
    """The hazard itself, so the fix below is not answering a hypothetical."""
    from services.ai_engine.app.chatbot.rag import RAGChatbot

    db_path = tmp_path / "lancedb"
    _seed_indexed_table(db_path, 1024)

    invented = RAGChatbot(db_uri=str(db_path), embedding_dim=768)
    with pytest.raises(RuntimeError, match="reindex required"):
        invented.table
    # ...and this is the same call `/providers/switch` makes before it
    # reconfigures anything, which is why the recovery path was unreachable.
    with pytest.raises(RuntimeError, match="reindex required"):
        invented.has_indexed_documents()


def test_unselected_provider_adopts_the_width_already_on_disk(tmp_path, monkeypatch):
    from services.ai_engine.app.chatbot.rag import RAGChatbot

    db_path = tmp_path / "lancedb"
    _seed_indexed_table(db_path, 1024)

    monkeypatch.setattr(main, "VECTOR_DB_PATH", str(db_path))
    monkeypatch.setattr(main, "llm_client", SimpleNamespace(embedding_config=None))

    assert main._existing_table_dimensions(str(db_path)) == 1024
    assert main._bootstrap_embedding_dim() == 1024, (
        "a width was invented for a table that could have been asked"
    )

    # The recovery path is now reachable: the table opens, and
    # `has_indexed_documents()` answers instead of exploding, so
    # `/providers/switch` reaches its own (correct) refusal with a fix in it.
    recovered = RAGChatbot(
        db_uri=str(db_path), embedding_dim=main._bootstrap_embedding_dim()
    )
    assert recovered.has_indexed_documents() is True


def test_bootstrap_width_is_used_only_when_there_is_nothing_to_read(
    tmp_path, monkeypatch
):
    """Non-vacuity: with no table on disk, the bootstrap width still applies.

    Without this, a `_bootstrap_embedding_dim` hardwired to 1024 would satisfy
    the probe above.
    """
    monkeypatch.setattr(main, "VECTOR_DB_PATH", str(tmp_path / "absent"))
    monkeypatch.setattr(main, "llm_client", SimpleNamespace(embedding_config=None))

    assert main._existing_table_dimensions(str(tmp_path / "absent")) is None
    assert main._bootstrap_embedding_dim() == main.UNSELECTED_EMBEDDING_DIM


def test_a_selected_provider_still_wins_over_the_table(tmp_path, monkeypatch):
    """The catalog dimension is authoritative when a provider IS selected.

    Adopting the on-disk width there would silently write the wrong-width
    vectors the mismatch check exists to prevent.
    """
    db_path = tmp_path / "lancedb"
    _seed_indexed_table(db_path, 1024)

    monkeypatch.setattr(main, "VECTOR_DB_PATH", str(db_path))
    monkeypatch.setattr(
        main,
        "llm_client",
        SimpleNamespace(embedding_config=SimpleNamespace(dimensions=768)),
    )

    assert main._bootstrap_embedding_dim() == 768


def test_current_embedding_dimensions_keeps_the_tables_width_when_unselected(
    monkeypatch,
):
    """`reconfigure_embeddings()` must never be handed an invented width."""
    monkeypatch.setattr(main, "llm_client", SimpleNamespace(embedding_config=None))
    monkeypatch.setattr(main, "chatbot", SimpleNamespace(embedding_dim=1536))

    assert main._current_embedding_dimensions() == 1536


@pytest.mark.asyncio
async def test_health_names_the_fix_when_the_vector_store_will_not_open(monkeypatch):
    """503 is allowed; a 503 nobody can act on is not.

    /health is the one thing still reachable when the store will not open, so
    it carries the same `{code, cause, fix}` shape as every other refusal.
    """

    class BrokenChatbot:
        db_uri = "/data/lancedb"
        TABLE_NAME = "tab_organizer_docs"
        embedding_dim = 768

        @property
        def table(self):
            raise RuntimeError(
                "Existing LanceDB table uses embeddings that do not match the "
                "configured dimension 768; reindex required"
            )

    monkeypatch.setattr(main, "chatbot", BrokenChatbot())

    with pytest.raises(HTTPException) as exc:
        await main.health()

    assert exc.value.status_code == 503
    detail = exc.value.detail["vector_store"]
    assert detail["code"] == "vector_store_dimension_mismatch"
    assert detail["cause"]
    assert "reindex" in detail["fix"]
    assert "DELETE /documents" in detail["fix"]
