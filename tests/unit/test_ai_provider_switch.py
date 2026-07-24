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
async def test_ai_health_reports_degraded_when_runtime_config_is_unavailable(monkeypatch):
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
