"""AI Engine Service - Main Application."""

import asyncio
import hmac
import os
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .chatbot.rag import Document, RAGChatbot
from .clustering.pipeline import Tab, TabClusterer
from .core.llm_client import LLMClient

app = FastAPI(
    title="Tab Organizer - AI Engine",
    description="AI services for embeddings, clustering, and chat",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
llm_client = LLMClient()
clusterer = TabClusterer()
clusterer.set_llm_client(llm_client)
chatbot = RAGChatbot(
    db_uri=os.getenv("VECTOR_DB_PATH", "/data/lancedb"),
    embedding_dim=llm_client.embedding_config.dimensions,
)
chatbot.set_llm_client(llm_client)
provider_state_lock = asyncio.Lock()
UNAUTHENTICATED_TRUE_VALUES = {"1", "true", "yes", "on"}


def _current_embedding_model() -> Optional[str]:
    return getattr(llm_client.embedding_config, "model", None)


# Request models
class EmbedRequest(BaseModel):
    texts: list[str]


class ClusterRequest(BaseModel):
    session_id: str
    urls: list[dict]


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class IndexRequest(BaseModel):
    session_id: str
    documents: list[dict]


class GenerateRequest(BaseModel):
    prompt: str
    system: Optional[str] = None


class ProviderSwitchRequest(BaseModel):
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None


class AIConfigRequest(BaseModel):
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None
    api_keys: dict[str, str] = Field(default_factory=dict)


ALLOWED_RUNTIME_API_KEYS = {
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "DEEPSEEK_API_KEY",
    "GOOGLE_API_KEY",
}


def _require_ai_engine_auth(authorization: Optional[str] = Header(default=None)):
    """Protect generation and provider mutation endpoints with a shared token."""
    expected = os.getenv("AI_ENGINE_API_TOKEN", "").strip()
    if not expected:
        if os.getenv("AI_ENGINE_ALLOW_UNAUTHENTICATED", "").strip().lower() in (
            UNAUTHENTICATED_TRUE_VALUES
        ):
            return
        raise HTTPException(
            status_code=401,
            detail=(
                "AI Engine token is not configured; set AI_ENGINE_API_TOKEN or run "
                "scripts/cli.py start so local service calls are authenticated"
            ),
        )

    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(token.strip(), expected):
        raise HTTPException(status_code=401, detail="Invalid AI Engine token")


# Health check
@app.get("/")
async def root():
    return {
        "service": "ai-engine",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
async def health():
    try:
        chatbot.table
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "vector_store": {
                    "type": "lancedb",
                    "path": chatbot.db_uri,
                    "table": chatbot.TABLE_NAME,
                    "ready": False,
                    "error": str(e),
                },
            },
        )

    runtime = llm_client.get_runtime_health()
    return {
        "status": "healthy" if runtime["ready"] else "degraded",
        "vector_store": {
            "type": "lancedb",
            "path": chatbot.db_uri,
            "table": chatbot.TABLE_NAME,
            "ready": True,
        },
        "runtime": runtime,
    }


# Provider info
@app.get("/providers")
async def get_providers():
    async with provider_state_lock:
        return llm_client.get_provider_info()


@app.post("/providers/switch")
async def switch_provider(
    request: ProviderSwitchRequest,
    _auth=Depends(_require_ai_engine_auth),
):
    try:
        async with provider_state_lock:
            current_embedding_model = _current_embedding_model()
            target_embedding_provider = (
                request.embedding_provider or llm_client.embedding_config.provider
            )
            target_embedding_model = request.embedding_model or current_embedding_model
            embedding_change_requested = bool(
                request.embedding_provider or request.embedding_model
            ) and (
                target_embedding_provider != llm_client.embedding_config.provider
                or target_embedding_model != current_embedding_model
            )
            if embedding_change_requested and chatbot.has_indexed_documents():
                raise ValueError(
                    "Cannot switch embedding provider or model while documents are "
                    "indexed; clear and reindex the vector store first"
                )

            llm_client.switch_provider(
                llm_provider=request.llm_provider,
                llm_model=request.llm_model,
                embedding_provider=request.embedding_provider,
                embedding_model=request.embedding_model,
            )
            if request.embedding_provider or request.embedding_model:
                chatbot.reconfigure_embeddings(llm_client.embedding_config.dimensions)
            return {
                "status": "switched",
                "providers": llm_client.get_provider_info(),
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/config")
async def update_runtime_config(
    request: AIConfigRequest,
    _auth=Depends(_require_ai_engine_auth),
):
    try:
        unknown_keys = sorted(set(request.api_keys) - ALLOWED_RUNTIME_API_KEYS)
        if unknown_keys:
            raise ValueError(
                "Unsupported API key environment variable(s): "
                + ", ".join(unknown_keys)
            )

        async with provider_state_lock:
            for key, value in request.api_keys.items():
                value = value.strip()
                if value:
                    os.environ[key] = value

            embedding_provider = request.embedding_provider
            embedding_model = request.embedding_model
            embedding_change_requested = bool(embedding_provider or embedding_model)
            current_embedding_model = _current_embedding_model()
            target_embedding_provider = (
                embedding_provider or llm_client.embedding_config.provider
            )
            target_embedding_model = embedding_model or current_embedding_model
            embedding_dimension_change_requested = embedding_change_requested and (
                target_embedding_provider != llm_client.embedding_config.provider
                or target_embedding_model != current_embedding_model
            )
            if (
                embedding_dimension_change_requested
                and chatbot.has_indexed_documents()
            ):
                raise ValueError(
                    "Cannot switch embedding provider or model while documents are "
                    "indexed; clear and reindex the vector store first"
                )

            llm_client.switch_provider(
                llm_provider=request.llm_provider,
                llm_model=request.llm_model,
                embedding_provider=request.embedding_provider,
                embedding_model=request.embedding_model,
            )
            llm_client.refresh_runtime_credentials()

            if embedding_dimension_change_requested:
                chatbot.reconfigure_embeddings(llm_client.embedding_config.dimensions)

            return {
                "status": "updated",
                "providers": llm_client.get_provider_info(),
                "health": llm_client.get_runtime_health(),
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Embedding endpoints
@app.post("/embed")
async def embed_texts(request: EmbedRequest, _auth=Depends(_require_ai_engine_auth)):
    try:
        async with provider_state_lock:
            embeddings = await llm_client.embed(request.texts)
        return {
            "embeddings": embeddings,
            "count": len(embeddings),
            "dimensions": len(embeddings[0]) if embeddings else 0,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Generation endpoints
@app.post("/generate")
async def generate_text(
    request: GenerateRequest,
    _auth=Depends(_require_ai_engine_auth),
):
    try:
        async with provider_state_lock:
            result = await llm_client.generate(request.prompt, request.system)
        return {"text": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Clustering endpoints
@app.post("/cluster")
async def cluster_urls(request: ClusterRequest, _auth=Depends(_require_ai_engine_auth)):
    try:
        # Convert to Tab objects
        tabs = [
            Tab(
                url=u.get("url", ""),
                title=u.get("title", ""),
                content=u.get("content", ""),
                metadata=u.get("metadata", {}),
            )
            for u in request.urls
        ]

        # Cluster
        async with provider_state_lock:
            clusters = await clusterer.cluster(tabs)

        return {
            "session_id": request.session_id,
            "clusters": clusterer.to_dict(clusters),
            "cluster_count": len(clusters),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Chatbot endpoints
@app.post("/index")
async def index_documents(request: IndexRequest, _auth=Depends(_require_ai_engine_auth)):
    try:
        documents = [
            Document(
                id=d.get("id", d.get("url", "")),
                url=d.get("url", ""),
                title=d.get("title", ""),
                content=d.get("content", ""),
                metadata=d.get("metadata", {}),
            )
            for d in request.documents
        ]

        async with provider_state_lock:
            count = await chatbot.index_documents(documents, request.session_id)
        return {"indexed": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(request: ChatRequest, _auth=Depends(_require_ai_engine_auth)):
    try:
        async with provider_state_lock:
            result = await chatbot.chat(
                query=request.query,
                session_id=request.session_id,
            )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search(request: ChatRequest, _auth=Depends(_require_ai_engine_auth)):
    try:
        async with provider_state_lock:
            results = await chatbot.search(
                query=request.query,
                session_id=request.session_id,
            )
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/summarize/{session_id}")
async def summarize_session(session_id: str, _auth=Depends(_require_ai_engine_auth)):
    try:
        async with provider_state_lock:
            summary = await chatbot.summarize_session(session_id)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{session_id}")
async def delete_session_documents(
    session_id: str,
    _auth=Depends(_require_ai_engine_auth),
):
    try:
        async with provider_state_lock:
            count = chatbot.delete_session_documents(session_id)
        return {"deleted": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
