"""AI Engine Service - Main Application."""

import asyncio
import hmac
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.config_loader import get_ai_config
from services.cors import allowed_origins
from services.observability import RequestIDMiddleware, configure_logging, log_event

from .chatbot.rag import Document, RAGChatbot
from .clustering.pipeline import Tab, TabClusterer
from .core.llm_client import LLMClient, ProviderSelectionError

configure_logging("ai-engine")

VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "/data/lancedb")

# Vector width used to bootstrap a LanceDB table that DOES NOT EXIST YET, when
# no embedding provider has been selected. It is not a provider default and
# never becomes one: with no embedding provider, `llm_client.embed()` raises,
# so no row can be written at this width, and `reconfigure_embeddings()`
# resizes the (necessarily empty) table the moment a provider is chosen.
#
# It is used ONLY when there is nothing on disk to read a width from. Applying
# it to a volume that already holds rows bricked the service on upgrade: the
# rows are 1024- or 1536-wide from an older default, `_ensure_table()` refuses
# the mismatch, `GET /health` 503s on it, and `/providers/switch` calls
# `has_indexed_documents()` -- which opens the same table -- BEFORE it
# reconfigures anything, so the user could not select the provider that would
# have fixed it. An unselected table is not an empty table; ask it its width.
UNSELECTED_EMBEDDING_DIM = 768

# Global instances
llm_client = LLMClient()
clusterer = TabClusterer()
clusterer.set_llm_client(llm_client)


def _existing_table_dimensions(db_uri: str) -> Optional[int]:
    """Read the embedding width already on disk, or None if there is none.

    Best-effort and read-only: it never creates the directory or the table
    (`RAGChatbot.db` would), and any failure degrades to None rather than
    preventing startup -- the caller's fallback is the bootstrap width, and a
    service that starts and reports its state beats one that cannot start.
    """
    if not os.path.isdir(db_uri):
        return None
    try:
        import lancedb
        import pyarrow as pa

        db = lancedb.connect(db_uri)
        if RAGChatbot.TABLE_NAME not in db.table_names():
            return None
        field = db.open_table(RAGChatbot.TABLE_NAME).schema.field("embedding")
        if pa.types.is_fixed_size_list(field.type):
            return int(field.type.list_size)
    except Exception as error:  # noqa: BLE001 - diagnostic, never fatal
        log_event(
            "vector_store.width_probe_failed",
            level=logging.WARNING,
            path=db_uri,
            reason=str(error),
        )
    return None


def _bootstrap_embedding_dim() -> int:
    """Pick the width the vector table opens with, without inventing one."""
    config = llm_client.embedding_config
    if config:
        return config.dimensions

    existing = _existing_table_dimensions(VECTOR_DB_PATH)
    if existing:
        log_event(
            "vector_store.width_adopted",
            dimensions=existing,
            reason=(
                "no embedding provider is selected; adopting the width of the "
                "table already on disk instead of assuming one"
            ),
        )
        return existing
    return UNSELECTED_EMBEDDING_DIM


chatbot = RAGChatbot(
    db_uri=VECTOR_DB_PATH,
    embedding_dim=_bootstrap_embedding_dim(),
)
chatbot.set_llm_client(llm_client)
provider_state_lock = asyncio.Lock()
UNAUTHENTICATED_TRUE_VALUES = {"1", "true", "yes", "on"}


def _announce_active_providers() -> None:
    """Emit one `provider.active` line per role (R4, `announce_active_provider`).

    A user must never have to discover after the fact which provider answered.
    Emitted unconditionally at startup — including when a role has no provider,
    where the line carries the structured `{code, cause, fix}` instead. The
    summary is catalog-derived and contains no credentials.
    """
    for role, summary in llm_client.get_active_providers().items():
        log_event(
            "provider.active",
            level=(logging.INFO if summary.get("provider") else logging.ERROR),
            role=role,
            **summary,
        )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Fail fast on malformed provider config; log (never crash) an unusable one.

    Schema errors (finding 27) mean ai_models.yaml itself is broken — refuse to
    start. An unusable *selected* provider (WI0 B1, e.g. openrouter with no API
    key) is a runtime/credentials fact, not a schema error: log it structurally
    and let `/health` report it as degraded so the service stays diagnosable
    instead of going dark.
    """
    errors = get_ai_config().validate_config()
    if errors:
        log_event("config.invalid", level=logging.CRITICAL, errors=errors)
        raise RuntimeError("AI model configuration is invalid: " + "; ".join(errors))

    _announce_active_providers()

    runtime = llm_client.get_runtime_health()
    if not runtime["ready"]:
        log_event(
            "provider.unusable_at_startup",
            level=logging.ERROR,
            llm_provider=runtime["llm"].get("provider"),
            llm_reason=runtime["llm"].get("reason"),
            embedding_provider=runtime["embeddings"].get("provider"),
            embedding_reason=runtime["embeddings"].get("reason"),
        )
    yield


app = FastAPI(
    title="Tab Organizer - AI Engine",
    description="AI services for embeddings, clustering, and chat",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware. Scoped to the Web UI origin, credentials never allowed --
# `allow_origins=["*"]` + `allow_credentials=True` made every endpoint here
# reachable from any page the user had open. Web UI calls this service
# server-side, so no browser-side caller is lost. See services/cors.py.
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Starlette wraps middleware in reverse add order (last added = outermost), so
# RequestIDMiddleware must be added last to wrap CORS -- otherwise a CORS
# preflight (OPTIONS) short-circuits inside CORSMiddleware before ever
# reaching this middleware and comes back with no X-Request-ID.
app.add_middleware(RequestIDMiddleware, service="ai-engine")


def _current_embedding_model() -> Optional[str]:
    return getattr(llm_client.embedding_config, "model", None)


def _current_embedding_provider() -> Optional[str]:
    """None when no embedding provider was selected -- never a stand-in."""
    return getattr(llm_client.embedding_config, "provider", None)


def _current_embedding_dimensions() -> int:
    """With no provider selected, keep the table's width -- never invent one."""
    config = llm_client.embedding_config
    return config.dimensions if config else chatbot.embedding_dim


# Request models
class EmbedRequest(BaseModel):
    texts: list[str]


class ClusterRequest(BaseModel):
    session_id: str
    urls: list[dict]


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    top_k: int = 5


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


def _provider_error(exc: ProviderSelectionError, **context) -> HTTPException:
    """Turn a selection/availability failure into a 503 that keeps its shape.

    503, not 500: nothing is wrong with the request, the service simply has no
    provider it is allowed to answer with. The `{code, cause, fix}` survives
    into the body instead of being flattened to a string, so the caller (and
    the UI) can act on it rather than parse prose.
    """
    log_event(
        "provider.request_refused",
        level=logging.ERROR,
        code=exc.code,
        reason=exc.cause,
        **context,
    )
    return HTTPException(status_code=503, detail=exc.to_dict())


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


def _vector_store_failure(error: Exception) -> dict:
    """Name the fix for an unopenable table instead of only the symptom.

    The only routinely-recoverable cause is a width mismatch between the
    selected embedding model and the rows already on disk. Reported as
    `{code, cause, fix}` like every other refusal on this service, because
    `/health` is the one thing a user can still reach when the store will not
    open, and "reindex required" buried in an exception string is not a fix
    anyone can act on. Carries no credential: the message is generated by
    `RAGChatbot._ensure_table` and names dimensions and a path only.
    """
    reason = str(error)
    if "do not match" in reason or "dimension" in reason:
        return {
            "code": "vector_store_dimension_mismatch",
            "cause": reason,
            "fix": (
                "The rows in this LanceDB volume were written at a different "
                f"vector width than the selected embedding model "
                f"({chatbot.embedding_dim}-d). Either select the embedding "
                "model that wrote them, or clear the store "
                "(DELETE /documents/{session_id} per session, or remove the "
                "lancedb-data volume) and reindex."
            ),
        }
    return {
        "code": "vector_store_unavailable",
        "cause": reason,
        "fix": (
            f"The LanceDB store at {chatbot.db_uri} could not be opened. Check "
            "the volume is mounted and writable, then restart ai-engine."
        ),
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
                    **_vector_store_failure(e),
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
        # `providers` is the attribution contract (R4): who is answering and
        # what it costs. It sits beside `runtime` rather than inside it because
        # `runtime` is the diagnostic blob (availability reasons, api-key
        # presence, installed Ollama models) whose shape follows internal
        # needs, while this block is a small stable payload the UI badge and
        # the TS facade read. Neither carries a token or an API key -- /health
        # is the one unauthenticated endpoint on this service.
        "providers": llm_client.get_active_providers(),
        "runtime": runtime,
    }


# Provider info. Authenticated like every other endpoint except /health: the
# payload discloses which API keys are configured (`api_key_configured` per
# provider) plus the whole model catalog and the structured selection errors.
# CLAUDE.md already stated "ai-engine endpoints (except /health) require
# AI_ENGINE_API_TOKEN"; this endpoint was the one that did not. Its only
# non-test caller, `services/web-ui/src/api/client.py:get_providers()`, already
# sends the bearer token.
@app.get("/providers")
async def get_providers(_auth=Depends(_require_ai_engine_auth)):
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
                request.embedding_provider or _current_embedding_provider()
            )
            target_embedding_model = request.embedding_model or current_embedding_model
            embedding_change_requested = bool(
                request.embedding_provider or request.embedding_model
            ) and (
                target_embedding_provider != _current_embedding_provider()
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
                chatbot.reconfigure_embeddings(_current_embedding_dimensions())
            llm_config = getattr(llm_client, "llm_config", None)
            log_event(
                "provider.switched",
                llm_provider=getattr(llm_config, "provider", None),
                llm_model=getattr(llm_config, "model", None),
                embedding_provider=_current_embedding_provider(),
                embedding_model=_current_embedding_model(),
            )
            return {
                "status": "switched",
                "providers": llm_client.get_provider_info(),
            }
    except Exception as e:
        log_event("provider.switch_failed", level=logging.WARNING, reason=str(e))
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
                embedding_provider or _current_embedding_provider()
            )
            target_embedding_model = embedding_model or current_embedding_model
            embedding_dimension_change_requested = embedding_change_requested and (
                target_embedding_provider != _current_embedding_provider()
                or target_embedding_model != current_embedding_model
            )
            if embedding_dimension_change_requested and chatbot.has_indexed_documents():
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
                chatbot.reconfigure_embeddings(_current_embedding_dimensions())

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
    except ProviderSelectionError as e:
        raise _provider_error(e, endpoint="/embed")
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
    except ProviderSelectionError as e:
        raise _provider_error(e, endpoint="/generate")
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
            payload = clusterer.to_dict(clusters)
            label_failures = clusterer.count_label_failures(clusters)

        if label_failures:
            log_event(
                "cluster.labels_incomplete",
                level=logging.WARNING,
                session_id=request.session_id,
                cluster_count=len(clusters),
                label_failures=label_failures,
            )

        return {
            "session_id": request.session_id,
            "clusters": payload,
            "cluster_count": len(clusters),
            # Best-effort label generation is counted and surfaced, per the
            # repo's batch convention. A caller must be able to tell a run
            # where every name is a placeholder from a successful one.
            "label_failures": label_failures,
        }
    except ProviderSelectionError as e:
        raise _provider_error(e, endpoint="/cluster", session_id=request.session_id)
    except Exception as e:
        log_event(
            "cluster.failed",
            level=logging.ERROR,
            session_id=request.session_id,
            url_count=len(request.urls),
            reason=str(e),
        )
        raise HTTPException(status_code=500, detail=str(e))


# Chatbot endpoints
@app.post("/index")
async def index_documents(
    request: IndexRequest, _auth=Depends(_require_ai_engine_auth)
):
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
        log_event(
            "index.completed",
            session_id=request.session_id,
            document_count=len(documents),
            indexed=count,
        )
        return {"indexed": count}
    except ProviderSelectionError as e:
        raise _provider_error(e, endpoint="/index", session_id=request.session_id)
    except Exception as e:
        log_event(
            "index.failed",
            level=logging.ERROR,
            session_id=request.session_id,
            document_count=len(request.documents),
            reason=str(e),
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(request: ChatRequest, _auth=Depends(_require_ai_engine_auth)):
    try:
        async with provider_state_lock:
            result = await chatbot.chat(
                query=request.query,
                session_id=request.session_id,
                top_k=request.top_k,
            )
        return result
    except ProviderSelectionError as e:
        raise _provider_error(e, endpoint="/chat", session_id=request.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search(request: ChatRequest, _auth=Depends(_require_ai_engine_auth)):
    try:
        async with provider_state_lock:
            results = await chatbot.search(
                query=request.query,
                session_id=request.session_id,
                top_k=request.top_k,
            )
        return {"results": results}
    except ProviderSelectionError as e:
        raise _provider_error(e, endpoint="/search", session_id=request.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/summarize/{session_id}")
async def summarize_session(session_id: str, _auth=Depends(_require_ai_engine_auth)):
    try:
        async with provider_state_lock:
            summary = await chatbot.summarize_session(session_id)
        return {"summary": summary}
    except ProviderSelectionError as e:
        raise _provider_error(e, endpoint="/summarize", session_id=session_id)
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
