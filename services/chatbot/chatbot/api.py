"""FastAPI application factory for the chatbot service."""

from __future__ import annotations

import importlib
from contextlib import asynccontextmanager
import sys
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from qdrant_client import QdrantClient

from .clients import create_ollama_client, create_qdrant_client
from .config import Settings
from .logging import configure_logging
from .models import ChatMessage, ChatResponse, FeedbackRequest
from .service import ChatbotService
from .store import ConversationStore


def create_app(
    *,
    settings: Optional[Settings] = None,
    qdrant_client: Optional[QdrantClient] = None,
    ollama_client: Optional[httpx.AsyncClient] = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""

    configure_logging()

    cfg = settings or Settings()
    conversations = ConversationStore()
    qdrant = qdrant_client or create_qdrant_client(cfg)
    ollama = ollama_client or create_ollama_client(cfg)
    service = ChatbotService(
        cfg,
        qdrant_client=qdrant,
        ollama_client=ollama,
        conversations=conversations,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            await service.aclose()

    app = FastAPI(
        title="Chatbot Service",
        description="Natural language interface for exploring scraped content and clusters",
        version="1.0.0",
        lifespan=lifespan,
    )

    robot_parser_path = "urllib.robotparser.RobotFileParser"

    def resolve_robot_parser_cls():
        module_name, attribute = robot_parser_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, attribute)

    app.state.settings = cfg
    app.state.chatbot_service = service
    app.state.qdrant_client = qdrant
    app.state.robot_parser_cls_path = robot_parser_path
    app.state.resolve_robot_parser_cls = resolve_robot_parser_cls
    app.state.robot_parser_cls = resolve_robot_parser_cls()

    def resolve_service() -> ChatbotService:
        main_module = sys.modules.get("main")
        if main_module and hasattr(main_module, "chatbot_service"):
            return getattr(main_module, "chatbot_service")
        return app.state.chatbot_service

    @app.post("/chat/message", response_model=ChatResponse)
    async def send_message(request: ChatMessage):
        """Send a message to the chatbot and get a response."""
        svc = resolve_service()
        response = await svc.process_message(
            request.session_id,
            request.message,
            request.context,
        )
        return response

    @app.get("/chat/history/{session_id}")
    async def get_conversation_history(session_id: str):
        """Get conversation history for a session."""
        svc = resolve_service()
        return {
            "session_id": session_id,
            "history": svc.conversation_history(session_id),
        }

    @app.delete("/chat/history/{session_id}")
    async def clear_conversation_history(session_id: str):
        """Clear conversation history for a session."""
        svc = resolve_service()
        svc.clear_history(session_id)
        return {"message": "Conversation history cleared"}

    @app.post("/chat/feedback")
    async def provide_feedback(request: FeedbackRequest):
        """Provide feedback on a chatbot response."""
        feedback_entry = {
            "message_id": request.message_id,
            "feedback": request.feedback,
            "comment": request.comment,
        }
        return {"message": "Feedback received", "feedback_id": feedback_entry["message_id"]}

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        svc = resolve_service()
        try:
            svc.qdrant_client.get_collections()
            response = await svc.ollama_client.get("/api/tags")
            response.raise_for_status()
            return {
                "status": "healthy",
                "services": {"qdrant": "connected", "ollama": "connected"},
            }
        except Exception as exc:  # pragma: no cover - surfaced to tests via FastAPI
            raise HTTPException(status_code=503, detail=f"Service unhealthy: {exc}") from exc

    return app
