"""Compatibility module for legacy imports of the chatbot service."""

from __future__ import annotations

import sys

from chatbot import (
    ChatbotService,
    ChatMessage,
    ChatResponse,
    FeedbackRequest,
    Settings,
    app,
    chatbot_service,
    conversations,
    create_app,
    qdrant_client,
)

# Maintain backward-compatible module name for mocks such as patch("main.qdrant_client").
sys.modules.setdefault("main", sys.modules[__name__])

__all__ = [
    "ChatbotService",
    "ChatMessage",
    "ChatResponse",
    "FeedbackRequest",
    "Settings",
    "create_app",
    "app",
    "chatbot_service",
    "conversations",
    "qdrant_client",
]


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8092)

