"""Chatbot service package exports."""

from __future__ import annotations

import sys

from .api import create_app
from .config import Settings
from .models import ChatMessage, ChatResponse, FeedbackRequest
from .service import ChatbotService
from .store import ConversationStore

app = create_app()
chatbot_service: ChatbotService = app.state.chatbot_service
conversations: ConversationStore = chatbot_service.conversations
qdrant_client = app.state.qdrant_client

# Ensure compatibility imports when package is used directly.
sys.modules.setdefault("services.chatbot.main", sys.modules[__name__])

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

