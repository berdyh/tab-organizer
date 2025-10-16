"""Domain and API models for the chatbot service."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import AnyHttpUrl, BaseModel


class ChatMessage(BaseModel):
    """Incoming chat message payload."""

    session_id: str
    message: str
    context: List[Dict[str, Any]] = []


class ArticleLink(BaseModel):
    """Lightweight reference to an article."""

    title: str = ""
    url: Optional[AnyHttpUrl] = None
    domain: Optional[str] = None


class SearchResult(BaseModel):
    """Semantic search hit returned to the client."""

    title: str = ""
    url: Optional[AnyHttpUrl] = None
    snippet: str = ""
    cluster: Optional[str] = None
    relevance_score: Optional[float] = None
    domain: Optional[str] = None


class ClusterSummary(BaseModel):
    """High-level representation of a cluster."""

    title: str
    description: str
    count: int
    sample_articles: List[ArticleLink] = []


class StatisticCard(BaseModel):
    """Summary statistics returned to the UI."""

    title: str
    description: str
    metadata: Dict[str, str]


class ChatResponse(BaseModel):
    """Response envelope returned from the chatbot."""

    response: str
    sources: List[Dict[str, Any]] = []
    suggestions: List[str] = []
    conversation_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    """Feedback payload from the UI."""

    message_id: str
    feedback: str
    comment: Optional[str] = None


class ConversationEntry(BaseModel):
    """Persisted conversational memory entry."""

    timestamp: datetime
    user_message: str
    bot_response: str
    intent: str
    sources_count: int

