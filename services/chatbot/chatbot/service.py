"""Core chatbot orchestration logic."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional

import httpx
import structlog
from fastapi import HTTPException
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from .clients import create_qdrant_client
from .config import Settings
from .models import (
    ArticleLink,
    ChatResponse,
    ClusterSummary,
    ConversationEntry,
    SearchResult,
    StatisticCard,
)
from .store import ConversationStore

logger = structlog.get_logger(__name__)


class ChatbotService:
    """Encapsulates chatbot domain logic and integrations."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        qdrant_client: Optional[QdrantClient] = None,
        ollama_client: Optional[httpx.AsyncClient] = None,
        conversations: Optional[ConversationStore] = None,
    ) -> None:
        self.settings = settings or Settings()
        self._explicit_qdrant_client = qdrant_client
        self._cached_qdrant_client: Optional[QdrantClient] = None
        self.ollama_client = ollama_client or httpx.AsyncClient(
            base_url=str(self.settings.ollama_base_url),
            timeout=self.settings.ollama_timeout,
        )
        self.conversations = conversations or ConversationStore()

    async def aclose(self) -> None:
        """Close async resources."""
        await self.ollama_client.aclose()

    @property
    def qdrant_client(self) -> QdrantClient:
        """Return the active Qdrant client, considering compatibility fallbacks."""
        if self._explicit_qdrant_client is not None:
            return self._explicit_qdrant_client

        if self._cached_qdrant_client is not None:
            return self._cached_qdrant_client

        main_client = self._resolve_main_qdrant_client()
        if main_client is not None:
            return main_client

        self._cached_qdrant_client = create_qdrant_client(self.settings)
        return self._cached_qdrant_client

    def _resolve_main_qdrant_client(self) -> Optional[QdrantClient]:
        """Retrieve qdrant_client from the compatibility module when available."""
        try:
            main_module = sys.modules.get("main")
            if main_module and hasattr(main_module, "qdrant_client"):
                return getattr(main_module, "qdrant_client")
        except Exception:
            return None
        return None

    # Business logic -----------------------------------------------------

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Ollama."""
        try:
            response = await self.ollama_client.post(
                "/api/embeddings",
                json={
                    "model": self.settings.default_embedding_model,
                    "prompt": text,
                },
            )
            response.raise_for_status()
            result = response.json()
            return result["embedding"]
        except Exception as exc:  # pragma: no cover - surfaced to API
            logger.error("Failed to generate embedding", error=str(exc))
            raise HTTPException(status_code=500, detail="Failed to generate embedding")

    async def search_similar_content(
        self, query: str, session_id: str, limit: Optional[int] = None
    ) -> List[SearchResult]:
        """Search for similar content using semantic search."""
        collection_name = self._collection_name(session_id)
        search_limit = limit or self.settings.default_search_limit

        try:
            query_embedding = await self.generate_embedding(query)
            hits = self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=search_limit,
                with_payload=True,
            )

            results: List[SearchResult] = []
            for hit in hits:
                payload = hit.payload or {}
                results.append(
                    SearchResult(
                        title=payload.get("title", ""),
                        url=payload.get("url") or None,
                        snippet=f"{payload.get('content', '')[:200]}...",
                        cluster=payload.get("cluster_label"),
                        relevance_score=round(float(hit.score), 3)
                        if hit.score is not None
                        else None,
                        domain=payload.get("domain"),
                    )
                )
            return results
        except Exception as exc:
            logger.error(
                "Failed to search content",
                error=str(exc),
                session_id=session_id,
            )
            return []

    async def get_session_stats(self, session_id: str) -> Dict[str, int]:
        """Get statistics about the session content."""
        collection_name = self._collection_name(session_id)
        try:
            # verify collection exists
            self.qdrant_client.get_collection(collection_name)

            documents, _ = self.qdrant_client.scroll(
                collection_name=collection_name,
                limit=self.settings.max_scroll_limit,
                with_payload=True,
            )

            total_docs = len(documents)
            domains = set()
            clusters = set()
            total_words = 0

            for doc in documents:
                payload = doc.payload or {}
                domain = payload.get("domain")
                cluster_label = payload.get("cluster_label")
                if domain:
                    domains.add(domain)
                if cluster_label:
                    clusters.add(cluster_label)
                total_words += int(payload.get("word_count") or 0)

            average_words = total_words // total_docs if total_docs else 0
            return {
                "total_documents": total_docs,
                "unique_domains": len(domains),
                "clusters_count": len(clusters),
                "average_words": int(average_words),
                "top_domains": list(domains)[:10],
                "clusters": list(clusters),
            }
        except Exception as exc:
            logger.error(
                "Failed to get session stats",
                error=str(exc),
                session_id=session_id,
            )
            return {
                "total_documents": 0,
                "unique_domains": 0,
                "clusters_count": 0,
                "average_words": 0,
                "top_domains": [],
                "clusters": [],
            }

    async def get_cluster_info(
        self, session_id: str, cluster_name: Optional[str] = None
    ) -> List[ClusterSummary]:
        """Get information about clusters in the session."""
        collection_name = self._collection_name(session_id)
        try:
            scroll_filter = None
            if cluster_name:
                scroll_filter = Filter(
                    must=[
                        FieldCondition(
                            key="cluster_label",
                            match=MatchValue(value=cluster_name),
                        )
                    ]
                )

            documents, _ = self.qdrant_client.scroll(
                collection_name=collection_name,
                limit=self.settings.max_scroll_limit,
                with_payload=True,
                scroll_filter=scroll_filter,
            )

            clusters: Dict[str, Dict[str, object]] = {}
            for doc in documents:
                payload = doc.payload or {}
                label = payload.get("cluster_label") or "Unclustered"
                cluster_entry = clusters.setdefault(
                    label,
                    {
                        "count": 0,
                        "documents": [],
                        "domains": set(),
                    },
                )
                cluster_entry["count"] = int(cluster_entry["count"]) + 1
                sample = ArticleLink(
                    title=payload.get("title", ""),
                    url=payload.get("url") or None,
                    domain=payload.get("domain"),
                )
                cluster_entry["documents"].append(sample)
                domain = payload.get("domain")
                if domain:
                    cluster_entry["domains"].add(domain)

            summaries: List[ClusterSummary] = []
            for label, info in clusters.items():
                summaries.append(
                    ClusterSummary(
                        title=label,
                        description=(
                            f"Contains {info['count']} articles "
                            f"from {len(info['domains'])} domains"
                        ),
                        count=int(info["count"]),
                        sample_articles=[
                            article for article in info["documents"][:3]
                        ],
                    )
                )

            summaries.sort(key=lambda item: item.count, reverse=True)
            return summaries
        except Exception as exc:
            logger.error(
                "Failed to get cluster info",
                error=str(exc),
                session_id=session_id,
            )
            return []

    async def generate_llm_response(
        self, prompt: str, context: str = ""
    ) -> str:  # pragma: no cover - network interactions mocked in tests
        """Generate response using local LLM."""
        try:
            full_prompt = (
                "You are a helpful assistant that helps users explore their scraped "
                "web content. You can answer questions about articles, topics, and "
                "clusters."
                f"\n\nContext: {context}\n\nUser question: {prompt}\n\n"
                "Provide a helpful, concise response. If you're showing results, "
                "format them clearly."
            )
            response = await self.ollama_client.post(
                "/api/generate",
                json={
                    "model": self.settings.default_llm_model,
                    "prompt": full_prompt,
                    "stream": False,
                },
            )
            response.raise_for_status()
            result = response.json()
            return result["response"]
        except Exception as exc:
            logger.error("Failed to generate LLM response", error=str(exc))
            return (
                "I'm having trouble generating a response right now. "
                "Please try again."
            )

    def extract_intent(self, message: str) -> Dict[str, object]:
        """Extract intent and entities from user message."""
        message_lower = message.lower()

        if any(
            word in message_lower
            for word in ["show", "find", "search", "articles about", "content about"]
        ):
            topic_keywords = [
                word
                for word in message.split()
                if word.lower()
                not in {
                    "show",
                    "me",
                    "find",
                    "articles",
                    "about",
                    "content",
                    "the",
                    "a",
                    "an",
                }
            ]
            return {
                "intent": "search_content",
                "topic": " ".join(topic_keywords) if topic_keywords else message,
                "entities": topic_keywords,
            }

        if any(word in message_lower for word in ["cluster", "topic", "group", "category"]):
            return {"intent": "explore_clusters", "entities": []}

        if any(word in message_lower for word in ["summary", "overview", "stats", "statistics"]):
            return {"intent": "get_summary", "entities": []}

        if any(word in message_lower for word in ["domain", "website", "site"]):
            return {"intent": "domain_analysis", "entities": []}

        return {"intent": "general_query", "topic": message, "entities": []}

    async def process_message(
        self, session_id: str, message: str, context: List[Dict[str, object]]
    ) -> ChatResponse:
        """Process user message and generate response."""
        intent_data = self.extract_intent(message)
        intent = intent_data["intent"]

        sources: List[Dict[str, object]] = []
        suggestions: List[str] = []
        response_text = ""

        if intent == "search_content":
            topic = str(intent_data.get("topic") or message)
            search_results = await self.search_similar_content(topic, session_id)
            if search_results:
                response_text = f"I found {len(search_results)} articles related to '{topic}':"
                sources = [
                    result.model_dump(exclude_none=True) for result in search_results
                ]
                first_cluster = search_results[0].cluster or "this topic"
                first_title = search_results[0].title[:30]
                suggestions = [
                    f"Tell me more about {first_cluster}",
                    f"Find similar articles to '{first_title}...'",
                    "Show me the most recent articles on this topic",
                ]
            else:
                response_text = (
                    f"I couldn't find any articles specifically about '{topic}'. "
                    "Try a different search term or ask me about the available topics."
                )
                suggestions = [
                    "What topics are available in my data?",
                    "Show me all clusters",
                    "Give me a content summary",
                ]

        elif intent == "explore_clusters":
            clusters = await self.get_cluster_info(session_id)
            if clusters:
                response_text = f"I found {len(clusters)} topic clusters in your content:"
                sources = [
                    cluster.model_dump(exclude_none=True) for cluster in clusters[:5]
                ]
                suggestions = [
                    f"Show me articles in the {clusters[0].title} cluster",
                    "What's the largest cluster?",
                    "Find articles that don't fit in any cluster",
                ]
            else:
                response_text = (
                    "No clusters have been created yet. Make sure your content has been "
                    "processed and clustered."
                )
                suggestions = [
                    "Show me all my content",
                    "How many articles do I have?",
                    "What domains are in my data?",
                ]

        elif intent == "get_summary":
            stats = await self.get_session_stats(session_id)
            response_text = "Here's a summary of your scraped content:"
            sources = [
                StatisticCard(
                    title="Content Overview",
                    description=(
                        f"Total articles: {stats['total_documents']} | "
                        f"Unique domains: {stats['unique_domains']} | "
                        f"Clusters: {stats['clusters_count']}"
                    ),
                    metadata={
                        "Total Articles": str(stats["total_documents"]),
                        "Unique Domains": str(stats["unique_domains"]),
                        "Topic Clusters": str(stats["clusters_count"]),
                        "Average Words": str(stats["average_words"]),
                    },
                ).model_dump()
            ]
            suggestions = [
                "Show me the most active domains",
                "What are the main topics?",
                "Find the longest articles",
            ]

        elif intent == "domain_analysis":
            stats = await self.get_session_stats(session_id)
            domains = stats.get("top_domains", [])
            if domains:
                response_text = "Here are some of the most active domains in your data:"
                sources = [
                    {
                        "title": "Top Domains",
                        "domains": domains,
                        "description": f"Found {len(domains)} active domains.",
                    }
                ]
            else:
                response_text = (
                    "I couldn't find any domain information yet. "
                    "Try clustering or scraping more content."
                )
            suggestions = [
                "Show me clusters for a domain",
                "What topics are trending?",
                "Find articles from a specific site",
            ]

        else:
            stats = await self.get_session_stats(session_id)
            context_str = (
                f"Session has {stats['total_documents']} documents across "
                f"{stats['clusters_count']} clusters and {stats['unique_domains']} domains."
            )
            response_text = await self.generate_llm_response(message, context_str)
            suggestions = [
                "Show me articles about a specific topic",
                "What clusters were created?",
                "Give me a content summary",
            ]

        entry = ConversationEntry(
            timestamp=datetime.now(timezone.utc),
            user_message=message,
            bot_response=response_text,
            intent=str(intent),
            sources_count=len(sources),
        )
        self.conversations.append(session_id, entry)

        return ChatResponse(
            response=response_text,
            sources=sources,
            suggestions=suggestions,
            conversation_id=session_id,
        )

    # Helpers -------------------------------------------------------------

    def conversation_history(self, session_id: str):
        """Return serialized conversation history for a session."""
        return [
            entry.model_dump() for entry in self.conversations.get(session_id)
        ]

    def clear_history(self, session_id: str) -> None:
        """Clear stored conversation history."""
        self.conversations.clear(session_id)

    def _collection_name(self, session_id: str) -> str:
        return f"{self.settings.qdrant_collection_prefix}{session_id}"
