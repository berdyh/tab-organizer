"""Chatbot Service - Natural language interface for content exploration."""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
from fastapi import FastAPI, HTTPException, Request, Depends
from pydantic import BaseModel
import structlog
from qdrant_client import QdrantClient

from llm_client import LLMFactory

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

app = FastAPI(
    title="Chatbot Service",
    description="Natural language interface for exploring scraped content and clusters",
    version="1.0.0"
)

# Initialize clients
qdrant_client = QdrantClient(
    host=os.getenv("QDRANT_HOST", "qdrant"),
    port=int(os.getenv("QDRANT_PORT", "6333"))
)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

# Data Models
class ChatMessage(BaseModel):
    session_id: str
    message: str
    context: Optional[List[Dict[str, Any]]] = []

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]] = []
    suggestions: List[str] = []
    conversation_id: Optional[str] = None

class FeedbackRequest(BaseModel):
    message_id: str
    feedback: str
    comment: Optional[str] = None

# Conversation memory
conversations = {}

class ChatbotService:
    def __init__(self):
        self.default_client = LLMFactory.get_client("ollama", base_url=OLLAMA_BASE_URL)

    def get_llm_client(self, request: Request):
        provider = request.headers.get("X-LLM-Provider", "ollama")
        
        api_key = None
        if provider == "openai":
            api_key = request.headers.get("X-OpenAI-Key")
        elif provider == "deepseek":
            api_key = request.headers.get("X-DeepSeek-Key")
        elif provider == "gemini":
            api_key = request.headers.get("X-Gemini-Key")

        return LLMFactory.get_client(provider, api_key=api_key, base_url=OLLAMA_BASE_URL if provider=="ollama" else None)

    def get_embedding_client(self, request: Request):
        provider = request.headers.get("X-Embedding-Provider", "local")
        # Map 'local' to 'ollama' for this service as it used Ollama for embeddings
        if provider == "local":
            provider = "ollama"

        api_key = None
        if provider == "openai":
            api_key = request.headers.get("X-OpenAI-Key")
        elif provider == "gemini":
            api_key = request.headers.get("X-Gemini-Key")

        return LLMFactory.get_client(provider, api_key=api_key, base_url=OLLAMA_BASE_URL if provider=="ollama" else None)

    async def generate_embedding(self, text: str, request: Request) -> List[float]:
        client = self.get_embedding_client(request)
        model = request.headers.get("X-Embedding-Model")
        try:
            return await client.get_embedding(text, model=model)
        except Exception as e:
            logger.error("Failed to generate embedding", error=str(e))
            raise HTTPException(status_code=500, detail="Failed to generate embedding")

    async def search_similar_content(self, query: str, session_id: str, request: Request, limit: int = 5) -> List[Dict[str, Any]]:
        try:
            query_embedding = await self.generate_embedding(query, request)
            
            search_result = qdrant_client.search(
                collection_name=f"session_{session_id}",
                query_vector=query_embedding,
                limit=limit,
                with_payload=True
            )
            
            results = []
            for hit in search_result:
                results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "title": hit.payload.get("title", ""),
                    "url": hit.payload.get("url", ""),
                    "content": hit.payload.get("content", "")[:200] + "...",
                    "cluster": hit.payload.get("cluster_label", ""),
                    "domain": hit.payload.get("domain", "")
                })
            
            return results
        except Exception as e:
            logger.error("Failed to search content", error=str(e), session_id=session_id)
            return []

    # ... (get_session_stats and get_cluster_info remain mostly same, omitted for brevity but should be kept if not dependent on LLM)
    # Re-implementing helper methods to keep file complete
    async def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        try:
            search_result = qdrant_client.scroll(
                collection_name=f"session_{session_id}",
                limit=1000,
                with_payload=True
            )
            documents = search_result[0]
            total_docs = len(documents)
            domains = set()
            clusters = set()
            total_words = 0
            for doc in documents:
                payload = doc.payload
                if payload.get("domain"): domains.add(payload["domain"])
                if payload.get("cluster_label"): clusters.add(payload["cluster_label"])
                if payload.get("word_count"): total_words += payload["word_count"]
            
            return {
                "total_documents": total_docs,
                "unique_domains": len(domains),
                "clusters_count": len(clusters),
                "average_words": total_words // max(total_docs, 1)
            }
        except:
            return {"total_documents": 0, "unique_domains": 0, "clusters_count": 0}

    async def get_cluster_info(self, session_id: str) -> List[Dict[str, Any]]:
        try:
            search_result = qdrant_client.scroll(
                collection_name=f"session_{session_id}",
                limit=1000,
                with_payload=True
            )
            documents = search_result[0]
            clusters = {}
            for doc in documents:
                payload = doc.payload
                cluster_label = payload.get("cluster_label", "Unclustered")
                if cluster_label not in clusters:
                    clusters[cluster_label] = {"name": cluster_label, "count": 0, "documents": [], "domains": set()}
                clusters[cluster_label]["count"] += 1
                clusters[cluster_label]["documents"].append({
                    "title": payload.get("title", ""),
                    "url": payload.get("url", ""),
                    "domain": payload.get("domain", "")
                })
                if payload.get("domain"): clusters[cluster_label]["domains"].add(payload["domain"])
            
            cluster_list = []
            for name, data in clusters.items():
                cluster_list.append({
                    "name": name,
                    "count": data["count"],
                    "description": f"Contains {data['count']} articles",
                    "sample_articles": data["documents"][:3]
                })
            cluster_list.sort(key=lambda x: x["count"], reverse=True)
            return cluster_list
        except:
            return []

    async def generate_llm_response(self, prompt: str, context: str, request: Request) -> str:
        client = self.get_llm_client(request)
        model = request.headers.get("X-LLM-Model")

        full_prompt = f"""You are a helpful assistant.
Context from database: {context}

User question: {prompt}"""

        try:
            return await client.generate(prompt, context, model=model)
        except Exception as e:
            logger.error("Failed to generate LLM response", error=str(e))
            return "I'm having trouble generating a response right now."

    async def process_message(self, session_id: str, message: str, context: List[Dict[str, Any]], request: Request) -> ChatResponse:
        try:
            # Simple intent logic
            intent = "general"
            if "show" in message.lower() or "find" in message.lower():
                intent = "search"
            
            response_text = ""
            sources = []
            
            if intent == "search":
                results = await self.search_similar_content(message, session_id, request)
                if results:
                    response_text = f"Found {len(results)} articles:"
                    sources = results
                else:
                    response_text = "No articles found."
            else:
                # LLM Generation
                # Build context string
                ctx_str = ""
                if context:
                    ctx_str = "\n".join([f"{m['role']}: {m['content']}" for m in context[-3:]])
                
                # Retrieve some stats to enrich context
                stats = await self.get_session_stats(session_id)
                ctx_str += f"\nSession Stats: {stats}"
                
                response_text = await self.generate_llm_response(message, ctx_str, request)
            
            # Save conversation
            if session_id not in conversations: conversations[session_id] = []
            conversations[session_id].append({
                "timestamp": datetime.now().isoformat(),
                "user_message": message,
                "bot_response": response_text
            })
            
            return ChatResponse(
                response=response_text,
                sources=sources,
                conversation_id=session_id
            )
            
        except Exception as e:
            logger.error("Failed to process message", error=str(e))
            return ChatResponse(response="Error processing message.")

chatbot_service = ChatbotService()

@app.post("/chat/message", response_model=ChatResponse)
async def send_message(request: ChatMessage, req: Request):
    return await chatbot_service.process_message(
        request.session_id,
        request.message,
        request.context,
        req
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8092)
