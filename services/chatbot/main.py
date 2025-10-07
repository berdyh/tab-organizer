"""Chatbot Service - Natural language interface for content exploration."""

import os
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import structlog
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchRequest
import numpy as np

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
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "phi4:3.8b")
DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "mxbai-embed-large")

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
    feedback: str  # 'positive' or 'negative'
    comment: Optional[str] = None

# Conversation memory
conversations = {}

class ChatbotService:
    def __init__(self):
        self.ollama_client = httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=60.0)
        
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Ollama."""
        try:
            response = await self.ollama_client.post(
                "/api/embeddings",
                json={
                    "model": DEFAULT_EMBEDDING_MODEL,
                    "prompt": text
                }
            )
            response.raise_for_status()
            result = response.json()
            return result["embedding"]
        except Exception as e:
            logger.error("Failed to generate embedding", error=str(e))
            raise HTTPException(status_code=500, detail="Failed to generate embedding")

    async def search_similar_content(self, query: str, session_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar content using semantic search."""
        try:
            # Generate embedding for query
            query_embedding = await self.generate_embedding(query)
            
            # Search in Qdrant
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

    async def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics about the session content."""
        try:
            # Get collection info
            collection_info = qdrant_client.get_collection(f"session_{session_id}")
            
            # Get sample of documents to analyze
            search_result = qdrant_client.scroll(
                collection_name=f"session_{session_id}",
                limit=1000,
                with_payload=True
            )
            
            documents = search_result[0]
            
            # Calculate statistics
            total_docs = len(documents)
            domains = set()
            clusters = set()
            total_words = 0
            
            for doc in documents:
                payload = doc.payload
                if payload.get("domain"):
                    domains.add(payload["domain"])
                if payload.get("cluster_label"):
                    clusters.add(payload["cluster_label"])
                if payload.get("word_count"):
                    total_words += payload["word_count"]
            
            return {
                "total_documents": total_docs,
                "unique_domains": len(domains),
                "clusters_count": len(clusters),
                "average_words": total_words // max(total_docs, 1),
                "top_domains": list(domains)[:10],
                "clusters": list(clusters)
            }
        except Exception as e:
            logger.error("Failed to get session stats", error=str(e), session_id=session_id)
            return {"total_documents": 0, "unique_domains": 0, "clusters_count": 0}

    async def get_cluster_info(self, session_id: str, cluster_name: str = None) -> List[Dict[str, Any]]:
        """Get information about clusters in the session."""
        try:
            # Get all documents with cluster information
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
                    clusters[cluster_label] = {
                        "name": cluster_label,
                        "count": 0,
                        "documents": [],
                        "domains": set()
                    }
                
                clusters[cluster_label]["count"] += 1
                clusters[cluster_label]["documents"].append({
                    "title": payload.get("title", ""),
                    "url": payload.get("url", ""),
                    "domain": payload.get("domain", "")
                })
                if payload.get("domain"):
                    clusters[cluster_label]["domains"].add(payload["domain"])
            
            # Convert to list and add descriptions
            cluster_list = []
            for cluster_name, cluster_data in clusters.items():
                cluster_list.append({
                    "name": cluster_name,
                    "count": cluster_data["count"],
                    "description": f"Contains {cluster_data['count']} articles from {len(cluster_data['domains'])} domains",
                    "top_domains": list(cluster_data["domains"])[:5],
                    "sample_articles": cluster_data["documents"][:3]
                })
            
            # Sort by count
            cluster_list.sort(key=lambda x: x["count"], reverse=True)
            return cluster_list
            
        except Exception as e:
            logger.error("Failed to get cluster info", error=str(e), session_id=session_id)
            return []

    async def generate_llm_response(self, prompt: str, context: str = "") -> str:
        """Generate response using local LLM."""
        try:
            full_prompt = f"""You are a helpful assistant that helps users explore their scraped web content. 
You have access to their content database and can answer questions about articles, topics, and clusters.

Context: {context}

User question: {prompt}

Provide a helpful, concise response. If you're showing results, format them clearly."""

            response = await self.ollama_client.post(
                "/api/generate",
                json={
                    "model": DEFAULT_LLM_MODEL,
                    "prompt": full_prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            result = response.json()
            return result["response"]
        except Exception as e:
            logger.error("Failed to generate LLM response", error=str(e))
            return "I'm having trouble generating a response right now. Please try again."

    def extract_intent(self, message: str) -> Dict[str, Any]:
        """Extract intent and entities from user message."""
        message_lower = message.lower()
        
        # Define intent patterns
        if any(word in message_lower for word in ["show", "find", "search", "articles about", "content about"]):
            # Extract topic
            topic_keywords = []
            for word in message.split():
                if word.lower() not in ["show", "me", "find", "articles", "about", "content", "the", "a", "an"]:
                    topic_keywords.append(word)
            
            return {
                "intent": "search_content",
                "topic": " ".join(topic_keywords) if topic_keywords else message,
                "entities": topic_keywords
            }
        
        elif any(word in message_lower for word in ["cluster", "topic", "group", "category"]):
            return {
                "intent": "explore_clusters",
                "entities": []
            }
        
        elif any(word in message_lower for word in ["summary", "overview", "stats", "statistics"]):
            return {
                "intent": "get_summary",
                "entities": []
            }
        
        elif any(word in message_lower for word in ["domain", "website", "site"]):
            return {
                "intent": "domain_analysis",
                "entities": []
            }
        
        else:
            return {
                "intent": "general_query",
                "topic": message,
                "entities": []
            }

    async def process_message(self, session_id: str, message: str, context: List[Dict[str, Any]]) -> ChatResponse:
        """Process user message and generate response."""
        try:
            # Extract intent
            intent_data = self.extract_intent(message)
            intent = intent_data["intent"]
            
            sources = []
            suggestions = []
            response_text = ""
            
            if intent == "search_content":
                # Search for relevant content
                topic = intent_data.get("topic", message)
                search_results = await self.search_similar_content(topic, session_id)
                
                if search_results:
                    response_text = f"I found {len(search_results)} articles related to '{topic}':"
                    sources = [
                        {
                            "title": result["title"],
                            "url": result["url"],
                            "snippet": result["content"],
                            "cluster": result["cluster"],
                            "relevance_score": round(result["score"], 3)
                        }
                        for result in search_results
                    ]
                    suggestions = [
                        f"Tell me more about {search_results[0]['cluster']}",
                        f"Find similar articles to '{search_results[0]['title'][:30]}...'",
                        "Show me the most recent articles on this topic"
                    ]
                else:
                    response_text = f"I couldn't find any articles specifically about '{topic}'. Try a different search term or ask me about the available topics."
                    suggestions = [
                        "What topics are available in my data?",
                        "Show me all clusters",
                        "Give me a content summary"
                    ]
            
            elif intent == "explore_clusters":
                # Get cluster information
                clusters = await self.get_cluster_info(session_id)
                
                if clusters:
                    response_text = f"I found {len(clusters)} topic clusters in your content:"
                    sources = [
                        {
                            "title": cluster["name"],
                            "description": cluster["description"],
                            "count": cluster["count"],
                            "sample_articles": cluster["sample_articles"]
                        }
                        for cluster in clusters[:5]  # Show top 5 clusters
                    ]
                    suggestions = [
                        f"Show me articles in the {clusters[0]['name']} cluster",
                        "What's the largest cluster?",
                        "Find articles that don't fit in any cluster"
                    ]
                else:
                    response_text = "No clusters have been created yet. Make sure your content has been processed and clustered."
                    suggestions = [
                        "Show me all my content",
                        "How many articles do I have?",
                        "What domains are in my data?"
                    ]
            
            elif intent == "get_summary":
                # Get session statistics
                stats = await self.get_session_stats(session_id)
                
                response_text = "Here's a summary of your scraped content:"
                sources = [
                    {
                        "title": "Content Overview",
                        "description": f"Total articles: {stats['total_documents']} | Unique domains: {stats['unique_domains']} | Clusters: {stats['clusters_count']}",
                        "metadata": {
                            "Total Articles": str(stats["total_documents"]),
                            "Unique Domains": str(stats["unique_domains"]),
                            "Topic Clusters": str(stats["clusters_count"]),
                            "Average Words": str(stats["average_words"])
                        }
                    }
                ]
                suggestions = [
                    "Show me the most active domains",
                    "What are the main topics?",
                    "Find the longest articles"
                ]
            
            else:
                # General query - use LLM with context
                context_str = f"Session has {await self.get_session_stats(session_id)} documents"
                response_text = await self.generate_llm_response(message, context_str)
                suggestions = [
                    "Show me articles about a specific topic",
                    "What clusters were created?",
                    "Give me a content summary"
                ]
            
            # Store conversation
            if session_id not in conversations:
                conversations[session_id] = []
            
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "user_message": message,
                "bot_response": response_text,
                "intent": intent,
                "sources_count": len(sources)
            }
            conversations[session_id].append(conversation_entry)
            
            return ChatResponse(
                response=response_text,
                sources=sources,
                suggestions=suggestions,
                conversation_id=session_id
            )
            
        except Exception as e:
            logger.error("Failed to process message", error=str(e), session_id=session_id)
            return ChatResponse(
                response="I encountered an error while processing your request. Please try again or rephrase your question.",
                sources=[],
                suggestions=[
                    "Try asking about your content topics",
                    "Ask for a content summary",
                    "Search for specific articles"
                ]
            )

# Initialize service
chatbot_service = ChatbotService()

@app.post("/chat/message", response_model=ChatResponse)
async def send_message(request: ChatMessage):
    """Send a message to the chatbot and get a response."""
    logger.info("Processing chat message", session_id=request.session_id, message_length=len(request.message))
    
    response = await chatbot_service.process_message(
        request.session_id,
        request.message,
        request.context
    )
    
    logger.info("Chat response generated", session_id=request.session_id, sources_count=len(response.sources))
    return response

@app.get("/chat/history/{session_id}")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session."""
    history = conversations.get(session_id, [])
    return {"session_id": session_id, "history": history}

@app.delete("/chat/history/{session_id}")
async def clear_conversation_history(session_id: str):
    """Clear conversation history for a session."""
    if session_id in conversations:
        del conversations[session_id]
    return {"message": "Conversation history cleared"}

@app.post("/chat/feedback")
async def provide_feedback(request: FeedbackRequest):
    """Provide feedback on a chatbot response."""
    logger.info("Received feedback", message_id=request.message_id, feedback=request.feedback)
    
    # Store feedback (in a real implementation, this would go to a database)
    feedback_entry = {
        "message_id": request.message_id,
        "feedback": request.feedback,
        "comment": request.comment,
        "timestamp": datetime.now().isoformat()
    }
    
    return {"message": "Feedback received", "feedback_id": request.message_id}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check Qdrant connection
        qdrant_client.get_collections()
        
        # Check Ollama connection
        response = await chatbot_service.ollama_client.get("/api/tags")
        response.raise_for_status()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "qdrant": "connected",
                "ollama": "connected"
            }
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service unhealthy")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8092)