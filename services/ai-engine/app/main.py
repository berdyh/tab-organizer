"""AI Engine Service - Main Application."""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from .core.llm_client import LLMClient
from .clustering.pipeline import TabClusterer, Tab
from .chatbot.rag import RAGChatbot, Document

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
    qdrant_host=os.getenv("QDRANT_HOST", "qdrant"),
    qdrant_port=int(os.getenv("QDRANT_PORT", 6333)),
    embedding_dim=int(os.getenv("EMBEDDING_DIMENSIONS", 768)),
)
chatbot.set_llm_client(llm_client)


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
    embedding_provider: Optional[str] = None


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
    return {"status": "healthy"}


# Provider info
@app.get("/providers")
async def get_providers():
    return llm_client.get_provider_info()


@app.post("/providers/switch")
async def switch_provider(request: ProviderSwitchRequest):
    try:
        llm_client.switch_provider(
            llm_provider=request.llm_provider,
            embedding_provider=request.embedding_provider,
        )
        return {
            "status": "switched",
            "providers": llm_client.get_provider_info(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Embedding endpoints
@app.post("/embed")
async def embed_texts(request: EmbedRequest):
    try:
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
async def generate_text(request: GenerateRequest):
    try:
        result = await llm_client.generate(request.prompt, request.system)
        return {"text": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Clustering endpoints
@app.post("/cluster")
async def cluster_urls(request: ClusterRequest):
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
async def index_documents(request: IndexRequest):
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
        
        count = await chatbot.index_documents(documents, request.session_id)
        return {"indexed": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        result = await chatbot.chat(
            query=request.query,
            session_id=request.session_id,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search(request: ChatRequest):
    try:
        results = await chatbot.search(
            query=request.query,
            session_id=request.session_id,
        )
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/summarize/{session_id}")
async def summarize_session(session_id: str):
    try:
        summary = await chatbot.summarize_session(session_id)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{session_id}")
async def delete_session_documents(session_id: str):
    try:
        count = chatbot.delete_session_documents(session_id)
        return {"deleted": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
