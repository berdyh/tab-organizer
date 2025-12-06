"""Content Analyzer Service - Generates embeddings and summaries with configurable models."""

import asyncio
import json
import os
import time
import uuid
import re
from typing import Dict, List, Optional, Any

import numpy as np
import structlog
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import tiktoken

from app.core.llm_client import LLMFactory

# Configure structured logging
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
    title="Content Analyzer Service",
    description="Generates embeddings and summaries for scraped content with configurable models",
    version="1.0.0"
)

# Pydantic models for API
class ContentItem(BaseModel):
    id: str
    content: str
    title: Optional[str] = None
    url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AnalysisRequest(BaseModel):
    content_items: Optional[List[ContentItem]] = None
    session_id: str
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None
    generate_summary: bool = True
    extract_keywords: bool = True
    assess_quality: bool = True

class AnalysisResponse(BaseModel):
    job_id: str
    status: str
    message: str

class TextChunker:
    """Handle text chunking with overlap preservation."""

    def __init__(self):
        self.logger = structlog.get_logger("text_chunker")
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            self.logger.warning("Could not load tiktoken, using character-based chunking", error=str(e))
            self.tokenizer = None

    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks."""
        if not text or not text.strip():
            return []

        try:
            if self.tokenizer:
                return self._chunk_by_tokens(text, chunk_size, overlap)
            else:
                return self._chunk_by_characters(text, chunk_size * 4, overlap * 4)
        except Exception as e:
            self.logger.error("Text chunking failed", error=str(e))
            return [text]

    def _chunk_by_tokens(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        tokens = self.tokenizer.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(self.tokenizer.decode(chunk_tokens))
            start = end - overlap
            if end >= len(tokens):
                break
        return chunks

    def _chunk_by_characters(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - overlap
            if end >= len(text):
                break
        return chunks

text_chunker = TextChunker()

class QdrantManager:
    def __init__(self):
        self.client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "qdrant"),
            port=int(os.getenv("QDRANT_PORT", "6333"))
        )

    async def ensure_collection_exists(self, collection_name: str, vector_size: int):
        try:
            collections = self.client.get_collections().collections
            if collection_name not in [c.name for c in collections]:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance="Cosine")
                )
        except Exception as e:
            logger.error("Qdrant collection ensure failed", error=str(e))

    async def store_analyzed_content(self, collection_name, items, embedding_model, llm_model):
        points = []
        for item in items:
            points.append(PointStruct(
                id=f"{item['content_id']}_{item['chunk_index']}",
                vector=item["embedding"],
                payload={
                    "content_id": item["content_id"],
                    "text": item["text"],
                    "title": item["title"],
                    "url": item["url"],
                    "summary": item.get("summary"),
                    "keywords": item.get("keywords"),
                    "quality_assessment": item.get("quality_assessment"),
                    "embedding_model": embedding_model,
                    "llm_model": llm_model
                }
            ))
        if points:
            self.client.upsert(collection_name=collection_name, points=points)
        return len(points)

    async def fetch_content_for_session(self, session_id: str) -> List[ContentItem]:
        try:
            collection_name = f"session_{session_id}"
            collections = self.client.get_collections().collections
            if collection_name not in [c.name for c in collections]:
                return []

            points, _ = self.client.scroll(
                collection_name=collection_name,
                limit=1000,
                with_payload=True,
                with_vectors=False
            )

            items = []
            seen_ids = set()
            for point in points:
                cid = point.payload.get("content_id", str(point.id))
                if cid not in seen_ids:
                    items.append(ContentItem(
                        id=cid,
                        content=point.payload.get("text", "") or point.payload.get("content", ""),
                        title=point.payload.get("title"),
                        url=point.payload.get("url"),
                        metadata=point.payload
                    ))
                    seen_ids.add(cid)
            return items
        except Exception as e:
            logger.error("Fetch content failed", error=str(e))
            return []

qdrant_manager = QdrantManager()

async def process_analysis_job(
    job_id: str,
    content_items: List[ContentItem],
    session_id: str,
    llm_config: Dict[str, Any],
    embedding_config: Dict[str, Any],
    options: Dict[str, bool]
):
    logger.info("Starting analysis job", job_id=job_id)
    try:
        # If no items passed, try to fetch from DB
        if not content_items:
            content_items = await qdrant_manager.fetch_content_for_session(session_id)
            if not content_items:
                logger.warning("No content found for session", session_id=session_id)
                return

        # Initialize clients
        llm_client = LLMFactory.get_client(
            llm_config.get("provider", "ollama"),
            api_key=llm_config.get("api_key"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        )

        emb_provider = embedding_config.get("provider", "local")
        embedding_model_name = embedding_config.get("model")

        local_embedder = None
        cloud_embedder = None
        vector_size = 384

        if emb_provider == "local":
             if not embedding_model_name or "minilm" in embedding_model_name.lower():
                 local_embedder = SentenceTransformer("all-MiniLM-L6-v2")
                 vector_size = 384
             else:
                 cloud_embedder = LLMFactory.get_client("ollama", base_url=os.getenv("OLLAMA_BASE_URL"))
                 vector_size = 768
        else:
            cloud_embedder = LLMFactory.get_client(emb_provider, api_key=embedding_config.get("api_key"))
            # Fix: Determine vector size dynamically for cloud models
            if cloud_embedder:
                try:
                    probe_text = "probe"
                    probe_embedding = await cloud_embedder.get_embedding(probe_text, model=embedding_model_name)
                    if probe_embedding:
                        vector_size = len(probe_embedding)
                        logger.info("Determined vector size from probe", size=vector_size, model=embedding_model_name)
                except Exception as e:
                    logger.error("Failed to probe vector size", error=str(e))
                    # Fallback defaults if probe fails
                    if emb_provider == "openai": vector_size = 1536
                    elif emb_provider == "gemini": vector_size = 768

        collection_name = f"session_{session_id}"
        await qdrant_manager.ensure_collection_exists(collection_name, vector_size)

        processed_items = []

        for item in content_items:
            chunks = text_chunker.chunk_text(item.content)

            for i, chunk_text in enumerate(chunks):
                embedding = []
                if local_embedder:
                    embedding = local_embedder.encode(chunk_text).tolist()
                elif cloud_embedder:
                    embedding = await cloud_embedder.get_embedding(chunk_text, model=embedding_model_name)

                summary = ""
                keywords = ""
                quality = ""

                if options.get("generate_summary"):
                    try:
                        summary = await llm_client.generate(f"Summarize this: {chunk_text[:2000]}", model=llm_config.get("model"))
                    except: pass

                if options.get("extract_keywords"):
                     try:
                        keywords = await llm_client.generate(f"Extract keywords: {chunk_text[:2000]}", model=llm_config.get("model"))
                     except: pass

                if options.get("assess_quality"):
                     try:
                        quality = await llm_client.generate(f"Rate quality 1-10: {chunk_text[:2000]}", model=llm_config.get("model"))
                     except: pass

                processed_items.append({
                    "content_id": item.id,
                    "chunk_index": i,
                    "text": chunk_text,
                    "title": item.title,
                    "url": item.url,
                    "embedding": embedding,
                    "summary": summary,
                    "keywords": keywords,
                    "quality_assessment": quality
                })

        await qdrant_manager.store_analyzed_content(
            collection_name,
            processed_items,
            embedding_model=embedding_model_name or "default",
            llm_model=llm_config.get("model") or "default"
        )

        logger.info("Analysis job completed", job_id=job_id)

    except Exception as e:
        logger.error("Analysis job failed", job_id=job_id, error=str(e))


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_content(request: AnalysisRequest, req: Request, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())

    llm_config = {
        "provider": req.headers.get("X-LLM-Provider", "ollama"),
        "model": req.headers.get("X-LLM-Model"),
        "api_key": req.headers.get("X-OpenAI-Key") if req.headers.get("X-LLM-Provider") == "openai" else
                   req.headers.get("X-DeepSeek-Key") if req.headers.get("X-LLM-Provider") == "deepseek" else
                   req.headers.get("X-Gemini-Key")
    }

    embedding_config = {
        "provider": req.headers.get("X-Embedding-Provider", "local"),
        "model": req.headers.get("X-Embedding-Model"),
        "api_key": req.headers.get("X-OpenAI-Key") if req.headers.get("X-Embedding-Provider") == "openai" else
                   req.headers.get("X-Gemini-Key")
    }

    options = {
        "generate_summary": request.generate_summary,
        "extract_keywords": request.extract_keywords,
        "assess_quality": request.assess_quality
    }

    background_tasks.add_task(
        process_analysis_job,
        job_id,
        request.content_items,
        request.session_id,
        llm_config,
        embedding_config,
        options
    )

    return AnalysisResponse(
        job_id=job_id,
        status="processing",
        message="Started analysis"
    )

@app.get("/status")
async def get_status(session_id: str):
    return {"status": "unknown", "message": "Check logs for progress"}

@app.get("/search")
async def search_content(query: str, session_id: str, req: Request):
    embedding_config = {
        "provider": req.headers.get("X-Embedding-Provider", "local"),
        "model": req.headers.get("X-Embedding-Model"),
        "api_key": req.headers.get("X-OpenAI-Key") if req.headers.get("X-Embedding-Provider") == "openai" else
                   req.headers.get("X-Gemini-Key")
    }

    emb_provider = embedding_config.get("provider", "local")
    cloud_embedder = None
    local_embedder = None
    embedding = []

    if emb_provider == "local":
        local_embedder = SentenceTransformer("all-MiniLM-L6-v2")
        embedding = local_embedder.encode(query).tolist()
    else:
        cloud_embedder = LLMFactory.get_client(emb_provider, api_key=embedding_config.get("api_key"))
        try:
            embedding = await cloud_embedder.get_embedding(query, model=embedding_config.get("model"))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

    try:
        client = QdrantClient(host=os.getenv("QDRANT_HOST", "qdrant"), port=int(os.getenv("QDRANT_PORT", "6333")))
        results = client.search(
            collection_name=f"session_{session_id}",
            query_vector=embedding,
            limit=5,
            with_payload=True
        )
        return [{
            "title": r.payload.get("title"),
            "url": r.payload.get("url"),
            "summary": r.payload.get("summary"),
            "score": r.score
        } for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8084)
