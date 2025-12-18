"""RAG-based chatbot for querying scraped content."""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter


@dataclass
class Document:
    """A document chunk for RAG."""
    id: str
    url: str
    title: str
    content: str
    embedding: Optional[list[float]] = None
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RAGChatbot:
    """RAG-based chatbot for querying scraped content."""
    
    COLLECTION_NAME = "tab_organizer_docs"
    
    def __init__(
        self,
        qdrant_host: str = "qdrant",
        qdrant_port: int = 6333,
        embedding_dim: int = 768,
    ):
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.embedding_dim = embedding_dim
        self._client: Optional[QdrantClient] = None
        self._llm_client = None
    
    @property
    def client(self) -> QdrantClient:
        """Lazy-load Qdrant client."""
        if self._client is None:
            self._client = QdrantClient(
                host=self.qdrant_host,
                port=self.qdrant_port,
            )
            self._ensure_collection()
        return self._client
    
    def set_llm_client(self, client) -> None:
        """Set LLM client for generation."""
        self._llm_client = client
    
    def _ensure_collection(self) -> None:
        """Ensure the collection exists."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.COLLECTION_NAME not in collection_names:
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
    
    async def index_documents(
        self,
        documents: list[Document],
        session_id: Optional[str] = None,
    ) -> int:
        """Index documents into Qdrant."""
        if not self._llm_client:
            raise RuntimeError("LLM client not set")
        
        # Generate embeddings for documents without them
        docs_needing_embeddings = [d for d in documents if d.embedding is None]
        if docs_needing_embeddings:
            contents = [d.content for d in docs_needing_embeddings]
            embeddings = await self._llm_client.embed(contents)
            for doc, emb in zip(docs_needing_embeddings, embeddings):
                doc.embedding = emb
        
        # Prepare points for Qdrant
        points = []
        for i, doc in enumerate(documents):
            if doc.embedding is None:
                continue
            
            payload = {
                "url": doc.url,
                "title": doc.title,
                "content": doc.content[:10000],  # Limit content size
                **doc.metadata,
            }
            if session_id:
                payload["session_id"] = session_id
            
            points.append(PointStruct(
                id=hash(doc.id) % (2**63),  # Convert to int64
                vector=doc.embedding,
                payload=payload,
            ))
        
        if points:
            self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=points,
            )
        
        return len(points)
    
    async def search(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: int = 5,
    ) -> list[dict]:
        """Search for relevant documents."""
        if not self._llm_client:
            raise RuntimeError("LLM client not set")
        
        # Generate query embedding
        query_embedding = await self._llm_client.embed_single(query)
        
        # Build filter
        query_filter = None
        if session_id:
            query_filter = Filter(
                must=[
                    {"key": "session_id", "match": {"value": session_id}}
                ]
            )
        
        # Search
        results = self.client.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=top_k,
        )
        
        return [
            {
                "url": r.payload.get("url", ""),
                "title": r.payload.get("title", ""),
                "content": r.payload.get("content", ""),
                "score": r.score,
            }
            for r in results
        ]
    
    async def chat(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: int = 5,
    ) -> dict:
        """
        Chat with the indexed content using RAG.
        
        Returns:
            Dict with 'answer', 'sources', and 'context'.
        """
        if not self._llm_client:
            raise RuntimeError("LLM client not set")
        
        # Search for relevant documents
        results = await self.search(query, session_id, top_k)
        
        if not results:
            return {
                "answer": "I don't have any relevant information to answer your question. Please make sure you've scraped some URLs first.",
                "sources": [],
                "context": "",
            }
        
        # Build context from results
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[{i}] {result['title']}\n"
                f"URL: {result['url']}\n"
                f"Content: {result['content'][:1000]}...\n"
            )
        context = "\n".join(context_parts)
        
        # Generate answer
        system_prompt = """You are a helpful assistant that answers questions based on the provided context from web pages.
Always cite your sources using the reference numbers [1], [2], etc.
If the context doesn't contain relevant information, say so honestly.
Be concise but thorough."""

        prompt = f"""Context from scraped web pages:

{context}

Question: {query}

Please answer the question based on the context above. Cite sources using [1], [2], etc."""

        answer = await self._llm_client.generate(prompt, system=system_prompt)
        
        return {
            "answer": answer,
            "sources": [
                {"url": r["url"], "title": r["title"], "score": r["score"]}
                for r in results
            ],
            "context": context,
        }
    
    async def summarize_session(self, session_id: str) -> str:
        """Generate a summary of all content in a session."""
        if not self._llm_client:
            raise RuntimeError("LLM client not set")
        
        # Get all documents for session
        results = self.client.scroll(
            collection_name=self.COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    {"key": "session_id", "match": {"value": session_id}}
                ]
            ),
            limit=100,
        )
        
        points, _ = results
        if not points:
            return "No content found for this session."
        
        # Build summary context
        summaries = []
        for point in points[:20]:  # Limit to 20 documents
            title = point.payload.get("title", "Untitled")
            content = point.payload.get("content", "")[:500]
            summaries.append(f"- {title}: {content}...")
        
        prompt = f"""Summarize the following collection of web pages in 2-3 paragraphs:

{chr(10).join(summaries)}

Provide a cohesive summary that captures the main themes and topics."""

        return await self._llm_client.generate(prompt)
    
    def delete_session_documents(self, session_id: str) -> int:
        """Delete all documents for a session."""
        # Get points to delete
        results = self.client.scroll(
            collection_name=self.COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    {"key": "session_id", "match": {"value": session_id}}
                ]
            ),
            limit=10000,
        )
        
        points, _ = results
        if not points:
            return 0
        
        point_ids = [p.id for p in points]
        self.client.delete(
            collection_name=self.COLLECTION_NAME,
            points_selector=point_ids,
        )
        
        return len(point_ids)
