"""RAG-based chatbot for querying scraped content."""

import os
from dataclasses import dataclass
from typing import Optional

import lancedb
import pandas as pd


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

    TABLE_NAME = "tab_organizer_docs"

    def __init__(
        self,
        db_uri: str = "/data/lancedb",
        embedding_dim: int = 1024,
    ):
        self.db_uri = db_uri
        self.embedding_dim = embedding_dim
        self._db = None
        self._table = None
        self._llm_client = None

    @property
    def db(self):
        """Lazy-load LanceDB connection."""
        if self._db is None:
            os.makedirs(self.db_uri, exist_ok=True)
            self._db = lancedb.connect(self.db_uri)
        return self._db

    @property
    def table(self):
        """Lazy-load LanceDB table."""
        if self._table is None:
            self._ensure_table()
        return self._table

    def set_llm_client(self, client) -> None:
        """Set LLM client for generation."""
        self._llm_client = client

    def _ensure_table(self) -> None:
        """Ensure LanceDB table exists with expected schema."""
        table_names = self.db.table_names()
        if self.TABLE_NAME in table_names:
            self._table = self.db.open_table(self.TABLE_NAME)
            return

        seed_df = pd.DataFrame(
            [
                {
                    "id": "__seed__",
                    "session_id": "__seed__",
                    "url": "",
                    "title": "",
                    "content": "",
                    "embedding": [0.0] * self.embedding_dim,
                    "metadata": "{}",
                }
            ]
        )
        table = self.db.create_table(self.TABLE_NAME, data=seed_df)
        table.delete("id = '__seed__'")
        self._table = table

    async def index_documents(
        self,
        documents: list[Document],
        session_id: Optional[str] = None,
    ) -> int:
        """Index documents into LanceDB."""
        if not self._llm_client:
            raise RuntimeError("LLM client not set")

        docs_needing_embeddings = [d for d in documents if d.embedding is None]
        if docs_needing_embeddings:
            contents = [d.content for d in docs_needing_embeddings]
            embeddings = await self._llm_client.embed(contents)
            for doc, emb in zip(docs_needing_embeddings, embeddings):
                doc.embedding = emb

        rows = []
        for doc in documents:
            if doc.embedding is None:
                continue
            rows.append(
                {
                    "id": doc.id,
                    "session_id": session_id or "",
                    "url": doc.url,
                    "title": doc.title,
                    "content": doc.content[:10000],
                    "embedding": doc.embedding,
                    "metadata": str(doc.metadata or {}),
                }
            )

        if rows:
            ids = [row["id"] for row in rows if row.get("id")]
            if ids:
                id_filter = " OR ".join(
                    f"id = '{doc_id.replace(chr(39), chr(39) * 2)}'" for doc_id in ids
                )
                try:
                    self.table.delete(id_filter)
                except Exception:
                    # Best-effort cleanup for upsert semantics.
                    pass

            self.table.add(pd.DataFrame(rows))

        return len(rows)

    async def search(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: int = 5,
    ) -> list[dict]:
        """Search for relevant documents using LanceDB vector search."""
        if not self._llm_client:
            raise RuntimeError("LLM client not set")

        query_embedding = await self._llm_client.embed_single(query)
        if not query_embedding:
            return []

        results = self.table.search(query_embedding)
        if session_id:
            escaped = session_id.replace("'", "''")
            results = results.where(f"session_id = '{escaped}'")

        df = results.limit(top_k).to_pandas()
        if df.empty:
            return []

        return [
            {
                "url": row.get("url", ""),
                "title": row.get("title", ""),
                "content": row.get("content", ""),
                # LanceDB returns distance (smaller is better). Convert to a bounded similarity-like score.
                "score": 1.0 / (1.0 + float(row.get("_distance", 0.0))),
            }
            for row in df.to_dict("records")
        ]

    async def chat(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: int = 5,
    ) -> dict:
        """Chat with indexed content using RAG."""
        if not self._llm_client:
            raise RuntimeError("LLM client not set")

        results = await self.search(query, session_id, top_k)
        if not results:
            return {
                "answer": "I don't have any relevant information to answer your question. Please make sure you've scraped some URLs first.",
                "sources": [],
                "context": "",
            }

        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[{i}] {result['title']}\n"
                f"URL: {result['url']}\n"
                f"Content: {result['content'][:1000]}...\n"
            )
        context = "\n".join(context_parts)

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

        escaped = session_id.replace("'", "''")
        df = self.table.search().where(f"session_id = '{escaped}'").limit(20).to_pandas()
        if df.empty:
            return "No content found for this session."

        summaries = []
        for row in df.to_dict("records"):
            title = row.get("title", "Untitled")
            content = row.get("content", "")[:500]
            summaries.append(f"- {title}: {content}...")

        prompt = f"""Summarize the following collection of web pages in 2-3 paragraphs:

{chr(10).join(summaries)}

Provide a cohesive summary that captures the main themes and topics."""

        return await self._llm_client.generate(prompt)

    def delete_session_documents(self, session_id: str) -> int:
        """Delete all documents for a session."""
        escaped = session_id.replace("'", "''")
        count = self.table.count_rows(filter=f"session_id = '{escaped}'")
        if count == 0:
            return 0

        self.table.delete(f"session_id = '{escaped}'")
        return count
