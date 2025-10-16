"""Background tasks executed by the analyzer service."""

from __future__ import annotations

import re
import time
from typing import List

import structlog

from .schemas import ContentItem
from .state import AnalyzerState

logger = structlog.get_logger("analyzer_tasks")


async def process_embeddings_job(
    state: AnalyzerState,
    job_id: str,
    content_items: List[ContentItem],
    session_id: str,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Background task to process embeddings."""
    generator = state.embedding_generator
    chunker = state.text_chunker
    qdrant = state.qdrant_manager
    monitor = state.performance_monitor

    if not all([generator, chunker, qdrant]):
        raise RuntimeError("Embedding job requested before analyzer components were initialized")

    logger.info("Starting embedding job", job_id=job_id, items_count=len(content_items))

    try:
        collection_name = f"session_{session_id}"
        model_info = generator.get_current_model_info()
        dimensions = model_info.get("dimensions", 384)

        await qdrant.ensure_collection_exists(collection_name, dimensions)

        processed_items = []

        for item in content_items:
            chunks = chunker.chunk_text(item.content, chunk_size=chunk_size, overlap=chunk_overlap)

            chunk_texts = [chunk["text"] for chunk in chunks]
            start_time = time.time()
            embeddings = generator.generate_embeddings(chunk_texts)
            embedding_time = time.time() - start_time

            if monitor:
                monitor.record_model_performance(
                    model_id=generator.current_model_id,  # type: ignore[arg-type]
                    model_type="embedding",
                    success=True,
                    response_time=embedding_time,
                    resource_usage={"chunks_processed": len(chunks)},
                )

            for chunk, embedding in zip(chunks, embeddings):
                processed_items.append(
                    {
                        "content_id": item.id,
                        "chunk_index": chunk["chunk_index"],
                        "text": chunk["text"],
                        "title": item.title,
                        "url": item.url,
                        "token_count": chunk["token_count"],
                        "embedding": embedding.tolist(),
                        "embedding_generation_time": embedding_time / max(len(chunks), 1),
                        "original_metadata": item.metadata,
                    }
                )

        if processed_items:
            points_stored = await qdrant.store_analyzed_content(
                collection_name=collection_name,
                content_items=processed_items,
                embedding_model=generator.current_model_id,  # type: ignore[arg-type]
            )

            logger.info("Embedding job completed", job_id=job_id, points_created=points_stored)

    except Exception as exc:
        logger.error("Embedding job failed", job_id=job_id, error=str(exc))
        if monitor:
            monitor.record_model_performance(
                model_id=generator.current_model_id if generator else "unknown",
                model_type="embedding",
                success=False,
                response_time=0.0,
            )


async def process_complete_analysis_job(
    state: AnalyzerState,
    job_id: str,
    content_items: List[ContentItem],
    session_id: str,
    llm_model: str,
    generate_summary: bool,
    extract_keywords: bool,
    assess_quality: bool,
) -> None:
    """Background task to process complete content analysis."""
    generator = state.embedding_generator
    chunker = state.text_chunker
    qdrant = state.qdrant_manager
    ollama = state.ollama_client
    monitor = state.performance_monitor

    if not all([generator, chunker, qdrant, ollama]):
        raise RuntimeError("Complete analysis requested before analyzer components were initialized")

    logger.info(
        "Starting complete analysis job",
        job_id=job_id,
        items_count=len(content_items),
        llm_model=llm_model,
    )

    try:
        collection_name = f"session_{session_id}"
        model_info = generator.get_current_model_info()
        dimensions = model_info.get("dimensions", 384)

        await qdrant.ensure_collection_exists(collection_name, dimensions)

        processed_items = []

        for item in content_items:
            logger.info("Processing content item", content_id=item.id)
            chunks = chunker.chunk_text(item.content)

            chunk_texts = [chunk["text"] for chunk in chunks]
            embedding_start = time.time()
            embeddings = generator.generate_embeddings(chunk_texts)
            embedding_time = time.time() - embedding_start

            if monitor:
                monitor.record_model_performance(
                    model_id=generator.current_model_id,  # type: ignore[arg-type]
                    model_type="embedding",
                    success=True,
                    response_time=embedding_time,
                    resource_usage={"chunks_processed": len(chunks)},
                )

            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_analysis = {
                    "content_id": item.id,
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                    "title": item.title,
                    "url": item.url,
                    "token_count": chunk["token_count"],
                    "embedding": embedding.tolist(),
                    "embedding_generation_time": embedding_time / max(len(chunks), 1),
                    "original_metadata": item.metadata,
                }

                llm_start_time = time.time()
                llm_success = True

                try:
                    if generate_summary:
                        summary_result = await ollama.summarize_content(chunk["text"], llm_model)
                        if summary_result["success"]:
                            chunk_analysis["summary"] = summary_result["response"]
                        else:
                            logger.warning(
                                "Summary generation failed",
                                content_id=item.id,
                                chunk_index=idx,
                                error=summary_result.get("error"),
                            )

                    if extract_keywords:
                        keywords_result = await ollama.extract_keywords(chunk["text"], llm_model)
                        if keywords_result["success"]:
                            chunk_analysis["keywords"] = keywords_result["response"]
                        else:
                            logger.warning(
                                "Keyword extraction failed",
                                content_id=item.id,
                                chunk_index=idx,
                                error=keywords_result.get("error"),
                            )

                    if assess_quality:
                        quality_result = await ollama.assess_content_quality(chunk["text"], llm_model)
                        if quality_result["success"]:
                            chunk_analysis["quality_assessment"] = quality_result["response"]
                            try:
                                score_match = re.search(r"(\d+(?:\.\d+)?)/10", quality_result["response"])
                                if score_match:
                                    chunk_analysis["quality_score"] = float(score_match.group(1))
                            except Exception:
                                pass
                        else:
                            logger.warning(
                                "Quality assessment failed",
                                content_id=item.id,
                                chunk_index=idx,
                                error=quality_result.get("error"),
                            )

                except Exception as exc:
                    logger.error(
                        "LLM analysis failed",
                        content_id=item.id,
                        chunk_index=idx,
                        error=str(exc),
                    )
                    llm_success = False

                llm_time = time.time() - llm_start_time
                chunk_analysis["llm_processing_time"] = llm_time

                if monitor:
                    monitor.record_model_performance(
                        model_id=llm_model,
                        model_type="llm",
                        success=llm_success,
                        response_time=llm_time,
                        resource_usage={"token_count": chunk["token_count"]},
                    )

                processed_items.append(chunk_analysis)

        if processed_items:
            points_stored = await qdrant.store_analyzed_content(
                collection_name=collection_name,
                content_items=processed_items,
                embedding_model=generator.current_model_id,  # type: ignore[arg-type]
                llm_model=llm_model,
            )

            logger.info(
                "Complete analysis job completed",
                job_id=job_id,
                points_created=points_stored,
                embedding_model=generator.current_model_id,
                llm_model=llm_model,
            )

        if monitor:
            monitor.record_system_resources()

    except Exception as exc:
        logger.error("Complete analysis job failed", job_id=job_id, error=str(exc))
        if monitor:
            monitor.record_model_performance(
                model_id=llm_model,
                model_type="llm",
                success=False,
                response_time=0.0,
            )


__all__ = ["process_complete_analysis_job", "process_embeddings_job"]
