"""
Vector similarity search and recommendation engine utilities.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .clients.qdrant import qdrant_client
from .config import QDRANT_SCROLL_LIMIT
from .executor import executor
from .logging import logger


class SimilaritySearchEngine:
    """Vector similarity search and recommendation engine."""

    def __init__(self) -> None:
        self.user_interaction_history: Dict[str, List[Dict[str, Any]]] = {}
        self.content_profiles: Dict[str, Dict[str, Any]] = {}

    async def vector_similarity_search(
        self,
        session_id: str,
        query_embedding: np.ndarray,
        top_k: int = 10,
        similarity_threshold: float = 0.7,
        filter_cluster_id: Optional[int] = None,
        use_reduced_embeddings: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Perform vector similarity search using embeddings.

        Returns:
            Tuple of (search_results, search_metadata).
        """
        start_time = time.time()

        logger.info(
            "Starting vector similarity search",
            session_id=session_id,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            filter_cluster_id=filter_cluster_id,
            use_reduced_embeddings=use_reduced_embeddings,
        )

        # Load embeddings from Qdrant
        collection_name = f"session_{session_id}"

        # Retrieve all points with embeddings
        search_result = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: qdrant_client.scroll(
                collection_name=collection_name,
                limit=QDRANT_SCROLL_LIMIT,
                with_payload=True,
                with_vectors=True,
            ),
        )

        points = search_result[0]

        if not points:
            return [], {"message": "No embeddings found in session"}

        # Extract embeddings and metadata
        embeddings: List[List[float]] = []
        metadata: List[Dict[str, Any]] = []

        for point in points:
            # Apply cluster filter if specified
            if filter_cluster_id is not None:
                point_cluster_id = point.payload.get("cluster_id", -1)
                if point_cluster_id != filter_cluster_id:
                    continue

            # Use appropriate embeddings
            if use_reduced_embeddings and "reduced_embedding" in point.payload:
                embeddings.append(point.payload["reduced_embedding"])
            elif not use_reduced_embeddings and point.vector is not None:
                embeddings.append(point.vector)
            else:
                continue

            metadata.append(
                {
                    "point_id": point.id,
                    "title": point.payload.get("title", ""),
                    "url": point.payload.get("url", ""),
                    "cluster_id": point.payload.get("cluster_id", -1),
                    "quality_score": point.payload.get("quality_score", 0.0),
                    "embedding_model": point.payload.get("embedding_model", "unknown"),
                    "content_type": point.payload.get("content_type", "unknown"),
                    "timestamp": point.payload.get("timestamp", 0),
                }
            )

        if not embeddings:
            return [], {"message": "No suitable embeddings found for search"}

        embeddings_array = np.array(embeddings)

        # Calculate cosine similarities
        similarities = await self._calculate_cosine_similarities(query_embedding, embeddings_array)

        # Filter by similarity threshold and get top-k
        valid_indices = np.where(similarities >= similarity_threshold)[0]

        if len(valid_indices) == 0:
            return [], {
                "message": "No results above similarity threshold",
                "max_similarity": float(np.max(similarities)),
                "threshold": similarity_threshold,
            }

        # Sort by similarity and take top-k
        valid_similarities = similarities[valid_indices]
        sorted_indices = np.argsort(valid_similarities)[::-1][:top_k]

        # Prepare results
        results: List[Dict[str, Any]] = []
        for idx in sorted_indices:
            original_idx = valid_indices[idx]
            result = metadata[original_idx].copy()
            result["similarity_score"] = float(valid_similarities[idx])
            results.append(result)

        # Calculate search metadata
        processing_time = time.time() - start_time
        search_metadata = {
            "processing_time_seconds": processing_time,
            "total_candidates": len(embeddings),
            "results_returned": len(results),
            "similarity_threshold": similarity_threshold,
            "average_similarity": float(np.mean([r["similarity_score"] for r in results])) if results else 0.0,
            "max_similarity": float(np.max([r["similarity_score"] for r in results])) if results else 0.0,
            "min_similarity": float(np.min([r["similarity_score"] for r in results])) if results else 0.0,
            "filter_cluster_id": filter_cluster_id,
            "use_reduced_embeddings": use_reduced_embeddings,
        }

        logger.info(
            "Vector similarity search completed",
            session_id=session_id,
            results_count=len(results),
            processing_time=processing_time,
        )

        return results, search_metadata

    async def _calculate_cosine_similarities(
        self,
        query_embedding: np.ndarray,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """Calculate cosine similarities between query and all embeddings."""

        def compute_similarities() -> np.ndarray:
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            similarities_local = np.dot(embeddings_norm, query_norm)
            return similarities_local

        similarities = await asyncio.get_event_loop().run_in_executor(executor, compute_similarities)
        return similarities

    async def content_based_recommendations(
        self,
        session_id: str,
        user_interactions: List[Dict[str, Any]],
        top_k: int = 10,
        diversity_factor: float = 0.3,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Generate content-based recommendations based on user interactions.
        """
        start_time = time.time()

        logger.info(
            "Generating content-based recommendations",
            session_id=session_id,
            interactions_count=len(user_interactions),
            top_k=top_k,
            diversity_factor=diversity_factor,
        )

        if not user_interactions:
            return [], {"message": "No user interactions provided"}

        # Build user profile from interactions
        user_profile = await self._build_user_profile(session_id, user_interactions)

        if user_profile is None:
            return [], {"message": "Could not build user profile"}

        # Find similar content
        similar_items, search_metadata = await self.vector_similarity_search(
            session_id=session_id,
            query_embedding=user_profile["profile_embedding"],
            top_k=top_k * 3,
            similarity_threshold=0.1,
            use_reduced_embeddings=False,
        )

        if not similar_items:
            return [], {"message": "No similar content found"}

        # Apply diversity filtering
        diverse_recommendations = await self._apply_diversity_filtering(similar_items, top_k, diversity_factor)

        # Add recommendation scores and reasons
        recommendations: List[Dict[str, Any]] = []
        for item in diverse_recommendations:
            recommendation = item.copy()
            recommendation["recommendation_score"] = self._calculate_recommendation_score(item, user_profile)
            recommendation["recommendation_reason"] = self._generate_recommendation_reason(item, user_profile)
            recommendations.append(recommendation)

        # Calculate recommendation metadata
        processing_time = time.time() - start_time
        recommendation_metadata = {
            "processing_time_seconds": processing_time,
            "user_profile_summary": {
                "preferred_clusters": user_profile.get("preferred_clusters", []),
                "interaction_count": len(user_interactions),
                "content_types": user_profile.get("content_types", []),
            },
            "diversity_factor": diversity_factor,
            "candidates_considered": len(similar_items),
            "recommendations_returned": len(recommendations),
            "average_recommendation_score": float(
                np.mean([r["recommendation_score"] for r in recommendations])
            )
            if recommendations
            else 0.0,
            "search_metadata": search_metadata,
        }

        logger.info(
            "Content-based recommendations completed",
            session_id=session_id,
            recommendations_count=len(recommendations),
            processing_time=processing_time,
        )

        return recommendations, recommendation_metadata

    async def _build_user_profile(
        self,
        session_id: str,
        user_interactions: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Build user profile from interaction history."""
        self.user_interaction_history.setdefault(session_id, []).extend(user_interactions)

        weights_used: List[float] = []
        interacted_embeddings: List[List[float]] = []
        cluster_preferences: Dict[int, float] = {}
        content_type_preferences: Dict[str, float] = {}

        for interaction in user_interactions:
            weight = self._calculate_interaction_weight(
                interaction.get("type", "view"),
                interaction.get("timestamp", 0.0),
            )
            cluster_id = interaction.get("cluster_id", -1)
            content_type = interaction.get("content_type", "unknown")
            embedding = interaction.get("embedding")

            if embedding is None:
                point_id = interaction.get("point_id")
                if point_id is not None:
                    collection_name = f"session_{session_id}"
                    try:
                        retrieved = await asyncio.get_event_loop().run_in_executor(
                            executor,
                            lambda pid=point_id: qdrant_client.retrieve(
                                collection_name=collection_name,
                                ids=[pid],
                                with_vectors=True,
                                with_payload=True,
                            ),
                        )
                    except Exception as error:  # pragma: no cover - network failure
                        logger.warning(
                            "Failed to retrieve interaction embedding from Qdrant",
                            session_id=session_id,
                            point_id=point_id,
                            error=str(error),
                        )
                        retrieved = None

                    if retrieved:
                        point = retrieved[0]
                        embedding = point.vector or point.payload.get("reduced_embedding")
                        cluster_id = point.payload.get("cluster_id", cluster_id)
                        content_type = point.payload.get("content_type", content_type)

            if embedding is None:
                continue

            interacted_embeddings.append(embedding)
            weights_used.append(weight)

            cluster_preferences[cluster_id] = cluster_preferences.get(cluster_id, 0.0) + weight
            content_type_preferences[content_type] = content_type_preferences.get(content_type, 0.0) + weight

        if not interacted_embeddings:
            return None

        profile_embedding = np.average(interacted_embeddings, axis=0, weights=weights_used)

        preferred_clusters = sorted(cluster_preferences.items(), key=lambda item: item[1], reverse=True)[:3]
        preferred_content_types = sorted(
            content_type_preferences.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:3]

        user_profile = {
            "profile_embedding": profile_embedding,
            "preferred_clusters": [cluster_id for cluster_id, _weight in preferred_clusters],
            "content_types": [content_type for content_type, _weight in preferred_content_types],
            "interaction_count": len(user_interactions),
            "total_weight": float(np.sum(weights_used)),
        }

        self.content_profiles[session_id] = user_profile
        return user_profile

    def _calculate_interaction_weight(self, interaction_type: str, timestamp: float) -> float:
        """Calculate weight for user interaction based on type and recency."""
        type_weights = {
            "view": 1.0,
            "click": 2.0,
            "like": 3.0,
            "share": 4.0,
            "bookmark": 5.0,
            "download": 3.0,
        }

        base_weight = type_weights.get(interaction_type, 1.0)
        current_time = time.time()
        hours_ago = (current_time - timestamp) / 3600 if timestamp else float("inf")

        if hours_ago <= 24:
            recency_factor = 1.0
        elif hours_ago <= 168:
            recency_factor = 0.8
        elif hours_ago <= 720:
            recency_factor = 0.6
        else:
            recency_factor = 0.4

        return base_weight * recency_factor

    async def _apply_diversity_filtering(
        self,
        candidates: List[Dict[str, Any]],
        top_k: int,
        diversity_factor: float,
    ) -> List[Dict[str, Any]]:
        """Apply diversity filtering to recommendation candidates."""
        if diversity_factor <= 0 or len(candidates) <= top_k:
            return candidates[:top_k]

        selected: List[Dict[str, Any]] = []
        remaining = candidates.copy()

        if remaining:
            selected.append(remaining.pop(0))

        while len(selected) < top_k and remaining:
            best_candidate = None
            best_score = -1.0
            best_idx = -1

            for idx, candidate in enumerate(remaining):
                diversity_score = self._calculate_diversity_score(candidate, selected)
                relevance_score = candidate["similarity_score"]
                combined_score = (1 - diversity_factor) * relevance_score + diversity_factor * diversity_score

                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = candidate
                    best_idx = idx

            if best_candidate is not None and best_idx >= 0:
                selected.append(best_candidate)
                remaining.pop(best_idx)

        return selected

    def _calculate_diversity_score(
        self,
        candidate: Dict[str, Any],
        selected: List[Dict[str, Any]],
    ) -> float:
        """Calculate diversity score for a candidate relative to selected items."""
        if not selected:
            return 1.0

        candidate_cluster = candidate.get("cluster_id", -1)
        selected_clusters = [item.get("cluster_id", -1) for item in selected]

        if candidate_cluster not in selected_clusters:
            return 1.0

        same_cluster_count = selected_clusters.count(candidate_cluster)
        return 1.0 / (1 + same_cluster_count)

    def _calculate_recommendation_score(
        self,
        item: Dict[str, Any],
        user_profile: Dict[str, Any],
    ) -> float:
        """Calculate final recommendation score for an item."""
        base_score = item.get("similarity_score", 0.0)

        cluster_id = item.get("cluster_id", -1)
        cluster_boost = 0.1 if cluster_id in user_profile.get("preferred_clusters", []) else 0.0

        content_type = item.get("content_type", "unknown")
        content_type_boost = 0.05 if content_type in user_profile.get("content_types", []) else 0.0

        quality_boost = item.get("quality_score", 0.0) * 0.1

        final_score = base_score + cluster_boost + content_type_boost + quality_boost
        return min(final_score, 1.0)

    def _generate_recommendation_reason(
        self,
        item: Dict[str, Any],
        user_profile: Dict[str, Any],
    ) -> str:
        """Generate human-readable reason for recommendation."""
        reasons: List[str] = []

        similarity = item.get("similarity_score", 0.0)
        if similarity > 0.8:
            reasons.append("highly similar to your interests")
        elif similarity > 0.6:
            reasons.append("similar to your interests")
        else:
            reasons.append("related to your interests")

        cluster_id = item.get("cluster_id", -1)
        if cluster_id in user_profile.get("preferred_clusters", []):
            reasons.append("from your preferred topic area")

        content_type = item.get("content_type", "unknown")
        if content_type in user_profile.get("content_types", []):
            reasons.append(f"matches your preference for {content_type} content")

        quality_score = item.get("quality_score", 0.0)
        if quality_score > 0.8:
            reasons.append("high quality content")

        if not reasons:
            return "recommended based on your activity"

        return "Recommended because it's " + " and ".join(reasons)

    async def collaborative_filtering_recommendations(
        self,
        session_id: str,
        user_interactions: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Generate collaborative filtering recommendations.

        Currently falls back to content-based recommendations with a note.
        """
        recommendations, metadata = await self.content_based_recommendations(
            session_id, user_interactions, top_k, diversity_factor=0.2
        )

        metadata["recommendation_type"] = "collaborative_filtering_fallback"
        metadata["note"] = "Using content-based filtering as collaborative data is limited"

        return recommendations, metadata


__all__ = ["SimilaritySearchEngine"]
