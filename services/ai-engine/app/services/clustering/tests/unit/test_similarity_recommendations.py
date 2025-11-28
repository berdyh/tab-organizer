"""Unit tests for similarity search and recommendation engine."""

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from services.clustering.app.similarity import SimilaritySearchEngine


class TestSimilaritySearchEngine:
    """Test cases for similarity search and recommendation functionality."""
    
    @pytest.fixture
    def search_engine(self):
        """Create SimilaritySearchEngine instance for testing."""
        return SimilaritySearchEngine()
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        # Create 5 embeddings with known similarities
        embeddings = [
            np.array([1.0, 0.0, 0.0]),  # Query will be similar to this
            np.array([0.9, 0.1, 0.0]),  # Very similar to query
            np.array([0.0, 1.0, 0.0]),  # Orthogonal to query
            np.array([0.0, 0.0, 1.0]),  # Orthogonal to query
            np.array([-1.0, 0.0, 0.0])  # Opposite to query
        ]
        return embeddings
    
    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata for testing."""
        return [
            {
                "point_id": "1",
                "title": "Similar Content 1",
                "url": "http://example.com/1",
                "cluster_id": 0,
                "quality_score": 0.9,
                "embedding_model": "test-model",
                "content_type": "article",
                "timestamp": time.time()
            },
            {
                "point_id": "2", 
                "title": "Similar Content 2",
                "url": "http://example.com/2",
                "cluster_id": 0,
                "quality_score": 0.8,
                "embedding_model": "test-model",
                "content_type": "article",
                "timestamp": time.time()
            },
            {
                "point_id": "3",
                "title": "Different Content 1",
                "url": "http://example.com/3", 
                "cluster_id": 1,
                "quality_score": 0.7,
                "embedding_model": "test-model",
                "content_type": "blog",
                "timestamp": time.time()
            },
            {
                "point_id": "4",
                "title": "Different Content 2",
                "url": "http://example.com/4",
                "cluster_id": 1,
                "quality_score": 0.6,
                "embedding_model": "test-model",
                "content_type": "blog",
                "timestamp": time.time()
            },
            {
                "point_id": "5",
                "title": "Opposite Content",
                "url": "http://example.com/5",
                "cluster_id": 2,
                "quality_score": 0.5,
                "embedding_model": "test-model",
                "content_type": "video",
                "timestamp": time.time()
            }
        ]
    
    @pytest.fixture
    def sample_user_interactions(self):
        """Create sample user interactions for testing."""
        current_time = time.time()
        return [
            {
                "point_id": "1",
                "type": "view",
                "timestamp": current_time - 3600  # 1 hour ago
            },
            {
                "point_id": "2",
                "type": "click",
                "timestamp": current_time - 1800  # 30 minutes ago
            },
            {
                "point_id": "1",
                "type": "like",
                "timestamp": current_time - 900   # 15 minutes ago
            }
        ]
    
    @pytest.mark.asyncio
    async def test_calculate_cosine_similarities(self, search_engine):
        """Test cosine similarity calculation."""
        query = np.array([1.0, 0.0, 0.0])
        embeddings = np.array([
            [1.0, 0.0, 0.0],  # Identical - similarity = 1.0
            [0.0, 1.0, 0.0],  # Orthogonal - similarity = 0.0
            [-1.0, 0.0, 0.0], # Opposite - similarity = -1.0
            [0.5, 0.5, 0.0]   # 45 degrees - similarity ≈ 0.707
        ])
        
        similarities = await search_engine._calculate_cosine_similarities(query, embeddings)
        
        assert len(similarities) == 4
        assert abs(similarities[0] - 1.0) < 1e-6  # Identical
        assert abs(similarities[1] - 0.0) < 1e-6  # Orthogonal
        assert abs(similarities[2] - (-1.0)) < 1e-6  # Opposite
        assert abs(similarities[3] - 0.707) < 0.01  # 45 degrees
    
    def test_calculate_interaction_weight(self, search_engine):
        """Test interaction weight calculation."""
        current_time = time.time()
        
        # Test different interaction types
        view_weight = search_engine._calculate_interaction_weight("view", current_time)
        click_weight = search_engine._calculate_interaction_weight("click", current_time)
        like_weight = search_engine._calculate_interaction_weight("like", current_time)
        
        assert click_weight > view_weight
        assert like_weight > click_weight
        
        # Test recency decay
        recent_weight = search_engine._calculate_interaction_weight("view", current_time)
        old_weight = search_engine._calculate_interaction_weight("view", current_time - 86400 * 7)  # 1 week ago
        
        assert recent_weight > old_weight
    
    def test_calculate_diversity_score(self, search_engine):
        """Test diversity score calculation."""
        candidate = {"cluster_id": 1}
        
        # Empty selection - should return 1.0
        diversity_score = search_engine._calculate_diversity_score(candidate, [])
        assert diversity_score == 1.0
        
        # Different cluster - should return 1.0
        selected = [{"cluster_id": 0}]
        diversity_score = search_engine._calculate_diversity_score(candidate, selected)
        assert diversity_score == 1.0
        
        # Same cluster - should return lower score
        selected = [{"cluster_id": 1}]
        diversity_score = search_engine._calculate_diversity_score(candidate, selected)
        assert diversity_score == 0.5
        
        # Multiple same cluster items - should return even lower score
        selected = [{"cluster_id": 1}, {"cluster_id": 1}]
        diversity_score = search_engine._calculate_diversity_score(candidate, selected)
        assert diversity_score == 1.0 / 3.0
    
    def test_calculate_recommendation_score(self, search_engine):
        """Test recommendation score calculation."""
        user_profile = {
            "preferred_clusters": [0, 1],
            "content_types": ["article", "blog"]
        }
        
        # Item with preferred cluster and content type
        item1 = {
            "similarity_score": 0.8,
            "cluster_id": 0,
            "content_type": "article",
            "quality_score": 0.9
        }
        
        score1 = search_engine._calculate_recommendation_score(item1, user_profile)
        
        # Item without preferences
        item2 = {
            "similarity_score": 0.8,
            "cluster_id": 2,
            "content_type": "video",
            "quality_score": 0.5
        }
        
        score2 = search_engine._calculate_recommendation_score(item2, user_profile)
        
        # Item with preferences should have higher score
        assert score1 > score2
        assert score1 <= 1.0  # Should be capped at 1.0
    
    def test_generate_recommendation_reason(self, search_engine):
        """Test recommendation reason generation."""
        user_profile = {
            "preferred_clusters": [0],
            "content_types": ["article"]
        }
        
        item = {
            "similarity_score": 0.9,
            "cluster_id": 0,
            "content_type": "article",
            "quality_score": 0.9
        }
        
        reason = search_engine._generate_recommendation_reason(item, user_profile)
        
        assert isinstance(reason, str)
        assert len(reason) > 0
        assert "highly similar" in reason or "similar" in reason
        assert "preferred topic" in reason or "preference for article" in reason
    
    @pytest.mark.asyncio
    async def test_apply_diversity_filtering(self, search_engine):
        """Test diversity filtering of recommendations."""
        candidates = [
            {"similarity_score": 0.9, "cluster_id": 0},
            {"similarity_score": 0.8, "cluster_id": 0},  # Same cluster as first
            {"similarity_score": 0.7, "cluster_id": 1},  # Different cluster
            {"similarity_score": 0.6, "cluster_id": 2},  # Different cluster
            {"similarity_score": 0.5, "cluster_id": 0}   # Same cluster as first
        ]
        
        # Test with diversity factor
        diverse_results = await search_engine._apply_diversity_filtering(
            candidates, top_k=3, diversity_factor=0.5
        )
        
        assert len(diverse_results) == 3
        
        # Should include items from different clusters
        cluster_ids = [item["cluster_id"] for item in diverse_results]
        unique_clusters = set(cluster_ids)
        assert len(unique_clusters) > 1  # Should have diversity
        
        # Test without diversity (diversity_factor = 0)
        no_diversity_results = await search_engine._apply_diversity_filtering(
            candidates, top_k=3, diversity_factor=0.0
        )
        
        assert len(no_diversity_results) == 3
        # Should just return top 3 by similarity
        assert no_diversity_results == candidates[:3]


class TestSimilaritySearchIntegration:
    """Integration tests for similarity search functionality."""
    
    @pytest.fixture
    def search_engine(self):
        return SimilaritySearchEngine()
    
    @pytest.mark.asyncio
    async def test_vector_similarity_search_mock(self, search_engine):
        """Test vector similarity search with mocked Qdrant."""
        
        # Mock Qdrant client response
        mock_points = [
            Mock(
                id="1",
                vector=[1.0, 0.0, 0.0],
                payload={
                    "title": "Test Content 1",
                    "url": "http://example.com/1",
                    "cluster_id": 0,
                    "quality_score": 0.9,
                    "embedding_model": "test-model",
                    "content_type": "article",
                    "timestamp": time.time()
                }
            ),
            Mock(
                id="2",
                vector=[0.9, 0.1, 0.0],
                payload={
                    "title": "Test Content 2", 
                    "url": "http://example.com/2",
                    "cluster_id": 0,
                    "quality_score": 0.8,
                    "embedding_model": "test-model",
                    "content_type": "article",
                    "timestamp": time.time()
                }
            )
        ]
        
        with patch('services.clustering.app.similarity.qdrant_client') as mock_qdrant:
            mock_qdrant.scroll.return_value = (mock_points, None)
            
            query_embedding = np.array([1.0, 0.0, 0.0])
            
            results, metadata = await search_engine.vector_similarity_search(
                session_id="test_session",
                query_embedding=query_embedding,
                top_k=2,
                similarity_threshold=0.5
            )
            
            assert len(results) == 2
            assert results[0]["similarity_score"] > results[1]["similarity_score"]
            assert "processing_time_seconds" in metadata
            assert metadata["results_returned"] == 2
    
    @pytest.mark.asyncio
    async def test_build_user_profile_mock(self, search_engine):
        """Test user profile building with mocked Qdrant."""
        
        user_interactions = [
            {
                "point_id": "1",
                "type": "view",
                "timestamp": time.time() - 3600
            },
            {
                "point_id": "2", 
                "type": "like",
                "timestamp": time.time() - 1800
            }
        ]
        
        # Mock Qdrant retrieve responses
        mock_points = [
            Mock(
                vector=[1.0, 0.0, 0.0],
                payload={
                    "cluster_id": 0,
                    "content_type": "article"
                }
            ),
            Mock(
                vector=[0.8, 0.2, 0.0],
                payload={
                    "cluster_id": 0,
                    "content_type": "article"
                }
            )
        ]
        
        with patch('services.clustering.app.similarity.qdrant_client') as mock_qdrant:
            mock_qdrant.retrieve.side_effect = [[mock_points[0]], [mock_points[1]]]
            
            user_profile = await search_engine._build_user_profile(
                "test_session", user_interactions
            )
            
            assert user_profile is not None
            assert "profile_embedding" in user_profile
            assert "preferred_clusters" in user_profile
            assert "content_types" in user_profile
            assert user_profile["interaction_count"] == 2
            assert 0 in user_profile["preferred_clusters"]
            assert "article" in user_profile["content_types"]


class TestRecommendationEngine:
    """Test cases for recommendation engine functionality."""
    
    @pytest.fixture
    def search_engine(self):
        return SimilaritySearchEngine()
    
    @pytest.mark.asyncio
    async def test_content_based_recommendations_mock(self, search_engine):
        """Test content-based recommendations with mocked dependencies."""
        
        user_interactions = [
            {
                "point_id": "1",
                "type": "view",
                "timestamp": time.time() - 3600
            }
        ]
        
        # Mock user profile building
        mock_profile = {
            "profile_embedding": np.array([1.0, 0.0, 0.0]),
            "preferred_clusters": [0],
            "content_types": ["article"],
            "interaction_count": 1,
            "total_weight": 1.0
        }
        
        # Mock similarity search results
        mock_similar_items = [
            {
                "point_id": "2",
                "title": "Similar Article",
                "similarity_score": 0.9,
                "cluster_id": 0,
                "content_type": "article",
                "quality_score": 0.8
            },
            {
                "point_id": "3",
                "title": "Another Article",
                "similarity_score": 0.7,
                "cluster_id": 1,
                "content_type": "blog",
                "quality_score": 0.6
            }
        ]
        
        with patch.object(search_engine, '_build_user_profile', return_value=mock_profile):
            with patch.object(search_engine, 'vector_similarity_search', 
                            return_value=(mock_similar_items, {"test": "metadata"})):
                
                recommendations, metadata = await search_engine.content_based_recommendations(
                    session_id="test_session",
                    user_interactions=user_interactions,
                    top_k=2,
                    diversity_factor=0.3
                )
                
                assert len(recommendations) <= 2
                assert all("recommendation_score" in rec for rec in recommendations)
                assert all("recommendation_reason" in rec for rec in recommendations)
                assert "processing_time_seconds" in metadata
                assert "user_profile_summary" in metadata
    
    @pytest.mark.asyncio
    async def test_content_based_recommendations_empty_interactions(self, search_engine):
        """Test content-based recommendations with empty interactions."""
        
        recommendations, metadata = await search_engine.content_based_recommendations(
            session_id="test_session",
            user_interactions=[],
            top_k=10,
            diversity_factor=0.3
        )
        
        assert len(recommendations) == 0
        assert metadata["message"] == "No user interactions provided"
    
    @pytest.mark.asyncio
    async def test_collaborative_filtering_fallback(self, search_engine):
        """Test collaborative filtering fallback to content-based."""
        
        user_interactions = [
            {
                "point_id": "1",
                "type": "view", 
                "timestamp": time.time()
            }
        ]
        
        with patch.object(search_engine, 'content_based_recommendations', 
                         return_value=([], {"test": "metadata"})) as mock_content:
            
            recommendations, metadata = await search_engine.collaborative_filtering_recommendations(
                session_id="test_session",
                user_interactions=user_interactions,
                top_k=10
            )
            
            mock_content.assert_called_once()
            assert metadata["recommendation_type"] == "collaborative_filtering_fallback"
            assert "note" in metadata


class TestPerformanceAndEdgeCases:
    """Test performance characteristics and edge cases."""
    
    @pytest.fixture
    def search_engine(self):
        return SimilaritySearchEngine()
    
    @pytest.mark.asyncio
    async def test_large_embedding_similarity_calculation(self, search_engine):
        """Test similarity calculation with large embeddings."""
        
        # Test with high-dimensional embeddings
        query = np.random.randn(1024)
        embeddings = np.random.randn(100, 1024)
        
        start_time = time.time()
        similarities = await search_engine._calculate_cosine_similarities(query, embeddings)
        processing_time = time.time() - start_time
        
        assert len(similarities) == 100
        assert all(-1 <= sim <= 1 for sim in similarities)  # Valid cosine similarity range
        assert processing_time < 1.0  # Should be reasonably fast
    
    def test_interaction_weight_edge_cases(self, search_engine):
        """Test interaction weight calculation edge cases."""
        
        current_time = time.time()
        
        # Test unknown interaction type
        unknown_weight = search_engine._calculate_interaction_weight("unknown_type", current_time)
        assert unknown_weight == 1.0  # Should default to 1.0
        
        # Test very old interaction
        very_old_weight = search_engine._calculate_interaction_weight("view", current_time - 86400 * 365)
        assert very_old_weight > 0  # Should still have some weight
        assert very_old_weight < 1.0  # But less than recent
    
    @pytest.mark.asyncio
    async def test_diversity_filtering_edge_cases(self, search_engine):
        """Test diversity filtering edge cases."""
        
        # Test with fewer candidates than requested
        candidates = [
            {"similarity_score": 0.9, "cluster_id": 0},
            {"similarity_score": 0.8, "cluster_id": 1}
        ]
        
        results = await search_engine._apply_diversity_filtering(
            candidates, top_k=5, diversity_factor=0.5
        )
        
        assert len(results) == 2  # Should return all available candidates
        
        # Test with zero diversity factor
        results_no_diversity = await search_engine._apply_diversity_filtering(
            candidates, top_k=2, diversity_factor=0.0
        )
        
        assert results_no_diversity == candidates  # Should return in original order
    
    def test_recommendation_score_bounds(self, search_engine):
        """Test that recommendation scores are properly bounded."""
        
        user_profile = {
            "preferred_clusters": [0],
            "content_types": ["article"]
        }
        
        # Test with very high similarity and quality scores
        item = {
            "similarity_score": 1.0,
            "cluster_id": 0,
            "content_type": "article", 
            "quality_score": 1.0
        }
        
        score = search_engine._calculate_recommendation_score(item, user_profile)
        assert score <= 1.0  # Should be capped at 1.0
        assert score >= 0.0  # Should not be negative


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
