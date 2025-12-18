"""Unit tests for Content Deduplication."""

import pytest
import numpy as np
import sys
sys.path.insert(0, '/app')

from services.backend_core.app.url_input.dedup import ContentDeduplicator, URLDeduplicator


class TestContentDeduplicator:
    """Tests for ContentDeduplicator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.dedup = ContentDeduplicator(similarity_threshold=0.95)
    
    def test_compute_content_hash(self):
        """Test content hash computation."""
        content1 = "Hello World"
        content2 = "Hello World"
        content3 = "Different content"
        
        hash1 = self.dedup.compute_content_hash(content1)
        hash2 = self.dedup.compute_content_hash(content2)
        hash3 = self.dedup.compute_content_hash(content3)
        
        assert hash1 == hash2
        assert hash1 != hash3
    
    def test_compute_content_hash_whitespace_normalized(self):
        """Test that whitespace is normalized in hash."""
        content1 = "Hello   World"
        content2 = "Hello World"
        
        hash1 = self.dedup.compute_content_hash(content1)
        hash2 = self.dedup.compute_content_hash(content2)
        
        assert hash1 == hash2
    
    def test_compute_simhash(self):
        """Test SimHash computation."""
        content1 = "The quick brown fox jumps over the lazy dog"
        content2 = "The quick brown fox jumps over the lazy cat"
        content3 = "Completely different text about programming"
        
        hash1 = self.dedup.compute_simhash(content1)
        hash2 = self.dedup.compute_simhash(content2)
        hash3 = self.dedup.compute_simhash(content3)
        
        # Similar content should have small Hamming distance
        dist_similar = self.dedup.hamming_distance(hash1, hash2)
        dist_different = self.dedup.hamming_distance(hash1, hash3)
        
        assert dist_similar < dist_different
    
    def test_is_near_duplicate_simhash(self):
        """Test near-duplicate detection with SimHash."""
        content1 = "The quick brown fox jumps over the lazy dog"
        content2 = "The quick brown fox jumps over the lazy cat"
        
        hash1 = self.dedup.compute_simhash(content1)
        hash2 = self.dedup.compute_simhash(content2)
        
        # Should be near-duplicates
        assert self.dedup.is_near_duplicate_simhash(hash1, hash2, threshold=10)
    
    def test_check_exact_duplicate(self):
        """Test exact duplicate detection."""
        content = "This is some content"
        
        # First check - not a duplicate
        result1 = self.dedup.check_exact_duplicate(content, "url1")
        assert result1 is None
        
        # Second check - is a duplicate
        result2 = self.dedup.check_exact_duplicate(content, "url2")
        assert result2 == "url1"
    
    def test_cosine_similarity(self):
        """Test cosine similarity computation."""
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([1, 0, 0])
        vec3 = np.array([0, 1, 0])
        
        sim_same = self.dedup.cosine_similarity(vec1, vec2)
        sim_orthogonal = self.dedup.cosine_similarity(vec1, vec3)
        
        assert sim_same == pytest.approx(1.0)
        assert sim_orthogonal == pytest.approx(0.0)
    
    def test_check_near_duplicate_embedding(self):
        """Test near-duplicate detection with embeddings."""
        emb1 = np.array([1, 0, 0, 0])
        emb2 = np.array([0.99, 0.01, 0, 0])  # Very similar
        emb3 = np.array([0, 1, 0, 0])  # Different
        
        # Add first embedding
        result1 = self.dedup.check_near_duplicate_embedding(emb1, "url1")
        assert result1 is None
        
        # Check similar embedding
        self.dedup._embeddings.clear()
        self.dedup.add_embedding("url1", emb1)
        result2 = self.dedup.check_near_duplicate_embedding(emb2, "url2")
        assert result2 is not None
        assert result2[0] == "url1"
        
        # Check different embedding
        self.dedup._embeddings.clear()
        self.dedup.add_embedding("url1", emb1)
        result3 = self.dedup.check_near_duplicate_embedding(emb3, "url3")
        assert result3 is None
    
    def test_find_similar(self):
        """Test finding similar embeddings."""
        self.dedup.add_embedding("url1", np.array([1, 0, 0]))
        self.dedup.add_embedding("url2", np.array([0.9, 0.1, 0]))
        self.dedup.add_embedding("url3", np.array([0, 1, 0]))
        
        query = np.array([1, 0, 0])
        results = self.dedup.find_similar(query, top_k=2, min_similarity=0.5)
        
        assert len(results) == 2
        assert results[0][0] == "url1"  # Most similar
        assert results[0][1] == pytest.approx(1.0)
    
    def test_clear(self):
        """Test clearing stored data."""
        self.dedup.check_exact_duplicate("content", "url1")
        self.dedup.add_embedding("url1", np.array([1, 0, 0]))
        
        self.dedup.clear()
        
        assert len(self.dedup._content_hashes) == 0
        assert len(self.dedup._embeddings) == 0


class TestURLDeduplicator:
    """Tests for URLDeduplicator class."""
    
    def test_extract_domain(self):
        """Test domain extraction."""
        assert URLDeduplicator.extract_domain("https://example.com/page") == "example.com"
        assert URLDeduplicator.extract_domain("https://www.example.com/page") == "example.com"
        assert URLDeduplicator.extract_domain("https://sub.example.com") == "sub.example.com"
    
    def test_extract_base_domain(self):
        """Test base domain extraction."""
        assert URLDeduplicator.extract_base_domain("https://sub.example.com") == "example.com"
        assert URLDeduplicator.extract_base_domain("https://www.example.co.uk") == "example.co.uk"
    
    def test_is_same_page(self):
        """Test same page detection."""
        url1 = "https://example.com/page"
        url2 = "https://example.com/page#section"
        url3 = "https://example.com/other"
        
        assert URLDeduplicator.is_same_page(url1, url2) is True
        assert URLDeduplicator.is_same_page(url1, url3) is False
    
    def test_group_by_domain(self):
        """Test grouping URLs by domain."""
        urls = [
            "https://example.com/page1",
            "https://example.com/page2",
            "https://other.com/page",
        ]
        
        groups = URLDeduplicator.group_by_domain(urls)
        
        assert len(groups) == 2
        assert len(groups["example.com"]) == 2
        assert len(groups["other.com"]) == 1
