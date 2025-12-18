"""Unit tests for URL Store."""

import pytest
import sys
sys.path.insert(0, '/app')

from services.backend_core.app.url_input.store import URLStore, URLRecord


class TestURLStore:
    """Tests for URLStore class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.store = URLStore()
    
    def test_normalize_basic(self):
        """Test basic URL normalization."""
        url = "https://Example.COM/path/"
        normalized = self.store.normalize(url)
        assert normalized == "https://example.com/path"
    
    def test_normalize_removes_www(self):
        """Test that www prefix is removed."""
        url = "https://www.example.com/page"
        normalized = self.store.normalize(url)
        assert normalized == "https://example.com/page"
    
    def test_normalize_removes_tracking_params(self):
        """Test that tracking parameters are removed."""
        url = "https://example.com/page?utm_source=google&utm_medium=cpc&id=123"
        normalized = self.store.normalize(url)
        assert "utm_source" not in normalized
        assert "utm_medium" not in normalized
        assert "id=123" in normalized
    
    def test_normalize_removes_fbclid(self):
        """Test that Facebook click ID is removed."""
        url = "https://example.com/page?fbclid=abc123&real_param=value"
        normalized = self.store.normalize(url)
        assert "fbclid" not in normalized
        assert "real_param=value" in normalized
    
    def test_add_new_url(self):
        """Test adding a new URL."""
        is_new, record = self.store.add("https://example.com/page")
        assert is_new is True
        assert record.original == "https://example.com/page"
        assert len(self.store) == 1
    
    def test_add_duplicate_url(self):
        """Test adding a duplicate URL."""
        self.store.add("https://example.com/page")
        is_new, record = self.store.add("https://example.com/page")
        assert is_new is False
        assert len(self.store) == 1
    
    def test_add_normalized_duplicate(self):
        """Test that normalized duplicates are detected."""
        self.store.add("https://www.example.com/page/")
        is_new, _ = self.store.add("https://example.com/page")
        assert is_new is False
        assert len(self.store) == 1
    
    def test_add_batch(self):
        """Test batch URL addition."""
        urls = [
            "https://example.com/page1",
            "https://example.com/page2",
            "https://example.com/page1",  # duplicate
            "https://example.com/page3",
        ]
        added, duplicates, records = self.store.add_batch(urls)
        assert added == 3
        assert duplicates == 1
        assert len(records) == 3
    
    def test_get_url(self):
        """Test getting a URL record."""
        self.store.add("https://example.com/page")
        record = self.store.get("https://example.com/page")
        assert record is not None
        assert record.original == "https://example.com/page"
    
    def test_get_nonexistent_url(self):
        """Test getting a nonexistent URL."""
        record = self.store.get("https://nonexistent.com")
        assert record is None
    
    def test_update_status(self):
        """Test updating URL status."""
        self.store.add("https://example.com/page")
        success = self.store.update_status("https://example.com/page", "scraped")
        assert success is True
        record = self.store.get("https://example.com/page")
        assert record.status == "scraped"
    
    def test_get_by_status(self):
        """Test getting URLs by status."""
        self.store.add("https://example.com/page1")
        self.store.add("https://example.com/page2")
        self.store.update_status("https://example.com/page1", "scraped")
        
        pending = self.store.get_by_status("pending")
        scraped = self.store.get_by_status("scraped")
        
        assert len(pending) == 1
        assert len(scraped) == 1
    
    def test_contains(self):
        """Test URL containment check."""
        self.store.add("https://example.com/page")
        assert "https://example.com/page" in self.store
        assert "https://other.com" not in self.store
    
    def test_remove(self):
        """Test URL removal."""
        self.store.add("https://example.com/page")
        success = self.store.remove("https://example.com/page")
        assert success is True
        assert len(self.store) == 0
    
    def test_clear(self):
        """Test clearing all URLs."""
        self.store.add("https://example.com/page1")
        self.store.add("https://example.com/page2")
        count = self.store.clear()
        assert count == 2
        assert len(self.store) == 0
    
    def test_count_by_status(self):
        """Test counting URLs by status."""
        self.store.add("https://example.com/page1")
        self.store.add("https://example.com/page2")
        self.store.add("https://example.com/page3")
        self.store.update_status("https://example.com/page1", "scraped")
        self.store.update_status("https://example.com/page2", "scraped")
        
        counts = self.store.count_by_status()
        assert counts["pending"] == 1
        assert counts["scraped"] == 2
    
    def test_content_hash_unique(self):
        """Test content hash for unique content."""
        self.store.add("https://example.com/page")
        is_unique, duplicate = self.store.set_content_hash(
            "https://example.com/page",
            "This is unique content"
        )
        assert is_unique is True
        assert duplicate is None
    
    def test_content_hash_duplicate(self):
        """Test content hash for duplicate content."""
        self.store.add("https://example.com/page1")
        self.store.add("https://example.com/page2")
        
        self.store.set_content_hash("https://example.com/page1", "Same content")
        is_unique, duplicate = self.store.set_content_hash(
            "https://example.com/page2",
            "Same content"
        )
        
        assert is_unique is False
        assert duplicate is not None
