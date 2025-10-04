"""Unit tests for Web Scraper Service."""

import pytest
import hashlib
from unittest.mock import Mock, patch, MagicMock
from bs4 import BeautifulSoup

from main import (
    ContentExtractor, 
    DuplicateDetector, 
    RobotsChecker,
    content_hashes
)


class TestContentExtractor:
    """Test content extraction functionality."""
    
    def test_extract_content_with_trafilatura_success(self):
        """Test successful content extraction with trafilatura."""
        html = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <article>
                    <h1>Main Article</h1>
                    <p>This is the main content of the article.</p>
                    <p>Another paragraph with useful information.</p>
                </article>
                <script>console.log('ignore this');</script>
            </body>
        </html>
        """
        
        extractor = ContentExtractor()
        with patch('main.trafilatura.extract') as mock_extract:
            mock_extract.return_value = "Main Article This is the main content of the article. Another paragraph with useful information."
            
            result = extractor.extract_content(html.encode('utf-8'), "https://example.com", "text/html")
            
            assert result['title'] == "Test Page"
            assert "Main Article" in result['content']
            assert "useful information" in result['content']
            assert result['word_count'] > 0
            assert "console.log" not in result['content']
    
    def test_extract_content_fallback_to_beautifulsoup(self):
        """Test fallback to Beautiful Soup when trafilatura fails."""
        html = """
        <html>
            <head><title>Fallback Test</title></head>
            <body>
                <main>
                    <h1>Main Content</h1>
                    <p>This content should be extracted by Beautiful Soup.</p>
                </main>
                <script>var x = 1;</script>
                <style>.hidden { display: none; }</style>
            </body>
        </html>
        """
        
        extractor = ContentExtractor()
        with patch('main.trafilatura.extract') as mock_extract:
            mock_extract.return_value = ""  # Simulate trafilatura failure
            
            result = extractor.extract_content(html.encode('utf-8'), "https://example.com", "text/html")
            
            assert result['title'] == "Fallback Test"
            assert "Main Content" in result['content']
            assert "Beautiful Soup" in result['content']
            assert "var x = 1" not in result['content']
            assert ".hidden" not in result['content']
    
    def test_extract_content_empty_html(self):
        """Test extraction with empty or invalid HTML."""
        extractor = ContentExtractor()
        result = extractor.extract_content(b"", "https://example.com", "text/html")
        
        assert result['title'] == ""
        assert result['content'] == ""
        assert result['word_count'] == 0
    
    def test_extract_content_no_title(self):
        """Test extraction when no title is present."""
        html = """
        <html>
            <body>
                <p>Content without title</p>
            </body>
        </html>
        """
        
        extractor = ContentExtractor()
        with patch('main.trafilatura.extract') as mock_extract:
            mock_extract.return_value = "Content without title that is long enough to pass the minimum length check"
            
            result = extractor.extract_content(html.encode('utf-8'), "https://example.com", "text/html")
            
            assert result['title'] == ""
            assert "Content without title" in result['content']
            assert result['word_count'] > 3


class TestDuplicateDetector:
    """Test duplicate detection functionality."""
    
    def setup_method(self):
        """Clear content hashes before each test."""
        content_hashes.clear()
    
    def test_generate_content_hash(self):
        """Test content hash generation."""
        detector = DuplicateDetector()
        content1 = "This is some test content."
        content2 = "This is some test content."
        content3 = "This is different content."
        
        hash1 = detector.generate_content_hash(content1)
        hash2 = detector.generate_content_hash(content2)
        hash3 = detector.generate_content_hash(content3)
        
        assert hash1 == hash2  # Same content should have same hash
        assert hash1 != hash3  # Different content should have different hash
        assert len(hash1) == 64  # SHA-256 hash length
    
    def test_generate_content_hash_normalization(self):
        """Test that content is normalized before hashing."""
        detector = DuplicateDetector()
        content1 = "This   is    some   test content."
        content2 = "THIS IS SOME TEST CONTENT."
        content3 = "  This is some test content.  "
        
        hash1 = detector.generate_content_hash(content1)
        hash2 = detector.generate_content_hash(content2)
        hash3 = detector.generate_content_hash(content3)
        
        assert hash1 == hash2 == hash3  # All should normalize to same hash
    
    def test_is_duplicate_first_occurrence(self):
        """Test duplicate detection for first occurrence."""
        detector = DuplicateDetector()
        content = "This is unique content."
        url = "https://example.com/page1"
        
        is_duplicate, content_hash, similarity = detector.is_duplicate(content, url)
        
        assert not is_duplicate
        assert len(content_hash) == 64
        assert content_hash in content_hashes
        assert content_hashes[content_hash] == url
        assert similarity == 0.0
    
    def test_is_duplicate_second_occurrence(self):
        """Test duplicate detection for duplicate content."""
        detector = DuplicateDetector()
        content = "This is duplicate content."
        url1 = "https://example.com/page1"
        url2 = "https://example.com/page2"
        
        # First occurrence
        is_duplicate1, hash1, similarity1 = detector.is_duplicate(content, url1)
        assert not is_duplicate1
        assert similarity1 == 0.0
        
        # Second occurrence (duplicate)
        is_duplicate2, hash2, similarity2 = detector.is_duplicate(content, url2)
        assert is_duplicate2
        assert hash1 == hash2
        assert similarity2 == 1.0
        assert content_hashes[hash1] == url1  # Original URL preserved


class TestRobotsChecker:
    """Test robots.txt compliance checking."""
    
    def test_can_fetch_allowed_url(self):
        """Test fetching allowed URL."""
        checker = RobotsChecker()
        
        with patch('main.RobotFileParser') as mock_parser_class:
            mock_parser = Mock()
            mock_parser.can_fetch.return_value = True
            mock_parser_class.return_value = mock_parser
            
            result = checker.can_fetch("https://example.com/allowed-page")
            
            assert result is True
            mock_parser.set_url.assert_called_with("https://example.com/robots.txt")
            mock_parser.read.assert_called_once()
    
    def test_can_fetch_disallowed_url(self):
        """Test fetching disallowed URL."""
        checker = RobotsChecker()
        
        with patch('main.RobotFileParser') as mock_parser_class:
            mock_parser = Mock()
            mock_parser.can_fetch.return_value = False
            mock_parser_class.return_value = mock_parser
            
            result = checker.can_fetch("https://example.com/disallowed-page")
            
            assert result is False
    
    def test_can_fetch_robots_txt_error(self):
        """Test handling robots.txt read errors."""
        checker = RobotsChecker()
        
        with patch('main.RobotFileParser') as mock_parser_class:
            mock_parser = Mock()
            mock_parser.read.side_effect = Exception("Network error")
            mock_parser.can_fetch.return_value = True
            mock_parser_class.return_value = mock_parser
            
            result = checker.can_fetch("https://example.com/some-page")
            
            assert result is True  # Should default to allowing
    
    def test_can_fetch_caching(self):
        """Test that robots.txt results are cached per domain."""
        checker = RobotsChecker()
        
        with patch('main.RobotFileParser') as mock_parser_class:
            mock_parser = Mock()
            mock_parser.can_fetch.return_value = True
            mock_parser_class.return_value = mock_parser
            
            # First call
            checker.can_fetch("https://example.com/page1")
            # Second call to same domain
            checker.can_fetch("https://example.com/page2")
            
            # Should only create parser once per domain
            assert mock_parser_class.call_count == 1
            assert mock_parser.read.call_count == 1


class TestIntegration:
    """Integration tests for scraper components."""
    
    def setup_method(self):
        """Clear content hashes before each test."""
        content_hashes.clear()
    
    def test_full_content_processing_pipeline(self):
        """Test complete content processing from HTML to duplicate detection."""
        html = """
        <html>
            <head><title>Integration Test</title></head>
            <body>
                <article>
                    <h1>Test Article</h1>
                    <p>This is test content for integration testing.</p>
                    <p>It should be processed correctly through the pipeline.</p>
                </article>
                <script>ignore_this();</script>
            </body>
        </html>
        """
        
        # Extract content
        extractor = ContentExtractor()
        with patch('main.trafilatura.extract') as mock_extract:
            mock_extract.return_value = "Test Article This is test content for integration testing. It should be processed correctly through the pipeline."
            
            extracted = extractor.extract_content(html.encode('utf-8'), "https://example.com", "text/html")
        
        # Check for duplicates
        detector = DuplicateDetector()
        is_duplicate, content_hash, similarity = detector.is_duplicate(extracted['content'], "https://example.com")
        
        # Verify results
        assert extracted['title'] == "Integration Test"
        assert "Test Article" in extracted['content']
        assert "integration testing" in extracted['content']
        assert extracted['word_count'] > 0
        assert not is_duplicate
        assert len(content_hash) == 64
        
        # Test duplicate detection on same content
        is_duplicate2, hash2, similarity2 = detector.is_duplicate(extracted['content'], "https://different.com")
        assert is_duplicate2
        assert hash2 == content_hash
        assert similarity2 == 1.0


if __name__ == "__main__":
    pytest.main([__file__])