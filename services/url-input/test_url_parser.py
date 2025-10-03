"""Unit tests for URL Input Service."""

import pytest
import json
from io import StringIO
from main import URLValidator, URLParser, InputFormatDetector, URLEntry


class TestURLValidator:
    """Test URL validation functionality."""
    
    def test_valid_urls(self):
        """Test validation of valid URLs."""
        valid_urls = [
            "https://example.com",
            "http://example.com",
            "https://subdomain.example.com",
            "https://example.com/path",
            "https://example.com/path?query=value",
            "https://example.com:8080",
            "http://localhost:3000",
            "https://192.168.1.1",
        ]
        
        for url in valid_urls:
            assert URLValidator.is_valid_url(url), f"URL should be valid: {url}"
            is_valid, error = URLValidator.validate_url(url)
            assert is_valid, f"URL should be valid: {url}"
            assert error is None, f"Should not have error for valid URL: {url}"
    
    def test_invalid_urls(self):
        """Test validation of invalid URLs."""
        invalid_urls = [
            "",
            "not-a-url",
            "ftp://example.com",  # Only http/https supported
            "example.com",  # Missing protocol
            "https://",  # Incomplete
            "https:// example.com",  # Space in URL
        ]
        
        for url in invalid_urls:
            assert not URLValidator.is_valid_url(url), f"URL should be invalid: {url}"
            is_valid, error = URLValidator.validate_url(url)
            assert not is_valid, f"URL should be invalid: {url}"
            assert error is not None, f"Should have error for invalid URL: {url}"
    
    def test_empty_and_none_urls(self):
        """Test handling of empty and None URLs."""
        test_cases = [None, "", "   ", "\n", "\t"]
        
        for url in test_cases:
            assert not URLValidator.is_valid_url(url), f"Should be invalid: {repr(url)}"
            is_valid, error = URLValidator.validate_url(url or "")
            assert not is_valid, f"Should be invalid: {repr(url)}"
            assert "empty" in error.lower(), f"Error should mention empty: {error}"


class TestInputFormatDetector:
    """Test input format detection."""
    
    def test_detect_file_type_by_extension(self):
        """Test file type detection by extension."""
        test_cases = [
            ("urls.txt", "", "text"),
            ("data.json", "", "json"),
            ("urls.csv", "", "csv"),
            ("data.xlsx", "", "excel"),
            ("file.tsv", "", "csv"),
        ]
        
        for filename, content, expected in test_cases:
            result = InputFormatDetector.detect_file_type(filename, content)
            assert result == expected, f"Expected {expected} for {filename}, got {result}"
    
    def test_detect_file_type_by_content(self):
        """Test file type detection by content."""
        json_content = '{"urls": ["https://example.com"]}'
        csv_content = "url,category\nhttps://example.com,test"
        
        assert InputFormatDetector.detect_file_type("unknown", json_content) == "json"
        assert InputFormatDetector.detect_file_type("unknown", csv_content) == "csv"
        assert InputFormatDetector.detect_file_type("unknown", "plain text") == "text"
    
    def test_extract_url_patterns(self):
        """Test URL pattern extraction from text."""
        text = """
        Check out https://example.com and http://test.org
        Also visit https://subdomain.example.com/path?query=value
        Invalid: not-a-url
        """
        
        urls = InputFormatDetector.extract_url_patterns(text)
        expected_urls = [
            "https://example.com",
            "http://test.org", 
            "https://subdomain.example.com/path?query=value"
        ]
        
        assert len(urls) == 3
        for expected_url in expected_urls:
            assert expected_url in urls


class TestURLParser:
    """Test URL parsing from different formats."""
    
    def test_parse_text_file(self):
        """Test parsing plain text files."""
        content = """
        https://example.com
        http://test.org
        # This is a comment
        
        https://another.com
        invalid-url
        """
        
        urls = URLParser.parse_text_file(content)
        
        assert len(urls) == 4  # 3 valid + 1 invalid
        
        # Check valid URLs
        valid_urls = [url for url in urls if url.validated]
        assert len(valid_urls) == 3
        assert "https://example.com" in [url.url for url in valid_urls]
        assert "http://test.org" in [url.url for url in valid_urls]
        assert "https://another.com" in [url.url for url in valid_urls]
        
        # Check invalid URL
        invalid_urls = [url for url in urls if not url.validated]
        assert len(invalid_urls) == 1
        assert invalid_urls[0].url == "invalid-url"
        assert invalid_urls[0].validation_error is not None
    
    def test_parse_json_file_simple_list(self):
        """Test parsing JSON with simple URL list."""
        content = json.dumps([
            "https://example.com",
            "http://test.org",
            "invalid-url"
        ])
        
        urls = URLParser.parse_json_file(content)
        
        assert len(urls) == 3
        valid_urls = [url for url in urls if url.validated]
        assert len(valid_urls) == 2
    
    def test_parse_json_file_structured(self):
        """Test parsing JSON with structured data."""
        content = json.dumps({
            "urls": [
                {
                    "url": "https://example.com",
                    "category": "test",
                    "priority": "high"
                },
                {
                    "url": "http://test.org",
                    "category": "demo"
                }
            ],
            "metadata": {
                "source": "test_data"
            }
        })
        
        urls = URLParser.parse_json_file(content)
        
        assert len(urls) == 2
        assert all(url.validated for url in urls)
        
        # Check metadata
        first_url = urls[0]
        assert first_url.category == "test"
        assert first_url.priority == "high"
        assert "source" in first_url.source_metadata
    
    def test_parse_json_file_invalid(self):
        """Test parsing invalid JSON."""
        with pytest.raises(ValueError, match="Invalid JSON format"):
            URLParser.parse_json_file("invalid json content")
    
    def test_parse_csv_file(self):
        """Test parsing CSV files."""
        content = """url,category,priority,notes
https://example.com,test,high,Test site
http://test.org,demo,medium,Demo site
invalid-url,error,low,Invalid URL
"""
        
        urls = URLParser.parse_csv_file(content)
        
        assert len(urls) == 3
        
        # Check valid URLs
        valid_urls = [url for url in urls if url.validated]
        assert len(valid_urls) == 2
        
        first_url = valid_urls[0]
        assert first_url.url == "https://example.com"
        assert first_url.category == "test"
        assert first_url.priority == "high"
        assert first_url.notes == "Test site"
    
    def test_parse_csv_file_auto_detect_url_column(self):
        """Test CSV parsing with auto-detection of URL column."""
        content = """website,description
https://example.com,Example site
http://test.org,Test site
"""
        
        urls = URLParser.parse_csv_file(content)
        
        assert len(urls) == 2
        assert all(url.validated for url in urls)
        assert urls[0].url == "https://example.com"
    
    def test_parse_csv_file_invalid(self):
        """Test parsing invalid CSV."""
        with pytest.raises(ValueError, match="Invalid CSV format"):
            URLParser.parse_csv_file("invalid,csv,content\nwith,mismatched,columns,extra")


if __name__ == "__main__":
    pytest.main([__file__])