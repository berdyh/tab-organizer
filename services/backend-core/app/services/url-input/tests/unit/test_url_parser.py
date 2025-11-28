"""Unit tests for the URL Input service parsing and validation helpers."""

import json
from io import StringIO

import pytest

from main import (
    BatchProcessor,
    InputFormatDetector,
    URLEntry,
    URLEnricher,
    URLDeduplicator,
    URLMetadata,
    URLParser,
    URLValidator,
)


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
            "https://subdomain.example.com/path?query=value",
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

    def test_parse_json_input(self):
        """Test parsing JSON input."""
        json_data = json.dumps({
            "urls": [
                {"url": "https://example.com", "category": "news"},
                {"url": "https://test.org", "notes": "Important"}
            ]
        })

        urls = URLParser.parse_json_input(json_data)

        assert len(urls) == 2
        assert urls[0].url == "https://example.com"
        assert urls[0].category == "news"
        assert urls[1].url == "https://test.org"
        assert urls[1].notes == "Important"

    def test_parse_csv_input(self):
        """Test parsing CSV input."""
        csv_content = StringIO("url,category\nhttps://example.com,news\nhttp://test.org,blog")

        urls = URLParser.parse_csv_input(csv_content)

        assert len(urls) == 2
        assert urls[0].url == "https://example.com"
        assert urls[0].category == "news"
        assert urls[1].url == "http://test.org"
        assert urls[1].category == "blog"


class TestURLEnricher:
    """Test URL enrichment functionality."""

    def test_extract_metadata(self):
        """Test metadata extraction."""
        url = "https://docs.python.org/3/tutorial/"
        metadata = URLEnricher.extract_metadata(url)

        assert isinstance(metadata, URLMetadata)
        assert metadata.domain == "python.org"
        assert metadata.path_segments == ["3", "tutorial"]

    def test_enrich_entry(self):
        """Test entry enrichment."""
        entry = URLEntry(url="https://docs.python.org/3/tutorial/", validated=True)
        enriched_entry = URLEnricher.enrich_entry(entry)

        assert enriched_entry.enriched
        assert enriched_entry.metadata.domain == "python.org"
        assert enriched_entry.metadata.subdomain == "docs"


class TestURLDeduplicator:
    """Test URL deduplication functionality."""

    def test_normalize_url(self):
        """Test URL normalization for deduplication."""
        test_cases = [
            ("https://example.com/page", "https://example.com/page"),
            ("https://example.com/page/", "https://example.com/page"),
            ("https://EXAMPLE.COM/PAGE", "https://example.com/page"),
            ("https://example.com/page?utm_source=google", "https://example.com/page"),
            (
                "https://example.com/page?param=value&utm_campaign=test",
                "https://example.com/page?param=value",
            ),
        ]

        for original, expected in test_cases:
            normalized = URLDeduplicator.normalize_url(original)
            assert normalized == expected

    def test_find_duplicates(self):
        """Test duplicate detection."""
        entries = [
            URLEntry(url="https://example.com/page", validated=True),
            URLEntry(url="https://example.com/page/", validated=True),
            URLEntry(url="https://example.com/different", validated=True),
            URLEntry(url="https://EXAMPLE.COM/page", validated=True),
        ]

        duplicates = URLDeduplicator.find_duplicates(entries)

        # Should find one group with duplicates
        assert len(duplicates) == 1
        duplicate_group = list(duplicates.values())[0]
        assert len(duplicate_group) == 3  # Three variations of the same URL

    def test_mark_duplicates(self):
        """Test marking duplicates in URL list."""
        entries = [
            URLEntry(url="https://example.com/page", validated=True),
            URLEntry(url="https://example.com/page/", validated=True),
            URLEntry(url="https://example.com/different", validated=True),
        ]

        marked_entries = URLDeduplicator.mark_duplicates(entries)

        # First entry should be primary, second should be marked as duplicate
        assert marked_entries[0].duplicate_of is None
        assert marked_entries[1].duplicate_of == "https://example.com/page"
        assert marked_entries[2].duplicate_of is None

    def test_get_similarity_groups(self):
        """Test grouping URLs by domain similarity."""
        entries = [
            URLEntry(
                url="https://example.com/page1",
                validated=True,
                metadata=URLEnricher.extract_metadata("https://example.com/page1"),
            ),
            URLEntry(
                url="https://example.com/page2",
                validated=True,
                metadata=URLEnricher.extract_metadata("https://example.com/page2"),
            ),
            URLEntry(
                url="https://test.org/page",
                validated=True,
                metadata=URLEnricher.extract_metadata("https://test.org/page"),
            ),
        ]

        groups = URLDeduplicator.get_similarity_groups(entries)

        assert len(groups) == 2
        assert "example.com" in groups
        assert "test.org" in groups
        assert len(groups["example.com"]) == 2
        assert len(groups["test.org"]) == 1


class TestBatchProcessor:
    """Test batch processing functionality."""

    def test_process_urls_batch(self):
        """Test batch processing of URLs."""
        entries = [
            URLEntry(url="https://example.com/page1", validated=True),
            URLEntry(url="https://example.com/page2", validated=True),
            URLEntry(url="https://example.com/page1", validated=True),  # Duplicate
            URLEntry(url="invalid-url", validated=False),
        ]

        processed = BatchProcessor.process_urls_batch(entries, batch_size=2)

        # Should have enriched valid URLs and marked duplicates
        valid_entries = [e for e in processed if e.validated]
        assert all(e.enriched for e in valid_entries)

        # Should have marked duplicates
        duplicates = [e for e in processed if e.duplicate_of]
        assert len(duplicates) == 1

    def test_get_processing_stats(self):
        """Test processing statistics generation."""
        entries = [
            URLEntry(
                url="https://github.com/repo1",
                validated=True,
                enriched=True,
                category="documentation",
            ),
            URLEntry(
                url="https://github.com/repo2",
                validated=True,
                enriched=True,
                category="documentation",
            ),
            URLEntry(
                url="https://news.example.com",
                validated=True,
                enriched=True,
                category="news_media",
            ),
            URLEntry(
                url="https://github.com/repo1",
                validated=True,
                enriched=True,
                category="documentation",
                duplicate_of="https://github.com/repo1",
            ),
            URLEntry(url="invalid-url", validated=False),
        ]

        # Add metadata to valid entries
        for entry in entries:
            if entry.validated:
                entry.metadata = URLEnricher.extract_metadata(entry.url)

        stats = BatchProcessor.get_processing_stats(entries)

        assert stats["total_urls"] == 5
        assert stats["valid_urls"] == 4
        assert stats["invalid_urls"] == 1
        assert stats["enriched_urls"] == 4
        assert stats["duplicate_urls"] == 1
        assert stats["unique_urls"] == 3
        assert stats["categories"]["documentation"] == 3
        assert stats["categories"]["news_media"] == 1
        assert "github.com" in stats["top_domains"]
        assert stats["processing_complete"] is True
