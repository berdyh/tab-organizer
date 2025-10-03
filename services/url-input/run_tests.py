#!/usr/bin/env python3
"""Simple test runner for URL Input Service."""

import sys
import traceback
from main import (URLValidator, URLParser, InputFormatDetector, URLEntry, 
                  URLEnricher, URLDeduplicator, BatchProcessor, URLMetadata)

def test_url_validator():
    """Test URL validation functionality."""
    print("Testing URL Validator...")
    
    # Test valid URLs
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
        if not URLValidator.is_valid_url(url):
            print(f"❌ FAIL: {url} should be valid")
            return False
        is_valid, error = URLValidator.validate_url(url)
        if not is_valid or error is not None:
            print(f"❌ FAIL: {url} should be valid, got error: {error}")
            return False
    
    # Test invalid URLs
    invalid_urls = [
        "",
        "not-a-url",
        "ftp://example.com",
        "example.com",
        "https://",
        "https:// example.com",
    ]
    
    for url in invalid_urls:
        if URLValidator.is_valid_url(url):
            print(f"❌ FAIL: {url} should be invalid")
            return False
        is_valid, error = URLValidator.validate_url(url)
        if is_valid or error is None:
            print(f"❌ FAIL: {url} should be invalid")
            return False
    
    print("✅ URL Validator tests passed")
    return True

def test_text_parser():
    """Test text file parsing."""
    print("Testing Text Parser...")
    
    content = """
    https://example.com
    http://test.org
    # This is a comment
    
    https://another.com
    invalid-url
    """
    
    urls = URLParser.parse_text_file(content)
    
    if len(urls) != 4:
        print(f"❌ FAIL: Expected 4 URLs, got {len(urls)}")
        return False
    
    valid_urls = [url for url in urls if url.validated]
    if len(valid_urls) != 3:
        print(f"❌ FAIL: Expected 3 valid URLs, got {len(valid_urls)}")
        return False
    
    invalid_urls = [url for url in urls if not url.validated]
    if len(invalid_urls) != 1:
        print(f"❌ FAIL: Expected 1 invalid URL, got {len(invalid_urls)}")
        return False
    
    if invalid_urls[0].url != "invalid-url":
        print(f"❌ FAIL: Expected invalid URL to be 'invalid-url', got {invalid_urls[0].url}")
        return False
    
    print("✅ Text Parser tests passed")
    return True

def test_json_parser():
    """Test JSON file parsing."""
    print("Testing JSON Parser...")
    
    import json
    
    # Test simple list
    content = json.dumps([
        "https://example.com",
        "http://test.org",
        "invalid-url"
    ])
    
    urls = URLParser.parse_json_file(content)
    
    if len(urls) != 3:
        print(f"❌ FAIL: Expected 3 URLs, got {len(urls)}")
        return False
    
    valid_urls = [url for url in urls if url.validated]
    if len(valid_urls) != 2:
        print(f"❌ FAIL: Expected 2 valid URLs, got {len(valid_urls)}")
        return False
    
    # Test structured format
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
    
    if len(urls) != 2:
        print(f"❌ FAIL: Expected 2 URLs, got {len(urls)}")
        return False
    
    if not all(url.validated for url in urls):
        print("❌ FAIL: All URLs should be valid")
        return False
    
    first_url = urls[0]
    if first_url.category != "test" or first_url.priority != "high":
        print(f"❌ FAIL: Metadata not parsed correctly")
        return False
    
    print("✅ JSON Parser tests passed")
    return True

def test_csv_parser():
    """Test CSV file parsing."""
    print("Testing CSV Parser...")
    
    content = """url,category,priority,notes
https://example.com,test,high,Test site
http://test.org,demo,medium,Demo site
invalid-url,error,low,Invalid URL
"""
    
    urls = URLParser.parse_csv_file(content)
    
    if len(urls) != 3:
        print(f"❌ FAIL: Expected 3 URLs, got {len(urls)}")
        return False
    
    valid_urls = [url for url in urls if url.validated]
    if len(valid_urls) != 2:
        print(f"❌ FAIL: Expected 2 valid URLs, got {len(valid_urls)}")
        return False
    
    first_url = valid_urls[0]
    if (first_url.url != "https://example.com" or 
        first_url.category != "test" or 
        first_url.priority != "high" or
        first_url.notes != "Test site"):
        print(f"❌ FAIL: CSV metadata not parsed correctly")
        return False
    
    print("✅ CSV Parser tests passed")
    return True

def test_format_detector():
    """Test format detection."""
    print("Testing Format Detector...")
    
    test_cases = [
        ("urls.txt", "", "text"),
        ("data.json", "", "json"),
        ("urls.csv", "", "csv"),
        ("data.xlsx", "", "excel"),
        ("file.tsv", "", "csv"),
    ]
    
    for filename, content, expected in test_cases:
        result = InputFormatDetector.detect_file_type(filename, content)
        if result != expected:
            print(f"❌ FAIL: Expected {expected} for {filename}, got {result}")
            return False
    
    # Test URL pattern extraction
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
    
    if len(urls) != 3:
        print(f"❌ FAIL: Expected 3 URLs in pattern extraction, got {len(urls)}")
        return False
    
    for expected_url in expected_urls:
        if expected_url not in urls:
            print(f"❌ FAIL: Expected URL {expected_url} not found in extracted URLs")
            return False
    
    print("✅ Format Detector tests passed")
    return True

def test_url_enricher():
    """Test URL enrichment functionality."""
    print("Testing URL Enricher...")
    
    # Test metadata extraction
    url = "https://subdomain.example.com:8080/path/to/page?param1=value1&param2=value2#section"
    metadata = URLEnricher.extract_metadata(url)
    
    if not isinstance(metadata, URLMetadata):
        print(f"❌ FAIL: Expected URLMetadata instance")
        return False
    
    if metadata.domain != "example.com":
        print(f"❌ FAIL: Expected domain 'example.com', got '{metadata.domain}'")
        return False
    
    if metadata.subdomain != "subdomain":
        print(f"❌ FAIL: Expected subdomain 'subdomain', got '{metadata.subdomain}'")
        return False
    
    if metadata.parameter_count != 2:
        print(f"❌ FAIL: Expected 2 parameters, got {metadata.parameter_count}")
        return False
    
    if metadata.path_depth != 3:
        print(f"❌ FAIL: Expected path depth 3, got {metadata.path_depth}")
        return False
    
    # Test URL categorization
    test_cases = [
        ("https://twitter.com/user", "social_media"),
        ("https://github.com/repo", "documentation"),
        ("https://news.example.com", "news_media"),
        ("https://shop.example.com", "ecommerce"),
        ("https://university.edu", "education"),
        ("https://example.com", "general")
    ]
    
    for url, expected_category in test_cases:
        entry = URLEntry(url=url, validated=True)
        entry.metadata = URLEnricher.extract_metadata(url)
        category = URLEnricher.categorize_url(entry)
        if category != expected_category:
            print(f"❌ FAIL: Expected category '{expected_category}' for {url}, got '{category}'")
            return False
    
    # Test complete enrichment
    entry = URLEntry(url="https://docs.python.org/3/tutorial/", validated=True)
    enriched_entry = URLEnricher.enrich_url_entry(entry)
    
    if not enriched_entry.enriched:
        print(f"❌ FAIL: Entry should be marked as enriched")
        return False
    
    if enriched_entry.metadata is None:
        print(f"❌ FAIL: Entry should have metadata")
        return False
    
    if enriched_entry.category != "documentation":
        print(f"❌ FAIL: Expected category 'documentation', got '{enriched_entry.category}'")
        return False
    
    print("✅ URL Enricher tests passed")
    return True

def test_url_deduplicator():
    """Test URL deduplication functionality."""
    print("Testing URL Deduplicator...")
    
    # Test URL normalization
    test_cases = [
        ("https://example.com/page", "https://example.com/page"),
        ("https://example.com/page/", "https://example.com/page"),
        ("https://EXAMPLE.COM/PAGE", "https://example.com/page"),
        ("https://example.com/page?utm_source=google", "https://example.com/page"),
    ]
    
    for original, expected in test_cases:
        normalized = URLDeduplicator.normalize_url(original)
        if normalized != expected:
            print(f"❌ FAIL: Expected '{expected}' for '{original}', got '{normalized}'")
            return False
    
    # Test duplicate detection
    entries = [
        URLEntry(url="https://example.com/page", validated=True),
        URLEntry(url="https://example.com/page/", validated=True),
        URLEntry(url="https://example.com/different", validated=True),
        URLEntry(url="https://EXAMPLE.COM/page", validated=True),
    ]
    
    duplicates = URLDeduplicator.find_duplicates(entries)
    
    if len(duplicates) != 1:
        print(f"❌ FAIL: Expected 1 duplicate group, got {len(duplicates)}")
        return False
    
    duplicate_group = list(duplicates.values())[0]
    if len(duplicate_group) != 3:
        print(f"❌ FAIL: Expected 3 URLs in duplicate group, got {len(duplicate_group)}")
        return False
    
    # Test marking duplicates
    marked_entries = URLDeduplicator.mark_duplicates(entries)
    
    if marked_entries[0].duplicate_of is not None:
        print(f"❌ FAIL: First entry should not be marked as duplicate")
        return False
    
    if marked_entries[1].duplicate_of != "https://example.com/page":
        print(f"❌ FAIL: Second entry should be marked as duplicate of first")
        return False
    
    print("✅ URL Deduplicator tests passed")
    return True

def test_batch_processor():
    """Test batch processing functionality."""
    print("Testing Batch Processor...")
    
    # Create test entries
    entries = [
        URLEntry(url="https://example.com/page1", validated=True),
        URLEntry(url="https://example.com/page2", validated=True),
        URLEntry(url="https://example.com/page1", validated=True),  # Duplicate
        URLEntry(url="invalid-url", validated=False),
    ]
    
    # Process in batches
    processed = BatchProcessor.process_urls_batch(entries, batch_size=2)
    
    if len(processed) != 4:
        print(f"❌ FAIL: Expected 4 processed entries, got {len(processed)}")
        return False
    
    # Check that valid URLs were enriched
    valid_entries = [e for e in processed if e.validated]
    if not all(e.enriched for e in valid_entries):
        print(f"❌ FAIL: All valid entries should be enriched")
        return False
    
    # Check that duplicates were marked
    duplicates = [e for e in processed if e.duplicate_of]
    if len(duplicates) != 1:
        print(f"❌ FAIL: Expected 1 duplicate, got {len(duplicates)}")
        return False
    
    # Test processing statistics
    stats = BatchProcessor.get_processing_stats(processed)
    
    expected_stats = {
        "total_urls": 4,
        "valid_urls": 3,
        "invalid_urls": 1,
        "duplicate_urls": 1,
        "unique_urls": 2
    }
    
    for key, expected_value in expected_stats.items():
        if stats[key] != expected_value:
            print(f"❌ FAIL: Expected {key}={expected_value}, got {stats[key]}")
            return False
    
    print("✅ Batch Processor tests passed")
    return True

def test_enriched_parsing():
    """Test parsing with enrichment enabled."""
    print("Testing Enriched Parsing...")
    
    # Test text parsing with enrichment
    content = """
    https://github.com/user/repo
    https://stackoverflow.com/questions/123
    https://github.com/user/repo
    """
    
    urls = URLParser.parse_text_file(content, enrich=True)
    
    if len(urls) != 3:
        print(f"❌ FAIL: Expected 3 URLs, got {len(urls)}")
        return False
    
    # Check enrichment
    valid_urls = [url for url in urls if url.validated]
    if not all(url.enriched for url in valid_urls):
        print(f"❌ FAIL: All valid URLs should be enriched")
        return False
    
    # Check categorization
    if not all(url.category == "documentation" for url in valid_urls):
        print(f"❌ FAIL: All URLs should be categorized as documentation")
        return False
    
    # Check deduplication
    duplicates = [url for url in urls if url.duplicate_of]
    if len(duplicates) != 1:
        print(f"❌ FAIL: Expected 1 duplicate, got {len(duplicates)}")
        return False
    
    print("✅ Enriched Parsing tests passed")
    return True

def main():
    """Run all tests."""
    print("🧪 Running URL Input Service Tests")
    print("=" * 50)
    
    tests = [
        test_url_validator,
        test_text_parser,
        test_json_parser,
        test_csv_parser,
        test_format_detector,
        test_url_enricher,
        test_url_deduplicator,
        test_batch_processor,
        test_enriched_parsing,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ FAIL: {test.__name__} raised exception: {e}")
            traceback.print_exc()
            failed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())