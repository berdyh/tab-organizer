#!/usr/bin/env python3
"""Simple test runner for URL Input Service."""

import sys
import traceback
from main import URLValidator, URLParser, InputFormatDetector, URLEntry

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