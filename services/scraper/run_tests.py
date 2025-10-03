#!/usr/bin/env python3
"""Test runner for Web Scraper Service."""

import subprocess
import sys
import os

def run_tests():
    """Run all tests for the scraper service."""
    print("Running Web Scraper Service Tests...")
    print("=" * 50)
    
    # Set environment variables for testing
    os.environ['PYTHONPATH'] = '/app'
    
    try:
        # Run pytest with verbose output
        result = subprocess.run([
            'python', '-m', 'pytest', 
            'test_scraper.py',
            'test_auth_integration.py',
            '-v',
            '--tb=short',
            '--color=yes'
        ], capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n" + "=" * 50)
            print("✅ All tests passed!")
            return True
        else:
            print("\n" + "=" * 50)
            print("❌ Some tests failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)