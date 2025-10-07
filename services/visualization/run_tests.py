#!/usr/bin/env python3
"""
Test runner for Visualization Service
"""

import subprocess
import sys

def run_tests():
    """Run all tests for the visualization service"""
    print("=" * 70)
    print("Running Visualization Service Tests")
    print("=" * 70)
    
    # Run pytest with coverage
    result = subprocess.run(
        ["pytest", "test_visualization.py", "-v", "--tb=short"],
        capture_output=False
    )
    
    if result.returncode == 0:
        print("\n" + "=" * 70)
        print("✅ All tests passed!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ Some tests failed!")
        print("=" * 70)
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
