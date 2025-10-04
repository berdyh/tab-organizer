#!/usr/bin/env python3
"""Test runner for clustering service."""

import subprocess
import sys
import os

def run_tests():
    """Run all clustering service tests."""
    print("🧪 Running Clustering Service Tests")
    print("=" * 50)
    
    # Change to clustering service directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run pytest with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        "test_umap_dimensionality.py",
        "-v",
        "--tb=short",
        "--color=yes"
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ All clustering tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code {e.returncode}")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)