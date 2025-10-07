#!/usr/bin/env python3
"""
Test runner for the monitoring service.
Runs comprehensive tests for monitoring functionality.
"""

import sys
import subprocess
import os
from pathlib import Path

def run_tests():
    """Run monitoring service tests."""
    print("🧪 Running Monitoring Service Tests")
    print("=" * 50)
    
    # Change to monitoring service directory
    os.chdir(Path(__file__).parent)
    
    # Install test dependencies if needed
    try:
        import pytest
        import pytest_asyncio
    except ImportError:
        print("Installing test dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest", "pytest-asyncio"], check=True)
    
    # Run tests with pytest
    test_command = [
        sys.executable, "-m", "pytest",
        "test_monitoring.py",
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ]
    
    try:
        result = subprocess.run(test_command, check=True)
        print("\n✅ All monitoring tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code {e.returncode}")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)