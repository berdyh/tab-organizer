#!/usr/bin/env python3
"""Test runner for Export Service."""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run all tests for the export service."""
    print("🧪 Running Export Service Tests")
    print("=" * 50)
    
    # Change to the service directory
    service_dir = Path(__file__).parent
    
    try:
        # Run pytest with coverage
        cmd = [
            sys.executable, "-m", "pytest",
            "test_export_integration.py",
            "-v",
            "--tb=short",
            "--color=yes"
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        print("-" * 50)
        
        result = subprocess.run(cmd, cwd=service_dir, capture_output=False)
        
        if result.returncode == 0:
            print("\n✅ All tests passed!")
        else:
            print(f"\n❌ Tests failed with return code: {result.returncode}")
            
        return result.returncode
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)