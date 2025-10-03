#!/usr/bin/env python3
"""Test runner for the analyzer service."""

import subprocess
import sys
import os
from pathlib import Path


def run_tests():
    """Run all tests for the analyzer service."""
    print("🧪 Running Analyzer Service Tests")
    print("=" * 40)
    
    # Get the directory containing this script
    test_dir = Path(__file__).parent
    
    # Test files to run
    test_files = [
        "test_embedding_generation.py",
        "test_model_switching.py", 
        "test_hardware_detection.py"
    ]
    
    all_passed = True
    
    # Set environment variables for testing
    os.environ["PYTHONPATH"] = str(test_dir)
    
    for test_file in test_files:
        test_path = test_dir / test_file
        if not test_path.exists():
            print(f"❌ Test file not found: {test_file}")
            all_passed = False
            continue
        
        print(f"\n🔍 Running {test_file}...")
        print("-" * 30)
        
        try:
            # Run pytest on the specific test file
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                str(test_path), 
                "-v",  # Verbose output
                "--tb=short",  # Short traceback format
                "--color=yes",  # Colored output
                "--no-header",  # Less verbose header
                "-x"  # Stop on first failure for faster feedback
            ], cwd=test_dir, capture_output=False)
            
            if result.returncode == 0:
                print(f"✅ {test_file} passed")
            else:
                print(f"❌ {test_file} failed")
                all_passed = False
                
        except Exception as e:
            print(f"❌ Error running {test_file}: {e}")
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1


def run_single_test(test_name):
    """Run a single test file."""
    test_dir = Path(__file__).parent
    test_path = test_dir / f"test_{test_name}.py"
    
    if not test_path.exists():
        print(f"❌ Test file not found: test_{test_name}.py")
        return 1
    
    os.environ["PYTHONPATH"] = str(test_dir)
    
    print(f"🔍 Running test_{test_name}.py...")
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        str(test_path), 
        "-v", "--tb=short", "--color=yes"
    ], cwd=test_dir)
    
    return result.returncode


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1]
        sys.exit(run_single_test(test_name))
    else:
        # Run all tests
        sys.exit(run_tests())