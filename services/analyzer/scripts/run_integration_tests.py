#!/usr/bin/env python3
"""
Docker-based integration test runner for the analyzer service.
This runs tests inside the Docker container with mocked external dependencies.
"""

import os
import sys
import subprocess
import tempfile
import json
from pathlib import Path

def setup_test_environment():
    """Setup test environment with mock configurations."""
    # Create temporary config directory
    config_dir = Path("/tmp/test_config")
    config_dir.mkdir(exist_ok=True)
    
    # Create mock models.json
    mock_models_config = {
        "llm_models": {
            "llama3.2:3b": {
                "name": "Llama 3.2 3B",
                "size": "2GB",
                "speed": "fast",
                "quality": "good",
                "min_ram_gb": 4,
                "description": "Test model",
                "provider": "Meta",
                "recommended": True
            },
            "llama3.2:1b": {
                "name": "Llama 3.2 1B", 
                "size": "1.3GB",
                "speed": "fastest",
                "quality": "basic",
                "min_ram_gb": 2,
                "description": "Lightweight test model",
                "provider": "Meta",
                "recommended": False
            }
        },
        "embedding_models": {
            "nomic-embed-text": {
                "name": "Nomic Embed Text",
                "size": "274MB",
                "dimensions": 768,
                "quality": "high",
                "min_ram_gb": 1,
                "description": "Test embedding model",
                "provider": "Nomic AI",
                "recommended": True
            },
            "all-minilm": {
                "name": "All-MiniLM",
                "size": "90MB", 
                "dimensions": 384,
                "quality": "good",
                "min_ram_gb": 0.5,
                "description": "Lightweight embedding model",
                "provider": "Microsoft",
                "recommended": False
            }
        }
    }
    
    with open(config_dir / "models.json", "w") as f:
        json.dump(mock_models_config, f, indent=2)
    
    # Set environment variables for testing
    os.environ["TESTING"] = "true"
    os.environ["CONFIG_PATH"] = str(config_dir)
    os.environ["QDRANT_URL"] = "http://mock-qdrant:6333"
    os.environ["OLLAMA_URL"] = "http://mock-ollama:11434"
    
    return config_dir

def run_tests():
    """Run the integration tests."""
    print("🚀 Setting up test environment...")
    config_dir = setup_test_environment()
    
    try:
        print("📋 Running integration tests...")
        
        # Run the tests with verbose output
        cmd = [
            sys.executable, "-m", "pytest", 
            "test_multi_model_integration.py",
            "-v",
            "--tb=short",
            "--no-header",
            "--disable-warnings"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("📊 Test Results:")
        print("=" * 60)
        print(result.stdout)
        
        if result.stderr:
            print("⚠️  Warnings/Errors:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed!")
            
        return result.returncode
        
    except Exception as e:
        print(f"💥 Error running tests: {e}")
        return 1
    
    finally:
        # Cleanup
        try:
            import shutil
            shutil.rmtree(config_dir)
        except Exception:
            pass

def run_unit_tests():
    """Run unit tests for core components."""
    print("🧪 Running unit tests...")
    
    unit_test_files = [
        "test_core_components.py",
        "test_hardware_detection.py", 
        "test_model_switching.py",
        "test_embedding_generation.py"
    ]
    
    total_passed = 0
    total_failed = 0
    
    for test_file in unit_test_files:
        if Path(test_file).exists():
            print(f"\n📝 Running {test_file}...")
            
            cmd = [
                sys.executable, "-m", "pytest",
                test_file,
                "-v",
                "--tb=short",
                "--disable-warnings"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {test_file} - PASSED")
                total_passed += 1
            else:
                print(f"❌ {test_file} - FAILED")
                print(result.stdout)
                total_failed += 1
        else:
            print(f"⚠️  {test_file} not found, skipping...")
    
    print(f"\n📊 Unit Test Summary: {total_passed} passed, {total_failed} failed")
    return total_failed == 0

def main():
    """Main test runner."""
    print("🔬 Analyzer Service Test Suite")
    print("=" * 50)
    
    # Change to the analyzer service directory
    os.chdir(Path(__file__).parent)
    
    # Check if we're in the right directory
    if not Path("main.py").exists():
        print("❌ Error: main.py not found. Make sure you're in the analyzer service directory.")
        return 1
    
    # Run unit tests first
    unit_success = run_unit_tests()
    
    # Run integration tests
    integration_result = run_tests()
    
    print("\n" + "=" * 60)
    if unit_success and integration_result == 0:
        print("🎉 All tests completed successfully!")
        return 0
    else:
        print("💥 Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())