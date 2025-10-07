#!/usr/bin/env python3
"""Run tests for the Chatbot Service using Docker Compose."""

import subprocess
import sys
import os

def run_tests():
    """Run all tests for the chatbot service in Docker containers."""
    print("🤖 Running Chatbot Service Tests in Docker")
    print("=" * 50)
    
    # Change to the service directory
    service_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(service_dir)
    
    try:
        # Clean up any existing test containers
        print("🧹 Cleaning up existing test containers...")
        subprocess.run([
            "docker", "compose", "-f", "docker-compose.test.yml", "down", "-v"
        ], capture_output=True)
        
        # Build and run tests
        print("🔨 Building test containers...")
        result = subprocess.run([
            "docker", "compose", "-f", "docker-compose.test.yml", "build"
        ], check=True, capture_output=True, text=True)
        
        print("🚀 Running containerized tests...")
        result = subprocess.run([
            "docker", "compose", "-f", "docker-compose.test.yml", "run", "--rm", "test-runner"
        ], check=True, capture_output=True, text=True)
        
        print("✅ All tests passed!")
        print(result.stdout)
        
        # Clean up
        print("🧹 Cleaning up test containers...")
        subprocess.run([
            "docker", "compose", "-f", "docker-compose.test.yml", "down", "-v"
        ], capture_output=True)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print("❌ Tests failed!")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        
        # Clean up on failure
        subprocess.run([
            "docker", "compose", "-f", "docker-compose.test.yml", "down", "-v"
        ], capture_output=True)
        
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)