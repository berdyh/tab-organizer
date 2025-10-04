#!/bin/bash
"""
Docker test runner script for analyzer service.
This script builds and runs tests inside a Docker container.
"""

set -e

echo "🐳 Building analyzer service Docker image for testing..."

# Build the analyzer service image
docker build -t analyzer-test -f Dockerfile .

echo "📋 Running integration tests in Docker container..."

# Run tests in container with mocked dependencies
docker run --rm \
    -v "$(pwd)":/app/test_src \
    -e TESTING=true \
    -e PYTHONPATH=/app \
    analyzer-test \
    python -m pytest test_multi_model_integration.py -v --tb=short

echo "✅ Docker tests completed!"