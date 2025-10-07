#!/bin/bash

# Script to run visualization service tests in Docker container

set -e

echo "=========================================="
echo "Running Visualization Service Tests"
echo "=========================================="

# Build and run the test container
docker compose -f docker-compose.test.yml build visualization-test
docker compose -f docker-compose.test.yml run --rm visualization-test

echo ""
echo "=========================================="
echo "✅ Visualization tests completed!"
echo "=========================================="
