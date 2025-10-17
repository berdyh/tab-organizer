#!/bin/bash
# Run a single test to see detailed output

echo "Building Docker image..."
docker build -f Dockerfile.test -t web-ui-test . > /dev/null 2>&1

echo "Running tests..."
docker run --rm -v $(pwd):/app -v /app/node_modules web-ui-test 2>&1
