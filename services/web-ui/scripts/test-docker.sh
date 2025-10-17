#!/bin/bash

# Web UI Docker Test Runner
# This script runs tests inside a Docker container

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "================================"
echo "Web UI Test Runner (Docker)"
echo "================================"
echo ""
echo "Working directory: $(pwd)"
echo ""

# Build the test image
echo "Building test image..."
docker build -t web-ui-test -f - . <<'EOF'
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package.json ./

# Install dependencies (use npm install since package-lock.json may not exist)
RUN npm install

# Copy source code
COPY . .

# Set environment for CI
ENV CI=true

CMD ["npm", "run", "test:ci"]
EOF

echo ""
echo "Running tests..."
docker run --rm \
  -v "$(pwd)/coverage:/app/coverage" \
  web-ui-test

echo ""
echo "================================"
echo "Tests completed!"
echo "================================"
echo ""
echo "Coverage report available in ./coverage directory"
