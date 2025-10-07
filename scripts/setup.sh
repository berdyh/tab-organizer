#!/bin/bash

# Setup script for Web Scraping & Clustering Tool
set -e

echo "🚀 Setting up Web Scraping & Clustering Tool..."

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not available. Please install Docker Compose V2 or update Docker."
    echo "💡 Try: 'docker --version' and 'docker compose version'"
    exit 1
fi

# Create logs directory
echo "📁 Creating logs directory..."
mkdir -p logs

# Copy environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your configuration before starting services"
fi

# Pull required Docker images
echo "📦 Pulling Docker images..."
docker compose pull qdrant ollama

# Build services
echo "🔨 Building services..."
docker compose build

echo "✅ Setup complete!"
echo ""
echo "🚀 Quick Start:"
echo "1. Run: ./scripts/start.sh"
echo "2. Access the Web UI at http://localhost:8089"
echo "3. Access the API Gateway at http://localhost:8080"
echo ""
echo "📝 Optional Configuration:"
echo "1. Edit .env file with your configuration (already has good defaults)"
echo "2. For advanced model selection (after starting services):"
echo "   - 'python3 scripts/model-manager.py auto' - Hardware-optimized model selection"
echo "   - './scripts/setup-models.sh' - Interactive model setup"
echo ""
echo "💡 The .env file already contains recommended models for your system:"
echo "  - LLM: phi4:3.8b (Microsoft's latest, GPU optimized)"
echo "  - Embedding: mxbai-embed-large (highest quality)"
echo ""
echo "⚠️  Note: Model setup tools require services to be running first!"
echo "   Start with './scripts/start.sh' then use model management tools."