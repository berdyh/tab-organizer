#!/bin/bash

# Setup script for Web Scraping & Clustering Tool
set -e

echo "🚀 Setting up Web Scraping & Clustering Tool..."

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
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
docker-compose pull qdrant ollama-cpu

# Build services
echo "🔨 Building services..."
docker-compose build

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Choose your AI models:"
echo "   - './scripts/setup-models.sh auto' for automatic setup (recommended)"
echo "   - './scripts/setup-models.sh' for interactive setup with manual choice"
echo "3. Run: ./scripts/start.sh"
echo "4. Access the API Gateway at http://localhost:8080"
echo ""
echo "💡 Model setup options:"
echo "  - './scripts/setup-models.sh auto' - Fully automatic based on hardware"
echo "  - './scripts/setup-models.sh' - Interactive with automatic/manual choice"
echo "  - './scripts/setup-models.sh list' - See all available models"
echo "  - 'python3 scripts/model-manager.py auto' - Advanced automatic setup"