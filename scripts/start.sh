#!/bin/bash

# Start script for Web Scraping & Clustering Tool
set -e

echo "🚀 Starting Web Scraping & Clustering Tool..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found. Please run ./scripts/setup.sh first."
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Determine which profile to use (GPU or CPU)
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "🎮 GPU detected, using GPU profile for Ollama..."
    PROFILE="--profile gpu"
else
    echo "💻 No GPU detected, using CPU profile for Ollama..."
    PROFILE="--profile cpu"
fi

# Start services
echo "🔄 Starting services..."
docker-compose $PROFILE up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 10

# Check service health
echo "🏥 Checking service health..."
for service in qdrant ollama api-gateway; do
    echo -n "  Checking $service... "
    for i in {1..30}; do
        if docker-compose ps $service | grep -q "healthy\|Up"; then
            echo "✅ healthy"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "❌ timeout"
        fi
        sleep 2
    done
done

# Download Ollama models if needed
echo "📥 Setting up Ollama models..."

# Read models from .env file
if [ -f .env ]; then
    OLLAMA_MODEL=$(grep "^OLLAMA_MODEL=" .env | cut -d'=' -f2)
    OLLAMA_EMBEDDING_MODEL=$(grep "^OLLAMA_EMBEDDING_MODEL=" .env | cut -d'=' -f2)
else
    OLLAMA_MODEL="llama3.2:3b"
    OLLAMA_EMBEDDING_MODEL="nomic-embed-text"
fi

echo "  Pulling LLM model: $OLLAMA_MODEL"
docker-compose exec -T ollama ollama pull "$OLLAMA_MODEL" || echo "⚠️  Failed to pull $OLLAMA_MODEL model"

echo "  Pulling embedding model: $OLLAMA_EMBEDDING_MODEL"
docker-compose exec -T ollama ollama pull "$OLLAMA_EMBEDDING_MODEL" || echo "⚠️  Failed to pull $OLLAMA_EMBEDDING_MODEL model"

echo ""
echo "💡 To change models or add more, run: ./scripts/setup-models.sh"

echo ""
echo "✅ Services started successfully!"
echo ""
echo "🌐 Access points:"
echo "  - API Gateway: http://localhost:8080"
echo "  - Qdrant UI: http://localhost:6333/dashboard"
echo "  - Health Check: http://localhost:8080/health"
echo ""
echo "📊 Service Status:"
curl -s http://localhost:8080/health | python3 -m json.tool 2>/dev/null || echo "API Gateway not ready yet"