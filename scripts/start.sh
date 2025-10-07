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

# Check Docker Compose availability
if ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose V2 is not available. Please update Docker."
    exit 1
fi

# Check if local Ollama is already running
echo "🔍 Checking for existing Ollama installation..."
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "✅ Local Ollama detected and running on port 11434"
    echo "🔗 Will use existing Ollama installation instead of Docker container"
    USE_LOCAL_OLLAMA=true
    PROFILE=""
    
    # Update .env to point to local Ollama
    if [ -f .env ]; then
        sed -i 's|OLLAMA_URL=http://ollama:11434|OLLAMA_URL=http://localhost:11434|g' .env
        echo "📝 Updated .env to use local Ollama"
    fi
else
    echo "🐳 No local Ollama found, will use Docker container"
    USE_LOCAL_OLLAMA=false
    
    # Determine which profile to use (GPU or CPU)
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        echo "🎮 GPU detected, using GPU profile for Ollama..."
        PROFILE="--profile gpu"
    else
        echo "💻 No GPU detected, using CPU profile for Ollama..."
        PROFILE="--profile cpu"
    fi
    
    # Update .env to point to Docker Ollama
    if [ -f .env ]; then
        sed -i 's|OLLAMA_URL=http://localhost:11434|OLLAMA_URL=http://ollama:11434|g' .env
        echo "📝 Updated .env to use Docker Ollama"
    fi
fi

# Start services
echo "🔄 Starting services..."
if [ "$USE_LOCAL_OLLAMA" = true ]; then
    echo "🔗 Using local Ollama installation"
    docker compose -f docker-compose.yml -f docker-compose.local-ollama.yml up -d --quiet-pull 2>/dev/null
else
    echo "🐳 Using Docker Ollama container"
    docker compose $PROFILE up -d --quiet-pull 2>/dev/null
fi

# Show a clean summary
echo "📊 Service startup summary:"
docker compose ps --format "table {{.Name}}\t{{.Status}}" | head -20

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
echo "   This may take 30-60 seconds for first startup..."

# Check service health with cleaner output
echo "🏥 Checking service health..."

# Function to check service health quietly
check_service_health() {
    local service=$1
    local max_attempts=15
    
    for i in $(seq 1 $max_attempts); do
        if docker compose ps $service 2>/dev/null | grep -q "healthy\|Up"; then
            return 0
        fi
        sleep 2
    done
    return 1
}

if [ "$USE_LOCAL_OLLAMA" = true ]; then
    services_to_check="qdrant api-gateway"
    echo -n "  Local Ollama... "
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "✅"
    else
        echo "❌"
    fi
else
    services_to_check="qdrant ollama api-gateway"
fi

for service in $services_to_check; do
    echo -n "  $service... "
    if check_service_health $service; then
        echo "✅"
    else
        echo "❌ (timeout)"
    fi
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
docker compose exec -T ollama ollama pull "$OLLAMA_MODEL" || echo "⚠️  Failed to pull $OLLAMA_MODEL model"

echo "  Pulling embedding model: $OLLAMA_EMBEDDING_MODEL"
docker compose exec -T ollama ollama pull "$OLLAMA_EMBEDDING_MODEL" || echo "⚠️  Failed to pull $OLLAMA_EMBEDDING_MODEL model"

echo ""
echo "💡 To change models or add more, run: ./scripts/setup-models.sh"

echo ""
echo "✅ Services started successfully!"
echo ""
echo "🌐 Access points:"
echo "  - 🌐 Web UI: http://localhost:8089"
echo "  - 🔗 API Gateway: http://localhost:8080"
echo "  - 📊 Qdrant Dashboard: http://localhost:6333/dashboard"
echo "  - ❤️  Health Check: http://localhost:8080/health"
echo ""
echo "💡 Next steps:"
echo "  1. Open Web UI: http://localhost:8089"
echo "  2. Setup AI models: python3 scripts/model-manager.py auto"
echo "  3. Create a session and start scraping!"
echo ""
echo "🛑 To stop: docker compose down"