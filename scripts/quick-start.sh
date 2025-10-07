#!/bin/bash

# Quick Start Script - Complete setup and start
set -e

echo "🚀 Quick Start: Web Scraping & Clustering Tool"
echo "=============================================="

# Check if setup has been run
if [ ! -f .env ]; then
    echo "📋 Running initial setup..."
    ./scripts/setup.sh
else
    echo "✅ Setup already completed"
fi

echo ""
echo "🔄 Starting all services..."
echo "   (This may take a moment, please wait...)"
./scripts/start.sh > /tmp/start.log 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Services started successfully"
else
    echo "❌ Service startup failed. Check logs:"
    tail -10 /tmp/start.log
    exit 1
fi

echo ""
echo "⏳ Waiting for services to be fully ready..."
echo "   Checking service health..."

# Wait for key services to be healthy
for i in {1..30}; do
    if docker compose ps qdrant | grep -q "healthy"; then
        echo "✅ Qdrant is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "⚠️  Qdrant taking longer than expected"
    fi
    sleep 2
done

echo ""
echo "🤖 Setting up optimal AI models for your hardware..."
python3 scripts/model-manager.py auto

echo ""
echo "🎉 Quick start complete!"
echo ""
echo "🌐 Your system is ready:"
echo "  - Web UI: http://localhost:8089"
echo "  - API Gateway: http://localhost:8080"
echo "  - Qdrant Dashboard: http://localhost:6333/dashboard"
echo ""
echo "💡 Next steps:"
echo "  1. Open the Web UI at http://localhost:8089"
echo "  2. Create a new session"
echo "  3. Add URLs to scrape"
echo "  4. Use the chatbot to explore your content!"
echo ""
echo "🛑 To stop all services: docker compose down"