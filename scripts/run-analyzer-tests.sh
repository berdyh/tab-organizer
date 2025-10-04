#!/bin/bash
# Comprehensive test runner for analyzer service.
# This script can run tests in different modes: Docker, local, or with full stack.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Default mode
MODE="docker"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--mode docker|local|stack]"
            echo ""
            echo "Modes:"
            echo "  docker  - Run tests in isolated Docker container (default)"
            echo "  local   - Run tests locally (requires Python environment)"
            echo "  stack   - Run tests with full Docker Compose stack"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

print_status "Running analyzer service tests in mode: $MODE"

case $MODE in
    "docker")
        print_status "Building analyzer service test image..."
        
        # Build the analyzer service image
        docker build -t analyzer-service-test ./services/analyzer
        
        print_status "Running tests in isolated Docker container..."
        
        # Run tests with mocked dependencies
        docker run --rm \
            -v "$(pwd)/config:/app/config" \
            -e TESTING=true \
            -e PYTHONPATH=/app \
            analyzer-service-test \
            python -m pytest test_multi_model_integration.py -v --tb=short --disable-warnings
        
        if [ $? -eq 0 ]; then
            print_success "Docker tests completed successfully!"
        else
            print_error "Docker tests failed!"
            exit 1
        fi
        ;;
        
    "local")
        print_status "Running tests locally..."
        
        cd services/analyzer
        
        # Check if virtual environment exists
        if [ ! -d "venv" ]; then
            print_warning "No virtual environment found. Creating one..."
            python3 -m venv venv
            source venv/bin/activate
            pip install -r requirements.txt
        else
            source venv/bin/activate
        fi
        
        # Set environment variables
        export TESTING=true
        export PYTHONPATH=$(pwd)
        
        # Run tests
        python -m pytest test_multi_model_integration.py -v --tb=short --disable-warnings
        
        if [ $? -eq 0 ]; then
            print_success "Local tests completed successfully!"
        else
            print_error "Local tests failed!"
            exit 1
        fi
        ;;
        
    "stack")
        print_status "Starting required services..."
        
        # Start only required services for testing
        docker compose up -d qdrant
        
        # Wait for services to be healthy
        print_status "Waiting for services to be ready..."
        sleep 10
        
        # Check if qdrant is healthy
        if ! docker compose ps qdrant | grep -q "healthy"; then
            print_warning "Qdrant service is not healthy, but continuing with tests..."
        fi
        
        print_status "Running tests with Docker Compose stack..."
        
        # Run tests in analyzer service
        docker compose run --rm \
            -e TESTING=true \
            -e PYTHONPATH=/app \
            analyzer-service \
            python -m pytest test_multi_model_integration.py -v --tb=short --disable-warnings
        
        if [ $? -eq 0 ]; then
            print_success "Stack tests completed successfully!"
        else
            print_error "Stack tests failed!"
            exit 1
        fi
        
        # Cleanup
        print_status "Cleaning up services..."
        docker compose down
        ;;
        
    *)
        print_error "Unknown mode: $MODE"
        exit 1
        ;;
esac

print_success "All tests completed!"