#!/bin/bash

# Comprehensive test runner script for containerized testing
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TEST_TYPE="${1:-all}"
PARALLEL="${2:-false}"
CLEANUP="${3:-true}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Web Scraping Tool - Test Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Create directories for test results
mkdir -p test-results coverage test-reports logs

# Function to run tests
run_tests() {
    local test_category=$1
    echo -e "${YELLOW}Running ${test_category} tests...${NC}"
    
    case $test_category in
        unit)
            echo -e "${BLUE}Starting unit tests for all services...${NC}"
            docker compose -f docker-compose.test.yml up --build --abort-on-container-exit \
                url-input-unit-test \
                auth-unit-test \
                scraper-unit-test \
                analyzer-unit-test \
                clustering-unit-test \
                export-unit-test \
                session-unit-test \
                visualization-unit-test \
                api-gateway-unit-test \
                web-ui-unit-test
            ;;
        integration)
            echo -e "${BLUE}Starting integration tests for all services...${NC}"
            # Start infrastructure first
            docker compose -f docker-compose.test.yml up -d test-qdrant test-ollama
            sleep 10
            
            # Run integration tests
            docker compose -f docker-compose.test.yml up --build --abort-on-container-exit \
                url-input-integration-test \
                auth-integration-test \
                scraper-integration-test \
                analyzer-integration-test \
                clustering-integration-test \
                export-integration-test \
                session-integration-test \
                api-gateway-integration-test
            ;;
        e2e)
            echo -e "${BLUE}Starting end-to-end tests...${NC}"
            # Start full test environment
            docker compose -f docker-compose.test.yml up -d test-qdrant test-ollama test-api-gateway test-web-ui
            sleep 20
            
            # Run E2E tests
            docker compose -f docker-compose.test.yml up --build --abort-on-container-exit e2e-test-runner
            ;;
        performance)
            echo -e "${BLUE}Starting performance tests...${NC}"
            # Start infrastructure
            docker compose -f docker-compose.test.yml up -d test-qdrant test-ollama test-api-gateway
            sleep 20
            
            # Run load tests
            docker compose -f docker-compose.test.yml up --abort-on-container-exit load-test-runner
            ;;
        all)
            echo -e "${BLUE}Running all tests sequentially...${NC}"
            run_tests unit
            run_tests integration
            run_tests e2e
            ;;
        *)
            echo -e "${RED}Unknown test type: $test_category${NC}"
            exit 1
            ;;
    esac
}

# Function to collect test results
collect_results() {
    echo -e "${YELLOW}Collecting test results...${NC}"
    
    # Copy test results from containers
    for container in $(docker ps -a --filter "name=test" --format "{{.Names}}"); do
        echo "Collecting results from $container..."
        docker cp $container:/app/test-results/. ./test-results/ 2>/dev/null || true
        docker cp $container:/app/coverage/. ./coverage/ 2>/dev/null || true
    done
    
    echo -e "${GREEN}Test results collected in ./test-results and ./coverage${NC}"
}

# Function to generate reports
generate_reports() {
    echo -e "${YELLOW}Generating test reports...${NC}"
    
    # Run report aggregator
    docker compose -f docker-compose.test.yml up --build test-report-aggregator
    
    echo -e "${GREEN}Reports generated in ./test-reports${NC}"
}

# Function to cleanup
cleanup() {
    if [ "$CLEANUP" = "true" ]; then
        echo -e "${YELLOW}Cleaning up containers and volumes...${NC}"
        docker compose -f docker-compose.test.yml down -v
        echo -e "${GREEN}Cleanup complete${NC}"
    else
        echo -e "${YELLOW}Skipping cleanup (containers still running)${NC}"
    fi
}

# Trap to ensure cleanup on exit
trap cleanup EXIT

# Main execution
echo -e "${BLUE}Test Configuration:${NC}"
echo -e "  Test Type: ${YELLOW}$TEST_TYPE${NC}"
echo -e "  Parallel: ${YELLOW}$PARALLEL${NC}"
echo -e "  Cleanup: ${YELLOW}$CLEANUP${NC}"
echo ""

# Run tests
run_tests $TEST_TYPE

# Collect results
collect_results

# Generate reports
generate_reports

# Print summary
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Test execution completed!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Results available in:"
echo -e "  - ${YELLOW}./test-results${NC} - JUnit XML reports"
echo -e "  - ${YELLOW}./coverage${NC} - Coverage reports"
echo -e "  - ${YELLOW}./test-reports${NC} - Aggregated reports"
echo ""

# Check for failures
if grep -q "FAILED" test-results/*.xml 2>/dev/null; then
    echo -e "${RED}Some tests failed. Check test-results for details.${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi
