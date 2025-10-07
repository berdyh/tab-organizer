#!/bin/bash

# Validate containerized testing setup
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Test Setup Validation${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Track validation status
ERRORS=0
WARNINGS=0

# Function to check command exists
check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1 is installed"
        return 0
    else
        echo -e "${RED}✗${NC} $1 is not installed"
        ERRORS=$((ERRORS + 1))
        return 1
    fi
}

# Function to check file exists
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $1 exists"
        return 0
    else
        echo -e "${RED}✗${NC} $1 not found"
        ERRORS=$((ERRORS + 1))
        return 1
    fi
}

# Function to check directory exists
check_directory() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} $1 exists"
        return 0
    else
        echo -e "${YELLOW}⚠${NC} $1 not found (will be created)"
        WARNINGS=$((WARNINGS + 1))
        mkdir -p "$1"
        return 1
    fi
}

# Check required commands
echo -e "${BLUE}Checking required commands...${NC}"
check_command docker
check_command make
echo ""

# Check Docker is running
echo -e "${BLUE}Checking Docker status...${NC}"
if docker info &> /dev/null; then
    echo -e "${GREEN}✓${NC} Docker is running"
else
    echo -e "${RED}✗${NC} Docker is not running"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check Docker Compose files
echo -e "${BLUE}Checking Docker Compose files...${NC}"
check_file "docker-compose.yml"
check_file "docker-compose.dev.yml"
check_file "docker-compose.test.yml"
echo ""

# Check test directories
echo -e "${BLUE}Checking test directories...${NC}"
check_directory "tests"
check_directory "tests/e2e"
check_directory "tests/load"
check_directory "test-results"
check_directory "coverage"
check_directory "test-reports"
check_directory "logs"
echo ""

# Check test files
echo -e "${BLUE}Checking test files...${NC}"
check_file "tests/e2e/Dockerfile"
check_file "tests/e2e/test_full_workflow.py"
check_file "tests/load/locustfile.py"
check_file "pytest.ini"
check_file "requirements-dev.txt"
echo ""

# Check scripts
echo -e "${BLUE}Checking test scripts...${NC}"
check_file "scripts/run-all-tests.sh"
check_file "Makefile"

# Check if scripts are executable
if [ -x "scripts/run-all-tests.sh" ]; then
    echo -e "${GREEN}✓${NC} scripts/run-all-tests.sh is executable"
else
    echo -e "${YELLOW}⚠${NC} scripts/run-all-tests.sh is not executable"
    chmod +x scripts/run-all-tests.sh
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

# Check CI/CD configuration
echo -e "${BLUE}Checking CI/CD configuration...${NC}"
check_directory ".github"
check_directory ".github/workflows"
check_file ".github/workflows/ci-cd.yml"
echo ""

# Check documentation
echo -e "${BLUE}Checking documentation...${NC}"
check_file "docs/TESTING.md"
check_file "tests/README.md"
echo ""

# Validate Docker Compose configurations
echo -e "${BLUE}Validating Docker Compose configurations...${NC}"

if docker compose -f docker-compose.yml config > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} docker-compose.yml is valid"
else
    echo -e "${RED}✗${NC} docker-compose.yml has errors"
    ERRORS=$((ERRORS + 1))
fi

if docker compose -f docker-compose.dev.yml config > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} docker-compose.dev.yml is valid"
else
    echo -e "${RED}✗${NC} docker-compose.dev.yml has errors"
    ERRORS=$((ERRORS + 1))
fi

if docker compose -f docker-compose.test.yml config > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} docker-compose.test.yml is valid"
else
    echo -e "${RED}✗${NC} docker-compose.test.yml has errors"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check service test files
echo -e "${BLUE}Checking service test files...${NC}"
SERVICES=("url-input" "auth" "scraper" "analyzer" "clustering" "export" "session" "visualization" "api-gateway")

for service in "${SERVICES[@]}"; do
    service_dir="services/$service"
    if [ -d "$service_dir" ]; then
        # Check for at least one test file
        if ls $service_dir/test_*.py 1> /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC} $service has test files"
        else
            echo -e "${YELLOW}⚠${NC} $service has no test files"
            WARNINGS=$((WARNINGS + 1))
        fi
    else
        echo -e "${YELLOW}⚠${NC} $service directory not found"
        WARNINGS=$((WARNINGS + 1))
    fi
done
echo ""

# Test Docker network connectivity
echo -e "${BLUE}Testing Docker network...${NC}"
if docker network ls | grep -q test_network; then
    echo -e "${GREEN}✓${NC} test_network exists"
else
    echo -e "${YELLOW}⚠${NC} test_network will be created on first test run"
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

# Check disk space
echo -e "${BLUE}Checking disk space...${NC}"
AVAILABLE_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -gt 10 ]; then
    echo -e "${GREEN}✓${NC} Sufficient disk space (${AVAILABLE_SPACE}GB available)"
else
    echo -e "${YELLOW}⚠${NC} Low disk space (${AVAILABLE_SPACE}GB available, recommend 10GB+)"
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Validation Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo ""
    echo -e "${BLUE}You can now run tests:${NC}"
    echo -e "  ${YELLOW}make test${NC}           - Run unit tests"
    echo -e "  ${YELLOW}make test-all${NC}       - Run all tests"
    echo -e "  ${YELLOW}make dev${NC}            - Start development environment"
    echo ""
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ Validation completed with $WARNINGS warning(s)${NC}"
    echo -e "${YELLOW}Tests should work, but some optional components are missing${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Validation failed with $ERRORS error(s) and $WARNINGS warning(s)${NC}"
    echo ""
    echo -e "${RED}Please fix the errors before running tests${NC}"
    echo ""
    exit 1
fi
