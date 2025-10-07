.PHONY: help test test-unit test-integration test-e2e test-performance test-all clean coverage dev dev-up dev-down build deploy lint format security

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)Web Scraping Tool - Make Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

# ==================== TESTING ====================

test: test-unit ## Run all unit tests (default)

test-unit: ## Run unit tests for all services
	@echo "$(BLUE)Running unit tests...$(NC)"
	@./scripts/run-all-tests.sh unit

test-integration: ## Run integration tests for all services
	@echo "$(BLUE)Running integration tests...$(NC)"
	@./scripts/run-all-tests.sh integration

test-e2e: ## Run end-to-end tests
	@echo "$(BLUE)Running end-to-end tests...$(NC)"
	@./scripts/run-all-tests.sh e2e

test-performance: ## Run performance and load tests
	@echo "$(BLUE)Running performance tests...$(NC)"
	@./scripts/run-all-tests.sh performance

test-all: ## Run all tests (unit, integration, e2e)
	@echo "$(BLUE)Running all tests...$(NC)"
	@./scripts/run-all-tests.sh all

test-service: ## Run tests for specific service (usage: make test-service SERVICE=analyzer)
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(RED)Error: SERVICE not specified. Usage: make test-service SERVICE=analyzer$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Running tests for $(SERVICE)...$(NC)"
	@docker compose -f docker-compose.test.yml up --build --abort-on-container-exit $(SERVICE)-unit-test

test-watch: ## Run tests in watch mode for development
	@echo "$(BLUE)Running tests in watch mode...$(NC)"
	@docker compose -f docker-compose.dev.yml up -d
	@echo "$(GREEN)Development environment started. Tests will run on file changes.$(NC)"

# ==================== COVERAGE ====================

coverage: ## Generate coverage reports
	@echo "$(BLUE)Generating coverage reports...$(NC)"
	@./scripts/run-all-tests.sh unit
	@echo "$(GREEN)Coverage reports generated in ./coverage$(NC)"
	@echo "$(YELLOW)Open coverage/index.html to view reports$(NC)"

coverage-report: coverage ## Generate and open coverage report
	@if command -v xdg-open > /dev/null; then \
		xdg-open coverage/index.html; \
	elif command -v open > /dev/null; then \
		open coverage/index.html; \
	else \
		echo "$(YELLOW)Please open coverage/index.html manually$(NC)"; \
	fi

# ==================== DEVELOPMENT ====================

dev: dev-up ## Start development environment (alias for dev-up)

dev-up: ## Start development environment with hot-reload
	@echo "$(BLUE)Starting development environment...$(NC)"
	@docker compose -f docker-compose.dev.yml up -d
	@echo "$(GREEN)Development environment started!$(NC)"
	@echo "$(YELLOW)Services available at:$(NC)"
	@echo "  - API Gateway: http://localhost:8080"
	@echo "  - Web UI: http://localhost:8089"
	@echo "  - Qdrant: http://localhost:6333"
	@echo "  - Ollama: http://localhost:11434"

dev-down: ## Stop development environment
	@echo "$(BLUE)Stopping development environment...$(NC)"
	@docker compose -f docker-compose.dev.yml down
	@echo "$(GREEN)Development environment stopped$(NC)"

dev-logs: ## View development environment logs
	@docker compose -f docker-compose.dev.yml logs -f

dev-restart: ## Restart development environment
	@echo "$(BLUE)Restarting development environment...$(NC)"
	@docker compose -f docker-compose.dev.yml restart
	@echo "$(GREEN)Development environment restarted$(NC)"

dev-rebuild: ## Rebuild and restart development environment
	@echo "$(BLUE)Rebuilding development environment...$(NC)"
	@docker compose -f docker-compose.dev.yml up -d --build
	@echo "$(GREEN)Development environment rebuilt$(NC)"

# ==================== PRODUCTION ====================

build: ## Build production Docker images
	@echo "$(BLUE)Building production images...$(NC)"
	@docker compose build
	@echo "$(GREEN)Production images built$(NC)"

up: ## Start production environment
	@echo "$(BLUE)Starting production environment...$(NC)"
	@docker compose up -d
	@echo "$(GREEN)Production environment started!$(NC)"

down: ## Stop production environment
	@echo "$(BLUE)Stopping production environment...$(NC)"
	@docker compose down
	@echo "$(GREEN)Production environment stopped$(NC)"

logs: ## View production logs
	@docker compose logs -f

restart: ## Restart production environment
	@docker compose restart

# ==================== CODE QUALITY ====================

lint: ## Run linting checks
	@echo "$(BLUE)Running linting checks...$(NC)"
	@docker run --rm -v $(PWD):/app -w /app python:3.11-slim sh -c "\
		pip install flake8 pylint > /dev/null 2>&1 && \
		echo '$(YELLOW)Running flake8...$(NC)' && \
		flake8 services/ --count --select=E9,F63,F7,F82 --show-source --statistics && \
		echo '$(YELLOW)Running pylint...$(NC)' && \
		find services/ -name '*.py' | xargs pylint --exit-zero"
	@echo "$(GREEN)Linting complete$(NC)"

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	@docker run --rm -v $(PWD):/app -w /app python:3.11-slim sh -c "\
		pip install black isort > /dev/null 2>&1 && \
		black services/ && \
		isort services/"
	@echo "$(GREEN)Code formatted$(NC)"

format-check: ## Check code formatting without modifying
	@echo "$(BLUE)Checking code formatting...$(NC)"
	@docker run --rm -v $(PWD):/app -w /app python:3.11-slim sh -c "\
		pip install black isort > /dev/null 2>&1 && \
		black --check services/ && \
		isort --check-only services/"

security: ## Run security checks
	@echo "$(BLUE)Running security checks...$(NC)"
	@docker run --rm -v $(PWD):/app -w /app python:3.11-slim sh -c "\
		pip install bandit safety > /dev/null 2>&1 && \
		bandit -r services/ -f json -o bandit-report.json && \
		safety check"
	@echo "$(GREEN)Security checks complete$(NC)"

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checks...$(NC)"
	@docker run --rm -v $(PWD):/app -w /app python:3.11-slim sh -c "\
		pip install mypy > /dev/null 2>&1 && \
		mypy services/ --ignore-missing-imports"

quality: lint format-check type-check security ## Run all code quality checks

# ==================== CLEANUP ====================

clean: ## Clean up containers, volumes, and test artifacts
	@echo "$(BLUE)Cleaning up...$(NC)"
	@docker compose -f docker-compose.test.yml down -v 2>/dev/null || true
	@docker compose -f docker-compose.dev.yml down -v 2>/dev/null || true
	@docker compose down -v 2>/dev/null || true
	@rm -rf test-results coverage test-reports logs/*.log
	@echo "$(GREEN)Cleanup complete$(NC)"

clean-all: clean ## Clean everything including Docker images
	@echo "$(BLUE)Cleaning all Docker resources...$(NC)"
	@docker system prune -af --volumes
	@echo "$(GREEN)All Docker resources cleaned$(NC)"

# ==================== DATABASE ====================

db-reset: ## Reset Qdrant database
	@echo "$(BLUE)Resetting Qdrant database...$(NC)"
	@docker compose down qdrant
	@docker volume rm $(shell docker volume ls -q | grep qdrant) 2>/dev/null || true
	@docker compose up -d qdrant
	@echo "$(GREEN)Database reset complete$(NC)"

db-backup: ## Backup Qdrant database
	@echo "$(BLUE)Backing up Qdrant database...$(NC)"
	@mkdir -p backups
	@docker run --rm -v $(shell docker volume ls -q | grep qdrant):/data -v $(PWD)/backups:/backup alpine tar czf /backup/qdrant-backup-$(shell date +%Y%m%d-%H%M%S).tar.gz -C /data .
	@echo "$(GREEN)Database backup complete$(NC)"

# ==================== MONITORING ====================

stats: ## Show container resource usage
	@docker stats --no-stream

ps: ## Show running containers
	@docker compose ps

health: ## Check health of all services
	@echo "$(BLUE)Checking service health...$(NC)"
	@curl -s http://localhost:8080/health | jq . || echo "$(RED)API Gateway not responding$(NC)"

# ==================== DOCUMENTATION ====================

docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	@echo "$(YELLOW)Documentation available in docs/$(NC)"
	@ls -la docs/

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation...$(NC)"
	@docker run --rm -v $(PWD)/docs:/docs -p 8000:8000 python:3.11-slim sh -c "\
		cd /docs && python -m http.server 8000"

# ==================== UTILITIES ====================

shell: ## Open shell in a service container (usage: make shell SERVICE=analyzer)
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(RED)Error: SERVICE not specified. Usage: make shell SERVICE=analyzer$(NC)"; \
		exit 1; \
	fi
	@docker compose exec $(SERVICE)-service /bin/bash

logs-service: ## View logs for specific service (usage: make logs-service SERVICE=analyzer)
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(RED)Error: SERVICE not specified. Usage: make logs-service SERVICE=analyzer$(NC)"; \
		exit 1; \
	fi
	@docker compose logs -f $(SERVICE)-service

install: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	@pip install -r requirements-dev.txt 2>/dev/null || echo "$(YELLOW)requirements-dev.txt not found$(NC)"
	@echo "$(GREEN)Dependencies installed$(NC)"

version: ## Show version information
	@echo "$(BLUE)Version Information:$(NC)"
	@echo "Docker: $(shell docker --version)"
	@echo "Docker Compose: $(shell docker compose version)"
	@echo "Python: $(shell python --version 2>&1)"

# ==================== CI/CD ====================

ci-local: ## Simulate CI pipeline locally
	@echo "$(BLUE)Running CI pipeline locally...$(NC)"
	@make quality
	@make test-all
	@echo "$(GREEN)CI pipeline complete$(NC)"

ci-test: ## Run CI tests (used by GitHub Actions)
	@./scripts/run-all-tests.sh all true true
