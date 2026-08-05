.PHONY: help test test-unit test-integration test-e2e test-performance test-all smoke-test test-security clean coverage dev dev-up dev-down build deploy lint format security

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
	@./scripts/cli.py test --type unit

test-integration: ## Run integration tests for all services
	@echo "$(BLUE)Running integration tests...$(NC)"
	@./scripts/cli.py test --type integration

test-e2e: ## Run end-to-end tests
	@echo "$(BLUE)Running end-to-end tests...$(NC)"
	@./scripts/cli.py test --type e2e

test-performance: ## Run performance and load tests
	@echo "$(BLUE)Running performance tests...$(NC)"
	@echo "$(YELLOW)Performance/load tests are manual today. Start the app stack, then run:$(NC)"
	@echo "  locust -f tests/load/locustfile.py"

test-all: ## Run all tests (unit, integration, e2e)
	@echo "$(BLUE)Running all tests...$(NC)"
	@./scripts/cli.py test --type all

smoke-test: ## Run tests marked @pytest.mark.smoke (quick validation subset)
	@echo "$(BLUE)Running smoke tests...$(NC)"
	@docker compose --profile test-unit run --rm test-unit pytest tests/ -m smoke -q

test-service: ## Run tests for specific service (usage: make test-service SERVICE=backend-core)
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(RED)Error: SERVICE not specified. Usage: make test-service SERVICE=backend-core$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Running tests for $(SERVICE)...$(NC)"
	@echo "$(YELLOW)Note: Tests are organized by type (unit/integration/e2e), not by service$(NC)"
	@docker compose --profile test-unit up --build --abort-on-container-exit test-unit

test-backend: ## Run focused Backend Core unit tests
	@echo "$(BLUE)Running Backend Core focused tests...$(NC)"
	@docker compose --profile test-unit run --rm test-unit pytest \
		tests/unit/test_backend_tab_workflows.py \
		tests/unit/test_platform_backend.py \
		tests/unit/test_backend_callback_persistence.py \
		tests/unit/test_ingest_v1.py \
		tests/unit/test_session_persistence.py \
		tests/unit/test_scrape_callback.py \
		tests/unit/test_url_store.py \
		tests/unit/test_startup_config_validation.py -q

test-ai: ## Run focused AI Engine unit tests
	@echo "$(BLUE)Running AI Engine focused tests...$(NC)"
	@docker compose --profile test-unit run --rm test-unit pytest \
		tests/unit/test_ai_provider_switch.py \
		tests/unit/test_subscription_cli_providers.py \
		tests/unit/test_rag_lancedb_persistence.py \
		tests/unit/test_clustering.py \
		tests/unit/test_provider_routing.py \
		tests/unit/test_startup_config_validation.py -q

test-browser: ## Run focused Browser Engine unit tests
	@echo "$(BLUE)Running Browser Engine focused tests...$(NC)"
	@docker compose --profile test-unit run --rm test-unit pytest \
		tests/unit/test_browser_tab_harvester.py \
		tests/unit/test_cdp_second_hop.py \
		tests/unit/test_browser_engine_callbacks.py \
		tests/unit/test_auth_detector.py \
		tests/unit/test_startup_config_validation.py -q

test-web: ## Run focused Web UI unit tests
	@echo "$(BLUE)Running Web UI focused tests...$(NC)"
	@docker compose --profile test-unit run --rm test-unit pytest \
		tests/unit/test_web_ui_platform.py \
		tests/unit/test_web_ui_text.py -q

test-ops: ## Run focused CLI/config/runtime unit tests
	@echo "$(BLUE)Running Ops Tooling focused tests...$(NC)"
	@docker compose --profile test-unit run --rm test-unit pytest \
		tests/unit/test_cli_host_ai.py \
		tests/unit/test_cli_configure_provider.py \
		tests/unit/test_cli_check_provider.py \
		tests/unit/test_init_script.py \
		tests/unit/test_runtime_auth_config.py -q

test-security: ## Run the frozen security-invariant suite (tests/security)
	@echo "$(BLUE)Running security-invariant suite...$(NC)"
	@docker compose --profile test-unit run --rm test-unit \
		pytest tests/security -m "security and not integration" -q

test-watch: ## Run tests in watch mode for development
	@echo "$(BLUE)Running tests in watch mode...$(NC)"
	@./scripts/cli.py start -d --dev
	@echo "$(GREEN)Development environment started. Tests will run on file changes.$(NC)"

# ==================== COVERAGE ====================

coverage: ## Generate coverage reports
	@echo "$(BLUE)Generating coverage reports...$(NC)"
	@./scripts/cli.py test --type unit
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
	@./scripts/cli.py start -d --dev
	@echo "$(GREEN)Development environment started!$(NC)"
	@echo "$(YELLOW)Services available at:$(NC)"
	@echo "  - Backend Core: http://localhost:8080"
	@echo "  - AI Engine: http://localhost:8090"
	@echo "  - Browser Engine: http://localhost:8083"
	@echo "  - Web UI: http://localhost:8089"
	@echo "  - Ollama: http://localhost:11434"
	@echo "  - LanceDB: embedded in AI Engine (volume: lancedb-data)"

dev-down: ## Stop development environment
	@echo "$(BLUE)Stopping development environment...$(NC)"
	@docker compose --profile dev down
	@echo "$(GREEN)Development environment stopped$(NC)"

dev-logs: ## View development environment logs
	@docker compose --profile dev logs -f

dev-restart: ## Restart development environment
	@echo "$(BLUE)Restarting development environment...$(NC)"
	@docker compose --profile dev restart
	@echo "$(GREEN)Development environment restarted$(NC)"

dev-rebuild: ## Rebuild and restart development environment
	@echo "$(BLUE)Rebuilding development environment...$(NC)"
	@./scripts/cli.py start -d --build --dev
	@echo "$(GREEN)Development environment rebuilt$(NC)"

# ==================== PRODUCTION ====================

build: ## Build production Docker images
	@echo "$(BLUE)Building production images...$(NC)"
	@docker compose build
	@echo "$(GREEN)Production images built$(NC)"

up: ## Start production environment
	@echo "$(BLUE)Starting production environment...$(NC)"
	@./scripts/cli.py start -d
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
	@docker run --rm -v $(PWD):/app -w /app python:3.12-slim sh -c "\
		pip install flake8 pylint > /dev/null 2>&1 && \
		echo '$(YELLOW)Running flake8...$(NC)' && \
		flake8 services/ --count --select=E9,F63,F7,F82 --show-source --statistics && \
		echo '$(YELLOW)Running pylint...$(NC)' && \
		find services/ -name '*.py' | xargs pylint --exit-zero"
	@echo "$(GREEN)Linting complete$(NC)"

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	@docker run --rm -v $(PWD):/app -w /app python:3.12-slim sh -c "\
		pip install black==26.3.1 isort==6.1.0 > /dev/null 2>&1 && \
		black --target-version py312 services/ && \
		isort services/"
	@echo "$(GREEN)Code formatted$(NC)"

format-check: ## Check code formatting without modifying
	@echo "$(BLUE)Checking code formatting...$(NC)"
	@docker run --rm -v $(PWD):/app -w /app python:3.12-slim sh -c "\
		pip install black==26.3.1 isort==6.1.0 > /dev/null 2>&1 && \
		black --check --target-version py312 services/ && \
		isort --check-only services/"

security: ## Run security checks
	@echo "$(BLUE)Running security checks...$(NC)"
	@docker run --rm -v $(PWD):/app -w /app python:3.12-slim sh -c "\
		pip install bandit safety > /dev/null 2>&1 && \
		bandit -r services/ -f json -o bandit-report.json && \
		safety check"
	@echo "$(GREEN)Security checks complete$(NC)"

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checks...$(NC)"
	@docker run --rm -v $(PWD):/app -w /app python:3.12-slim sh -c "\
		pip install mypy > /dev/null 2>&1 && \
		mypy services/ --ignore-missing-imports"

quality: lint format-check type-check security ## Run all code quality checks

# ==================== CLEANUP ====================

clean: ## Clean up containers, volumes, and test artifacts
	@echo "$(BLUE)Cleaning up...$(NC)"
	@docker compose --profile test-unit --profile test-integration --profile test-e2e --profile test-performance --profile test-report down -v 2>/dev/null || true
	@docker compose --profile dev down -v 2>/dev/null || true
	@docker compose down -v 2>/dev/null || true
	@rm -rf test-results coverage test-reports logs/*.log
	@echo "$(GREEN)Cleanup complete$(NC)"

clean-all: clean ## Clean everything including Docker images
	@echo "$(BLUE)Cleaning all Docker resources...$(NC)"
	@docker system prune -af --volumes
	@echo "$(GREEN)All Docker resources cleaned$(NC)"

# ==================== DATABASE ====================

db-reset: ## Reset LanceDB vector store (deletes the lancedb-data volume)
	@echo "$(BLUE)Resetting LanceDB vector store...$(NC)"
	@docker compose stop ai-engine
	@docker volume rm $(shell docker volume ls -q | grep lancedb-data) 2>/dev/null || true
	@docker compose up -d ai-engine
	@echo "$(GREEN)Database reset complete$(NC)"

db-backup: ## Backup LanceDB vector store
	@echo "$(BLUE)Backing up LanceDB vector store...$(NC)"
	@mkdir -p backups
	@docker run --rm -v $(shell docker volume ls -q | grep lancedb-data):/data -v $(PWD)/backups:/backup alpine tar czf /backup/lancedb-backup-$(shell date +%Y%m%d-%H%M%S).tar.gz -C /data .
	@echo "$(GREEN)Database backup complete$(NC)"

# ==================== MONITORING ====================

stats: ## Show container resource usage
	@docker stats --no-stream

ps: ## Show running containers
	@docker compose ps

health: ## Check health of all services
	@echo "$(BLUE)Checking service health...$(NC)"
	@echo "Backend Core:"
	@curl -s http://localhost:8080/health | jq . || echo "$(RED)Backend Core not responding$(NC)"
	@echo "\nAI Engine:"
	@curl -s http://localhost:8090/health | jq . || echo "$(RED)AI Engine not responding$(NC)"
	@echo "\nBrowser Engine:"
	@curl -s http://localhost:8083/health | jq . || echo "$(RED)Browser Engine not responding$(NC)"

# ==================== DOCUMENTATION ====================

docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	@echo "$(YELLOW)Documentation available in docs/$(NC)"
	@ls -la docs/

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation...$(NC)"
	@docker run --rm -v $(PWD)/docs:/docs -p 8000:8000 python:3.12-slim sh -c "\
		cd /docs && python -m http.server 8000"

# ==================== UTILITIES ====================

shell: ## Open shell in a service container (usage: make shell SERVICE=backend-core)
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(RED)Error: SERVICE not specified. Usage: make shell SERVICE=backend-core$(NC)"; \
		echo "$(YELLOW)Available services: backend-core, ai-engine, browser-engine, web-ui, ollama$(NC)"; \
		exit 1; \
	fi
	@docker compose exec $(SERVICE) /bin/bash || docker compose exec $(SERVICE) /bin/sh

logs-service: ## View logs for specific service (usage: make logs-service SERVICE=backend-core)
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(RED)Error: SERVICE not specified. Usage: make logs-service SERVICE=backend-core$(NC)"; \
		echo "$(YELLOW)Available services: backend-core, ai-engine, browser-engine, web-ui, ollama$(NC)"; \
		exit 1; \
	fi
	@docker compose logs -f $(SERVICE)

install: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	@uv pip install -r tests/requirements.txt 2>/dev/null || pip install -r tests/requirements.txt
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
	@./scripts/cli.py test --type all
