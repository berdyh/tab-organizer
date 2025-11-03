# Test Harness

This directory holds cross-service test assets that complement the per-service unit/integration suites found under `services/<name>/tests/`.

## Layout

```
tests/
├── e2e/          # End-to-end workflow tests (pytest + Dockerfile)
├── load/         # Locust scenarios for performance checks
└── README.md
```

## Running Tests

Prefer the unified CLI so the same containers run locally and in CI:

```bash
./scripts/cli.py test --type unit        # All unit suites
./scripts/cli.py test --type integration # Integration suites (requires test-qdrant/test-ollama)
./scripts/cli.py test --type e2e         # Full workflow
./scripts/cli.py test --type performance # Locust load scenarios
```

Add `--skip-cleanup` to keep containers alive for debugging or `--skip-artifacts` during rapid iteration.

Invoke Docker Compose directly when you need to iterate on a single service:

```bash
docker compose --profile test-unit up --build --abort-on-container-exit analyzer-unit-test
docker compose --profile test-integration up --build --abort-on-container-exit clustering-integration-test
```

## Artefacts

- `coverage/` – aggregated coverage data (ignored from git)
- `test-results/` – JUnit XML from container runs
- `test-reports/` – HTML reports produced by the aggregator

All are produced automatically by the scripts and CI pipeline.
