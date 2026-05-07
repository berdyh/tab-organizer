# Test Harness

Cross-service test assets. The unified `docker-compose.yml` exposes one profile
per test type and `scripts/cli.py test` is a thin wrapper that runs the right
profile. The full pipeline (linting, formatting, unit, integration, e2e, build)
is mirrored by `.github/workflows/ci-cd.yml`.

> Authoritative testing guide: [`docs/TESTING.md`](../docs/TESTING.md).

## Layout

```
tests/
├── unit/           # Fast, isolated unit tests (pytest)
├── integration/    # Tests against running services
├── e2e/            # End-to-end workflow tests (full docker stack)
├── load/           # Locust scenarios (no CLI wrapper yet)
├── conftest.py     # Shared pytest fixtures
├── Dockerfile.test # Image used by every test-* compose service
├── requirements.txt
└── README.md
```

## Running tests

Prefer the CLI so the same containers run locally and in CI:

```bash
./scripts/cli.py test --type unit
./scripts/cli.py test --type integration   # boots the default stack first
./scripts/cli.py test --type e2e           # boots the default stack first
```

Or invoke `docker compose` directly when you need to iterate on a single
profile:

```bash
docker compose --profile test-unit up --build --abort-on-container-exit test-unit
docker compose --profile test-integration up --build --abort-on-container-exit test-integration
docker compose --profile test-e2e up --build --abort-on-container-exit test-e2e
```

There is no `--type performance` or `--type all` in `cli.py` today; run the
load suite manually from `tests/load/` if you need it.

## Artefacts

- `coverage/` — HTML coverage report (mounted from `test-unit`)
- `test-results/` — JUnit XML produced by integration and e2e runs

Both are gitignored and recreated on every run.
