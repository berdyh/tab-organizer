"""Static regressions for auth-aware local runtime entry points."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_ci_cross_service_jobs_export_local_service_tokens():
    workflow = (ROOT / ".github" / "workflows" / "ci-cd.yml").read_text()

    assert "AI_ENGINE_API_TOKEN: local-test-token" in workflow
    assert "BACKEND_CALLBACK_TOKEN: local-test-token" in workflow
    assert "BACKEND_AGENT_API_TOKEN: local-test-token" in workflow
    assert "PLATFORM_MAINTAINER_SIGNUP_CODE: local-maintainer" in workflow


def test_make_start_targets_use_cli_token_generation():
    makefile = (ROOT / "Makefile").read_text()

    assert "./scripts/cli.py start -d --dev" in makefile
    assert "./scripts/cli.py start -d --build --dev" in makefile
    assert "./scripts/cli.py start -d" in makefile


def test_direct_compose_docs_show_required_local_service_tokens():
    testing_doc = (ROOT / "docs" / "TESTING.md").read_text()
    test_readme = (ROOT / "tests" / "README.md").read_text()

    for text in (testing_doc, test_readme):
        assert "AI_ENGINE_API_TOKEN=local-test-token" in text
        assert "BACKEND_CALLBACK_TOKEN=local-test-token" in text
        assert "BACKEND_AGENT_API_TOKEN=local-test-token" in text
        assert "PLATFORM_MAINTAINER_SIGNUP_CODE=local-maintainer" in text


def test_published_ports_are_bound_to_loopback():
    """Every published port must be LAN-unreachable.

    The services are unauthenticated or thinly authenticated and hold the
    user's captured page content; container-to-container traffic uses docker
    service names, so nothing legitimate needs the 0.0.0.0 binding.
    """
    compose = (ROOT / "docker-compose.yml").read_text()

    published = re.findall(r'^\s+- "([^"]+)"\s*$', compose, flags=re.MULTILINE)
    mappings = [entry for entry in published if re.match(r"^[\d.:]+$", entry)]
    assert mappings, "no published port mappings found in docker-compose.yml"
    for mapping in mappings:
        assert mapping.startswith("127.0.0.1:"), (
            f"published port {mapping!r} is reachable from the whole LAN; "
            "bind it to 127.0.0.1 (see CLAUDE.md)"
        )


def test_web_ui_service_holds_only_the_tokens_it_uses():
    """web-ui calls backend routes that require the agent scope, and never
    uses a browser-engine token (its only browser-engine call is tokenless
    /health), so the grant set must match exactly that."""
    compose = (ROOT / "docker-compose.yml").read_text()
    web_ui_block = compose.split("  web-ui:", 1)[1].split("\n  ollama:", 1)[0]

    assert "- BACKEND_AGENT_API_TOKEN=${BACKEND_AGENT_API_TOKEN:-}" in web_ui_block
    assert "- BROWSER_ENGINE_API_TOKEN=" not in web_ui_block

    client = (ROOT / "services" / "web-ui" / "src" / "api" / "client.py").read_text()
    assert "BACKEND_AGENT_API_TOKEN" in client
    assert "BROWSER_ENGINE_API_TOKEN" not in client


def test_service_images_package_shared_url_safety_module():
    compose = (ROOT / "docker-compose.yml").read_text()
    backend_dockerfile = (ROOT / "services" / "backend-core" / "Dockerfile").read_text()
    browser_dockerfile = (
        ROOT / "services" / "browser-engine" / "Dockerfile"
    ).read_text()

    assert "dockerfile: services/backend-core/Dockerfile" in compose
    assert "dockerfile: services/browser-engine/Dockerfile" in compose
    assert "context: ./services/backend-core" not in compose
    assert "context: ./services/browser-engine" not in compose

    for dockerfile in (backend_dockerfile, browser_dockerfile):
        assert "COPY services/__init__.py ./services/__init__.py" in dockerfile
        assert "COPY services/url_safety.py ./services/url_safety.py" in dockerfile
