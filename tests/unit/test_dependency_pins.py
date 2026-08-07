"""Guards on the LanceDB pin, which lives in two files and ships in an image.

Two separate failures hid here at once and neither was visible from a green
test run:

1. ``lancedb==0.6.8`` was a real release that upstream later DELETED from PyPI
   (all of 0.6.6-0.6.12 went with it). The ai-engine image became unbuildable
   from scratch, but every machine with a cached image kept working, so the
   breakage stayed invisible for months while the fixes it blocked piled up.
2. The pin is duplicated: ``services/ai-engine/requirements.txt`` decides what
   ships, and ``tests/requirements.txt`` decides what
   ``test_rag_lancedb_persistence.py`` actually drives -- that suite uses a real
   LanceDB, not a mock. If the two drift, the LanceDB tests certify a version
   the service does not contain.

These tests cannot detect a deleted release (that needs the network, and unit
tests stay offline). They detect the two things that made the deletion so
expensive: a split pin, and a test image whose installed version no longer
matches the pin it claims to test -- i.e. a stale image.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
AI_ENGINE_REQUIREMENTS = REPO_ROOT / "services" / "ai-engine" / "requirements.txt"
TEST_REQUIREMENTS = REPO_ROOT / "tests" / "requirements.txt"


def _pinned_version(requirements_path: Path, package: str) -> str:
    """Return the exact pinned version of a package, or fail with why not."""
    if not requirements_path.exists():
        pytest.fail(f"{requirements_path} is missing")

    pattern = re.compile(
        rf"^{re.escape(package)}(\[[^\]]*\])?==([^\s;#]+)",
        re.IGNORECASE,
    )
    for line in requirements_path.read_text().splitlines():
        match = pattern.match(line.strip())
        if match:
            return match.group(2)

    pytest.fail(f"{requirements_path} does not pin {package}==<version>")


def test_lancedb_pin_is_identical_in_service_and_test_requirements():
    """A split pin makes the LanceDB tests certify a version that never ships."""
    service_pin = _pinned_version(AI_ENGINE_REQUIREMENTS, "lancedb")
    test_pin = _pinned_version(TEST_REQUIREMENTS, "lancedb")

    assert service_pin == test_pin, (
        "LanceDB pin drifted: services/ai-engine/requirements.txt ships "
        f"{service_pin} but tests/requirements.txt tests {test_pin}. "
        "tests/unit/test_rag_lancedb_persistence.py drives a real LanceDB, so "
        "the two pins must move together."
    )


def test_installed_lancedb_matches_the_pin_under_test():
    """A stale test image silently validates a version we no longer ship."""
    lancedb = pytest.importorskip("lancedb")

    pinned = _pinned_version(TEST_REQUIREMENTS, "lancedb")
    installed = lancedb.__version__

    assert installed == pinned, (
        f"This test image has lancedb {installed} installed but "
        f"tests/requirements.txt pins {pinned}. The image is stale: rebuild it "
        "with `docker compose --profile test-unit build test-unit` (and rebuild "
        "ai-engine too) before trusting any LanceDB result."
    )


def test_lancedb_stays_on_the_api_surface_rag_depends_on():
    """0.8+ rewrote this surface; rag.py is written against the 0.6 line."""
    pinned = _pinned_version(AI_ENGINE_REQUIREMENTS, "lancedb")
    major, minor = (int(part) for part in pinned.split(".")[:2])

    assert (major, minor) == (0, 6), (
        f"lancedb is pinned to {pinned}, outside the 0.6 line. "
        "services/ai-engine/app/chatbot/rag.py depends on the 0.6-era surface: "
        "create_table(data=<arrow table>) with a fixed_size_list vector column, "
        "table.to_pandas(), the no-arg table.search() empty-query builder used "
        "by summarize_session, and count_rows(filter=...). Moving off 0.6 also "
        "raises the pyarrow floor. If that move is intended, port the call "
        "sites and update this test and services/ai-engine/MODULE.md with it."
    )
