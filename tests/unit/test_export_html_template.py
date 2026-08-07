"""The HTML export must render its TEMPLATE, not quietly fall back.

There were no export tests at all before this file, which is how the
following went unnoticed: `templates/` was never copied into the
backend-core image, AND `Exporter` resolved the template directory by
walking five parents from `__file__` -- the repo root in a source checkout,
but `/` inside the container. So `export_html`'s `try/except` fell through
to `_generate_basic_html` on every deployed request, and the 185-line
`export.html.j2` was live only when someone ran from a checkout.

Two faults that mask each other: fixing only the Dockerfile still resolves
to `/`, and fixing only the path finds nothing to load. A test asserting
merely "HTML came back" passes in both broken states and in the fixed one,
which is why these assert on content the TEMPLATE produces and the
fallback does not.
"""

from pathlib import Path

import pytest

from services.backend_core.app.export.exporter import Exporter
from services.backend_core.app.sessions.manager import Session

REPO_TEMPLATES = Path(__file__).resolve().parents[2] / "templates"


def _session() -> Session:
    return Session(id="s-1", name="Reading list")


def test_export_html_renders_the_template_not_the_fallback():
    html = Exporter().export_html(_session())

    # `export.html.j2` sets a <title> of "<name> - Tab Organizer Export";
    # `_generate_basic_html` does not emit that suffix at all. Asserting on
    # "<html" or on the session name would pass against both.
    assert "Reading list - Tab Organizer Export" in html


def test_the_fallback_is_still_reachable_and_still_produces_html():
    """Non-vacuity guard, and a real contract.

    The fallback exists so a missing template degrades instead of 500ing.
    Without this test, `export_html` could be changed to raise and the test
    above would still pass.
    """
    html = Exporter(templates_dir="/nonexistent/templates").export_html(_session())

    assert "<html" in html.lower()
    assert "Reading list" in html
    assert "Tab Organizer Export" not in html


def test_the_template_directory_actually_ships():
    """The path resolution is only half the fix; the files must be present.

    In the container `_resolve_templates_dir` returns /app/templates, which
    exists only because the Dockerfile copies it. Here it resolves to the
    repo copy -- so this asserts the source of that COPY still exists and
    still carries the template `export_html` asks for by name.
    """
    assert (REPO_TEMPLATES / "export.html.j2").is_file()


@pytest.mark.parametrize("fmt", ["markdown", "json", "obsidian", "html"])
def test_every_dispatched_format_returns_content(fmt):
    """`export()`'s dispatcher is the API surface; notion was never in it."""
    assert Exporter().export(_session(), fmt)


def test_an_unsupported_format_is_refused():
    with pytest.raises(ValueError):
        Exporter().export(_session(), "notion")
