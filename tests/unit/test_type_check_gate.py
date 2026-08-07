"""Guards on the mypy gate, which was a gate in name only for its whole life.

``make type-check`` and CI's "Run mypy type checking" step both ran
``mypy services/ --ignore-missing-imports``. That command cannot work in this
repo: backend-core, ai-engine and browser-engine each ship a top-level package
named ``app``, so mypy exits 2 with ``Duplicate module named "app"`` before
analysing a single line. Locally that was a permanent red nobody could read a
result out of; in CI the step ended in ``|| true``, so it was green for two
independent wrong reasons at once. It was documented as broken in CLAUDE.md and
left that way -- the documented-but-false pattern this repo keeps repeating.

Fixed 2026-08-07: ``scripts/type-check.sh`` runs mypy once per service and both
callers go through it. These tests hold the three things that would quietly
undo that:

1. an unpinned mypy (a gate that changes its mind on someone else's release is
   a gate people learn to ignore);
2. either caller drifting back to its own copy of the command;
3. CI re-acquiring a result-swallowing suffix.

They are text assertions on purpose. Actually running mypy belongs to the gate
itself, not to the unit suite -- but a unit suite that stays green while the
gate is silently disconnected is how this got here in the first place.
"""

import re
import stat
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TYPE_CHECK_SCRIPT = REPO_ROOT / "scripts" / "type-check.sh"
MAKEFILE = REPO_ROOT / "Makefile"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci-cd.yml"


def _read(path: Path) -> str:
    if not path.exists():
        pytest.fail(f"{path} is missing")
    return path.read_text()


def _makefile_target(name: str) -> str:
    """The recipe lines of a single Makefile target."""
    text = _read(MAKEFILE)
    match = re.search(
        rf"^{re.escape(name)}:.*?$\n((?:\t.*\n|\n)*)",
        text,
        re.MULTILINE,
    )
    if not match:
        pytest.fail(f"Makefile has no `{name}:` target")
    return match.group(1)


def test_type_check_script_exists_and_is_executable():
    """CI invokes it as `./scripts/type-check.sh`, which needs the bit set."""
    _read(TYPE_CHECK_SCRIPT)
    mode = TYPE_CHECK_SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR, (
        f"{TYPE_CHECK_SCRIPT} is not executable; CI runs it directly as "
        "`./scripts/type-check.sh` and would fail with Permission denied."
    )


def _pin_list() -> list[str]:
    """The packages the gate installs, from the script's MYPY_PINS line."""
    script = _read(TYPE_CHECK_SCRIPT)
    match = re.search(r'^MYPY_PINS="([^"]*)"', script, re.MULTILINE)
    if not match:
        pytest.fail('scripts/type-check.sh has no MYPY_PINS="..." line')
    return match.group(1).split()


def test_every_package_the_gate_installs_is_pinned():
    """An unpinned type checker turns an unrelated release into a red gate."""
    pins = _pin_list()
    assert pins, "MYPY_PINS is empty"
    for pin in pins:
        assert re.fullmatch(r"[A-Za-z0-9_.\-]+==\S+", pin), (
            f"{pin!r} in scripts/type-check.sh is not an exact pin. An "
            "unpinned tool makes the gate's verdict depend on the day it ran."
        )
    assert any(pin.startswith("mypy==") for pin in pins), (
        "scripts/type-check.sh must pin mypy itself"
    )


def test_stub_packages_are_pinned_too():
    """A stub package is a dependency of the gate's verdict like any other.

    `types-requests` is what makes web-ui's `requests` calls checkable at all;
    without it mypy reports `import-untyped` (which --ignore-missing-imports
    does NOT suppress) and the gate is red for a missing install rather than
    for anything in the diff.
    """
    pins = _pin_list()
    for stub in ("types-requests", "types-PyYAML"):
        assert any(pin.startswith(f"{stub}==") for pin in pins), (
            f"scripts/type-check.sh must pin {stub}=="
        )


def test_make_type_check_delegates_to_the_script():
    """The Makefile must not carry its own copy of the mypy command."""
    recipe = _makefile_target("type-check")
    assert "scripts/type-check.sh" in recipe, (
        "`make type-check` must run scripts/type-check.sh. Two copies of the "
        "invocation is exactly how the local gate and the CI gate drifted."
    )
    assert "mypy services/" not in recipe, (
        "`mypy services/` in a single pass aborts with `Duplicate module named "
        '"app"` -- three services own a package called `app`.'
    )


def test_ci_runs_the_same_script_and_does_not_swallow_its_result():
    """CI's `|| true` is what kept a never-running gate looking green."""
    workflow = _read(CI_WORKFLOW)
    step = re.search(
        r"^(\s*)- name: .*mypy.*$\n((?:(?:\1\s+.*)?\n)*)",
        workflow,
        re.MULTILINE | re.IGNORECASE,
    )
    assert step, 'the CI workflow has no step whose name mentions "mypy"'
    body = step.group(2)
    assert "scripts/type-check.sh" in body, (
        "the CI type-check step must run scripts/type-check.sh, so it and "
        f"`make type-check` cannot diverge. Step body was:\n{body}"
    )
    assert "|| true" not in body, (
        f"CI swallows the type-check result. Step body was:\n{body}\n"
        "A gate whose failure is discarded is not a gate."
    )
    assert not re.search(r"^\s*mypy services/", workflow, re.MULTILINE), (
        "the CI workflow still runs the single-pass `mypy services/`, which "
        "cannot analyse this repo"
    )


def test_quality_target_still_includes_type_check():
    """`make quality` is the advertised umbrella; type-check must be under it."""
    text = _read(MAKEFILE)
    match = re.search(r"^quality:\s*(.*)$", text, re.MULTILINE)
    assert match, "Makefile has no `quality:` target"
    assert "type-check" in match.group(1), (
        "`make quality` no longer runs type-check; the README and CLAUDE.md "
        "both advertise it as one of the quality gates."
    )
