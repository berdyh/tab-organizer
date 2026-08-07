"""Loader + strict assertion runners for the language-neutral probe fixtures.

Plan decision 44 (``docs/ARCHITECTURE_PLAN.md``, Addendum 2026-08-05) is the
hedge against deferring ``SEC_BOOT_*_CMD`` boot mode to ~wk8: every
``sec_managed`` probe's INPUTS and EXPECTED REFUSALS live in
``tests/security/fixtures/*.json`` instead of inline in Python. Boot mode then
becomes a runner over data, a TypeScript port can consume the same files, and
softening the contract to fit whatever got built is a reviewable fixture diff
rather than a quiet edit inside a test body.

The runners below are deliberately STRICT: an unrecognised key inside a
contract block raises instead of being ignored, and an empty contract block
raises. A generic runner that silently skips keys it does not understand is
exactly how a frozen contract gets weakened by accident — one typo would turn
an assertion into a no-op, and the probe would still pass.

Contract block vocabulary
-------------------------

``argv`` (list of entries)
    ``{"flag": "-s", "value": "read-only"}`` — flag present AND the next argv
    element equals ``value``. ``{"flag": "--ephemeral"}`` — present only.

``text`` (object; used for stdin, argv values, response bodies)
    ``equals``            exact string equality
    ``starts_with``       prefix
    ``contains``          list of case-sensitive substrings
    ``contains_ci``       list of case-insensitive substrings
    ``not_contains``      list of case-sensitive substrings that must be absent
    ``ordered``           list of ``{"first", "second", "first_case_insensitive"?}``;
                          both must be present and ``first`` must occur earlier
    ``absent_or_after``   list of ``{"needle", "after"}``; ``needle`` is either
                          absent entirely or occurs strictly after ``after``

``response`` (object)
    ``status``            exact status code
    ``status_min``        status code >= this
    ``status_not``        status code != this
    ``text``              a nested ``text`` contract applied to the body
"""

from __future__ import annotations

import json
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"

_ARGV_KEYS = {"flag", "value"}
_TEXT_KEYS = {
    "equals",
    "starts_with",
    "contains",
    "contains_ci",
    "not_contains",
    "ordered",
    "absent_or_after",
}
_ORDERED_KEYS = {"first", "second", "first_case_insensitive"}
_ABSENT_OR_AFTER_KEYS = {"needle", "after"}
_RESPONSE_KEYS = {"status", "status_min", "status_not", "text"}


class FixtureContractError(AssertionError):
    """A fixture file is malformed, incomplete, or uses an unknown key."""


def load_fixture(name: str) -> dict:
    """Load ``fixtures/<name>.json``; raise if missing or empty."""
    path = FIXTURES_DIR / f"{name}.json"
    if not path.is_file():
        raise FixtureContractError(f"security fixture file missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data.get("probes"):
        raise FixtureContractError(f"security fixture has no `probes` block: {path}")
    return data


def probe_spec(fixture_name: str, probe_id: str) -> dict:
    """Return one probe's spec, raising if the fixture does not carry it."""
    probes = load_fixture(fixture_name)["probes"]
    if probe_id not in probes:
        raise FixtureContractError(
            f"{probe_id} has no entry in fixtures/{fixture_name}.json "
            f"(known: {sorted(probes)})"
        )
    spec = probes[probe_id]
    if not isinstance(spec, dict) or not spec:
        raise FixtureContractError(f"{probe_id} in {fixture_name}.json is empty")
    return spec


def expectations(spec: dict, probe_id: str) -> dict:
    """Return a probe spec's ``expect`` block, raising if it is missing/empty."""
    expect = spec.get("expect")
    if not isinstance(expect, dict) or not expect:
        raise FixtureContractError(f"{probe_id} carries no non-empty `expect` block")
    return expect


def _strip_notes(mapping: dict) -> dict:
    """Drop documentation-only keys (``_comment``, ``_note``, ...).

    Every fixture in this directory carries prose next to its data under an
    underscore-prefixed key. Those are the ONLY keys a runner may ignore; every
    other unrecognised key raises, so a mistyped assertion cannot go silently
    unenforced.
    """
    return {k: v for k, v in mapping.items() if not k.startswith("_")}


def assert_expect_keys_consumed(expect: dict, handled, *, label: str) -> None:
    """Fail if the fixture declares an expectation the probe never checks.

    This is the counterpart to the strict key vocabulary below. Without it a
    probe could be rewired to read only some of its fixture's ``expect`` keys
    and the rest would become decorative — the contract would look extracted
    while half of it no longer ran.
    """
    unhandled = set(_strip_notes(expect)) - set(handled)
    if unhandled:
        raise FixtureContractError(
            f"{label}: fixture declares expectations the probe does not check: "
            f"{sorted(unhandled)}"
        )


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def assert_argv(argv: list, contract: list, *, label: str) -> None:
    """Apply an ``argv`` contract to a recorded subprocess argv."""
    if not isinstance(contract, list) or not contract:
        raise FixtureContractError(f"{label}: empty argv contract")
    for entry in contract:
        if not isinstance(entry, dict):
            raise FixtureContractError(f"{label}: argv entry is not an object: {entry!r}")
        entry = _strip_notes(entry)
        unknown = set(entry) - _ARGV_KEYS
        if unknown:
            raise FixtureContractError(
                f"{label}: unknown argv-contract keys {sorted(unknown)}"
            )
        if "flag" not in entry:
            raise FixtureContractError(f"{label}: argv entry has no `flag`: {entry!r}")
        flag = entry["flag"]
        assert flag in argv, f"{label}: required flag {flag!r} absent from argv: {argv}"
        if "value" in entry:
            index = argv.index(flag)
            assert index + 1 < len(argv), (
                f"{label}: flag {flag!r} has no following value in argv: {argv}"
            )
            assert argv[index + 1] == entry["value"], (
                f"{label}: {flag!r} carried {argv[index + 1]!r}, "
                f"contract requires {entry['value']!r} (argv: {argv})"
            )


def assert_text(text: str, contract: dict, *, label: str) -> None:
    """Apply a ``text`` contract to an observed string."""
    if not isinstance(contract, dict) or not contract:
        raise FixtureContractError(f"{label}: empty text contract")
    contract = _strip_notes(contract)
    if not contract:
        raise FixtureContractError(f"{label}: text contract carries no assertions")
    unknown = set(contract) - _TEXT_KEYS
    if unknown:
        raise FixtureContractError(
            f"{label}: unknown text-contract keys {sorted(unknown)}"
        )

    lower = text.lower()

    if "equals" in contract:
        assert text == contract["equals"], (
            f"{label}: expected exactly {contract['equals']!r}, got {text[:200]!r}"
        )
    if "starts_with" in contract:
        assert text.startswith(contract["starts_with"]), (
            f"{label}: must start with {contract['starts_with']!r}, "
            f"got {text[:120]!r}"
        )
    for needle in contract.get("contains", []):
        assert needle in text, f"{label}: missing required text {needle!r}"
    for needle in contract.get("contains_ci", []):
        assert needle.lower() in lower, (
            f"{label}: missing required text {needle!r} (case-insensitive)"
        )
    for needle in contract.get("not_contains", []):
        assert needle not in text, f"{label}: forbidden text {needle!r} present"

    for entry in contract.get("ordered", []):
        entry = _strip_notes(entry)
        unknown = set(entry) - _ORDERED_KEYS
        if unknown:
            raise FixtureContractError(
                f"{label}: unknown `ordered` keys {sorted(unknown)}"
            )
        first, second = entry["first"], entry["second"]
        if entry.get("first_case_insensitive"):
            assert first.lower() in lower, f"{label}: ordering anchor {first!r} absent"
            first_pos = lower.index(first.lower())
        else:
            assert first in text, f"{label}: ordering anchor {first!r} absent"
            first_pos = text.index(first)
        assert second in text, f"{label}: ordering anchor {second!r} absent"
        second_pos = text.index(second)
        assert first_pos < second_pos, (
            f"{label}: {first!r} (at {first_pos}) must precede {second!r} "
            f"(at {second_pos})"
        )

    for entry in contract.get("absent_or_after", []):
        entry = _strip_notes(entry)
        unknown = set(entry) - _ABSENT_OR_AFTER_KEYS
        if unknown:
            raise FixtureContractError(
                f"{label}: unknown `absent_or_after` keys {sorted(unknown)}"
            )
        needle, after = entry["needle"], entry["after"]
        assert after in text, f"{label}: ordering anchor {after!r} absent"
        after_pos = text.index(after)
        needle_pos = text.find(needle)
        assert needle_pos == -1 or needle_pos > after_pos, (
            f"{label}: {needle!r} must be absent or appear only after "
            f"{after!r} (found at {needle_pos}, anchor at {after_pos})"
        )


def assert_response(response, contract: dict, *, label: str) -> None:
    """Apply a ``response`` contract to an httpx response."""
    if not isinstance(contract, dict) or not contract:
        raise FixtureContractError(f"{label}: empty response contract")
    contract = _strip_notes(contract)
    if not contract:
        raise FixtureContractError(f"{label}: response contract carries no assertions")
    unknown = set(contract) - _RESPONSE_KEYS
    if unknown:
        raise FixtureContractError(
            f"{label}: unknown response-contract keys {sorted(unknown)}"
        )
    if "status" in contract:
        assert response.status_code == contract["status"], (
            f"{label}: expected HTTP {contract['status']}, "
            f"got {response.status_code} {response.text[:200]}"
        )
    if "status_min" in contract:
        assert response.status_code >= contract["status_min"], (
            f"{label}: expected HTTP >= {contract['status_min']}, "
            f"got {response.status_code} {response.text[:200]}"
        )
    if "status_not" in contract:
        assert response.status_code != contract["status_not"], (
            f"{label}: HTTP {contract['status_not']} is forbidden here"
        )
    if "text" in contract:
        assert_text(response.text, contract["text"], label=f"{label} body")
