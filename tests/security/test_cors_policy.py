"""SEC-43: browser-origin (CORS) policy on all three services.

All three services used to run ``allow_origins=["*"]`` together with
``allow_credentials=True``. Verified live before the fix: a preflight claiming
``Origin: https://evil.example`` came back 200 echoing that origin with
``access-control-allow-credentials: true``. Combined with the unauthenticated
read endpoints on backend-core, that is a full cross-site corpus-exfiltration
primitive -- any page the user has open while the stack runs can fetch
``/api/v1/sessions`` then ``/api/v1/urls/{id}``.

None of these services has a legitimate browser-side caller: the Streamlit UI
calls them from the server side with ``requests``; the user's browser only ever
talks to Streamlit on :8089. The policy is therefore an explicit origin
allowlist with credentials disabled.

The suite had no probe that issued an ``Origin`` header anywhere before this,
so the hole was invisible to CI. These probes are pure HTTP and port directly
to the TypeScript implementation.
"""

import os

import pytest

from tests.security import contracts

pytestmark = [pytest.mark.security]

FIXTURE = "cors_policy"
_CORS = contracts.load_fixture(FIXTURE)
_SHARED = _CORS["shared"]
_CHECKS = _CORS["probes"]["SEC-43"]["checks"]

FOREIGN_ORIGIN = _SHARED["foreign_origin"]

# Same env contract the services read (services/cors.py). The TS port must
# honour the same variable and the same default.
CORS_ORIGINS_ENV = _SHARED["origins_env"]
DEFAULT_UI_ORIGIN = _SHARED["default_ui_origin"]

SERVICES = _SHARED["services"]
PROBE_PATH = _SHARED["probe_path"]

_CHECK_EXPECT_KEYS = {
    "access_control_allow_origin_not_in",
    "access_control_allow_origin_equals",
    "access_control_allow_credentials_not",
}


def _configured_origin() -> str:
    source = _CHECKS["configured_ui_origin_allowed"]["origin_source"]
    assert source["take"] == "first_non_empty_comma_separated_entry"
    assert source["strip_trailing_slash"] is True
    raw = os.getenv(_subst(source["env"]), "")
    for candidate in raw.split(","):
        candidate = candidate.strip().rstrip("/")
        if candidate:
            return candidate
    return _subst(source["default"])


def _subst(value: str, **extra) -> str:
    """Resolve a fixture placeholder against the shared block."""
    table = {
        "{foreign_origin}": FOREIGN_ORIGIN,
        "{origins_env}": CORS_ORIGINS_ENV,
        "{default_ui_origin}": DEFAULT_UI_ORIGIN,
        **{f"{{{k}}}": v for k, v in extra.items()},
    }
    if value in table:
        return table[value]
    if value.startswith("{") and value.endswith("}"):
        raise contracts.FixtureContractError(
            f"{FIXTURE}: unresolved placeholder {value!r}"
        )
    return value


def _assert_no_credentials(response, forbidden: str) -> None:
    allow_credentials = response.headers.get("access-control-allow-credentials")
    assert (allow_credentials or "").lower() != forbidden, (
        "access-control-allow-credentials must never be true: combined with a "
        "reflected origin it lets a foreign page read authenticated responses"
    )


def _run_check(client, service, check_name, **subs):
    """Issue one fixture-declared CORS check and apply its expectations."""
    check = _CHECKS[check_name]
    expect = check["expect"]
    unknown = set(expect) - _CHECK_EXPECT_KEYS
    if unknown:
        raise contracts.FixtureContractError(
            f"{FIXTURE}/{check_name}: unknown expectation keys {sorted(unknown)}"
        )
    headers = {k: _subst(v, **subs) for k, v in check["request_headers"].items()}
    response = client.request(
        check["method"], PROBE_PATH, token=None, headers=headers
    )
    allow_origin = response.headers.get("access-control-allow-origin")

    if "access_control_allow_origin_not_in" in expect:
        forbidden = [_subst(v, **subs) for v in expect["access_control_allow_origin_not_in"]]
        assert allow_origin not in forbidden, (
            f"{service}/{check_name} granted a forbidden origin "
            f"(access-control-allow-origin={allow_origin!r}, forbidden={forbidden!r})"
        )
    if "access_control_allow_origin_equals" in expect:
        required = _subst(expect["access_control_allow_origin_equals"], **subs)
        assert allow_origin == required, (
            f"{service} did not grant its own configured UI origin {required!r}; "
            "the allowlist is misconfigured or CORS was removed entirely"
        )
    assert "access_control_allow_credentials_not" in expect, (
        f"{FIXTURE}/{check_name}: every CORS check must forbid credentials"
    )
    _assert_no_credentials(response, expect["access_control_allow_credentials_not"])
    return response


@pytest.mark.parametrize("service", SERVICES)
def test_sec43_preflight_does_not_admit_foreign_origin(request, service):
    """A preflight from an arbitrary site must not be granted."""
    assert _CHECKS["foreign_origin_preflight_refused"]["managed_only"] is False
    _run_check(
        request.getfixturevalue(service), service, "foreign_origin_preflight_refused"
    )


@pytest.mark.parametrize("service", SERVICES)
def test_sec43_simple_request_does_not_echo_foreign_origin(request, service):
    """The actual (non-preflight) response must not be readable cross-site."""
    assert _CHECKS["foreign_origin_simple_request_not_echoed"]["managed_only"] is False
    _run_check(
        request.getfixturevalue(service),
        service,
        "foreign_origin_simple_request_not_echoed",
    )


@pytest.mark.sec_managed
@pytest.mark.parametrize("service", SERVICES)
def test_sec43_configured_ui_origin_is_still_allowed(request, service):
    """Non-vacuity: the policy is an allowlist, not a removed middleware.

    Managed-only because attached mode cannot know the running server's
    configured origin.
    """
    assert _CHECKS["configured_ui_origin_allowed"]["managed_only"] is True
    _run_check(
        request.getfixturevalue(service),
        service,
        "configured_ui_origin_allowed",
        configured_origin=_configured_origin(),
    )
