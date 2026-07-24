"""SEC-39 (WI0-B6): auth-wall behavioral fixtures.

Seam (``sec_seam``): ``AuthDetector.detect(url, status_code, headers, html)`` --
chosen because the black-box ``/detect-auth`` endpoint does not accept
``status_code``/``headers``, and the 403 branch is exactly what misfired in
WI0. TS porting rule: feed the same JSON fixtures to the TS auth-classifier
seam and apply the same assertions.

Challenge fixtures previously carried ``xfail(reason="WI0-B6 open")`` to
document the live bug (public bot-challenge pages classified as
credential-promptable auth walls). WI0-B6 is now closed: ``AuthDetector``
classifies Cloudflare/PerimeterX/generic-403 challenges as
``requires_auth=False`` (``blocked=True``, ``auth_type="bot_challenge"``), so
the marker is deleted and the cases assert for real. The real-auth
counter-fixtures guard against a fix that merely neuters all 403 handling.
"""

import pytest

from services.browser_engine.app.auth.detector import AuthDetector

pytestmark = [pytest.mark.security, pytest.mark.sec_seam]

CHALLENGE_FIXTURES = [
    "npmjs_cloudflare_challenge",
    "anthropic_docs_403_challenge",
    "perimeterx_denied_challenge",
]

REAL_AUTH_FIXTURES = [
    "basic_401_real",
    "regwall_form_real",
    "login_redirect_real",
]


def _detect(fixture: dict):
    detector = AuthDetector()
    return detector.detect(
        url=fixture["url"],
        status_code=fixture.get("status_code"),
        headers=fixture.get("headers"),
        html=fixture.get("html"),
    )


@pytest.mark.parametrize("name", CHALLENGE_FIXTURES)
def test_challenge_pages_are_not_auth_walls(name, load_fixture):
    fixture = load_fixture(name)
    result = _detect(fixture)
    assert result.requires_auth is False, (
        f"{name}: public bot-challenge page must not enter the credential queue"
    )


@pytest.mark.parametrize("name", REAL_AUTH_FIXTURES)
def test_real_auth_pages_still_detected(name, load_fixture):
    fixture = load_fixture(name)
    result = _detect(fixture)
    assert result.requires_auth is True, (
        f"{name}: genuine auth wall must still be detected"
    )
    expected_type = fixture.get("expect_auth_type")
    if expected_type:
        assert result.auth_type == expected_type
