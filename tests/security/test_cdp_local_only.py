"""SEC-11..15: CDP attach is local-only.

Probe surface: browser-engine ``POST /tabs/import`` and ``POST /tabs/open``.
Non-local endpoints are refused with an HTTP 400 validation error before any
connection attempt. WI0-B2 note: the exact accepted local-host set is NOT
frozen (the wk0 CDP fix may change it) -- only "non-local refused, local
loopback accepted at validation".
"""

import socket

import pytest

pytestmark = [pytest.mark.security]

CDP_ENDPOINTS = ["/tabs/import", "/tabs/open"]


def _cdp_request(client, endpoint, cdp_url, *, urls=None):
    if endpoint == "/tabs/open":
        body = {"urls": urls or ["http://example.com/"], "cdp_url": cdp_url}
    else:
        body = {"cdp_url": cdp_url}
    return client.post(endpoint, token=client.token, json=body)


NON_LOCAL_CDP = [
    "http://example.com:9222",
    "http://192.168.1.5:9222",
    "http://evil.internal:9222",
]

MALFORMED_CDP = [
    "http://localhost:9222/json",
    "http://localhost:9222?x=1",
    "http://user:p@localhost:9222",
]

BAD_SCHEME_CDP = [
    "ws://localhost:9222",
    "ftp://localhost:9222",
]


@pytest.mark.parametrize("endpoint", CDP_ENDPOINTS)
@pytest.mark.parametrize("cdp_url", NON_LOCAL_CDP)
def test_sec11_non_local_cdp_refused(browser, endpoint, cdp_url):
    response = _cdp_request(browser, endpoint, cdp_url)
    assert response.status_code == 400, f"{endpoint}: {cdp_url} not refused"


@pytest.mark.parametrize("endpoint", CDP_ENDPOINTS)
@pytest.mark.parametrize("cdp_url", MALFORMED_CDP)
def test_sec12_cdp_path_query_userinfo_refused(browser, endpoint, cdp_url):
    response = _cdp_request(browser, endpoint, cdp_url)
    assert response.status_code == 400, f"{endpoint}: {cdp_url} not refused"


@pytest.mark.parametrize("endpoint", CDP_ENDPOINTS)
@pytest.mark.parametrize("cdp_url", BAD_SCHEME_CDP)
def test_sec13_cdp_scheme_allowlist(browser, endpoint, cdp_url):
    response = _cdp_request(browser, endpoint, cdp_url)
    assert response.status_code == 400, f"{endpoint}: {cdp_url} not refused"


def test_sec14_local_cdp_passes_validation_then_connect_error(browser):
    """Valid-local must be distinguishable from a policy refusal: not a 400."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    unused_port = sock.getsockname()[1]
    sock.close()  # nothing is listening now -> connection-class failure

    response = browser.post(
        "/tabs/import",
        token=browser.token,
        json={"cdp_url": f"http://127.0.0.1:{unused_port}"},
    )
    assert response.status_code != 400, "valid-local endpoint was policy-refused"
    assert response.status_code in (500, 502)


def test_sec15_tabs_open_targets_pass_scrape_policy(browser, canary_listener):
    response = browser.post(
        "/tabs/open",
        token=browser.token,
        json={
            "urls": [f"http://127.0.0.1:{canary_listener.port}/"],
            "cdp_url": "http://127.0.0.1:9222",
        },
    )
    assert response.status_code == 400
    assert canary_listener.count == 0
