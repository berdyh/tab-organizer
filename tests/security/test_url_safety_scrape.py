"""SEC-1..10: outbound URL-safety policy at the scrape entry points.

Every hermetic case is parametrised over both entry points: browser-engine
``POST /scrape/single`` and backend-core ``POST /api/v1/urls``. The refusal is
HTTP 400 with a policy message; where a canary is addressable, zero canary
connections are additionally asserted.
"""

import os

import pytest

from tests.security import contracts

pytestmark = [pytest.mark.security]

ENTRIES = ["browser", "backend"]

NETWORK_GATED = pytest.mark.skipif(
    os.getenv("SEC_ALLOW_NETWORK") != "1", reason="needs internet"
)


def _probe(request, entry, url, *, use_browser=False):
    """Submit a single URL through the named entry point; return the response."""
    if entry == "browser":
        client = request.getfixturevalue("browser")
        body = {"url": url, "use_browser": use_browser}
        return client, client.post("/scrape/single", token=client.token, json=body)
    client = request.getfixturevalue("backend")
    return client, client.post("/api/v1/urls", json={"urls": [url]})


SCHEME_URLS = [
    "ftp://example.com/x",
    "file:///etc/passwd",
    "gopher://x/",
    "javascript:alert(1)",
    "data:text/html,x",
]

PRIVATE_IP_URLS = [
    "http://127.0.0.1/",
    "http://10.0.0.1/",
    "http://172.16.0.1/",
    "http://192.168.1.1/",
    "http://169.254.169.254/latest/meta-data/",
    "http://0.0.0.0/",
    "http://224.0.0.1/",
    "http://240.0.0.1/",
    "http://[::1]/",
    "http://[fe80::1]/",
    "http://[fc00::1]/",
]

INTERNAL_HOST_URLS = [
    "http://localhost/",
    "http://foo.localhost/",
    "http://printer.local/",
    "http://svc.internal/",
    "http://host.docker.internal/",
    "http://intranet/",
]


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("url", SCHEME_URLS)
def test_sec1_scheme_allowlist(request, entry, url):
    _client, response = _probe(request, entry, url)
    assert response.status_code == 400, f"{entry}: {url} was not refused"


@pytest.mark.parametrize("entry", ENTRIES)
def test_sec2_userinfo_refused(request, entry):
    _client, response = _probe(request, entry, "https://user:pass@example.com/")
    assert response.status_code == 400


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("url", PRIVATE_IP_URLS)
def test_sec3_private_ip_literals_refused(request, entry, url):
    _client, response = _probe(request, entry, url)
    assert response.status_code == 400, f"{entry}: {url} was not refused"


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("url", INTERNAL_HOST_URLS)
def test_sec4_internal_hostnames_refused(request, entry, url):
    _client, response = _probe(request, entry, url)
    assert response.status_code == 400, f"{entry}: {url} was not refused"


@pytest.mark.parametrize("entry", ENTRIES)
def test_sec5_no_connect_guarantee_single(request, entry, canary_listener):
    url = f"http://127.0.0.1:{canary_listener.port}/"
    _client, response = _probe(request, entry, url)
    assert response.status_code == 400
    assert canary_listener.count == 0, "a connection was attempted to a refused target"


def test_sec5_no_connect_guarantee_batch(browser, canary_listener):
    url = f"http://127.0.0.1:{canary_listener.port}/"
    response = browser.post(
        "/scrape",
        token=browser.token,
        json={"session_id": "sec5-batch", "urls": [url]},
    )
    # Whole-batch validation rejects the unsafe target before dispatch.
    assert response.status_code == 400
    assert canary_listener.count == 0


@NETWORK_GATED
@pytest.mark.integration
def test_sec6_dns_resolved_private_refused(browser, canary_listener):
    url = f"http://127.0.0.1.nip.io:{canary_listener.port}/"
    response = browser.post(
        "/scrape/single", token=browser.token, json={"url": url}
    )
    assert response.status_code == 400 or response.json().get("status") != "success"
    assert canary_listener.count == 0


@NETWORK_GATED
@pytest.mark.integration
def test_sec7_redirect_to_private_refused(browser, canary_listener):
    target = f"http://127.0.0.1:{canary_listener.port}/"
    url = f"https://httpbingo.org/redirect-to?url={target}"
    response = browser.post(
        "/scrape/single", token=browser.token, json={"url": url}
    )
    assert response.status_code == 200
    assert response.json().get("status") != "success"
    assert canary_listener.count == 0


@NETWORK_GATED
@pytest.mark.integration
def test_sec8_redirect_chain_bounded(browser):
    url = "https://httpbingo.org/absolute-redirect/15"
    response = browser.post(
        "/scrape/single", token=browser.token, json={"url": url}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload.get("status") != "success"
    assert "redirect" in (payload.get("error") or "").lower()


@NETWORK_GATED
@pytest.mark.integration
def test_sec9_dns_rebinding_pin(browser, canary_listener):
    """Residual gap: full flip-flop rebinding needs an attacker resolver; this
    hermetic-adjacent probe asserts the refusal surface across attempts."""
    url = f"http://7f000001.rebind.example:{canary_listener.port}/"
    for _ in range(5):
        response = browser.post(
            "/scrape/single", token=browser.token, json={"url": url}
        )
        assert response.status_code == 400 or response.json().get("status") != "success"
    assert canary_listener.count == 0


@pytest.mark.sec_managed
def test_sec10_escape_hatch_default_off(browser, canary_listener, monkeypatch):
    """SCRAPE_ALLOW_PRIVATE_NETWORKS is off by default and really opens when set.

    ``sec_managed``: proving the DEFAULT means deleting the variable from the
    service's environment, which attached mode cannot do — so the case lives in
    ``fixtures/url_safety_escape_hatch.json`` (plan decision 44) for the planned
    ``SEC_BOOT_*_CMD`` runner to drive against any language. Its siblings
    SEC-1..9 need no controlled env and stay inline.
    """
    spec = contracts.probe_spec("url_safety_escape_hatch", "SEC-10")
    staging = spec["staging"]
    assert staging["service"] == "browser"
    url = spec["input"]["url_template"].format(canary_port=canary_listener.port)

    steps = spec["steps"]
    assert [step["step"] for step in steps] == ["hatch_absent", "hatch_set"], (
        "SEC-10 must run both halves: default refused, flag-set permitted"
    )
    required_canary_key = {
        "hatch_absent": "canary_connections",
        "hatch_set": "canary_connections_min",
    }

    for step in steps:
        label = f"SEC-10[{step['step']}]"
        expect = step["expect"]
        contracts.assert_expect_keys_consumed(
            expect,
            ["response", "canary_connections", "canary_connections_min"],
            label=label,
        )
        assert required_canary_key[step["step"]] in expect, (
            f"{label}: the canary assertion is what makes 'refused' mean "
            f"'never dialled'; it may not be dropped"
        )

        for name in step.get("unset_env", []):
            monkeypatch.delenv(name, raising=False)
        for name, value in (step.get("set_env") or {}).items():
            monkeypatch.setenv(name, value)

        response = browser.request(
            staging["method"], staging["path"], token=browser.token, json={"url": url}
        )
        contracts.assert_response(response, expect["response"], label=label)
        if "canary_connections" in expect:
            assert canary_listener.count == expect["canary_connections"], (
                f"{label}: a connection was attempted to a refused target"
            )
        if "canary_connections_min" in expect:
            assert canary_listener.count >= expect["canary_connections_min"], (
                f"{label}: flag did not permit the loopback fetch"
            )
