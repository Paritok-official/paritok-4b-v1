"""/stats content-negotiates (#42): a browser gets the visual dashboard, every
programmatic caller (curl / hosted meter / Accept: */* or application/json) keeps
getting the JSON snapshot, so nothing that reads /stats today breaks."""
import re

import pytest
from starlette.testclient import TestClient

from paritok.proxy.server import create_app


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def test_stats_serves_html_dashboard_to_a_browser(client):
    r = client.get("/stats", headers={"accept": "text/html,application/xhtml+xml,*/*"})
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    body = r.text
    assert "<!doctype html>" in body.lower()
    assert "Paritok" in body
    assert 'id="chart"' in body          # the live token-savings chart
    assert 'headers:{Accept:"application/json"}' in body  # polls itself for JSON
    # the paged original→compressed before/after view (#42): one pair per page,
    # each fetched one at a time from /stats?sample=N
    assert "Recent compressions" in body
    assert 'id="sampleView"' in body
    assert 'id="pgLabel"' in body
    assert "/stats?sample=" in body


def test_dashboard_renders_sample_text_via_textcontent_not_innerhtml(client):
    """Sample fields (source/model/original/compressed) are attacker-influenced file
    content and paths — they must reach the DOM only via textContent, never HTML.
    Guard the invariant so a future 'let's syntax-highlight with innerHTML' edit
    can't silently reintroduce stored XSS (finding: test-adequacy-xss)."""
    body = client.get("/stats", headers={"accept": "text/html"}).text
    assert "e.textContent=txt" in body  # the el() helper sets user text as textContent
    for expr in re.findall(r"innerHTML\s*=\s*([^;\n]+)", body):
        for field in ("s.original", "s.compressed", "s.source", "s.model", ".sample"):
            assert field not in expr, f"sample data must not flow through innerHTML: {expr!r}"


def test_stats_serves_json_to_programmatic_callers(client):
    # explicit JSON, wildcard, and no Accept header must all still return JSON
    for headers in ({"accept": "application/json"}, {"accept": "*/*"}, {}):
        r = client.get("/stats", headers=headers)
        assert r.status_code == 200
        assert "application/json" in r.headers["content-type"], headers
        d = r.json()
        for k in ("total_requests", "tokens_saved", "compression_ratio",
                  "uptime_seconds", "compression_samples_count"):
            assert k in d, (k, headers)
        assert isinstance(d["uptime_seconds"], (int, float))
        assert isinstance(d["compression_samples_count"], int)
