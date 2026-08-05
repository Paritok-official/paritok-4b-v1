"""`check()` asks /test; `verify()` asks the endpoint that actually does the work.

/test has been observed reporting `gpu_available: true` while /compress returned
`gpu_available: false` and passed every request through uncompressed. Startup
reports healthy, traffic is silently uncompressed, and nothing errors -- so the
one case these tests care about is the disagreement: `check()` says yes and
`verify()` says no.
"""

import httpx

from paritok.config import GpuServerConfig
from paritok.strategies.gpu_server import GpuServerStrategy


def _endpoint(monkeypatch, *, test_says=True, compress_returns=""):
    """Wire /test and /compress independently so they can disagree."""

    def fake_get(url, **kwargs):
        return httpx.Response(
            200,
            json={"gpu_available": test_says, "message": "ok"},
            request=httpx.Request("GET", url),
        )

    def fake_post(url, **kwargs):
        body = kwargs["json"]["content"] if compress_returns is PASSTHROUGH else compress_returns
        return httpx.Response(
            200,
            json={"compressed": body, "gpu_available": True},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(httpx, "post", fake_post)
    return GpuServerStrategy(GpuServerConfig(api_key="k"))


PASSTHROUGH = object()


def test_empty_canary_means_relevance_filtering_is_running(monkeypatch):
    s = _endpoint(monkeypatch, compress_returns="")
    ok, msg = s.verify()
    assert ok
    assert "compressing" in msg


def test_a_compressed_canary_also_counts_as_working(monkeypatch):
    s = _endpoint(monkeypatch, compress_returns="banner helpers")
    ok, _ = s.verify()
    assert ok


def test_passthrough_is_caught_even_though_test_reports_healthy(monkeypatch):
    """The reported defect, in one assertion pair."""
    s = _endpoint(monkeypatch, test_says=True, compress_returns=PASSTHROUGH)

    available, _ = s.check()
    assert available is True          # /test is happy

    ok, msg = s.verify()
    assert ok is False                # /compress is not
    assert "NOT compressing" in msg


def test_verify_does_not_contradict_check_when_the_backend_is_fine(monkeypatch):
    s = _endpoint(monkeypatch, test_says=True, compress_returns="")
    assert s.check()[0] is True
    assert s.verify()[0] is True
