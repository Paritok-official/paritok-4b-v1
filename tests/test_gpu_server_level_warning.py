"""`level` is accepted, sent, and ignored by the hosted backend (#3).

Nothing on either side says so. A caller implementing the documented
L2 -> L1 -> L0 fallback ladder gets byte-identical output at every rung and
concludes their own logic is broken. One warning, once per strategy, removes an
afternoon of debugging.

Once, specifically: `compress()` is called per segment, so warning per call would
put thousands of identical lines through the proxy console -- the same reason
`_warn_invalid_key_once` exists.
"""

import logging

import httpx

from paritok.config import GpuServerConfig
from paritok.strategies.gpu_server import GpuServerStrategy

CONTENT = "def charge(amount):\n    return round(amount, 2)\n" * 20


def _strategy(monkeypatch):
    def fake_post(url, **kwargs):
        return httpx.Response(
            200,
            json={"compressed": "summary", "gpu_available": True},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    return GpuServerStrategy(GpuServerConfig(api_key="k"))


def test_passing_a_level_warns(monkeypatch, caplog):
    s = _strategy(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="paritok.gpu_server"):
        s.compress(CONTENT, query="q", level="L2")
    assert any("ignores it" in m for m in (r.getMessage() for r in caplog.records))


def test_it_warns_only_once_however_many_segments(monkeypatch, caplog):
    s = _strategy(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="paritok.gpu_server"):
        for _ in range(50):
            s.compress(CONTENT, query="q", level="L1")
    assert len(caplog.records) == 1


def test_no_warning_when_no_level_is_passed(monkeypatch, caplog):
    s = _strategy(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="paritok.gpu_server"):
        s.compress(CONTENT, query="q")
    assert caplog.records == []


def test_the_level_is_still_sent_unchanged(monkeypatch):
    """A warning, not a behaviour change: the payload is untouched."""
    seen = {}

    def fake_post(url, **kwargs):
        seen.update(kwargs["json"])
        return httpx.Response(
            200,
            json={"compressed": "summary", "gpu_available": True},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    GpuServerStrategy(GpuServerConfig(api_key="k")).compress(CONTENT, query="q", level="L0")
    assert seen["level"] == "L0"
