"""Both hosted calls must identify the client.

The endpoint rejects some default User-Agents with HTTP 403 and an empty body.
`urllib.request` is one of them, which is the natural choice for a
dependency-free repro script -- and 403 is the same status a rejected API key
returns, so the client prints "Paritok API key not valid" at someone whose key is
perfectly good.

Sending an explicit agent removes that failure mode and makes hosted traffic
attributable. `check()` matters as much as `compress()` here: it is the call that
runs at startup, so it is the one most likely to produce the misleading message.
"""

import httpx

from paritok import __version__
from paritok.config import GpuServerConfig
from paritok.strategies.gpu_server import GpuServerStrategy


def _capture(monkeypatch, method):
    seen = {}

    def fake(url, **kwargs):
        seen["headers"] = kwargs.get("headers") or {}
        return httpx.Response(
            200,
            json={"compressed": "x", "gpu_available": True, "message": "ok"},
            request=httpx.Request(method.upper(), url),
        )

    monkeypatch.setattr(httpx, method, fake)
    return seen


def test_compress_sends_an_explicit_user_agent(monkeypatch):
    seen = _capture(monkeypatch, "post")
    GpuServerStrategy(GpuServerConfig(api_key="k")).compress("x" * 400, query="q")
    assert seen["headers"].get("User-Agent", "").startswith("paritok/")


def test_check_sends_an_explicit_user_agent(monkeypatch):
    """startup path: the one that produces the misleading key warning"""
    seen = _capture(monkeypatch, "get")
    GpuServerStrategy(GpuServerConfig(api_key="k")).check()
    assert seen["headers"].get("User-Agent", "").startswith("paritok/")


def test_user_agent_carries_the_version(monkeypatch):
    seen = _capture(monkeypatch, "post")
    GpuServerStrategy(GpuServerConfig(api_key="k")).compress("x" * 400, query="q")
    assert __version__ in seen["headers"]["User-Agent"]


def test_authorization_is_still_sent(monkeypatch):
    seen = _capture(monkeypatch, "post")
    GpuServerStrategy(GpuServerConfig(api_key="secret")).compress("x" * 400, query="q")
    assert seen["headers"]["Authorization"] == "Bearer secret"


def test_user_agent_is_sent_even_without_a_key(monkeypatch):
    seen = _capture(monkeypatch, "post")
    GpuServerStrategy(GpuServerConfig(api_key="")).compress("x" * 400, query="q")
    assert "User-Agent" in seen["headers"]
    assert "Authorization" not in seen["headers"]
