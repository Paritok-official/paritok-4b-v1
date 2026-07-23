"""Regression for issue #2: GpuServerStrategy.check() dropped the API key, so an
auth-gated /test made `paritok up` falsely report the hosted endpoint as
unreachable even with a valid key. check() now sends Authorization like
compress(), distinguishes 401/403 (key) from other failures, and parses JSON
defensively.
"""
import httpx

from paritok.config import GpuServerConfig
from paritok.strategies.gpu_server import GpuServerStrategy


def _strategy(api_key="pk_live_test"):
    cfg = GpuServerConfig(
        base_url="https://www.paritok.com/api", model="paritok-4b-v1",
        api_key=api_key, timeout=10.0,
    )
    return GpuServerStrategy(cfg)


class _Resp:
    def __init__(self, status=200, json_data=None, raise_json=False):
        self.status_code = status
        self._json = json_data if json_data is not None else {}
        self._raise_json = raise_json

    def json(self):
        if self._raise_json:
            raise ValueError("not json")
        return self._json


def _patch_get(monkeypatch, resp=None, exc=None, capture=None):
    def fake_get(url, headers=None, timeout=None):
        if capture is not None:
            capture["url"] = url
            capture["headers"] = headers or {}
            capture["timeout"] = timeout
        if exc is not None:
            raise exc
        return resp
    monkeypatch.setattr(httpx, "get", fake_get)


def test_check_sends_authorization_header(monkeypatch):
    cap = {}
    _patch_get(monkeypatch, resp=_Resp(200, {"gpu_available": True, "message": "ok"}), capture=cap)
    available, msg = _strategy("pk_live_abc").check()
    assert available is True and msg == "ok"
    assert cap["headers"].get("Authorization") == "Bearer pk_live_abc"
    assert cap["url"].endswith("/test")


def test_check_no_key_sends_no_auth_header(monkeypatch):
    cap = {}
    _patch_get(monkeypatch, resp=_Resp(200, {"gpu_available": True}), capture=cap)
    _strategy(api_key="").check()
    assert "Authorization" not in cap["headers"]


def test_check_401_reports_rejected_key(monkeypatch):
    _patch_get(monkeypatch, resp=_Resp(401))
    available, msg = _strategy("pk_live_bad").check()
    assert available is False
    assert "401" in msg and "API key" in msg and "rejected" in msg


def test_check_403_without_key_says_no_key_sent(monkeypatch):
    _patch_get(monkeypatch, resp=_Resp(403))
    available, msg = _strategy(api_key="").check()
    assert available is False
    assert "no API key was sent" in msg


def test_check_gpu_offline_returns_false_with_server_message(monkeypatch):
    _patch_get(monkeypatch, resp=_Resp(200, {"gpu_available": False, "message": "GPU rebooting"}))
    available, msg = _strategy().check()
    assert available is False and msg == "GPU rebooting"


def test_check_gpu_available_true(monkeypatch):
    _patch_get(monkeypatch, resp=_Resp(200, {"gpu_available": True, "message": "online"}))
    assert _strategy().check() == (True, "online")


def test_check_non_json_response_handled(monkeypatch):
    _patch_get(monkeypatch, resp=_Resp(200, raise_json=True))
    available, msg = _strategy().check()
    assert available is False and "non-JSON" in msg


def test_check_5xx_returns_false(monkeypatch):
    _patch_get(monkeypatch, resp=_Resp(502))
    available, msg = _strategy().check()
    assert available is False and "502" in msg


def test_check_network_error_is_unreachable(monkeypatch):
    _patch_get(monkeypatch, exc=httpx.ConnectError("boom"))
    available, msg = _strategy().check()
    assert available is False and "Could not reach" in msg


def test_compress_invalid_key_passes_through_and_warns_once(monkeypatch, capsys):
    # /compress rejects a bad key with 401 -> content passes through uncompressed,
    # and the proxy console warns exactly once (not once per request).
    calls = {"n": 0}

    def fake_post(url, json=None, headers=None, timeout=None):
        calls["n"] += 1
        return _Resp(401)

    monkeypatch.setattr(httpx, "post", fake_post)
    s = _strategy("pk_live_bad")

    out1 = s.compress("hello world content", query="q")
    assert out1 == "hello world content"  # unchanged passthrough
    cap1 = capsys.readouterr().out
    assert "API key not valid" in cap1 and "401" in cap1

    out2 = s.compress("another chunk here", query="q")
    assert out2 == "another chunk here"
    cap2 = capsys.readouterr().out
    assert "API key not valid" not in cap2  # warned once, not again
    assert calls["n"] == 2  # but both requests were attempted


def test_compress_no_key_warning_says_no_key_configured(monkeypatch, capsys):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resp(403))
    out = _strategy(api_key="").compress("some content to compress", query="q")
    assert out == "some content to compress"
    assert "no API key configured" in capsys.readouterr().out
