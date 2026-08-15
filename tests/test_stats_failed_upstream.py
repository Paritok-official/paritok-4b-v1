"""Regression for issue #19: /stats must not credit savings to a request whose
upstream forward failed (e.g. Groq free-tier 429 TPM). The proxy compresses the
history BEFORE it forwards, so a naive counter folds the "savings" in even though
the agent never got a usable completion — inflating tokens_saved / total_requests
over the real successful-answer path.

Both tests drive the real Starlette app on /v1/chat/completions with a mock
upstream (success in one, HTTP 429 in the other), a tiny history window so the
threshold trips on a small fixture, and a stubbed 4B model (unavailable in CI).
The compression work is identical across the two — only the upstream outcome
differs — so the contrast isolates the accounting behaviour.
"""
import pytest

pytest.importorskip("starlette")
from starlette.testclient import TestClient  # noqa: E402

MODEL = "gpt-4o"
SUMMARY = "SUMMARY_OF_OLD_HISTORY"

CONFIG_YAML = """\
use_gpu_server: false
history:
  enabled: true
  keep_recent_turns: 2
  context_threshold: 0.5
  context_window: 1000
tool_discovery:
  strategy: passthrough
"""


class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload, self.status_code, self.headers = payload, status, {}

    def json(self):
        return self._payload


class _Upstream:
    """Records forwarded bodies; replies 200 with a plain turn, or a given error."""

    def __init__(self, status=200):
        self.status = status
        self.calls = []

    async def post(self, url, headers=None, json=None):
        self.calls.append(json)
        if 200 <= self.status < 300:
            return _FakeResp({
                "id": "chatcmpl-x", "object": "chat.completion",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "done"}}],
            })
        # Groq free tier: 429 with a rate-limit error body and no choices.
        return _FakeResp(
            {"error": {"message": "rate limit reached", "type": "tokens",
                       "code": "rate_limit_exceeded"}},
            status=self.status,
        )

    async def aclose(self):
        pass


def _long_history():
    """6 user/assistant turns; the first 4 are old, the last 2 recent. Big enough
    to exceed the tiny 500-token (1000 * 0.5) history threshold so compression fires."""
    filler = "context filler text " * 60
    msgs = []
    for t in range(6):
        marker = f"OLD_MARKER_{t}" if t < 4 else "RECENT_MARKER"
        msgs.append({"role": "user", "content": f"{marker} question {t} {filler}"})
        msgs.append({"role": "assistant", "content": f"{marker} answer {t} {filler}"})
    return msgs


def _app(tmp_path, monkeypatch, upstream):
    monkeypatch.setattr(
        "paritok.strategies.local_model.LocalModelStrategy.compress",
        lambda self, content, **kw: SUMMARY,
    )
    cfg = tmp_path / "paritok.yaml"
    cfg.write_text(CONFIG_YAML, encoding="utf-8")
    from paritok.proxy import server
    return server.create_app(config_path=str(cfg),
                             openai_base_url="http://upstream.test",
                             http_client=upstream)


def test_failed_upstream_is_not_credited(tmp_path, monkeypatch):
    upstream = _Upstream(status=429)
    app = _app(tmp_path, monkeypatch, upstream)

    with TestClient(app) as client:
        r = client.post("/v1/chat/completions",
                        json={"model": MODEL, "messages": _long_history()})
        assert r.status_code == 429, "the upstream 429 must be relayed to the caller"
        stats = client.get("/stats").json()

    # Compression still ran (the proxy forwarded a compressed body) ...
    assert len(upstream.calls) == 1
    # ... but NOTHING is counted: the turn produced no answer, so it neither bumps
    # total_requests nor folds any savings into the totals.
    assert stats["total_requests"] == 0, "a failed turn must not count as processed"
    assert stats["tokens_saved"] == 0


def test_successful_upstream_is_credited(tmp_path, monkeypatch):
    """Control: the identical request, but a 200 upstream, IS credited — proving
    the gate keys on the outcome, not on suppressing all accounting."""
    upstream = _Upstream(status=200)
    app = _app(tmp_path, monkeypatch, upstream)

    with TestClient(app) as client:
        r = client.post("/v1/chat/completions",
                        json={"model": MODEL, "messages": _long_history()})
        assert r.status_code == 200
        stats = client.get("/stats").json()

    assert stats["total_requests"] == 1
    assert stats["tokens_saved"] > 0
