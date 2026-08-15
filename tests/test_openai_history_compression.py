"""Repro + regression for issue #40: the OpenAI proxy path must apply history
compression, not discard it.

process_request() compresses old conversation history (step 3) and returns the
compressed messages as its first value. The Anthropic handler keeps it; the OpenAI
handler used to drop it (`_, processed_tools, ... = process_request(...)`) and
forward the ORIGINAL messages — so long OpenAI sessions went upstream uncompressed.

This drives the real Starlette app on /v1/chat/completions with a mock upstream
that records the forwarded body, a tiny history window (so the threshold trips on a
small fixture), and a stubbed 4B model (unavailable in CI). It asserts the forwarded
request carries the history summary and NOT the old turns.
"""
import json

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


class _FakeUpstream:
    """Records every forwarded body; replies with a plain assistant turn."""

    def __init__(self):
        self.calls = []

    async def post(self, url, headers=None, json=None):
        self.calls.append(json)
        return _FakeResp({
            "id": "chatcmpl-x", "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "done"}}],
        })

    async def aclose(self):
        pass


def _long_history():
    """6 user/assistant turns. First 4 turns carry OLD_MARKER_*, last 2 RECENT_MARKER.
    Big enough to exceed the tiny 500-token (1000 * 0.5) threshold."""
    filler = "context filler text " * 60
    msgs = []
    for t in range(6):
        marker = f"OLD_MARKER_{t}" if t < 4 else "RECENT_MARKER"
        msgs.append({"role": "user", "content": f"{marker} question {t} {filler}"})
        msgs.append({"role": "assistant", "content": f"{marker} answer {t} {filler}"})
    return msgs


def test_openai_path_compresses_history(tmp_path, monkeypatch):
    # Stub the 4B model: any history-summary call returns a fixed marker.
    monkeypatch.setattr(
        "paritok.strategies.local_model.LocalModelStrategy.compress",
        lambda self, content, **kw: SUMMARY,
    )

    cfg = tmp_path / "paritok.yaml"
    cfg.write_text(CONFIG_YAML, encoding="utf-8")

    from paritok.proxy import server
    upstream = _FakeUpstream()
    app = server.create_app(config_path=str(cfg),
                            openai_base_url="http://upstream.test",
                            http_client=upstream)

    body = {"model": MODEL, "messages": _long_history()}
    with TestClient(app) as client:
        r = client.post("/v1/chat/completions", json=body)
        assert r.status_code == 200
        stats = client.get("/stats").json()

    assert len(upstream.calls) == 1
    forwarded = json.dumps(upstream.calls[0]["messages"])

    # History was compressed: the summary reached upstream and the old turns did not.
    assert SUMMARY in forwarded, "history summary was not forwarded (history compression dropped)"
    assert "OLD_MARKER_0" not in forwarded, "old turn survived uncompressed"
    assert "OLD_MARKER_3" not in forwarded
    # Recent turns are kept intact.
    assert "RECENT_MARKER" in forwarded

    # /stats now reflects the history savings (folding a big blob into a tiny summary).
    assert stats["tokens_saved"] > 0
    assert stats["file_compression_saved"] > 0
