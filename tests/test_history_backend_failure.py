"""Regression (issue #36, history path): a compression-backend failure during HISTORY
summarization must not 500 the request. _compress_history must log a warning, keep the
original messages, and let the proxy forward them upstream — the same pass-through
contract CompressionPipeline.compress() already gives tool outputs.

Only (ConnectionError, TimeoutError, ValueError) used to be caught, so an Ollama HTTP
error (httpx.HTTPStatusError), a dropped connection (httpx.RemoteProtocolError) or an
unexpected response shape (TypeError) escaped process_request and Starlette returned
500 without ever calling upstream.
"""
import json
import logging

import httpx
import pytest

pytest.importorskip("starlette")
from starlette.testclient import TestClient  # noqa: E402

from paritok.config import ParitokConfig  # noqa: E402
from paritok.middleware import wrapper  # noqa: E402
from paritok.middleware.wrapper import CompressionStats, _compress_history  # noqa: E402
from paritok.pipelines.compress import CompressionPipeline  # noqa: E402

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


def _long_history():
    """6 user/assistant turns; first 4 old, last 2 recent. Exceeds the 500-token
    (1000 * 0.5) threshold so history compression fires."""
    filler = "context filler text " * 60
    msgs = []
    for t in range(6):
        marker = f"OLD_MARKER_{t}" if t < 4 else "RECENT_MARKER"
        msgs.append({"role": "user", "content": f"{marker} question {t} {filler}"})
        msgs.append({"role": "assistant", "content": f"{marker} answer {t} {filler}"})
    return msgs


def _req():
    return httpx.Request("POST", "http://localhost:11434/v1/chat/completions")


def _backend(mode):
    """A fake httpx.post for LocalModelStrategy's Ollama call."""
    def post(url, **kwargs):
        if mode == "http_500":
            return httpx.Response(500, json={"error": "model failed"}, request=_req())
        if mode == "remote_protocol":
            raise httpx.RemoteProtocolError("Server disconnected", request=_req())
        if mode == "null_content":
            return httpx.Response(200, json={"choices": [{"message": {"content": None}}]},
                                  request=_req())
        raise AssertionError(mode)
    return post


def _compress(msgs, stats):
    return _compress_history(msgs, CompressionPipeline(ParitokConfig()), stats, query="q",
                             keep_recent_turns=2, context_threshold=0.5,
                             context_window=1000)


@pytest.mark.parametrize("mode", ["http_500", "remote_protocol", "null_content"])
def test_history_backend_failure_returns_original_messages(mode, monkeypatch, caplog):
    monkeypatch.setattr(httpx, "post", _backend(mode))
    msgs = _long_history()
    stats = CompressionStats()
    with caplog.at_level(logging.WARNING, logger="paritok"):
        out = _compress(msgs, stats)
    assert out is msgs
    assert stats.history_turns_compressed == 0
    assert stats.items_compressed == 0
    assert any("History compression failed" in r.getMessage() for r in caplog.records)


def test_history_wrapper_bug_still_propagates(monkeypatch):
    # Only the model call is guarded: a bug in the wrapper's own post-processing
    # must still surface, not be swallowed as a "backend failure".
    monkeypatch.setattr(
        "paritok.strategies.local_model.LocalModelStrategy.compress",
        lambda self, content, **kw: "SUMMARY",
    )

    def boom(messages):
        raise RuntimeError("bug in demote")

    monkeypatch.setattr(wrapper, "_demote_orphan_tool_results", boom)
    with pytest.raises(RuntimeError, match="bug in demote"):
        _compress(_long_history(), CompressionStats())


ROUTES = [
    ("/v1/messages",
     {"model": "claude-sonnet-4", "max_tokens": 10},
     {"x-api-key": "k", "anthropic-version": "2023-06-01"}),
    ("/v1/chat/completions",
     {"model": "gpt-4o"},
     {"authorization": "Bearer sk"}),
]


def _upstream_client(calls):
    def handler(request):
        calls.append(json.loads(request.content))
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(200, json={
                "id": "chatcmpl-x", "object": "chat.completion",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "done"}}]})
        return httpx.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-4",
            "content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1}})
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("mode", ["http_500", "null_content"])
@pytest.mark.parametrize("path,extra,headers", ROUTES)
def test_proxy_forwards_original_history_on_backend_failure(
        mode, path, extra, headers, tmp_path, monkeypatch):
    monkeypatch.setattr(httpx, "post", _backend(mode))
    cfg = tmp_path / "paritok.yaml"
    cfg.write_text(CONFIG_YAML, encoding="utf-8")

    from paritok.proxy import server
    calls = []
    app = server.create_app(config_path=str(cfg),
                            anthropic_base_url="http://anth.test",
                            openai_base_url="http://oai.test",
                            http_client=_upstream_client(calls))
    with TestClient(app) as client:
        r = client.post(path, json={**extra, "messages": _long_history()}, headers=headers)

    assert r.status_code == 200, r.text
    assert len(calls) == 1
    forwarded = json.dumps(calls[0]["messages"])
    assert "OLD_MARKER_0" in forwarded
    assert "RECENT_MARKER" in forwarded
    assert "[Conversation Summary]" not in forwarded
