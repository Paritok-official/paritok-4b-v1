"""A streamed turn that calls read_original is resolved from the reply already buffered.

The proxy used to throw that reply away and re-run the whole request non-streaming just
to get the tool call again: one extra full-context upstream call per read_original
(loopbench r13: 19 discarded streams, 2.64M input tokens, 57% of all hidden cost), and
the re-run often chose a different turn than the one the model had made. Now the
buffered reply is reassembled — thinking signature included — and the loop continues
from it; only a reply that can't be reassembled exactly falls back to the re-run.
"""
import json

import pytest

pytest.importorskip("starlette")
from starlette.testclient import TestClient  # noqa: E402

MODEL = "claude-opus-5-5"
ORIGINAL = "\n".join(f"{i:6d}\tline of source code number {i}" for i in range(1, 200))
TOOLS = [{"name": "read_original", "description": "the original of a [REF:id]",
          "input_schema": {"type": "object", "properties": {"shadow_id": {"type": "string"}}}}]


def _sse(blocks, stop_reason):
    """A streamed message: thinking / text / tool_use blocks, the way the API sends them."""
    ev = [("message_start", {"type": "message_start", "message": {
        "id": "msg_s", "type": "message", "role": "assistant", "model": MODEL, "content": [],
        "usage": {"input_tokens": 3, "cache_read_input_tokens": 140000}}})]
    for i, b in enumerate(blocks):
        if b["type"] == "thinking":
            ev.append(("content_block_start", {"type": "content_block_start", "index": i,
                                                "content_block": {"type": "thinking", "thinking": ""}}))
            ev.append(("content_block_delta", {"type": "content_block_delta", "index": i,
                                                "delta": {"type": "thinking_delta", "thinking": b["thinking"]}}))
            if b.get("signature"):
                ev.append(("content_block_delta", {"type": "content_block_delta", "index": i,
                                                    "delta": {"type": "signature_delta",
                                                              "signature": b["signature"]}}))
        elif b["type"] == "text":
            ev.append(("content_block_start", {"type": "content_block_start", "index": i,
                                                "content_block": {"type": "text", "text": ""}}))
            ev.append(("content_block_delta", {"type": "content_block_delta", "index": i,
                                                "delta": {"type": "text_delta", "text": b["text"]}}))
        elif b["type"] == "tool_use":
            ev.append(("content_block_start", {"type": "content_block_start", "index": i,
                                                "content_block": {"type": "tool_use", "id": b["id"],
                                                                  "name": b["name"], "input": {}}}))
            js = json.dumps(b["input"])
            for part in (js[:7], js[7:]):  # the input arrives in fragments
                ev.append(("content_block_delta", {"type": "content_block_delta", "index": i,
                                                    "delta": {"type": "input_json_delta",
                                                              "partial_json": part}}))
        else:
            ev.append(("content_block_start", {"type": "content_block_start", "index": i,
                                                "content_block": b}))
        ev.append(("content_block_stop", {"type": "content_block_stop", "index": i}))
    ev.append(("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason},
                                 "usage": {"output_tokens": 50}}))
    ev.append(("message_stop", {"type": "message_stop"}))
    return "".join(f"event: {n}\ndata: {json.dumps(d)}\n\n" for n, d in ev).encode()


class _Resp:
    def __init__(self, payload):
        self._p, self.status_code, self.headers = payload, 200, {}

    def json(self):
        return self._p


class _StreamResp:
    def __init__(self, raw):
        self.raw, self.status_code, self.headers = raw, 200, {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def aiter_bytes(self):
        yield self.raw


class _Upstream:
    def __init__(self, streamed, posts):
        self.streamed, self.posts, self.calls = streamed, list(posts), []

    def stream(self, method, url, headers=None, json=None):
        self.calls.append(("stream", json))
        return _StreamResp(self.streamed)

    async def post(self, url, headers=None, json=None):
        self.calls.append(("post", json))
        return _Resp(self.posts.pop(0))

    async def aclose(self):
        pass


def _run(monkeypatch, streamed, posts, trace=None):
    from paritok.storage import MemoryShadowStorage
    import paritok.middleware.wrapper as wrapper
    from paritok.proxy import server

    storage = MemoryShadowStorage()
    sid = storage.store(ORIGINAL)
    monkeypatch.setattr(wrapper, "build_shadow_storage", lambda config: storage)
    if trace:
        monkeypatch.setenv("PARITOK_AGENT_TRACE", str(trace))
    else:
        monkeypatch.delenv("PARITOK_AGENT_TRACE", raising=False)
    up = _Upstream(streamed(sid), posts)
    app = server.create_app(anthropic_base_url="http://upstream.test", http_client=up)
    with TestClient(app) as client:
        r = client.post("/v1/messages", json={
            "model": MODEL, "max_tokens": 64, "stream": True, "tools": TOOLS,
            "messages": [{"role": "user", "content": "explain the file"}]})
        stats = client.get("/stats").json()
    assert r.status_code == 200
    _run.stats = stats
    return up, r.text


FINAL = {"id": "msg_f", "type": "message", "role": "assistant", "stop_reason": "end_turn",
         "content": [{"type": "text", "text": "the explanation"}],
         "usage": {"input_tokens": 3, "output_tokens": 9}}


def _blocks(sid, signature="sig-abc"):
    return [{"type": "thinking", "thinking": "I need the exact lines.", "signature": signature},
            {"type": "text", "text": "Let me look at the original."},
            {"type": "tool_use", "id": "tu_ro", "name": "read_original", "input": {"shadow_id": sid}}]


def test_buffered_reply_is_continued_not_rerun(monkeypatch, tmp_path):
    trace = tmp_path / "trace.jsonl"
    up, text = _run(monkeypatch, lambda sid: _sse(_blocks(sid), "tool_use"), [FINAL], trace)
    assert [k for k, _ in up.calls] == ["stream", "post"], "no re-run of the same request"
    msgs = up.calls[1][1]["messages"]
    # the turn the model actually made goes back verbatim, thinking signature included
    assert msgs[-2]["role"] == "assistant"
    assert msgs[-2]["content"][0] == {"type": "thinking", "thinking": "I need the exact lines.",
                                      "signature": "sig-abc"}
    call = msgs[-2]["content"][2]
    assert (call["id"], call["name"], set(call["input"])) == ("tu_ro", "read_original", {"shadow_id"})
    assert msgs[-1]["content"][0]["tool_use_id"] == "tu_ro"
    assert "line of source code number 42" in json.dumps(msgs[-1])
    assert "the explanation" in text
    ups = [json.loads(ln) for ln in trace.read_text(encoding="utf-8").splitlines()
           if json.loads(ln)["ev"] == "upstream"]
    assert [(u["kind"], u["outcome"]) for u in ups] == [("stream", "hidden"), ("resolve", "final")]
    assert ups[0]["virtual"][0]["name"] == "read_original"
    # /stats charges the hidden call's real usage against the savings
    s = _run.stats
    assert (s["hidden_rounds"], s["hidden_input_tokens"], s["hidden_output_tokens"]) == (1, 140003, 50)
    assert s["hidden_cost_usd"] > 0
    assert s["tokens_saved"] == s["input_tokens_original"] - s["input_tokens_compressed"] - 140003


def test_reply_that_cannot_be_reassembled_falls_back_to_a_rerun(monkeypatch):
    rerun = {"id": "msg_r", "type": "message", "role": "assistant", "stop_reason": "tool_use",
             "content": [{"type": "tool_use", "id": "tu_ro2", "name": "read_original",
                          "input": {"shadow_id": "x"}}]}
    up, text = _run(monkeypatch, lambda sid: _sse(_blocks(sid, signature=None), "tool_use"),
                    [rerun, FINAL])
    assert [k for k, _ in up.calls] == ["stream", "post", "post"]
    assert "the explanation" in text
    # both the thrown-away stream and the re-run's tool-call round are hidden calls
    assert (_run.stats["hidden_rounds"], _run.stats["hidden_input_tokens"]) == (2, 140003)


def test_text_that_only_mentions_read_original_is_not_rerun(monkeypatch):
    up, text = _run(monkeypatch, lambda sid: _sse(
        [{"type": "text", "text": "No need for read_original here."}], "end_turn"), [])
    assert [k for k, _ in up.calls] == ["stream"]
    assert "No need for read_original here." in text
    assert _run.stats["hidden_rounds"] == 0
