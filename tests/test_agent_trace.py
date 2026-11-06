"""Opt-in agent-loop trace (PARITOK_AGENT_TRACE): one JSONL event stream that
explains where an agent's extra turns went — per-tool_result decisions, hidden
read_original rounds, the turn the client finally sees, and Edit outcomes.

Drives the real proxy app with a mock upstream (the 4B isn't available in CI):
the shadow store is pre-seeded so the expand round has something to serve.
"""
import json

import pytest

pytest.importorskip("starlette")
from starlette.testclient import TestClient  # noqa: E402

MODEL = "claude-sonnet-4-5-20250929"
ORIGINAL = "\n".join(f"{i:6d}\tline of source code number {i}" for i in range(1, 400))


class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload, self.status_code, self.headers = payload, status, {}

    def json(self):
        return self._payload


class _FakeUpstream:
    """Turn 1 expands a ref (hidden round); turn 2 makes a real Edit call."""

    def __init__(self, shadow_id):
        self.calls = []
        self._queue = [
            {"id": "msg_hidden", "type": "message", "role": "assistant",
             "stop_reason": "tool_use",
             "usage": {"input_tokens": 10, "output_tokens": 5},
             "content": [{"type": "text", "text": "need the exact lines"},
                         {"type": "tool_use", "id": "tu_x", "name": "read_original",
                          "input": {"shadow_id": shadow_id}}]},
            {"id": "msg_final", "type": "message", "role": "assistant",
             "stop_reason": "tool_use",
             "usage": {"input_tokens": 20, "output_tokens": 7},
             "content": [{"type": "tool_use", "id": "tu_edit", "name": "Edit",
                          "input": {"file_path": "never_read.py",
                                    "old_string": "a = 1", "new_string": "a = 2"}}]},
        ]

    async def post(self, url, headers=None, json=None):
        self.calls.append(json)
        return _FakeResp(self._queue.pop(0))

    async def aclose(self):
        pass


def _body():
    return {
        "model": MODEL,
        "max_tokens": 128,
        "messages": [
            {"role": "user", "content": "fix the bug in foo.py"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_read", "name": "Read",
                 "input": {"file_path": "foo.py"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_read", "content": "x = 1\n"}]},
        ],
        "tools": [{"name": "Read", "description": "read",
                   "input_schema": {"type": "object", "properties": {}}},
                  {"name": "Edit", "description": "edit",
                   "input_schema": {"type": "object", "properties": {}}}],
    }


def _app(monkeypatch, tmp_path, trace=True):
    from paritok.storage import MemoryShadowStorage
    import paritok.middleware.wrapper as wrapper
    from paritok.proxy import server

    storage = MemoryShadowStorage()
    shadow_id = storage.store(ORIGINAL)
    monkeypatch.setattr(wrapper, "build_shadow_storage", lambda config: storage)
    path = tmp_path / "trace.jsonl"
    if trace:
        monkeypatch.setenv("PARITOK_AGENT_TRACE", str(path))
    else:
        monkeypatch.delenv("PARITOK_AGENT_TRACE", raising=False)
    upstream = _FakeUpstream(shadow_id)
    app = server.create_app(anthropic_base_url="http://upstream.test", http_client=upstream)
    return app, path, shadow_id


def _events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_trace_records_items_hidden_rounds_final_and_edit(monkeypatch, tmp_path):
    app, path, shadow_id = _app(monkeypatch, tmp_path)
    with TestClient(app) as client:
        r = client.post("/v1/messages", json=_body())
        assert r.status_code == 200

    ev = _events(path)
    kinds = [e["ev"] for e in ev]
    req = next(e for e in ev if e["ev"] == "request")
    rid = req["rid"]

    # The tiny Read result was passed through, with the reason, keyed by tool_use_id.
    (item,) = req["items_changed"]
    assert item["tool_use_id"] == "tu_read" and item["tool"] == "Read"
    assert item["action"] == "skipped" and item["reason"] == "below_min_tokens"
    assert item["input"] == {"file_path": "foo.py"}

    ups = [e for e in ev if e["ev"] == "upstream"]
    assert [u["outcome"] for u in ups] == ["hidden", "final"]
    hidden = ups[0]
    assert hidden["msg_id"] == "msg_hidden" and hidden["usage"]["in"] == 10
    (v,) = hidden["virtual"]
    assert v["name"] == "read_original" and v["ref"] == shadow_id and v["out_tok"] > 1000

    # The Edit targets a file the gateway never compressed.
    edit = next(e for e in ev if e["ev"] == "edit")
    assert edit["tool_use_id"] == "tu_edit" and edit["outcome"] == "no_shadow"

    fin = next(e for e in ev if e["ev"] == "final")
    assert fin["rid"] == rid and fin["msg_id"] == "msg_final"
    assert fin["upstream_calls"] == 2
    assert fin["client_tool_uses"] == [{"id": "tu_edit", "name": "Edit"}]
    assert kinds.index("request") < kinds.index("upstream") < kinds.index("final")


def test_unchanged_item_view_is_not_relogged(monkeypatch, tmp_path):
    """Every turn re-sends the whole conversation; an item is logged only when
    its forwarded view first appears or changes."""
    from paritok.proxy import server

    path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("PARITOK_AGENT_TRACE", str(path))
    plain = {"id": "m", "type": "message", "role": "assistant",
             "stop_reason": "end_turn", "content": [{"type": "text", "text": "ok"}]}
    up = _FakeUpstream("unused")
    up._queue = [dict(plain), dict(plain)]
    app = server.create_app(anthropic_base_url="http://upstream.test", http_client=up)
    with TestClient(app) as client:
        client.post("/v1/messages", json=_body())
        client.post("/v1/messages", json=_body())
    reqs = [e for e in _events(path) if e["ev"] == "request"]
    assert len(reqs) == 2
    assert len(reqs[0]["items_changed"]) == 1
    assert reqs[1]["items_changed"] == []
    assert reqs[1]["items_by_action"] == {"skipped": 1}


def test_recovered_edit_logs_model_and_client_versions(monkeypatch, tmp_path):
    """A rewritten Edit must log what the model wrote AND what the client will apply,
    so a mangled rewrite (e.g. lost indentation) can be told apart from a model slip."""
    from paritok.storage import MemoryShadowStorage
    import paritok.middleware.wrapper as wrapper
    from paritok.proxy import server

    raw = "def f(\n    a,\n    b,\n):\n    return a + b\n"
    numbered = "".join(f"{i}\t{ln}\n" for i, ln in enumerate(raw.splitlines(), 1))
    storage = MemoryShadowStorage()
    sid = storage.store(numbered)
    storage.set_shadow_for_path("/p/m.py", sid)
    monkeypatch.setattr(wrapper, "build_shadow_storage", lambda config: storage)
    path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("PARITOK_AGENT_TRACE", str(path))

    old = "def f(a, b):\n    return a + b"           # authored from a collapsed view
    new = "def f(a, b):\n    return a - b"
    up = _FakeUpstream("unused")
    up._queue = [{"id": "m1", "type": "message", "role": "assistant", "stop_reason": "tool_use",
                  "content": [{"type": "tool_use", "id": "tu_e", "name": "Edit",
                               "input": {"file_path": "/p/m.py", "old_string": old,
                                         "new_string": new}}]}]
    app = server.create_app(anthropic_base_url="http://upstream.test", http_client=up)
    with TestClient(app) as client:
        assert client.post("/v1/messages", json=_body()).status_code == 200

    (edit,) = [e for e in _events(path) if e["ev"] == "edit"]
    assert edit["outcome"] == "recovered"
    assert edit["model_old"] == old and edit["model_new"] == new
    assert "def f(\n    a,\n    b,\n):" in edit["client_old"]  # mapped back onto the real file
    assert "return a - b" in edit["client_new"]


def test_message_tokens_count_content_not_signatures():
    """msg_tokens_* must count what the messages say. json.dumps also counted the opaque
    thinking `signature` (base64, not billed as text) — on an Opus session that was
    ~80% of a late request's count and made compression look 4x weaker than it was."""
    from paritok.proxy.agent_trace import content_tokens
    from paritok.token_counter import count_tokens

    sig = "EuYBCkYICBgCKkB" + "x9Zq" * 4000  # an Opus-style signature, ~16k chars
    messages = [
        {"role": "user", "content": "fix the bug"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "look at deletion.py", "signature": sig},
            {"type": "tool_use", "id": "toolu_01ABCDEFGHIJKLMNOPQRSTUV", "name": "Read",
             "input": {"file_path": "deletion.py"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_01ABCDEFGHIJKLMNOPQRSTUV",
             "content": [{"type": "text", "text": "def delete(self):\n    pass\n"}]}]},
    ]
    n = content_tokens(messages, count_tokens)
    expected = sum(count_tokens(t) for t in (
        "fix the bug", "look at deletion.py", json.dumps({"file_path": "deletion.py"}),
        "def delete(self):\n    pass\n"))
    assert n == expected
    assert count_tokens(json.dumps(messages)) > 20 * n  # what the old count was dominated by


def test_trace_off_writes_nothing(monkeypatch, tmp_path):
    app, path, _ = _app(monkeypatch, tmp_path, trace=False)
    with TestClient(app) as client:
        assert client.post("/v1/messages", json=_body()).status_code == 200
    assert not path.exists()
