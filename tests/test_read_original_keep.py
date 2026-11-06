"""read_original(keep=true): the original stays for later steps, attached to the first
content sent after the turn that asked — never where the summary was, so nothing already
sent upstream changes (prompt cache prefix and thinking signatures stay valid)."""
import copy
import json

import pytest

from paritok.proxy.kept_originals import KeptOriginals, turn_fingerprint

ORIGINAL = "".join(f"{i:6d}\tline of source code number {i}\n" for i in range(1, 120))


def _msgs():
    return [
        {"role": "user", "content": "explain compiler.py"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "tu_1", "name": "Read", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "x"}]},
    ]


def test_attached_after_the_anchor_tool_result_and_identical_every_time():
    k = KeptOriginals()
    k.add("c", "ref1", ORIGINAL, "/p/compiler.py", "tu_1", 2)
    msgs = _msgs()
    before = copy.deepcopy(msgs)
    out1, n1 = k.attach("c", msgs)
    out2, _ = k.attach("c", msgs + [{"role": "assistant", "content": "next"}])
    assert msgs == before  # the input is untouched
    assert out1[2]["content"][0]["type"] == "tool_result"  # tool results stay first
    assert out1[2]["content"][1]["text"].endswith(ORIGINAL) and "REF:ref1" in out1[2]["content"][1]["text"]
    assert out1[:2] == msgs[:2] and n1 > 0
    assert json.dumps(out2[2]) == json.dumps(out1[2])  # byte-identical on later requests


def _no_tool_turn(reply="I have what I need."):
    return [{"role": "user", "content": "explain compiler.py"},
            {"role": "assistant", "content": [{"type": "text", "text": reply}]},
            {"role": "user", "content": "now write the notes"}]


def test_turn_without_tool_calls_anchors_on_the_next_message():
    k = KeptOriginals()
    msgs = _no_tool_turn()
    k.add("c", "ref1", ORIGINAL, None, None, 2, anchor_fp=turn_fingerprint(msgs[1]["content"]))
    out, _ = k.attach("c", msgs)
    assert out[2]["content"][0] == {"type": "text", "text": "now write the notes"}
    assert out[2]["content"][1]["text"].endswith(ORIGINAL)


def test_side_request_waits_and_rewritten_history_is_skipped_not_dropped():
    k = KeptOriginals()
    k.add("c", "ref1", ORIGINAL, None, "tu_1", 2)
    out, n = k.attach("c", _msgs()[:1])  # shorter side request: nothing yet, still kept
    assert n == 0 and k.refs("c") == ["ref1"]
    compacted = [{"role": "user", "content": "summary of earlier work"},
                 {"role": "assistant", "content": "ok"}, {"role": "user", "content": "go on"}]
    out, n = k.attach("c", compacted)  # anchor not in this history: nothing attached...
    assert n == 0 and out == compacted
    assert k.refs("c") == ["ref1"]  # ...but not dropped: it may be another session's


# ── two sessions opened with the same prompt share the conversation key ──

def _session(tool_id):
    return [{"role": "user", "content": "fix the bug"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": tool_id, "name": "Read", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": "x"}]}]


def test_another_session_with_the_same_first_message_does_not_drop_the_original():
    """Seen in review: session B's request (anchor not in its history) dropped session A's
    kept original, so A's next request lost a block A had already sent upstream."""
    k = KeptOriginals()
    k.add("c", "ref1", ORIGINAL, None, "tu_A", 2)
    a = _session("tu_A")
    first, _ = k.attach("c", a)
    b_out, b_n = k.attach("c", _session("tu_B"))
    assert b_n == 0 and b_out == _session("tu_B")  # nothing of A's leaks into B
    later, n = k.attach("c", a + [{"role": "assistant", "content": "next"}])
    assert n > 0 and json.dumps(later[2]) == json.dumps(first[2])  # A's history unchanged


def test_no_tool_call_anchor_only_attaches_after_the_same_turn():
    k = KeptOriginals()
    a = _no_tool_turn("Session A's answer.")
    k.add("c", "ref1", ORIGINAL, None, None, 2, anchor_fp=turn_fingerprint(a[1]["content"]))
    b = _no_tool_turn("Session B's different answer.")
    assert k.attach("c", b) == (b, 0)  # same index, other session's turn: not attached
    out, n = k.attach("c", a)
    assert n > 0 and out[2]["content"][1]["text"].endswith(ORIGINAL)


def test_no_tool_call_anchor_without_a_fingerprint_is_never_guessed():
    k = KeptOriginals()
    k.add("c", "ref1", ORIGINAL, None, None, 2)
    assert k.attach("c", _no_tool_turn()) == (_no_tool_turn(), 0)


def test_two_sessions_keeping_the_same_original_each_get_it_once():
    k = KeptOriginals()
    k.add("c", "ref1", ORIGINAL, None, "tu_A", 2)
    k.add("c", "ref1", ORIGINAL, None, "tu_B", 2)
    for tid in ("tu_A", "tu_B"):
        out, n = k.attach("c", _session(tid))
        kept = [b for m in out if isinstance(m.get("content"), list)
                for b in m["content"] if b.get("type") == "text" and "REF:ref1" in b.get("text", "")]
        assert n > 0 and len(kept) == 1


def test_turn_fingerprint_ignores_surrounding_whitespace_and_falls_back_to_signatures():
    assert turn_fingerprint([{"type": "text", "text": "done\n"}]) == turn_fingerprint("done")
    sig = [{"type": "thinking", "thinking": "...", "signature": "abc"}]
    assert turn_fingerprint(sig) and turn_fingerprint(sig) != turn_fingerprint(
        [{"type": "thinking", "thinking": "...", "signature": "xyz"}])
    assert turn_fingerprint([]) is None and turn_fingerprint("  ") is None


# ── memory is bounded ──

def test_least_recently_used_conversations_are_evicted_past_the_cap():
    k = KeptOriginals(max_conversations=3)
    for c in ("c1", "c2", "c3"):
        k.add(c, "ref", ORIGINAL, None, "tu_1", 2)
    k.attach("c1", _msgs())  # c1 used again: c2 is now the oldest
    k.add("c4", "ref", ORIGINAL, None, "tu_1", 2)
    assert k.refs("c2") == [] and k.refs("c1") == ["ref"] and k.refs("c4") == ["ref"]


def test_total_size_is_capped_but_the_newest_conversation_is_kept():
    k = KeptOriginals(max_total_chars=len(ORIGINAL) * 2)
    k.add("c1", "r1", ORIGINAL, None, "tu_1", 2)
    k.add("c2", "r2", ORIGINAL, None, "tu_1", 2)
    assert k.refs("c1") == []  # two originals plus headers exceed the cap: oldest goes
    assert k.refs("c2") == ["r2"]
    big = KeptOriginals(max_total_chars=10)
    big.add("only", "r", ORIGINAL, None, "tu_1", 2)
    assert big.refs("only") == ["r"]  # a single conversation is never evicted for size


def test_other_conversations_and_repeat_keeps_are_ignored():
    k = KeptOriginals()
    k.add("c", "ref1", ORIGINAL, None, "tu_1", 2)
    k.add("c", "ref1", "other text", None, "tu_9", 9)  # same ref kept again later: first wins
    assert k.attach("other", _msgs()) == (_msgs(), 0)
    out, _ = k.attach("c", _msgs())
    assert sum(1 for b in out[2]["content"] if b.get("type") == "text") == 1


# ── through the proxy: a three-request conversation ──

pytest.importorskip("starlette")
from starlette.testclient import TestClient  # noqa: E402

MODEL = "claude-opus-5-5"
TOOLS = [{"name": "read_original", "description": "x",
          "input_schema": {"type": "object", "properties": {"shadow_id": {"type": "string"},
                                                            "keep": {"type": "boolean"}}}}]


class _Resp:
    def __init__(self, p):
        self._p, self.status_code, self.headers = p, 200, {}

    def json(self):
        return self._p


class _Up:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    async def post(self, url, headers=None, json=None):
        self.calls.append(copy.deepcopy(json))
        return _Resp(self.replies.pop(0))

    async def aclose(self):
        pass


def _msg(content, stop="end_turn"):
    return {"id": "m", "type": "message", "role": "assistant", "stop_reason": stop, "content": content}


def _conversation(monkeypatch, keep):
    from paritok.storage import MemoryShadowStorage
    import paritok.middleware.wrapper as wrapper
    from paritok.proxy import server

    storage = MemoryShadowStorage()
    sid = storage.store(ORIGINAL)
    monkeypatch.setattr(wrapper, "build_shadow_storage", lambda config: storage)
    monkeypatch.delenv("PARITOK_AGENT_TRACE", raising=False)
    up = _Up([
        _msg([{"type": "tool_use", "id": "tu_ro", "name": "read_original",
               "input": {"shadow_id": sid, "keep": keep}}], "tool_use"),          # req 1: expand
        _msg([{"type": "text", "text": "noted"},
              {"type": "tool_use", "id": "tu_w", "name": "Write", "input": {"file_path": "/n.md"}}],
             "tool_use"),                                                          # req 1: final
        _msg([{"type": "tool_use", "id": "tu_g", "name": "Grep", "input": {}}], "tool_use"),  # req 2
        _msg([{"type": "text", "text": "done"}]),                                  # req 3
    ])
    app = server.create_app(anthropic_base_url="http://upstream.test", http_client=up)
    m0 = [{"role": "user", "content": "write notes on compiler.py"}]
    with TestClient(app) as client:
        r1 = client.post("/v1/messages", json={"model": MODEL, "max_tokens": 64, "tools": TOOLS, "messages": m0})
        a1 = {"role": "assistant", "content": r1.json()["content"]}
        m2 = m0 + [a1, {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_w", "content": "written"}]}]
        client.post("/v1/messages", json={"model": MODEL, "max_tokens": 64, "tools": TOOLS, "messages": m2})
        m3 = m2 + [{"role": "assistant", "content": [{"type": "tool_use", "id": "tu_g", "name": "Grep", "input": {}}]},
                   {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_g", "content": "hit"}]}]
        client.post("/v1/messages", json={"model": MODEL, "max_tokens": 64, "tools": TOOLS, "messages": m3})
    return up, m2, m3


def test_keep_true_stays_from_the_next_request_on_without_touching_history(monkeypatch):
    up, m2, m3 = _conversation(monkeypatch, keep=True)
    req2, req3 = up.calls[2]["messages"], up.calls[3]["messages"]
    # what the client sent before stays as sent
    assert req2[:2] == m2[:2]
    # the original rides on the first new content: the tool_result answering the turn
    assert req2[2]["content"][0] == m2[2]["content"][0]
    assert req2[2]["content"][1]["text"].endswith(ORIGINAL)
    # and the same bytes on the next request, nowhere else
    assert json.dumps(req3[2]) == json.dumps(req2[2])
    assert req3[4] == m3[4]


def test_keep_in_a_turn_without_tool_calls_rides_on_the_next_message(monkeypatch):
    from paritok.storage import MemoryShadowStorage
    import paritok.middleware.wrapper as wrapper
    from paritok.proxy import server

    storage = MemoryShadowStorage()
    sid = storage.store(ORIGINAL)
    monkeypatch.setattr(wrapper, "build_shadow_storage", lambda config: storage)
    monkeypatch.delenv("PARITOK_AGENT_TRACE", raising=False)
    up = _Up([
        _msg([{"type": "tool_use", "id": "tu_ro", "name": "read_original",
               "input": {"shadow_id": sid, "keep": True}}], "tool_use"),
        _msg([{"type": "text", "text": "I have the full file now."}]),   # ends the turn, no tool call
        _msg([{"type": "text", "text": "notes written"}]),
    ])
    app = server.create_app(anthropic_base_url="http://upstream.test", http_client=up)
    m0 = [{"role": "user", "content": "explain compiler.py"}]
    with TestClient(app) as client:
        r1 = client.post("/v1/messages", json={"model": MODEL, "max_tokens": 64, "tools": TOOLS, "messages": m0})
        m2 = m0 + [{"role": "assistant", "content": r1.json()["content"]},
                   {"role": "user", "content": "now write the notes"}]
        client.post("/v1/messages", json={"model": MODEL, "max_tokens": 64, "tools": TOOLS, "messages": m2})
    req2 = up.calls[2]["messages"]
    assert req2[:2] == m2[:2]
    assert req2[2]["content"][0] == {"type": "text", "text": "now write the notes"}
    assert req2[2]["content"][1]["text"].endswith(ORIGINAL)


def test_keep_false_is_not_kept(monkeypatch):
    up, m2, m3 = _conversation(monkeypatch, keep=False)
    assert up.calls[2]["messages"] == m2
    assert up.calls[3]["messages"] == m3


def test_cache_breakpoint_moves_onto_the_kept_original():
    """Seen live: appended after the block carrying the client's cache_control, the
    original was billed as uncached input once before entering the cache."""
    k = KeptOriginals()
    k.add("c", "ref1", ORIGINAL, None, "tu_1", 2)
    msgs = _msgs()
    msgs[2] = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "x",
                                            "cache_control": {"type": "ephemeral"}}]}
    out, _ = k.attach("c", msgs)
    assert "cache_control" not in out[2]["content"][0]
    assert out[2]["content"][1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" in msgs[2]["content"][0]  # the input is untouched
