"""End-to-end: a sonnet turn that fires expand_context must show up in /stats.

Drives the real Starlette proxy app (handle_anthropic -> _anthropic_resolve ->
record_expansion -> /stats) with a mock upstream standing in for sonnet. The
shadow store is pre-seeded to stand in for the compression step (the 4B model
isn't available in CI), so we can assert the re-delivered original is counted
onto the compressed side rather than vanishing from accounting.
"""
import json

import pytest

from paritok.token_counter import count_tokens

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
    """Sonnet stand-in: first turn calls expand_context, second turn ends."""

    def __init__(self, shadow_id):
        self.calls = []
        self._queue = [
            {"type": "message", "role": "assistant", "stop_reason": "tool_use",
             "content": [{"type": "tool_use", "id": "tu1", "name": "expand_context",
                          "input": {"shadow_id": shadow_id}}]},
            {"type": "message", "role": "assistant", "stop_reason": "end_turn",
             "content": [{"type": "text", "text": "done"}]},
        ]

    async def post(self, url, headers=None, json=None):
        self.calls.append(json)
        return _FakeResp(self._queue.pop(0))

    async def aclose(self):
        pass


def test_expand_context_counted_in_stats(monkeypatch):
    from paritok.storage import MemoryShadowStorage
    import paritok.middleware.wrapper as wrapper
    from paritok.proxy import server

    # Pre-seed the shadow store and force the engine to use this very instance.
    storage = MemoryShadowStorage()
    shadow_id = storage.store(ORIGINAL)
    monkeypatch.setattr(wrapper, "build_shadow_storage", lambda config: storage)

    upstream = _FakeUpstream(shadow_id)
    app = server.create_app(anthropic_base_url="http://upstream.test",
                            http_client=upstream)

    body = {
        "model": MODEL,
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "open the file and summarize it"}],
        # Client presents expand_context so the resolve path is exercised without
        # needing the (unavailable) 4B model to inject it via real compression.
        "tools": [{"name": "expand_context", "description": "expand a [REF:id]",
                   "input_schema": {"type": "object",
                                    "properties": {"shadow_id": {"type": "string"}}}}],
    }

    with TestClient(app) as client:
        r = client.post("/v1/messages", json=body)
        assert r.status_code == 200
        stats = client.get("/stats").json()

    expanded = count_tokens(ORIGINAL, MODEL)

    # The proxy really looped upstream twice (call -> resolve -> final turn),
    assert len(upstream.calls) == 2
    # and the second POST actually carried the expanded original back up as a
    # tool_result (structural check — json.dumps would escape tabs/newlines).
    second_post = json.dumps(upstream.calls[1])
    assert "line of source code number 42" in second_post

    # The re-delivered original is counted on the compressed side (the only
    # content here — nothing was really compressed, so it's a pure penalty). The
    # tool-schema block nets to zero (orig == comp), so the whole compressed-vs-
    # original gap and the negative savings are exactly the expanded original.
    assert stats["input_tokens_compressed"] - stats["input_tokens_original"] == expanded
    assert stats["tokens_saved"] == -expanded
    assert expanded > 1000  # sanity: this really is a big blob, not a rounding blip
