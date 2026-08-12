"""`/v1/messages/count_tokens` (Anthropic). Claude Code hits this every turn to size its
context; without it the endpoint 404s and Claude Code auto-compacts early over the full
uncompressed conversation instead of letting paritok's compression extend the window.

The route must exist, return `{"input_tokens": <int>}`, and never fail the meter (a
compression error must fall back to counting the raw payload, not 500). Compression-aware
counting (compressed < raw) needs a live 4B backend, so here we assert the contract and the
no-backend fallback; the compression-reflects-savings behaviour is covered end-to-end.
"""
import json

import httpx
from starlette.testclient import TestClient

from paritok.proxy.server import create_app
from paritok.token_counter import count_tokens


class _StubTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        # count_tokens never forwards upstream; this is only here so the app builds.
        return httpx.Response(200, json={"ok": True})


def _client():
    return TestClient(create_app(http_client=httpx.AsyncClient(transport=_StubTransport())))


BODY = {
    "model": "claude-sonnet-4-5",
    "system": "You are a helpful coding assistant.",
    "messages": [
        {"role": "user", "content": "Read config.py and fix the bug in load()."},
        {"role": "assistant", "content": "Let me read it."},
    ],
    "tools": [{"name": "Read", "description": "Read a file",
               "input_schema": {"type": "object", "properties": {}}}],
}


def test_count_tokens_route_exists_and_returns_input_tokens():
    r = _client().post("/v1/messages/count_tokens", json=BODY)
    assert r.status_code == 200            # not 404 — the whole point
    data = r.json()
    assert isinstance(data.get("input_tokens"), int)
    assert data["input_tokens"] > 0


def test_count_tokens_counts_system_and_tools_and_messages():
    # A payload with more content must count higher than a bare one.
    small = {"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "hi"}]}
    c = _client()
    n_small = c.post("/v1/messages/count_tokens", json=small).json()["input_tokens"]
    n_big = c.post("/v1/messages/count_tokens", json=BODY).json()["input_tokens"]
    assert n_big > n_small


def test_count_tokens_no_backend_is_in_the_raw_ballpark():
    # With no live 4B backend, compression skips/falls back, so the count is the raw payload
    # PLUS the fixed read_original tool schema paritok injects into tools[] (~a few hundred
    # tokens). It must never come back below raw (that'd mean lost content) or absurdly high.
    raw = (count_tokens(json.dumps(BODY["messages"]), BODY["model"])
           + count_tokens(json.dumps(BODY["system"]), BODY["model"])
           + count_tokens(json.dumps(BODY["tools"]), BODY["model"]))
    n = _client().post("/v1/messages/count_tokens", json=BODY).json()["input_tokens"]
    assert raw <= n <= raw + 3000  # raw + fixed virtual-tool injection, nothing lost/blown up


def test_count_tokens_handles_empty_messages():
    r = _client().post("/v1/messages/count_tokens",
                       json={"model": "claude-sonnet-4-5", "messages": []})
    assert r.status_code == 200
    assert isinstance(r.json().get("input_tokens"), int)
