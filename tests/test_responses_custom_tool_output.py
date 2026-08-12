"""Codex's newer custom tools (e.g. `exec`/shell) feed results back as
`custom_tool_call_output` items whose `output` is a LIST of
`{"type":"input_text","text":...}` blocks — not the plain-string `output` of a
`function_call_output`. paritok must compress the text inside those blocks too,
otherwise the file/shell content codex re-sends every turn is never compressed.

These cover the pure item-transform (`_compress_responses_item`) deterministically
with a fake compressor, plus a handler smoke test that the shape round-trips.
"""
import json
import httpx
from starlette.testclient import TestClient

from paritok.proxy.server import (
    _compress_responses_item,
    _normalize_crlf,
    create_app,
)


# ── CRLF normalization (codex reads via PowerShell → CRLF; unify to LF) ──

def test_normalize_crlf_strips_carriage_returns():
    crlf = "def f():\r\n    return 1\r\n\r\n"
    assert _normalize_crlf(crlf) == "def f():\n    return 1\n\n"


def test_normalize_crlf_leaves_lf_untouched():
    lf = "def f():\n    return 1\n\n"
    assert _normalize_crlf(lf) is lf  # no \r → returned as-is


def test_normalize_crlf_handles_lone_cr():
    assert _normalize_crlf("a\rb\r\nc") == "a\nb\nc"


# ── pure transform: _compress_responses_item(item, compress_text) ──

def _shrink_big(text: str):
    """Fake compressor: 'compress' anything over 50 chars to a marker, else skip."""
    return "COMPRESSED" if len(text) > 50 else None


def test_custom_tool_call_output_compresses_each_text_block():
    item = {
        "type": "custom_tool_call_output",
        "id": "ctco_1",
        "call_id": "call_1",
        "output": [
            {"type": "input_text", "text": "Script completed\nOutput:\n"},   # small → skipped
            {"type": "input_text", "text": "Exit code: 0\nOutput:\n" + "x" * 100},  # big → compressed
        ],
    }
    new, changed = _compress_responses_item(item, _shrink_big)
    assert changed is True
    assert new["output"][0]["text"] == "Script completed\nOutput:\n"  # untouched
    assert new["output"][1]["text"] == "COMPRESSED"                    # compressed
    # id / call_id / type preserved
    assert new["type"] == "custom_tool_call_output"
    assert new["call_id"] == "call_1" and new["id"] == "ctco_1"


def test_custom_tool_call_output_preserves_non_text_blocks():
    item = {
        "type": "custom_tool_call_output",
        "output": [
            {"type": "input_image", "image_url": "…"},          # non-text → untouched
            {"type": "input_text", "text": "y" * 100},           # big → compressed
        ],
    }
    new, changed = _compress_responses_item(item, _shrink_big)
    assert changed is True
    assert new["output"][0] == {"type": "input_image", "image_url": "…"}
    assert new["output"][1]["text"] == "COMPRESSED"


def test_custom_tool_call_output_all_small_is_unchanged():
    item = {
        "type": "custom_tool_call_output",
        "output": [{"type": "input_text", "text": "tiny"}],
    }
    new, changed = _compress_responses_item(item, _shrink_big)
    assert changed is False
    assert new is item  # returned as-is, not rebuilt


def test_custom_tool_call_output_does_not_mutate_input():
    item = {
        "type": "custom_tool_call_output",
        "output": [{"type": "input_text", "text": "z" * 100}],
    }
    _compress_responses_item(item, _shrink_big)
    assert item["output"][0]["text"] == "z" * 100  # original untouched (rebuilt, not mutated)


def test_function_call_output_string_still_compresses():
    item = {"type": "function_call_output", "call_id": "c1", "output": "q" * 100}
    new, changed = _compress_responses_item(item, _shrink_big)
    assert changed is True and new["output"] == "COMPRESSED"


def test_function_call_output_skip_leaves_unchanged():
    item = {"type": "function_call_output", "call_id": "c1", "output": "small"}
    new, changed = _compress_responses_item(item, _shrink_big)
    assert changed is False and new is item


def test_other_item_types_are_untouched():
    for item in (
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "x" * 100}]},
        {"type": "custom_tool_call", "name": "exec", "input": "shell(...)"},
        {"type": "reasoning", "summary": []},
        {"type": "custom_tool_call_output", "output": "a string, not a list"},  # defensive
    ):
        new, changed = _compress_responses_item(item, _shrink_big)
        assert changed is False and new is item


# ── handler smoke: a custom_tool_call_output request round-trips (no 500) ──

class _CaptureTransport(httpx.AsyncBaseTransport):
    def __init__(self):
        self.forwarded = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.forwarded = json.loads(request.content.decode())
        body = json.dumps({
            "id": "resp_1", "object": "response", "status": "completed",
            "output": [{"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": "ok"}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }).encode()
        return httpx.Response(200, headers={"content-type": "application/json"}, content=body)


def test_handler_accepts_custom_tool_call_output_and_keeps_shape():
    tr = _CaptureTransport()
    client = TestClient(create_app(http_client=httpx.AsyncClient(transport=tr)))
    req = {
        "model": "gpt-5",
        "input": [
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "check the dir"}]},
            {"type": "custom_tool_call_output", "call_id": "c1", "output": [
                {"type": "input_text", "text": "Script completed\nOutput:\n"},
                {"type": "input_text", "text": "Exit code: 0\nOutput:\n" + "line\n" * 50},
            ]},
        ],
        "stream": False,
    }
    r = client.post("/v1/responses", json=req)
    assert r.status_code == 200
    # The item was forwarded and its list-of-blocks shape is intact (no backend in
    # tests → compression skipped, but the item must not be corrupted).
    fco = [it for it in tr.forwarded["input"] if it.get("type") == "custom_tool_call_output"]
    assert len(fco) == 1
    assert isinstance(fco[0]["output"], list)
    assert [b["type"] for b in fco[0]["output"]] == ["input_text", "input_text"]
