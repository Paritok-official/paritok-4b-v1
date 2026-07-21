"""Regression: a Responses request carrying a function_call_output must not 500.

A loop var named `body` in handle_responses shadowed the request `body` dict, so
after compressing a tool output `body.get("model")` hit a str -> AttributeError ->
500. This drives the full handler (not just the header helper) to catch it.
"""
import json
import httpx
from starlette.testclient import TestClient

from paritok.proxy.server import create_app


class _RespTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = json.dumps({
            "id": "resp_1", "object": "response", "status": "completed",
            "output": [{"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": "ok"}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }).encode()
        return httpx.Response(200, headers={"content-type": "application/json"}, content=body)


def _client() -> TestClient:
    return TestClient(create_app(http_client=httpx.AsyncClient(transport=_RespTransport())))


def _req(output: str) -> dict:
    return {
        "model": "gpt-5",
        "input": [
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "fix the bug"}]},
            {"type": "function_call_output", "call_id": "c1", "output": output},
        ],
        "tools": [{"type": "function", "name": "foo",
                   "parameters": {"type": "object", "properties": {}}}],
        "stream": False,
    }


def test_function_call_output_does_not_clobber_request_body():
    # codex-style shell-wrapped output — exercises _split_codex_header + the record() call
    r = _client().post("/v1/responses", json=_req("Exit code: 0\nOutput:\nsome tool output"))
    assert r.status_code == 200


def test_plain_function_call_output_ok():
    r = _client().post("/v1/responses", json=_req("plain tool output, no header"))
    assert r.status_code == 200
