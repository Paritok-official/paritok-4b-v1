"""Regression: a streaming upstream that disconnects before sending a response
(httpx.RemoteProtocolError) must NOT surface as an unhandled 500. It should
retry once, and if it still fails, degrade to a clean 502.

Reproduces the traceback from a real report: `_stream_anthropic` ->
http_client.stream(...) raising "Server disconnected without sending a response".
"""
import json
import httpx
from starlette.testclient import TestClient

from paritok.proxy.server import create_app


class _FlakyTransport(httpx.AsyncBaseTransport):
    """Fails the first N streaming POSTs with RemoteProtocolError, then succeeds."""

    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.calls = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise httpx.RemoteProtocolError(
                "Server disconnected without sending a response.", request=request)
        # minimal valid non-streaming Anthropic message (no virtual markers)
        body = json.dumps({
            "id": "msg_1", "type": "message", "role": "assistant",
            "model": "claude-sonnet-4", "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }).encode()
        return httpx.Response(200, headers={"content-type": "application/json"}, content=body)


def _client(fail_times: int) -> TestClient:
    transport = _FlakyTransport(fail_times)
    http_client = httpx.AsyncClient(transport=transport)
    app = create_app(http_client=http_client)
    return TestClient(app)


_REQ = {"model": "claude-sonnet-4", "max_tokens": 16, "stream": True,
        "messages": [{"role": "user", "content": "hi"}]}


def test_transient_disconnect_retries_and_succeeds():
    # First stream attempt drops; the one retry succeeds -> not a 500.
    client = _client(fail_times=1)
    r = client.post("/v1/messages", json=_REQ)
    assert r.status_code == 200


def test_persistent_disconnect_degrades_to_502_not_500():
    # Both attempts drop -> clean 502, never an unhandled 500.
    client = _client(fail_times=2)
    r = client.post("/v1/messages", json=_REQ)
    assert r.status_code == 502
    assert "RemoteProtocolError" in r.json().get("error", "")
