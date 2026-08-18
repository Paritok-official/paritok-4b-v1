"""Regression: the proxy must not forward `accept-encoding` upstream (#28).

Forwarding the client's `accept-encoding: gzip, deflate, br, zstd` made the upstream answer
in br/zstd, which httpx can't decode without brotli/zstandard (the `[proxy]` extra ships
neither) — so `resp.json()` raised and every such request 502'd as "Upstream returned
invalid JSON". Stripping it lets httpx negotiate only codecs it can actually decode.
"""
import gzip
import importlib.util
import json

import pytest

from paritok.proxy.server import _forward_header_dict


def test_forward_header_dict_strips_accept_encoding():
    out = _forward_header_dict([
        ("Host", "api.anthropic.com"),
        ("Content-Length", "42"),
        ("Accept-Encoding", "gzip, deflate, br, zstd"),
        ("Authorization", "Bearer k"),
        ("anthropic-version", "2023-06-01"),
    ])
    lower = {k.lower() for k in out}
    assert "accept-encoding" not in lower          # the #28 fix
    assert "host" not in lower and "content-length" not in lower
    assert out.get("Authorization") == "Bearer k"  # non-hop headers pass through untouched
    assert out.get("anthropic-version") == "2023-06-01"


_HAS_DECODER = (importlib.util.find_spec("brotli") is not None
                or importlib.util.find_spec("brotlicffi") is not None
                or importlib.util.find_spec("zstandard") is not None)


@pytest.mark.skipif(_HAS_DECODER, reason="#28 reproduction needs NO brotli/zstandard decoder")
def test_br_upstream_no_longer_502s_end_to_end():
    # Faithful #28 repro: an upstream that answers in br whenever the forwarded request
    # advertises it (as Anthropic behind Cloudflare can). Pre-fix the proxy forwarded the
    # client's br/zstd and 502'd; post-fix it strips it, so httpx only asks for codecs it
    # can decode and the request succeeds.
    import httpx
    from starlette.testclient import TestClient
    from paritok.proxy.server import create_app

    good = {"id": "msg_1", "type": "message", "role": "assistant",
            "content": [{"type": "text", "text": "hi"}], "model": "claude-opus-5",
            "stop_reason": "end_turn", "usage": {"input_tokens": 5, "output_tokens": 2}}

    def handler(request):
        ae = request.headers.get("accept-encoding", "")
        if "br" in ae or "zstd" in ae:
            return httpx.Response(200, headers={"content-encoding": "br", "content-type": "application/json"},
                                  content=gzip.compress(json.dumps(good).encode()))
        return httpx.Response(200, json=good)

    mock = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(anthropic_base_url="http://fake-upstream", http_client=mock)
    client = TestClient(app)
    r = client.post(
        "/v1/messages",
        json={"model": "claude-opus-5", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]},
        headers={"authorization": "Bearer x", "anthropic-version": "2023-06-01",
                 "accept-encoding": "gzip, deflate, br, zstd"},
    )
    assert r.status_code == 200, r.text  # pre-#28-fix: 502 "Upstream returned invalid JSON"
