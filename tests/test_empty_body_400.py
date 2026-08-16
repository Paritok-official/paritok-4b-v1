"""Regression (issue #31): an empty or malformed request body must return HTTP 400,
not a 500. Every POST handler parsed `json.loads(await request.body())` directly, so
an empty body raised an unhandled JSONDecodeError and surfaced as an opaque 500 —
the natural thing to send when probing whether the proxy is up. `_read_json_body`
now turns a bad body into a clean 400 before any forwarding.
"""
import pytest

pytest.importorskip("starlette")
from starlette.testclient import TestClient  # noqa: E402

CONFIG_YAML = "use_gpu_server: false\ntool_discovery:\n  strategy: passthrough\n"

# Every POST route that parses a JSON body.
ENDPOINTS = [
    "/v1/messages",
    "/v1/chat/completions",
    "/v1/responses",
    "/v1/messages/count_tokens",
]


def _client(tmp_path):
    cfg = tmp_path / "paritok.yaml"
    cfg.write_text(CONFIG_YAML, encoding="utf-8")
    from paritok.proxy import server
    app = server.create_app(config_path=str(cfg), openai_base_url="http://upstream.test")
    return TestClient(app)


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_empty_body_is_400_not_500(tmp_path, endpoint):
    with _client(tmp_path) as client:
        r = client.post(endpoint, content=b"")
    assert r.status_code == 400, f"{endpoint}: empty body should 400, got {r.status_code}"
    assert "error" in r.json()


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_malformed_json_is_400_not_500(tmp_path, endpoint):
    with _client(tmp_path) as client:
        r = client.post(endpoint, content=b"{not valid json")
    assert r.status_code == 400, f"{endpoint}: malformed body should 400, got {r.status_code}"
