"""Flash pricing and caller-owned model settings through the real HTTP boundary."""
import json

import httpx
import pytest

pytest.importorskip("starlette")
from starlette.testclient import TestClient  # noqa: E402

from paritok.proxy.server import create_app  # noqa: E402


@pytest.mark.parametrize("region", ["io", "cn"])
@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("effort", [None, "high"])
@pytest.mark.parametrize(
    "model", ["MiniMax-M3.1-Flash", "MiniMax-M3", "MiniMax-M2.7", "custom-saved-model"]
)
def test_proxy_preserves_model_and_reasoning(tmp_path, region, protocol, model, effort):
    config = tmp_path / "paritok.yaml"
    config.write_text("history:\n  enabled: false\ntool_discovery:\n  strategy: passthrough\n")
    seen = []

    def upstream(request):
        seen.append(request)
        return httpx.Response(200, json={"content": [], "choices": []})

    transport = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    base = f"https://api.minimax.{region}"
    app = create_app(
        config_path=str(config), http_client=transport,
        openai_base_url=base + "/v1", anthropic_base_url=base + "/anthropic",
    )
    content = [{"type": "text", "text": "Describe this attachment."}]
    body = {"model": model, "messages": [{"role": "user", "content": content}],
            "max_tokens": 1024}
    if protocol == "openai":
        content.append({"type": "image_url", "image_url": {"url": "https://example.test/a.png"}})
        if effort is not None:
            body["reasoning_effort"] = effort
        path, expected_url = "/v1/chat/completions", base + "/v1/chat/completions"
    else:
        content.append({"type": "image", "source": {"type": "url", "url": "https://example.test/a.png"}})
        if effort is not None:
            body["thinking"] = {"type": "adaptive"}
            body["output_config"] = {"effort": effort}
        path, expected_url = "/v1/messages", base + "/anthropic/v1/messages"

    with TestClient(app) as client:
        assert client.post(path, json=body).status_code == 200
    assert len(seen) == 1
    assert str(seen[0].url) == expected_url
    outgoing = json.loads(seen[0].content)
    for key, value in body.items():
        assert outgoing[key] == value
    assert "service_tier" not in outgoing
    assert "reasoning_split" not in outgoing
    if effort is None:
        assert not {"thinking", "output_config", "reasoning_effort"} & outgoing.keys()


@pytest.mark.parametrize(
    ("model", "rate"),
    [("MiniMax-M3.1-Flash", 0.15), ("minimax/minimax-m3.1-flash:free", 0.0),
     ("minimax-m3.1-flash-free", 0.0)],
)
def test_flash_stats_after_http_forward(tmp_path, monkeypatch, model, rate):
    config = tmp_path / "paritok.yaml"
    config.write_text(
        "history:\n  enabled: true\n  keep_recent_turns: 2\n"
        "  context_threshold: 0.5\n  context_window: 1000\n"
        "tool_discovery:\n  strategy: passthrough\n"
    )
    monkeypatch.setattr("paritok.strategies.local_model.LocalModelStrategy.compress",
                        lambda self, content, **kw: "Summary of older history.")
    seen = []

    def upstream(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "role": "assistant", "content": "done"}}]})

    app = create_app(config_path=str(config), openai_base_url="https://upstream.test/v1",
                     http_client=httpx.AsyncClient(transport=httpx.MockTransport(upstream)))
    messages = [{"role": role, "content": f"Turn {i}: " + "history text " * 100}
                for i in range(6) for role in ("user", "assistant")]
    with TestClient(app) as client:
        for _ in range(2):
            assert client.post("/v1/chat/completions", json={
                "model": model, "messages": messages}).status_code == 200
        stats = client.get("/stats").json()
    assert len(seen) == 2
    assert all(body["model"] == model for body in seen)
    assert stats["total_requests"] == 2
    assert stats["tokens_saved"] > 0
    # The existing estimator uses 1.25x for the first turn and cache-read for the next.
    # This asserts its accounting convention, not a provider cache-write price.
    expected = round(stats["tokens_saved"] / 2 * rate / 1_000_000 * (1.25 + 0.1), 4)
    assert app.state.proxy_stats.estimated_cost_saved_usd == expected
    assert stats["estimated_cost_saved_usd"] == f"${expected:.2f}"
