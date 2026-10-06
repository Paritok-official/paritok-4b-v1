import httpx
from unittest.mock import patch, MagicMock
from paritok.config import ParitokConfig
from paritok.proxy.server import create_app
from starlette.testclient import TestClient

def test_proxy_initializes_client_with_config_timeout():
    """Verify that the proxy's internal HTTP client is created using the configured timeout."""
    with patch("httpx.AsyncClient") as mock_client:
        with patch("paritok.config.ParitokConfig.load") as mock_load:
            # Use the real ParitokConfig to avoid missing attributes during engine init
            cfg = ParitokConfig()
            cfg.upstream_timeout = 42.0
            mock_load.return_value = cfg

            # Create the app, which should trigger the client initialization
            create_app(config_path="fake_path")

            # Verify AsyncClient was called with the correct timeout from the config
            mock_client.assert_called_once_with(timeout=42.0)

def test_proxy_returns_504_on_upstream_timeout():
    """Verify that an upstream httpx.TimeoutException is surfaced as an HTTP 504 Gateway Timeout."""
    # Mock the http_client to simulate a timeout exception during the POST request
    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_client.post.side_effect = httpx.TimeoutException("Request timed out")

    # Inject the mocked client into the app
    app = create_app(http_client=mock_client)

    # Use Starlette's sync TestClient to call the app
    with TestClient(app) as client:
        # Call any endpoint that forwards to upstream (e.g., OpenAI Chat Completions)
        resp = client.post("/v1/chat/completions", json={"model": "gpt-4", "messages": []})

        assert resp.status_code == 504
        assert "Upstream timed out" in resp.text
