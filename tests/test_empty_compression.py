"""Regression (issue #38): a compression backend can return an empty or
whitespace-only string for valid non-empty input (the hosted /api/compress relayed
whatever the GPU worker produced). An empty result is NOT a 100%-savings win:
count_tokens("") == 0 makes savings_ratio ≈ 1.0, which sails past the refusal
threshold and would forward an EMPTY prompt downstream. compress() must treat it as
a failed attempt and keep the original content.
"""
from paritok.config import ParitokConfig
from paritok.pipelines.compress import CompressionPipeline


class _EmptyModel:
    """Stand-in backend that returns an empty/whitespace body for any input."""
    def __init__(self, out=""):
        self._out = out

    def compress(self, *args, **kwargs):
        return self._out


def _pipeline(out=""):
    cfg = ParitokConfig()
    cfg.compression.min_tokens = 1        # force a compression attempt
    cfg.compression.refusal_threshold = 0.0
    pipe = CompressionPipeline(cfg)
    pipe._model = _EmptyModel(out)        # inject the empty backend
    return pipe


TEXT = "\n".join(f"{i:>4}\tdef f_{i}(): return {i}" for i in range(200))


def test_empty_compression_keeps_original():
    pipe = _pipeline("")
    cr = pipe.compress(TEXT, query="summarize")
    assert cr.compressed == TEXT, "empty compression was forwarded instead of the original"
    assert cr.metadata.get("skipped") is True
    assert cr.metadata.get("reason") == "empty_compression"
    assert cr.compressed_tokens == cr.original_tokens


def test_whitespace_only_compression_keeps_original():
    pipe = _pipeline("   \n\t  ")
    cr = pipe.compress(TEXT, query="q")
    assert cr.compressed == TEXT
    assert cr.metadata.get("skipped") is True
    assert cr.metadata.get("reason") == "empty_compression"


# --- strategy-level defense: the hosted GpuServerStrategy itself must never hand
#     back an empty prompt for non-empty input, even called directly (SDK use). ---

class _FakeHTTPResp:
    def __init__(self, payload):
        self._payload, self.status_code = payload, 200

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def test_gpu_server_empty_body_returns_original(monkeypatch):
    import httpx
    from paritok.config import GpuServerConfig
    from paritok.strategies.gpu_server import GpuServerStrategy

    # Hosted endpoint says GPU is up but relays an empty compressed body.
    monkeypatch.setattr(httpx, "post",
                        lambda *a, **k: _FakeHTTPResp({"compressed": "", "gpu_available": True}))
    strat = GpuServerStrategy(GpuServerConfig(base_url="http://x", api_key="k"))
    out = strat.compress("real non-empty content that must survive")
    assert out == "real non-empty content that must survive"

