"""Regression (issue #36): a compression-backend failure (timeout / model loading /
connection error) must NOT crash the request. compress() should fall back to
forwarding the original content uncompressed, not raise."""
from paritok.config import ParitokConfig
from paritok.pipelines.compress import CompressionPipeline


class _BoomModel:
    """Stand-in compression backend that always fails, like a timed-out Ollama."""
    def compress(self, *args, **kwargs):
        raise TimeoutError("Compression request timed out after 0.05s.")


def _pipeline():
    cfg = ParitokConfig()
    cfg.compression.min_tokens = 1        # force a compression attempt
    cfg.compression.refusal_threshold = 0.0
    pipe = CompressionPipeline(cfg)
    pipe._model = _BoomModel()            # inject the failing backend
    return pipe


def test_backend_failure_falls_back_to_original():
    pipe = _pipeline()
    text = "\n".join(f"{i:>4}\tdef f_{i}(): return {i}" for i in range(200))
    cr = pipe.compress(text, query="summarize")
    # graceful passthrough: no exception, original content forwarded unchanged
    assert cr.metadata.get("skipped") is True
    assert cr.metadata.get("reason", "").startswith("backend_error")
    assert cr.compressed == text
    assert cr.compressed_tokens == cr.original_tokens


def test_backend_failure_reason_names_exception():
    pipe = _pipeline()
    text = "\n".join(f"{i:>4}\tline {i} content here" for i in range(200))
    cr = pipe.compress(text, query="q")
    assert cr.metadata["reason"] == "backend_error:TimeoutError"
