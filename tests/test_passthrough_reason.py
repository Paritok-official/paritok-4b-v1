"""Regression (#30): a backend that echoes the input verbatim is a PASSTHROUGH (the hosted
GPU does this on gpu_available:false), not a weak compression. It used to be recorded as
`below_refusal_threshold` — indistinguishable in the trace from content that genuinely
didn't compress. It now gets a specific reason (`gpu_unavailable` / `backend_passthrough`).
"""
from paritok.config import ParitokConfig
from paritok.pipelines.compress import CompressionPipeline

CONTENT = "\n".join(f"{i:>4}\tdef f_{i}(): return {i}" for i in range(200))


class _EchoModel:
    """Returns the input verbatim — what GpuServerStrategy does when the GPU is offline."""
    def compress(self, content, **kwargs):
        return content


class _WeakModel:
    """Compresses a little (below the refusal threshold) but does change the bytes."""
    def compress(self, content, **kwargs):
        return content[:-3] + "x"


def _pipe(use_gpu):
    cfg = ParitokConfig()
    cfg.use_gpu_server = use_gpu
    cfg.compression.min_tokens = 1          # force a compression attempt
    cfg.compression.refusal_threshold = 0.05
    return CompressionPipeline(cfg)


def test_gpu_passthrough_labelled_gpu_unavailable():
    pipe = _pipe(use_gpu=True)
    pipe._model = _EchoModel()
    r = pipe.compress(CONTENT, query="fix", kind="file_read")
    assert r.metadata.get("skipped") is True
    assert r.metadata.get("reason") == "gpu_unavailable", r.metadata


def test_local_passthrough_labelled_backend_passthrough():
    pipe = _pipe(use_gpu=False)
    pipe._model = _EchoModel()
    r = pipe.compress(CONTENT, query="fix", kind="file_read")
    assert r.metadata.get("reason") == "backend_passthrough", r.metadata


def test_genuine_weak_compression_still_below_refusal_threshold():
    pipe = _pipe(use_gpu=True)
    pipe._model = _WeakModel()
    r = pipe.compress(CONTENT, query="fix", kind="file_read")
    assert r.metadata.get("reason") == "below_refusal_threshold", r.metadata
