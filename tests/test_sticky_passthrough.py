"""An item forwarded verbatim because the backend couldn't compress it stays verbatim.

loopbench r11 (task 4): a Bash output passed through uncompressed during a GPU outage
(`gpu_unavailable`) got compressed 14 requests later when the GPU came back. That
changed bytes already in the upstream prompt cache, and everything cached after them
(156k tokens) was written again, ≈ $0.54 on one request.
"""
from paritok.config import ParitokConfig
from paritok.middleware.wrapper import ParitokEngine
from paritok.pipelines.compress import CompressionPipeline

TEXT = "\n".join(f"{n}\tvalue_{n} = compute_something_long(row_{n}, column_{n}, extra)"
                 for n in range(1, 120))


class _Backend:
    """Echoes its input (GPU unavailable), raises, or compresses, per call."""

    def __init__(self, *modes):
        self.modes, self.calls = list(modes), 0

    def compress(self, text, **kwargs):
        mode = self.modes[min(self.calls, len(self.modes) - 1)]
        self.calls += 1
        if mode == "echo":
            return text
        if mode == "raise":
            raise TimeoutError("backend timed out")
        if mode == "weak":
            return text[:-20]
        return "summary: 119 assignments computing values from rows and columns"


def _pipe(backend, gpu=True):
    cfg = ParitokConfig()
    cfg.use_gpu_server = gpu
    pipe = CompressionPipeline(cfg)
    pipe._model = backend
    return pipe


def test_gpu_unavailable_item_stays_verbatim_when_the_gpu_comes_back():
    backend = _Backend("echo", "ok")
    pipe = _pipe(backend)
    first = pipe.compress(TEXT, query="find the bug", item_key="tu1#0")
    assert first.metadata["reason"] == "gpu_unavailable" and first.compressed == TEXT
    again = pipe.compress(TEXT, query="now a different intent", item_key="tu1#0")
    assert again.compressed == TEXT and again.metadata.get("kept_verbatim")
    assert again.metadata["reason"] == "gpu_unavailable"
    assert backend.calls == 1, "a kept item is not sent to the backend again"
    # a NEW item (same content, e.g. the file read again later) is compressed normally
    fresh = pipe.compress(TEXT, query="x", item_key="tu2#0")
    assert not fresh.metadata.get("skipped") and fresh.compressed != TEXT


def test_backend_error_and_weak_compression_are_sticky_too():
    for mode, reason in (("raise", "backend_error:TimeoutError"),
                         ("weak", "below_refusal_threshold")):
        pipe = _pipe(_Backend(mode, "ok"))
        assert pipe.compress(TEXT, item_key="k#0").metadata["reason"] == reason
        assert pipe.compress(TEXT, item_key="k#0").compressed == TEXT


def test_deterministic_skips_are_not_recorded():
    pipe = _pipe(_Backend("ok"))
    small = "tiny output"
    assert pipe.compress(small, item_key="s#0").metadata["reason"] == "below_min_tokens"
    assert not pipe._kept_verbatim


def test_without_item_key_identical_content_is_the_same_item():
    pipe = _pipe(_Backend("echo", "ok"))
    pipe.compress(TEXT)
    assert pipe.compress(TEXT).compressed == TEXT


def test_engine_forwards_a_history_item_byte_identical_across_requests():
    """Through the engine: the tool_result forwarded verbatim on request 1 is forwarded
    byte-identical on request 2 (cache-safe), while request 2's new result compresses."""
    cfg = ParitokConfig()
    cfg.use_gpu_server = True
    cfg.history.enabled = False
    engine = ParitokEngine(cfg)
    engine.pipeline._model = _Backend("echo", "ok")

    def turn(tid):
        return [{"role": "assistant", "content": [
                    {"type": "tool_use", "id": tid, "name": "Bash", "input": {"command": "run"}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": tid, "content": TEXT + f"\n# {tid}"}]}]

    msgs1 = [{"role": "user", "content": "go"}] + turn("t1")
    out1, *_ = engine.process_request(msgs1, None, upstream_model="claude-sonnet-4-5-20250929")
    msgs2 = msgs1 + turn("t2")
    out2, *_ = engine.process_request(msgs2, None, upstream_model="claude-sonnet-4-5-20250929")
    assert out2[2] == out1[2], "the earlier tool_result must not change"
    assert out2[2]["content"][0]["content"] == TEXT + "\n# t1"
    assert out2[4]["content"][0]["content"].startswith("[REF:")
