"""Regression: widening the cache key must not disable the path short-circuit.

The intent-aware cache key (query/level/kind widened, see
test_cache_key_intent.py) stores results under `cache_key`, but the path
short-circuit at step 1b resolves a prior [REF] tag by the content-only
`sid`. If the result is only stored under `cache_key`, every re-read of the
same source file misses the short-circuit and re-invokes the model.

The fix indexes the tagged result under both keys.
"""

from paritok import CompressionPipeline, ParitokConfig


class _RecordingModel:
    def __init__(self):
        self.calls = []

    def compress(self, text, *, query=None, level=None, kind=None, **_):
        self.calls.append((query, level, kind))
        return f"compressed<{query}|{level}>"


def _pipeline():
    pipeline = CompressionPipeline(ParitokConfig.load())
    model = _RecordingModel()
    pipeline._model = model
    return pipeline, model


def _file(n):
    return "".join(f"def f{i}(a, b):\n    return a * b + {i}\n" for i in range(n))


def test_partial_re_read_still_short_circuits():
    """A re-read of the same source must reuse the existing [REF] tag."""
    pipeline, model = _pipeline()

    first = pipeline.compress(
        _file(200), query="first intent", kind="file_read", source="/app/tax.py"
    )
    assert first.metadata.get("cache_hit") is False

    second = pipeline.compress(
        _file(100), query="second intent", kind="file_read", source="/app/tax.py"
    )

    assert second.metadata.get("path_shortcircuit") is True
    assert second.shadow_id == first.shadow_id
    assert len(model.calls) == 1, "the re-read must not reach the model"


def test_full_re_read_still_short_circuits():
    pipeline, model = _pipeline()

    pipeline.compress(_file(200), query="one", kind="file_read", source="/app/tax.py")
    again = pipeline.compress(_file(200), query="two", kind="file_read", source="/app/tax.py")

    assert again.metadata.get("path_shortcircuit") is True
    assert len(model.calls) == 1


def test_intent_cache_isolation_survives_the_dual_write():
    """The sid-indexed copy must not leak across intents on the sourceless path."""
    pipeline, model = _pipeline()
    content = _file(200)

    first = pipeline.compress(content, query="fix the bug", kind="tool_output")
    second = pipeline.compress(content, query="explain the design", kind="tool_output")

    assert first.compressed != second.compressed
    assert second.metadata.get("cache_hit") is False
    assert len(model.calls) == 2
