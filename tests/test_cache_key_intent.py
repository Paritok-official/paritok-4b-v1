"""Regression: the compressed-result cache was keyed on content alone.

`sid = content_hash(content)` was used both as the shadow id and as the cache
key, so the first caller's answer was returned to every later caller regardless
of what they asked for:

    query="Fix the IntegrityError"             level=L0  -> 159 tok, cache_hit=False
    query="Explain the tax rounding TODO"      level=L3  -> 159 tok, cache_hit=True
    identical output: yes

That is worse than an ordinary stale-cache bug, because `query` is documented as
"USER INTENT -- the agent's current task. Drives keep/drop." Query-conditioned
compression stops happening the moment the same bytes are seen twice, which in a
coding agent is constant: the same file across turns, the same test output after
a re-run, the same schema every request.

`level` was affected the same way, and it looks like "levels do nothing" -- L0
through L3 come back byte-identical, and only the timings (9.6s, then 0.0s,
0.0s, 0.0s) reveal it as a cache hit.

The key now covers content, level, kind and query. `sid` stays content-only on
purpose: expand_context resolves originals by content and that is correct.
"""

from paritok import CompressionPipeline, ParitokConfig

CONTENT = (
    '  File "app/db/session.py", line 214, in commit\n'
    "sqlalchemy.exc.IntegrityError: duplicate key 0x1f4 violates \"orders_pkey\"\n"
    "def compute_tax(amount, rate):\n    return amount * rate  # TODO rounding\n"
) * 40


class _RecordingModel:
    """Returns a distinct answer per (query, level) so cache reuse is visible."""

    def __init__(self):
        self.calls = []

    def compress(self, text, *, query=None, level=None, kind=None, **_):
        self.calls.append((query, level, kind))
        return f"compressed for query={query!r} level={level!r}"


def _pipeline():
    pipeline = CompressionPipeline(ParitokConfig.load())
    model = _RecordingModel()
    pipeline._model = model
    return pipeline, model


def test_a_different_query_is_not_served_from_cache():
    pipeline, model = _pipeline()

    first = pipeline.compress(CONTENT, query="Fix the IntegrityError", kind="tool_output")
    second = pipeline.compress(CONTENT, query="Explain the rounding TODO", kind="tool_output")

    assert first.metadata.get("cache_hit") is not True
    assert second.metadata.get("cache_hit") is not True
    assert first.compressed != second.compressed
    assert len(model.calls) == 2, "the second intent must reach the model"


def test_a_different_level_is_not_served_from_cache():
    pipeline, model = _pipeline()

    low = pipeline.compress(CONTENT, query="same", kind="tool_output", level="L0")
    high = pipeline.compress(CONTENT, query="same", kind="tool_output", level="L3")

    assert low.compressed != high.compressed
    assert [c[1] for c in model.calls] == ["L0", "L3"]


def test_an_identical_call_still_hits_the_cache():
    """The fix must not turn the cache off."""
    pipeline, model = _pipeline()

    pipeline.compress(CONTENT, query="same", kind="tool_output", level="L1")
    again = pipeline.compress(CONTENT, query="same", kind="tool_output", level="L1")

    assert again.metadata.get("cache_hit") is True
    assert len(model.calls) == 1, "an identical call must not reach the model twice"


def test_shadow_id_stays_content_only():
    """expand_context resolves by content; widening the cache key must not touch it."""
    pipeline, _ = _pipeline()

    first = pipeline.compress(CONTENT, query="one intent", kind="tool_output", level="L0")
    second = pipeline.compress(CONTENT, query="another intent", kind="tool_output", level="L3")

    assert first.shadow_id == second.shadow_id
