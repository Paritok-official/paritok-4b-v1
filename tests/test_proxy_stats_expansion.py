"""Stats accounting for the server-side expand_context resolve loop.

When the model calls expand_context, the proxy hands back the full original it
had compressed to a [REF:id] stub. record_expansion folds that re-delivered
original onto the compressed side (original untouched), so /stats stops
over-reporting savings: tokens_saved falls, goes negative when a turn expands
more than it compressed, and compression_ratio climbs toward/past 1.0.
"""
from types import SimpleNamespace

from paritok.proxy.server import ProxyStats


def _stats(orig, comp, model="claude"):
    s = ProxyStats()
    s.record(
        SimpleNamespace(
            tools_filtered=0, items_compressed=1,
            original_tokens=orig, compressed_tokens=comp,
        ),
        model=model,
    )
    return s


def test_expansion_folds_original_onto_compressed_side():
    # Compressed 20000 -> a 20-token stub, then the model expanded it (orig=20000).
    s = _stats(orig=20000, comp=20)
    assert s.total_original_tokens == 20000
    assert s.total_saved_tokens == 19980  # looks great before the expand

    s.record_expansion(20000, model="claude")

    # original is left alone; the re-delivered original lands on compressed.
    assert s.total_original_tokens == 20000
    assert s.total_compressed_tokens == 20 + 20000
    # Net saving is negative — we sent the stub AND the full original.
    assert s.total_saved_tokens == -20
    snap = s.snapshot()
    assert snap["tokens_saved"] == -20
    assert snap["compression_ratio"] > 1.0  # honestly reports "no savings"


def test_expansion_cost_saved_can_go_negative():
    # A small real compression, then a large expansion dwarfs it.
    s = _stats(orig=1000, comp=100)
    assert s.estimated_cost_saved_usd > 0
    s.record_expansion(50000, model="claude")
    assert s.estimated_cost_saved_usd < 0


def test_record_expansion_ignores_nonpositive():
    s = _stats(orig=1000, comp=100)
    before = (s.total_compressed_tokens, dict(s.by_model))
    s.record_expansion(0, model="claude")
    s.record_expansion(-5, model="claude")
    assert s.total_compressed_tokens == before[0]


def test_expansion_without_prior_content_uses_first_slot():
    # expand fired for a model with no compressed content bucket yet: must not crash
    # and must still register on the compressed side.
    s = ProxyStats()
    s.record_expansion(1234, model="gemini")
    assert s.total_compressed_tokens == 1234
    assert s.by_model["gemini"]["content_first_comp"] == 1234
