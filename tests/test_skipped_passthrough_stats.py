"""Passthrough (skipped) content is excluded from original/compressed so the
compression ratio reflects only what paritok actually compressed; skipped tokens are
tracked separately, by reason (below_min_tokens, gpu_unavailable, backend errors, ...),
like output tokens — outside the compressor. Surfaced in /stats as tokens_skipped +
skipped_by_reason (and rendered on the dashboard)."""
from paritok.middleware.wrapper import CompressionStats
from paritok.pipelines.compress import CompressionResult
from paritok.proxy.server import ProxyStats


def _cr(orig, comp, **meta):
    return CompressionResult(compressed="x" * comp, original_tokens=orig,
                             compressed_tokens=comp, metadata=meta)


def test_skipped_excluded_from_ratio_tracked_by_reason():
    s = CompressionStats()
    s.record_result(_cr(1000, 300))                                    # real compression
    s.record_result(_cr(200, 200, skipped=True, reason="below_min_tokens"))
    s.record_result(_cr(500, 500, skipped=True, reason="gpu_unavailable"))

    assert s.original_tokens == 1000        # the 700 passthrough is NOT counted here
    assert s.compressed_tokens == 300
    assert s.items_compressed == 1
    assert s.items_skipped == 2
    assert s.skipped_tokens == 700
    assert s.skipped_by_reason == {"below_min_tokens": 200, "gpu_unavailable": 500}
    # ratio reflects only the compressed 1000->300, not diluted toward 1.0 by the 700
    assert s.ratio == round(1 - 300 / 1000, 3)


def test_cache_hit_and_shortcircuit_still_count_as_compressed():
    s = CompressionStats()
    s.record_result(_cr(1000, 120, cache_hit=True))          # re-sent earlier compression
    s.record_result(_cr(800, 90, path_shortcircuit=True))    # re-read short-circuit
    assert s.original_tokens == 1800 and s.compressed_tokens == 210
    assert s.cache_hits == 2 and s.items_compressed == 2
    assert s.skipped_tokens == 0


def test_proxystats_surfaces_skipped_and_keeps_ratio_clean():
    ps = ProxyStats()
    s = CompressionStats()
    s.record_result(_cr(1000, 300))
    s.record_result(_cr(200, 200, skipped=True, reason="below_min_tokens"))
    ps.record(s, model="claude-test")
    snap = ps.snapshot()

    assert snap["tokens_skipped"] == 200
    assert snap["skipped_by_reason"]["below_min_tokens"] == 200
    assert snap["input_tokens_original"] == 1000     # skip excluded from the ratio base
    assert snap["file_orig"] == 1000
    assert snap["compression_ratio"] == round(300 / 1000, 3)
