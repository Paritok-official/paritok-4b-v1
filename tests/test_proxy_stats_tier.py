"""Regression for #50: a >512K MiniMax-M3 request must be priced at the
long-context input tier ($0.60/M), not the standard $0.30/M — otherwise
estimated_cost_saved_usd under-counts large-context savings by ~2x. The tier is
picked from the request's original input length, threaded into the per-model
bucket key at record time.
"""
from types import SimpleNamespace

from paritok.proxy.server import ProxyStats

_WRITE = 1.25  # first turn per (model, tier) is a cache write at 1.25x base


def _record_one(orig, comp, model):
    s = ProxyStats()
    s.record(
        SimpleNamespace(
            tools_filtered=0, items_compressed=1,
            original_tokens=orig, compressed_tokens=comp,
        ),
        model=model,
    )
    return s


def test_minimax_over_512k_priced_at_long_context_rate():
    orig, comp = 600_000, 300_000     # >512K input -> $0.60/M tier
    s = _record_one(orig, comp, "MiniMax-M3")
    expected = round((orig - comp) * (0.60 / 1_000_000) * _WRITE, 4)
    assert s.estimated_cost_saved_usd == expected


def test_minimax_under_512k_priced_at_standard_rate():
    orig, comp = 100_000, 50_000      # <=512K input -> $0.30/M tier
    s = _record_one(orig, comp, "MiniMax-M3")
    expected = round((orig - comp) * (0.30 / 1_000_000) * _WRITE, 4)
    assert s.estimated_cost_saved_usd == expected


def test_over_512k_is_double_the_under_512k_rate():
    # The whole point of #50: same saved-token count, the >512K request is worth
    # exactly 2x the standard-tier request (0.60 vs 0.30), no longer under-counted.
    big = _record_one(600_000, 300_000, "MiniMax-M3").estimated_cost_saved_usd
    small_same_saving = _record_one(400_000, 100_000, "MiniMax-M3").estimated_cost_saved_usd
    # both saved 300_000 tokens; big is at 0.60, small at 0.30
    assert round(big / small_same_saving, 3) == 2.0
