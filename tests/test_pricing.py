import pytest

from paritok.proxy.pricing import input_price_per_token, input_usd_per_mtok


@pytest.mark.parametrize(
    ("model", "expected_price"),
    [
        ("MiniMax-M3", 0.30),
        ("MiniMax-M2.7", 0.30),
        ("minimax/MiniMax-M3", 0.30),
        ("minimax/MiniMax-M2.7", 0.30),
    ],
)
def test_minimax_input_prices(model, expected_price):
    # No input length given -> standard (<=512K) tier, unchanged from before #50.
    assert input_usd_per_mtok(model) == (expected_price, True)


@pytest.mark.parametrize(
    ("model", "input_tokens", "expected"),
    [
        # MiniMax-M3 is length-tiered: <=512K -> 0.30, >512K -> 0.60. Boundary
        # token (exactly 512,000) bills at the STANDARD tier (issue #50 comment).
        ("MiniMax-M3", 600_000, 0.60),
        ("MiniMax-M3", 100_000, 0.30),
        ("MiniMax-M3", 512_000, 0.30),        # boundary: standard
        ("MiniMax-M3", 512_001, 0.60),        # one over: long-context
        ("MiniMax-M3", None, 0.30),           # omitted -> standard
        # namespaced id resolves the same
        ("minimax/MiniMax-M3", 600_000, 0.60),
        ("minimax/MiniMax-M3", 100_000, 0.30),
        # M2.7 has no ladder -> flat rate, input length ignored
        ("MiniMax-M2.7", 600_000, 0.30),
    ],
)
def test_minimax_m3_length_tiers(model, input_tokens, expected):
    assert input_usd_per_mtok(model, input_tokens=input_tokens) == (expected, True)


@pytest.mark.parametrize(
    ("model", "input_tokens", "expected", "matched"),
    [
        # Non-tiered models ignore input_tokens entirely (existing callers unaffected).
        ("claude-sonnet-4-20250514", 999_999_999, 3.0, True),
        ("gpt-5", 700_000, 1.25, True),
        # Unknown model -> $3/M default, matched False, length irrelevant.
        ("some-unknown-model", 600_000, 3.0, False),
    ],
)
def test_non_tiered_models_ignore_length(model, input_tokens, expected, matched):
    assert input_usd_per_mtok(model, input_tokens=input_tokens) == (expected, matched)


def test_input_price_per_token_threads_length():
    assert input_price_per_token("MiniMax-M3", input_tokens=600_000) == 0.60 / 1_000_000
    assert input_price_per_token("MiniMax-M3", input_tokens=100_000) == 0.30 / 1_000_000
    assert input_price_per_token("MiniMax-M3") == 0.30 / 1_000_000  # omitted -> standard
