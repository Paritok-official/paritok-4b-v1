import pytest

from paritok.proxy.pricing import cache_read_multiplier, input_price_per_token, input_usd_per_mtok


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


@pytest.mark.parametrize("input_tokens", [None, 100_000, 600_000, 1_000_000])
@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("MiniMax-M3.1-Flash", 0.15),
        (" MINIMAX-M3.1-FLASH ", 0.15),
        ("minimax/MiniMax-M3.1-Flash", 0.15),
        ("minimax/minimax-m3.1-flash:free", 0.0),
        ("minimax-m3.1-flash-free", 0.0),
        ("opencode/minimax-m3.1-flash-free", 0.0),
    ],
)
def test_flash_flat_price_overrides_m3_tiers(model, expected, input_tokens):
    assert input_usd_per_mtok(model, input_tokens) == (expected, True)
    assert input_price_per_token(model, input_tokens) == expected / 1_000_000


def test_input_price_per_token_threads_length():
    assert input_price_per_token("MiniMax-M3", input_tokens=600_000) == 0.60 / 1_000_000
    assert input_price_per_token("MiniMax-M3", input_tokens=100_000) == 0.30 / 1_000_000
    assert input_price_per_token("MiniMax-M3") == 0.30 / 1_000_000  # omitted -> standard


# ── Claude Opus / Fable and GPT-6, against the providers' published tables ──

@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-fable-5-1", 10.0),
        ("claude-fable-5", 10.0),
        ("claude-opus-5-5", 4.0),
        ("claude-opus-5-5[1m]", 4.0),
        ("anthropic/claude-opus-5-5", 4.0),
        ("claude-opus-5", 5.0),
        ("claude-opus-4-8", 5.0),
        ("claude-opus-4-7", 5.0),
        ("claude-opus-4-6", 5.0),
        ("claude-opus-4-5-20251101", 5.0),
        ("claude-opus-4-1-20250805", 15.0),
        ("claude-opus-4-20250514", 15.0),
        ("claude-opus-6", 4.0),
        ("gpt-6-cyber", 20.0),
        ("gpt-6-rosalind", 15.0),
    ],
)
def test_opus_fable_and_flat_gpt6_input_prices(model, expected):
    assert input_usd_per_mtok(model) == (expected, True)


@pytest.mark.parametrize(
    ("model", "short", "long"),
    [
        ("gpt-6-astra", 10.0, 20.0),
        ("gpt-6-sol", 2.0, 4.0),
        ("gpt-6-luna", 0.10, 0.20),
        ("gpt-6.1-sol", 2.0, 4.0),
        ("gpt-6.1-terra", 1.0, 2.0),
        ("openai/gpt-6.1-sol", 2.0, 4.0),
    ],
)
def test_gpt6_doubles_above_272k(model, short, long):
    assert input_usd_per_mtok(model) == (short, True)
    assert input_usd_per_mtok(model, input_tokens=272_000) == (short, True)
    assert input_usd_per_mtok(model, input_tokens=272_001) == (long, True)


@pytest.mark.parametrize(
    ("model", "mult"),
    [
        ("claude-fable-5-1", 0.025),        # $0.25 / $10
        ("claude-fable-5", 0.1),            # $1 / $10
        ("claude-opus-5-5", 0.05),          # $0.20 / $4
        ("claude-opus-5-5[1m]", 0.05),
        ("anthropic/claude-opus-5-5", 0.05),
        ("claude-opus-5", 0.1),             # $0.50 / $5
        ("claude-opus-4-8", 0.1),
        ("gpt-6-astra", 0.1),               # $1 / $10
        ("gpt-6-sol", 0.1),                 # $0.20 / $2
        ("gpt-6-luna", 0.1),                # $0.01 / $0.10
        ("gpt-6.1-sol", 0.05),              # $0.10 / $2
        ("gpt-6.1-terra", 0.05),            # $0.05 / $1
    ],
)
def test_cache_read_multiplier_per_model(model, mult):
    assert cache_read_multiplier(model) == pytest.approx(mult)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-mythos-5-1", 10.0),
        ("claude-sonnet-5-5", 2.0),
        ("claude-sonnet-5", 2.0),
        ("claude-sonnet-4-6", 3.0),
        ("claude-sonnet-4-5-20250929", 3.0),
        ("claude-sonnet-4-20250514", 3.0),
        ("claude-sonnet-6", 2.0),
        ("claude-3-7-sonnet-20250219", 3.0),
        ("claude-haiku-4-5", 1.0),
        ("claude-haiku-6", 0.10),
        ("gpt-5.6-cyber", 12.5),
        ("gpt-5.5-cyber", 12.5),
        ("gpt-rosalind-research", 5.0),
        ("gpt-5.4-mini", 0.75),
        ("gpt-5.4-nano", 0.20),
        ("gpt-5.3-codex", 1.75),
        ("gpt-5.2", 1.75),
        ("gpt-5.2-pro", 21.0),
        ("gpt-5.1", 1.25),
        ("gpt-5-pro", 15.0),
        ("gpt-5", 1.25),
        ("o3-pro", 20.0),
    ],
)
def test_other_current_input_prices(model, expected):
    assert input_usd_per_mtok(model) == (expected, True)


@pytest.mark.parametrize(
    ("model", "limit", "short", "long"),
    [
        ("claude-haiku-5-5", 100_000, 0.10, 0.50),
        ("gpt-5.6-sol", 272_000, 4.0, 8.0),
        ("gpt-5.6-terra", 272_000, 2.0, 4.0),
        ("gpt-5.6-luna", 272_000, 0.20, 0.40),
        ("gpt-5.5", 272_000, 5.0, 10.0),
        ("gpt-5.5-pro", 272_000, 30.0, 60.0),
        ("gpt-5.4", 272_000, 2.50, 5.0),
        ("gpt-5.4-pro", 272_000, 30.0, 60.0),
    ],
)
def test_length_tiered_prices(model, limit, short, long):
    assert input_usd_per_mtok(model, input_tokens=limit) == (short, True)
    assert input_usd_per_mtok(model, input_tokens=limit + 1) == (long, True)


@pytest.mark.parametrize(
    ("model", "mult"),
    [
        ("claude-mythos-5-1", 0.025),
        ("claude-sonnet-5-5", 0.05),        # $0.10 / $2
        ("claude-sonnet-5", 0.1),           # $0.20 / $2
        ("claude-sonnet-4-6", 0.1),
        ("claude-haiku-5-5", 0.1),          # $0.01 / $0.10
        ("gpt-5.6-sol", 0.1),
        ("gpt-5.5", 0.1),
        ("gpt-5.4-mini", 0.1),
        ("gpt-5.2", 0.1),
        ("gpt-5.5-pro", 1.0),               # no cached-input rate
        ("gpt-5.4-pro", 1.0),
        ("gpt-5.2-pro", 1.0),
        ("gpt-5-pro", 1.0),
        ("o3-pro", 1.0),
        ("o3", 0.25),                       # $0.50 / $2
        ("gpt-4.1", 0.25),                  # $0.50 / $2
        ("gpt-4o", 0.5),                    # $1.25 / $2.50
        ("gpt-4o-mini", 0.5),               # $0.075 / $0.15
    ],
)
def test_other_cache_read_multipliers(model, mult):
    assert cache_read_multiplier(model) == pytest.approx(mult)
