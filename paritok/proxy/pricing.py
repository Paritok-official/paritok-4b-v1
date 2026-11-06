"""Per-model INPUT-token pricing for the /stats cost estimate.

USD per 1,000,000 INPUT tokens — public list prices, kept small and easy to edit.
Only input is priced: it's the part Paritok compresses; output is the provider's
and out of scope. A model is matched by its longest name prefix (so
`claude-sonnet-4-20250514` → `claude-sonnet-4`); an unknown model falls back to
$3/M (Claude Sonnet 4.x) — it's an estimate either way.

List prices as of November 2026 (models the providers no longer list keep their last
price); update them when providers change pricing.
https://platform.claude.com/docs/en/about-claude/pricing
https://developers.openai.com/api/docs/pricing
"""

from __future__ import annotations

# Fallback input price for a model not in the table, USD per 1M tokens
# (Claude Sonnet 4.x's rate).
DEFAULT_USD_PER_MTOK = 3.0

# $ per 1M input tokens.
INPUT_USD_PER_MTOK: dict[str, float] = {
    # Anthropic — Claude Code's models
    "claude-fable": 10.0,
    "claude-mythos": 10.0,
    "claude-opus-5-5": 4.0,
    "claude-opus-5": 5.0,
    "claude-opus-4-8": 5.0,
    "claude-opus-4-7": 5.0,
    "claude-opus-4-6": 5.0,
    "claude-opus-4-5": 5.0,
    "claude-opus-4-1": 15.0,
    "claude-opus-4": 15.0,
    "claude-opus": 4.0,  # newer than this table: assume the family's current rate
    "claude-sonnet-5-5": 2.0,
    "claude-sonnet-5": 2.0,
    "claude-sonnet-4": 3.0,
    "claude-sonnet": 2.0,  # newer than this table
    "claude-3-7-sonnet": 3.0,
    "claude-3-5-sonnet": 3.0,
    "claude-haiku-4-5": 1.0,
    "claude-haiku": 0.10,  # newer than this table; Haiku 5.5 is in INPUT_TIERS
    "claude-3-5-haiku": 0.80,
    # OpenAI — Codex / GPT. GPT-6, GPT-5.6, 5.5 and 5.4 are in INPUT_TIERS.
    "gpt-6-cyber": 20.0,
    "gpt-6-rosalind": 15.0,
    "gpt-rosalind": 5.0,
    "gpt-5.6-cyber": 12.5,
    "gpt-5.5-cyber": 12.5,
    "gpt-5.4-mini": 0.75,
    "gpt-5.4-nano": 0.20,
    "gpt-5.3-codex": 1.75,
    "gpt-5.2-pro": 21.0,
    "gpt-5.2": 1.75,
    "gpt-5.1": 1.25,
    "gpt-5-pro": 15.0,
    "gpt-5-mini": 0.25,
    "gpt-5-nano": 0.05,
    "gpt-5": 1.25,
    "gpt-4.1-mini": 0.40,
    "gpt-4.1-nano": 0.10,
    "gpt-4.1": 2.00,
    "gpt-4o-mini": 0.15,
    "gpt-4o": 2.50,
    "o4-mini": 1.10,
    "o3-pro": 20.0,
    "o3-mini": 1.10,
    "o3": 2.00,
    # MiniMax-M3.1-Flash has a flat rate, overriding the shorter M3 tier prefix.
    # https://platform.minimax.io/docs/guides/pricing-paygo
    "minimax-m3.1-flash": 0.15,
    # Free gateway variants must not inherit the direct API price.
    # https://openrouter.ai/api/v1/models
    # https://opencode.ai/docs/zen/
    "minimax-m3.1-flash:free": 0.0,
    "minimax-m3.1-flash-free": 0.0,
    # Preserve historical M2.7 estimates; M3 lives in INPUT_TIERS below.
    "minimax-m2.7": 0.30,
}

# Models whose INPUT price is tiered by input length. Each value is an ordered
# ladder of (max_input_tokens_inclusive | None, USD-per-1M-tokens) from the
# smallest tier up: a request's input length picks the FIRST tier whose inclusive
# max it does not exceed, and the final (None, price) tier is the unbounded top.
# The threshold lives here as DATA — the inclusive upper bound of each tier — not
# as a hardcoded comparison in the lookup, so a vendor that bills the boundary
# token at the higher tier is expressed by lowering that tier's max by one rather
# than by editing code. A model with no ladder just uses its flat INPUT_USD_PER_MTOK
# value.
INPUT_TIERS: dict[str, list[tuple[int | None, float]]] = {
    # MiniMax-M3 (the permanent 50% discount is already baked into these rates):
    # $0.30/M for input <= 512K tokens, $0.60/M above it. NOTE: MiniMax's pricing
    # page writes "512k" and never resolves it (512,000 vs 524,288). We take
    # 512,000, inclusive of the standard tier — an assumption, not a sourced number;
    # it errs toward charging the long-context rate slightly early, keeping the
    # reported saving conservative in the same direction the flat rate already chose.
    "minimax-m3": [(512_000, 0.30), (None, 0.60)],
    # Claude Haiku 5.5: a prompt over 100,000 tokens pays the higher price.
    "claude-haiku-5-5": [(100_000, 0.10), (None, 0.50)],
    # OpenAI: the input rate doubles above 272K input tokens. OpenAI writes "<=272K";
    # 272,000 is taken the same way as MiniMax's "512k" above.
    "gpt-6-astra": [(272_000, 10.0), (None, 20.0)],
    "gpt-6-sol": [(272_000, 2.0), (None, 4.0)],
    "gpt-6-luna": [(272_000, 0.10), (None, 0.20)],
    "gpt-6.1-sol": [(272_000, 2.0), (None, 4.0)],
    "gpt-6.1-terra": [(272_000, 1.0), (None, 2.0)],
    "gpt-5.6-sol": [(272_000, 4.0), (None, 8.0)],
    "gpt-5.6-terra": [(272_000, 2.0), (None, 4.0)],
    "gpt-5.6-luna": [(272_000, 0.20), (None, 0.40)],
    "gpt-5.5": [(272_000, 5.0), (None, 10.0)],
    "gpt-5.5-pro": [(272_000, 30.0), (None, 60.0)],
    "gpt-5.4": [(272_000, 2.50), (None, 5.0)],
    "gpt-5.4-pro": [(272_000, 30.0), (None, 60.0)],
}

# Cache-READ multiplier: what a provider charges for an input token served from
# its prompt cache, as a fraction of the base input price. Applied to the frozen
# (byte-stable) tool-schema block, which after the first turn is a cache hit on
# every subsequent turn — so its real per-turn saving is (orig-comp) * rate * this,
# not the full list price. Longest-prefix match, same as INPUT_USD_PER_MTOK.
# (Turn 1 is actually a cache *write* at ~1.25x, so this slightly under-counts a
# short session and converges to exact over a long one — conservative on purpose.)
CACHE_READ_MULT: dict[str, float] = {
    "claude": 0.1,      # Anthropic: cache read = 10% of base input, except:
    "claude-opus-5-5": 0.05,
    "claude-sonnet-5-5": 0.05,
    "claude-fable-5-1": 0.025,
    "claude-mythos-5-1": 0.025,
    "gpt-6": 0.1,       # OpenAI: astra, sol and luna $/10, except 6.1 sol and terra $/20
    "gpt-6.1-sol": 0.05,
    "gpt-6.1-terra": 0.05,
    "gpt-5": 0.1,       # OpenAI cached input, per-model
    # The pro models have no cached-input rate: a re-sent token pays the full price.
    "gpt-5.5-pro": 1.0,
    "gpt-5.4-pro": 1.0,
    "gpt-5.2-pro": 1.0,
    "gpt-5-pro": 1.0,
    "o3-pro": 1.0,
    "gpt-4.1": 0.25,
    "gpt-4o": 0.5,
    "o4": 0.25,
    "o3": 0.25,
    # Flash: $0.015/M cache read / $0.15/M input = 10%; M3 and legacy stay at 20%.
    "minimax-m3.1-flash": 0.1,
    "minimax": 0.2,
}
# Unknown model → assume the common 10% rather than a provider's specific rate.
DEFAULT_CACHE_READ_MULT = 0.1

# Cache-WRITE multiplier: the first turn a frozen prefix is cached costs a premium
# over base input. Anthropic: 1.25x for the 5-min cache, 2x for the 1-hour one, which
# Claude Code used in every loopbench run. OpenAI: 1.25x for the 30-min cache on GPT-6
# and 5.6, none listed on older models. We use 1.25x: for a 1-hour write it under-counts
# the saving. Applied once, to the first tool-bearing turn per model.
CACHE_WRITE_MULT = 1.25


def _normalize(model: str) -> str:
    m = (model or "").strip().lower()
    if "/" in m:  # drop a provider namespace, e.g. "anthropic/claude-..."
        m = m.split("/", 1)[1]
    return m


def _longest_prefix(m: str, keys) -> str | None:
    """The longest key in `keys` that is a prefix of the normalized name `m`, else None."""
    best = None
    for key in keys:
        if m.startswith(key) and (best is None or len(key) > len(best)):
            best = key
    return best


def cache_read_multiplier(model: str) -> float:
    """Fraction of base input price charged for a cached (prompt-cache read) token."""
    best = _longest_prefix(_normalize(model), CACHE_READ_MULT)
    return CACHE_READ_MULT[best] if best is not None else DEFAULT_CACHE_READ_MULT


def _tier_price(ladder: list[tuple[int | None, float]], input_tokens: int | None) -> float:
    """Pick a ladder tier's USD-per-1M by input length: the first tier whose inclusive
    max the length does not exceed. A None/omitted length uses the standard (smallest)
    tier — the conservative flat default #43 shipped."""
    n = 0 if input_tokens is None else input_tokens
    for max_tok, price in ladder:
        if max_tok is None or n <= max_tok:
            return price
    return ladder[-1][1]  # unreachable while the ladder ends in an unbounded (None, ...)


def input_usd_per_mtok(model: str, input_tokens: int | None = None) -> tuple[float, bool]:
    """(USD per 1M input tokens, matched?) for `model` via longest-prefix match.

    `input_tokens` is optional: pass a request's input length to price a
    length-tiered model (see INPUT_TIERS) at its correct tier; omit it and every
    tiered model falls back to its standard (smallest) tier, so all existing
    callers keep their previous result unchanged. `matched` is False when the model
    was unknown and the $3/M default was used.
    """
    m = _normalize(model)
    tier_key = _longest_prefix(m, INPUT_TIERS)
    flat_key = _longest_prefix(m, INPUT_USD_PER_MTOK)
    # The more specific (longer) match wins; a tier ladder wins an exact-length tie
    # because it carries the more precise, length-aware price.
    if tier_key is not None and (flat_key is None or len(tier_key) >= len(flat_key)):
        return _tier_price(INPUT_TIERS[tier_key], input_tokens), True
    if flat_key is not None:
        return INPUT_USD_PER_MTOK[flat_key], True
    return DEFAULT_USD_PER_MTOK, False


def input_price_per_token(model: str, input_tokens: int | None = None) -> float:
    """USD per single input token for `model` ($3/M default for unknown). Pass
    `input_tokens` to price a length-tiered model at its request-length tier."""
    return input_usd_per_mtok(model, input_tokens)[0] / 1_000_000
