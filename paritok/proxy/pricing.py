"""Per-model INPUT-token pricing for the /stats cost estimate.

USD per 1,000,000 INPUT tokens — public list prices, kept small and easy to edit.
Only input is priced: it's the part Paritok compresses; output is the provider's
and out of scope. A model is matched by its longest name prefix (so
`claude-sonnet-4-20250514` → `claude-sonnet`); an unknown model falls back to
$3/M (Claude Sonnet) — it's an estimate either way.

These are list prices as of early 2026; update them when providers change pricing.
"""

from __future__ import annotations

# Fallback input price for a model not in the table, USD per 1M tokens
# (Claude Sonnet's rate).
DEFAULT_USD_PER_MTOK = 3.0

# $ per 1M input tokens.
INPUT_USD_PER_MTOK: dict[str, float] = {
    # Anthropic — Claude Code's models
    "claude-opus": 15.0,
    "claude-3-7-sonnet": 3.0,
    "claude-3-5-sonnet": 3.0,
    "claude-sonnet": 3.0,
    "claude-3-5-haiku": 0.80,
    "claude-haiku": 1.0,
    # OpenAI — Codex / GPT
    "gpt-5-mini": 0.25,
    "gpt-5-nano": 0.05,
    "gpt-5": 1.25,
    "gpt-4.1-mini": 0.40,
    "gpt-4.1-nano": 0.10,
    "gpt-4.1": 2.00,
    "gpt-4o-mini": 0.15,
    "gpt-4o": 2.50,
    "o4-mini": 1.10,
    "o3-mini": 1.10,
    "o3": 2.00,
}

# Cache-READ multiplier: what a provider charges for an input token served from
# its prompt cache, as a fraction of the base input price. Applied to the frozen
# (byte-stable) tool-schema block, which after the first turn is a cache hit on
# every subsequent turn — so its real per-turn saving is (orig-comp) * rate * this,
# not the full list price. Longest-prefix match, same as INPUT_USD_PER_MTOK.
# (Turn 1 is actually a cache *write* at ~1.25x, so this slightly under-counts a
# short session and converges to exact over a long one — conservative on purpose.)
CACHE_READ_MULT: dict[str, float] = {
    "claude": 0.1,      # Anthropic: cache read = 10% of base input
    "gpt-5": 0.1,       # OpenAI cached input, per-model
    "gpt-4.1": 0.25,
    "gpt-4o": 0.5,
    "o4": 0.25,
    "o3": 0.25,
}
# Unknown model → assume the deepest discount (smallest saving) to avoid overstating.
DEFAULT_CACHE_READ_MULT = 0.1

# Cache-WRITE multiplier: the first turn a frozen prefix is cached costs a premium
# over base input (Anthropic's 5-min cache write is 1.25x; OpenAI writes at base).
# We use 1.25x — the Claude Code case, and the conservative (larger-write) end for
# OpenAI. Applied once, to the first tool-bearing turn per model.
CACHE_WRITE_MULT = 1.25


def cache_read_multiplier(model: str) -> float:
    """Fraction of base input price charged for a cached (prompt-cache read) token."""
    m = _normalize(model)
    best = None
    for key in CACHE_READ_MULT:
        if m.startswith(key) and (best is None or len(key) > len(best)):
            best = key
    return CACHE_READ_MULT[best] if best is not None else DEFAULT_CACHE_READ_MULT


def _normalize(model: str) -> str:
    m = (model or "").strip().lower()
    if "/" in m:  # drop a provider namespace, e.g. "anthropic/claude-..."
        m = m.split("/", 1)[1]
    return m


def input_usd_per_mtok(model: str) -> tuple[float, bool]:
    """(USD per 1M input tokens, matched?) for `model` via longest-prefix match.

    `matched` is False when the model was unknown and the $3/M default was used.
    """
    m = _normalize(model)
    best = None
    for key in INPUT_USD_PER_MTOK:
        if m.startswith(key) and (best is None or len(key) > len(best)):
            best = key
    if best is not None:
        return INPUT_USD_PER_MTOK[best], True
    return DEFAULT_USD_PER_MTOK, False


def input_price_per_token(model: str) -> float:
    """USD per single input token for `model` ($3/M default for unknown)."""
    return input_usd_per_mtok(model)[0] / 1_000_000
