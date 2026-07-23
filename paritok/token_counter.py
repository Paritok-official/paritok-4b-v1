"""Token counting utilities with cached encoder instances."""

from __future__ import annotations

import tiktoken

_encoder_cache: dict[str, tiktoken.Encoding] = {}

_DEFAULT_ENCODING = "cl100k_base"

# Two-tier model-id -> tiktoken encoding, mirroring tiktoken's own model.py so a
# `pip upgrade` isn't required to know newer models, and staying in sync with the
# hosted cost model in paritok-dashboard's pricing.js (self-hosted `/stats`, the
# dashboard, and the landing calculator all count with the tokenizer the provider
# actually bills on).
#
# Tier 1 — EXACT base ids (below). Tier 2 — `<base>-` prefixes (further down) whose
# trailing hyphen is a real boundary: it covers dated snapshots and sub-variants
# (gpt-4o-2024-08-06, o3-2025-04-16, gpt-4.1-mini, ...) WITHOUT matching a
# hyphen-less malformed id (o3random / gpt5x stay at the default). Anything neither
# tier knows -> default cl100k_base (also our Claude approximation); never raised.
_MODEL_TO_ENCODING: dict[str, str] = {
    # cl100k_base — GPT-4 / GPT-3.5 legacy
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-3.5": "cl100k_base",
    "gpt-35-turbo": "cl100k_base",  # Azure spelling
    # o200k_base — everything OpenAI has shipped since gpt-4o
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4.1": "o200k_base",
    "gpt-4.1-mini": "o200k_base",
    "gpt-4.1-nano": "o200k_base",
    "gpt-5": "o200k_base",
    "gpt-5-mini": "o200k_base",
    "gpt-5-nano": "o200k_base",
    "o1": "o200k_base",
    "o1-mini": "o200k_base",
    "o3": "o200k_base",
    "o3-mini": "o200k_base",
    "o4-mini": "o200k_base",
}

# Tier 2: `<base>-` prefixes (hyphen required). Matched by LONGEST prefix so
# `gpt-4o-` wins over `gpt-4-`, and `ft:gpt-4o` over `ft:gpt-4`.
_MODEL_PREFIX_TO_ENCODING: dict[str, str] = {
    # o200k_base
    "o1-": "o200k_base",
    "o3-": "o200k_base",
    "o4-mini-": "o200k_base",
    "gpt-5-": "o200k_base",
    "gpt-4.1-": "o200k_base",
    "gpt-4o-": "o200k_base",
    "chatgpt-4o-": "o200k_base",
    "ft:gpt-4o": "o200k_base",
    # cl100k_base
    "gpt-4-": "cl100k_base",
    "gpt-3.5-turbo-": "cl100k_base",
    "gpt-35-turbo-": "cl100k_base",
    "ft:gpt-4": "cl100k_base",
    "ft:gpt-3.5-turbo": "cl100k_base",
}

# tiktoken encoding names that may be passed through directly (e.g. "o200k_base").
try:
    _KNOWN_ENCODINGS = set(tiktoken.list_encoding_names())
except Exception:  # pragma: no cover - very old tiktoken without the helper
    _KNOWN_ENCODINGS = {"cl100k_base", "o200k_base", "p50k_base", "r50k_base", "gpt2"}


def _get_encoder(encoding: str = _DEFAULT_ENCODING) -> tiktoken.Encoding:
    if encoding not in _encoder_cache:
        try:
            _encoder_cache[encoding] = tiktoken.get_encoding(encoding)
        except (ValueError, KeyError):
            # Last-resort guard: an unresolved name must never crash token counting.
            _encoder_cache[encoding] = tiktoken.get_encoding(_DEFAULT_ENCODING)
    return _encoder_cache[encoding]


def _resolve_encoding(model_or_encoding: str) -> str:
    """Resolve a model name OR an encoding name to a tiktoken encoding name.

    Never raises: an unmapped model id falls back to the default encoding rather
    than being passed to tiktoken verbatim (which used to raise
    ``ValueError: Unknown encoding gpt-5`` for every newer model id).
    """
    name = (model_or_encoding or "").strip()
    if not name:
        return _DEFAULT_ENCODING

    # An explicit tiktoken encoding name passed through unchanged.
    if name in _KNOWN_ENCODINGS:
        return name

    # Case-insensitive; drop any "provider/" namespace (openai/gpt-5 -> gpt-5).
    key = name.lower()
    if "/" in key:
        key = key.split("/", 1)[1]

    # Tier 1: exact base id.
    if key in _MODEL_TO_ENCODING:
        return _MODEL_TO_ENCODING[key]

    # Tier 2: `<base>-` prefix (hyphen boundary), longest match wins. Covers dated
    # snapshots / sub-variants; a hyphen-less malformed id matches nothing here.
    best: str | None = None
    for prefix in _MODEL_PREFIX_TO_ENCODING:
        if key.startswith(prefix) and (best is None or len(prefix) > len(best)):
            best = prefix
    if best is not None:
        return _MODEL_PREFIX_TO_ENCODING[best]

    # Neither tier knows it: safe default, never guessed and never raised.
    return _DEFAULT_ENCODING


def count_tokens(text: str, model_or_encoding: str = _DEFAULT_ENCODING) -> int:
    encoding = _resolve_encoding(model_or_encoding)
    return len(_get_encoder(encoding).encode(text))


def token_cost_ratio(text: str, model_or_encoding: str = _DEFAULT_ENCODING) -> dict:
    """Analyze token efficiency: how many tokens per character."""
    if not text:
        return {"tokens": 0, "chars": 0, "ratio": 0.0}

    tokens = count_tokens(text, model_or_encoding)
    chars = len(text)
    return {
        "tokens": tokens,
        "chars": chars,
        "ratio": round(tokens / chars, 3) if chars > 0 else 0.0,
    }
