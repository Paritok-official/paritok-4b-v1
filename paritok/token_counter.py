"""Token counting utilities with cached encoder instances."""

from __future__ import annotations

import tiktoken

_encoder_cache: dict[str, tiktoken.Encoding] = {}

# Map model names to tiktoken encoding names.
# Claude uses a proprietary tokenizer; cl100k_base is an approximation.
# Add new models here as needed.
_MODEL_TO_ENCODING: dict[str, str] = {
    "gpt-4": "cl100k_base",
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-3.5-turbo": "cl100k_base",
}

_DEFAULT_ENCODING = "cl100k_base"


def _get_encoder(encoding: str = _DEFAULT_ENCODING) -> tiktoken.Encoding:
    if encoding not in _encoder_cache:
        _encoder_cache[encoding] = tiktoken.get_encoding(encoding)
    return _encoder_cache[encoding]


def _resolve_encoding(model_or_encoding: str) -> str:
    """Resolve a model name or encoding name to a tiktoken encoding name."""
    # Exact match in model mapping
    if model_or_encoding in _MODEL_TO_ENCODING:
        return _MODEL_TO_ENCODING[model_or_encoding]
    # Claude models: proprietary tokenizer, use cl100k_base as approximation
    if model_or_encoding.startswith("claude"):
        return _DEFAULT_ENCODING
    # Assume it's already an encoding name (e.g. "cl100k_base", "o200k_base")
    return model_or_encoding


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
