"""Cached tokenizer for Qwen2.5-Coder."""
from functools import lru_cache
from transformers import AutoTokenizer


@lru_cache(maxsize=1)
def get_tokenizer():
    return AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-3B-Instruct",
        trust_remote_code=True,
    )


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(get_tokenizer().encode(text, add_special_tokens=False))