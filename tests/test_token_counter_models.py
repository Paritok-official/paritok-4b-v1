"""Regression for issue #1: count_tokens() used to (a) raise
`ValueError: Unknown encoding gpt-5` for every newer model id passed explicitly,
and (b) silently fall back to cl100k_base for gpt-5 / gpt-4.1 / o3 metering, so
the dashboard billed savings on the wrong tokenizer (5-15% drift from o200k_base).

These lock in: no model id ever raises; frontier OpenAI families resolve to
o200k_base; legacy families and Claude stay cl100k_base; explicit encoding names
still pass through; and the upstream model threads through the compress pipeline
so original_tokens reflects the provider's real billing tokenizer.
"""
import types

import pytest

from paritok.token_counter import _resolve_encoding, count_tokens


# ── (a) no model id raises, and frontier families map to o200k_base ──
@pytest.mark.parametrize(
    "model",
    [
        "gpt-5", "gpt-5-mini", "gpt-5-nano",
        "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
        "o1", "o3", "o3-mini", "o4-mini",
        "gpt-4o", "gpt-4o-mini",
    ],
)
def test_frontier_models_resolve_to_o200k_without_raising(model):
    assert _resolve_encoding(model) == "o200k_base"
    # the actual count path must not raise either (the original bug)
    assert count_tokens("hello world, this is a test", model) > 0


@pytest.mark.parametrize("model", ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"])
def test_legacy_openai_models_stay_cl100k(model):
    assert _resolve_encoding(model) == "cl100k_base"


@pytest.mark.parametrize("model", ["claude-sonnet-4", "claude-3-5-haiku", "claude-opus-4-8"])
def test_claude_models_use_cl100k_approximation(model):
    assert _resolve_encoding(model) == "cl100k_base"


def test_provider_namespace_prefix_is_stripped():
    assert _resolve_encoding("openai/gpt-5") == "o200k_base"
    assert _resolve_encoding("anthropic/claude-sonnet-4") == "cl100k_base"


def test_explicit_encoding_names_pass_through():
    assert _resolve_encoding("o200k_base") == "o200k_base"
    assert _resolve_encoding("cl100k_base") == "cl100k_base"


def test_empty_and_unknown_fall_back_to_default_never_raise():
    assert _resolve_encoding("") == "cl100k_base"
    assert _resolve_encoding(None) == "cl100k_base"
    # a genuinely unknown id must not blow up — it degrades to the default
    assert _resolve_encoding("some-brand-new-model-9000") == "cl100k_base"
    assert count_tokens("x" * 50, "some-brand-new-model-9000") > 0


@pytest.mark.parametrize(
    "snapshot,enc",
    [
        # real OpenAI dated snapshots + sub-variants resolve via the `<base>-` tier
        ("gpt-4o-2024-08-06", "o200k_base"),
        ("gpt-4o-2024-11-20", "o200k_base"),
        ("gpt-4.1-2025-04-14", "o200k_base"),
        ("o3-2025-04-16", "o200k_base"),
        ("o4-mini-2025-04-16", "o200k_base"),
        ("gpt-5-2025-08-07", "o200k_base"),
        ("chatgpt-4o-latest", "o200k_base"),
        ("gpt-4-0613", "cl100k_base"),
        ("gpt-4-turbo-2024-04-09", "cl100k_base"),
        ("gpt-3.5-turbo-16k", "cl100k_base"),
        # fine-tuned model ids (ft:<base>-...:org::id)
        ("ft:gpt-4o-mini-2024-07-18:acme::abc", "o200k_base"),
        ("ft:gpt-3.5-turbo:acme::xyz", "cl100k_base"),
    ],
)
def test_dated_snapshots_and_variants_resolve_via_hyphen_prefix(snapshot, enc):
    assert _resolve_encoding(snapshot) == enc


@pytest.mark.parametrize(
    "malformed",
    ["o3random", "o1foo", "gpt5x", "gpt-4oevil", "gpt5nano", "o4mini"],
)
def test_hyphenless_malformed_ids_fall_back_to_default(malformed):
    # The trailing hyphen in the prefix table is the boundary that prevents
    # guessing: an id that is NOT `<base>-...` matches no prefix and takes the
    # safe default — so "o3random" never inherits o200k from "o3".
    assert _resolve_encoding(malformed) == "cl100k_base"


# o200k's biggest edge over cl100k is multilingual/non-ASCII text (CJK, accents),
# where plain ASCII code often tokenizes identically. Use non-ASCII content to
# prove the fix changes real counts — the drift the dashboard was billing on.
_DRIFT_TEXT = "naïve café résumé — 日本語のトークン化テスト、こんにちは世界。" * 20


def test_o200k_and_cl100k_actually_differ():
    o200k = count_tokens(_DRIFT_TEXT, "gpt-5")
    cl100k = count_tokens(_DRIFT_TEXT, "gpt-4")
    assert o200k != cl100k
    # sanity: o200k is the denser (fewer-token) multilingual tokenizer here
    assert o200k < cl100k


# ── (b) upstream model threads through the compress pipeline's metering ──
def test_pipeline_original_tokens_uses_upstream_tokenizer():
    from paritok.config import ParitokConfig
    from paritok.pipelines.compress import CompressionPipeline

    def fresh_pipe():
        p = CompressionPipeline(ParitokConfig())
        # stub the model so no Ollama/GPU is needed; force the full model-call path
        p._model = types.SimpleNamespace(compress=lambda *a, **k: "def f(): pass")
        return p

    # non-ASCII content so the two tokenizers genuinely disagree (see _DRIFT_TEXT)
    content = ("naïve café — 日本語のトークン化テスト、こんにちは世界。\n" * 60)

    r5 = fresh_pipe().compress(content, query="fix the bug", upstream_model="gpt-5")
    r4 = fresh_pipe().compress(content, query="fix the bug", upstream_model="gpt-4")

    # gpt-5 metering must equal the o200k count of the content, and differ from cl100k
    assert r5.original_tokens == count_tokens(content, "gpt-5")
    assert r5.original_tokens != r4.original_tokens
