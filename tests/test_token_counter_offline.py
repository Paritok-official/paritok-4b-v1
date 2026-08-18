"""Regression tests (#23): token counting must survive an unreachable tiktoken vocab
host, and the vocabs for both encodings our model map uses (cl100k_base and o200k_base)
are bundled so the compression hot path never depends on reaching
openaipublic.blob.core.windows.net."""
import hashlib
import os

import paritok.token_counter as tc


def test_both_vocabs_shipped():
    # cl100k_base (gpt-4/3.5, Claude approximation) AND o200k_base (gpt-4o and every OpenAI
    # model since) both ship in the wheel so get_encoding works fully offline for each.
    for name in ("cl100k_base", "o200k_base"):
        p = tc._bundled_vocab_path(name)
        assert os.path.exists(p), name
        assert os.path.getsize(p) > 1_000_000, name


def test_seed_copies_both_vocabs_into_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path))
    tc._seed_tiktoken_cache()
    for name in tc._BUNDLED_ENCODINGS:
        key = hashlib.sha1(tc._BLOB.format(name).encode()).hexdigest()
        assert (tmp_path / key).exists(), name  # resolvable by tiktoken without the network


def test_o200k_counts_exactly_offline(monkeypatch, tmp_path):
    # A gpt-4o/gpt-5-style model uses o200k_base; a fresh (no-cache) box must still count it
    # exactly from the bundled vocab, not fall back to a cl100k approximation.
    text = "def total(items):\n    return sum(i.price for i in items)\n" * 25
    warm = tc.count_tokens(text, "gpt-4o")          # warm cache = ground truth
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path))   # empty → forces the bundle path
    tc._encoder_cache.clear()
    try:
        assert tc.count_tokens(text, "gpt-4o") == warm
    finally:
        tc._encoder_cache.clear()


def test_count_tokens_never_crashes_when_vocab_host_unreachable(monkeypatch):
    # Before #23, _get_encoder caught only (ValueError, KeyError), so a blocked Azure blob
    # (403/ConnectionError) propagated up and crashed every compression request.
    def _boom(*a, **k):
        raise OSError("blocked host: openaipublic.blob.core.windows.net")
    monkeypatch.setattr(tc.tiktoken, "get_encoding", _boom)
    tc._encoder_cache.clear()
    try:
        assert tc.count_tokens("def f(x):\n    return x + 1\n" * 20, "cl100k_base") > 0
        assert tc.count_tokens("hello", "p50k_base") > 0   # non-bundled + offline
        assert tc.count_tokens("") == 0
    finally:
        tc._encoder_cache.clear()


def test_estimate_encoder_roundtrip():
    enc = tc._EstimateEncoder()
    assert len(enc.encode("a" * 400)) == 100          # ~4 chars / token
    assert enc.encode("") == []
    assert enc.decode(enc.encode("hello world")) == "hello world"
