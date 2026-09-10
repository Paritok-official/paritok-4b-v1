"""Regression (#51): the [REF:id ...] tag is part of the payload the pipeline
actually returns and caches, so its ~5-15 token overhead must count toward the
refusal-threshold effectiveness gate. Before this fix the gate measured the raw,
UNtagged model output, so a compression that only cleared the threshold before the
tag was added slipped through and shipped a payload that, once tagged, saved less
than the configured floor. The de-padded `model_text` denominator (which prevents
line-number-pad stripping from registering as savings) is unchanged.
"""
from paritok.config import ParitokConfig
from paritok.pipelines.compress import CompressionPipeline
from paritok.token_counter import _DEFAULT_ENCODING, count_tokens


def _str_of_tokens(n: int) -> str:
    """A string of approximately `n` cl100k tokens (never more)."""
    s = " w" * n
    while count_tokens(s, _DEFAULT_ENCODING) > n:
        s = s[:-1]
    return s


def _pipe(threshold: float = 0.05) -> CompressionPipeline:
    cfg = ParitokConfig()
    cfg.compression.min_tokens = 1            # force a compression attempt
    cfg.compression.refusal_threshold = threshold
    return CompressionPipeline(cfg)


def test_ref_tag_overhead_counts_toward_refusal_gate():
    thr = 0.05
    # Bare source (no line-number padding) -> model_text == content, so model_tokens
    # equals count_tokens(content) and the denominators line up cleanly.
    content = "\n".join(f"def f_{i}(): return {i}" for i in range(20))
    m_tokens = count_tokens(content, _DEFAULT_ENCODING)
    body = _str_of_tokens(int(m_tokens * (1 - thr)) - 1)   # untagged saving a hair above thr

    class _Model:
        def compress(self, _content, **_kwargs):
            return body

    pipe = _pipe(thr)
    pipe._model = _Model()

    # Precondition: the UNTAGGED output clears the threshold (it passed the old gate).
    assert 1 - count_tokens(body, _DEFAULT_ENCODING) / m_tokens >= thr

    # A source makes the tag "[REF:<sid> src=...] " — several extra tokens that push the
    # real (tagged) saving below the threshold, so the pipeline now refuses it.
    r = pipe.compress(content, kind="file_read", source="pkg/mod/thing.py")
    assert r.metadata.get("skipped") is True
    assert r.metadata.get("reason") == "below_refusal_threshold", r.metadata


def test_strong_compression_still_accepted_after_tagging():
    # A genuine compression whose saving dwarfs the tag overhead is still accepted,
    # and the returned payload is the tagged [REF:id] form.
    content = "\n".join(f"def f_{i}(): return {i}" for i in range(20))

    class _Model:
        def compress(self, _content, **_kwargs):
            return "ok"

    pipe = _pipe(0.05)
    pipe._model = _Model()
    r = pipe.compress(content, kind="file_read", source="pkg/mod/thing.py")
    assert r.metadata.get("skipped") is not True
    assert r.compressed.startswith("[REF:")
