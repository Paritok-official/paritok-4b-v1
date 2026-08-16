"""End-to-end contract for the irrelevant-drop stub (issues #20/#31/#32, option B).

When the hosted model completes but judges a segment irrelevant to the query, the
website now returns a tiny placeholder ("(omitted: ...)") instead of "" (which
blinded the agent, #20) or the full original (which wastes tokens on the noise the
model dropped). To the proxy that placeholder is just a very small non-empty
compression, so the pipeline must:

  * tag it [REF:id] and forward the stub (not the original) — real token savings,
  * store the UNTOUCHED original in shadow storage — so read_original recovers it.

This drives the real CompressionPipeline with a stubbed backend that returns the
placeholder (standing in for the hosted response), and asserts both halves. No
Python code changed for option B — the stub is produced server-side — so this
locks the contract the server relies on.
"""
from paritok.config import ParitokConfig
from paritok.pipelines.compress import CompressionPipeline

# Mirrors website compressResult.mjs IRRELEVANT_STUB — a real block compresses to
# this when nothing in it is relevant to the current query.
IRRELEVANT_STUB = "(omitted: no content in this block was relevant to the current task)"

BLOCK = "\n".join(f"{i:>4}\tlog line {i}: nothing here is about the current task"
                  for i in range(120))


class _DropModel:
    """Backend that judged the segment irrelevant: returns the placeholder stub."""
    def compress(self, *args, **kwargs):
        return IRRELEVANT_STUB


def _pipeline():
    cfg = ParitokConfig()
    cfg.compression.min_tokens = 1          # force a compression attempt
    cfg.compression.refusal_threshold = 0.0  # the stub is a huge saving; never refused
    pipe = CompressionPipeline(cfg)
    pipe._model = _DropModel()
    return pipe


def test_irrelevant_drop_is_tagged_and_not_the_original():
    pipe = _pipeline()
    cr = pipe.compress(BLOCK, query="how does the database connection pool retry")

    # The stub is forwarded, [REF]-tagged — NOT "" (issue #20) and NOT the original.
    assert cr.compressed != BLOCK, "the full original was forwarded — no saving on the noise"
    assert cr.compressed.startswith("[REF:"), "the drop must be [REF]-tagged so it can be expanded"
    assert IRRELEVANT_STUB in cr.compressed
    assert cr.compressed.strip(), "must never forward an empty body"
    # And it is a real saving (tiny stub vs a 120-line block).
    assert cr.compressed_tokens < cr.original_tokens


def test_dropped_original_is_recoverable():
    pipe = _pipeline()
    cr = pipe.compress(BLOCK, query="unrelated query")

    # The untouched original is in shadow storage under the tag's id — read_original
    # (gateway) resolves exactly this, so a wrong drop is never lossy.
    assert cr.shadow_id is not None
    assert pipe.storage.retrieve(cr.shadow_id) == BLOCK
