"""Regression test for eval_model.compress: the compression path must de-pad the
width-padded cat -n line numbers BEFORE the chunk-size decision, matching production
(paritok/pipelines/compress.py de-pads the whole content, then LocalModelStrategy
sizes/chunks against the real token count).

The bug: dataset.line_number_context emits padded `{n:6d}\t` (as a real Claude Code
Read does), which inflates the token count ~20%. compress_context used to test that
PADDED size against `chunk`, so a file that fits in one SEG de-padded (e.g.
quality_agent.py: 2,878 tok) was pushed over the 3,000 threshold (3,460 padded) and
split into 2 SEGs — nearly halving the compression the model achieves single-shot
(measured kept 0.39 vs 0.19 on the GPU). No model/network needed: the model call is
stubbed so this asserts the routing + de-pad, not the model output.

Run: python -m pytest eval_model/tests/
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval_model import compress as ec  # noqa: E402
from eval_model.dataset import line_number_context  # noqa: E402
from paritok.token_counter import count_tokens  # noqa: E402

ENC = "cl100k_base"
_PADDED = re.compile(r"(?m)^[ \t]+\d+\t")  # a width-padded "     123\t" line prefix


def _padded_body(n_lines=130):
    """A one-file context whose numbered body is > `chunk` padded but <= `chunk`
    de-padded, so the split decision differs between the two."""
    src = "\n".join(f"    def method_{i}(self, x): return x + {i}" for i in range(n_lines))
    return line_number_context(f"# File: t.py\n{src}\n")


def _run_capturing_chunks(full_context, chunk):
    calls = []

    def fake_seg(chunk_text, intent, level, seg_id, url, model, num_ctx, timeout, stats=None):
        if stats is not None:
            stats["chunks"] = stats.get("chunks", 0) + 1
        calls.append(chunk_text)
        return "<c>"

    orig = ec._seg_compress
    ec._seg_compress = fake_seg
    try:
        ec.compress_context(full_context, "do the task", chunk=chunk)
    finally:
        ec._seg_compress = orig
    return calls


def test_depad_before_chunk_decision_keeps_single_shot():
    ctx = _padded_body()
    body = ctx.split("\n", 1)[1]  # drop the "# File:" header line
    padded_tok = count_tokens(body, ENC)
    depadded_tok = count_tokens(ec._depad_line_numbers(body), ENC)
    # Sanity: the padding must actually straddle the threshold we pick, else the test
    # would pass trivially.
    assert depadded_tok < padded_tok
    chunk = (depadded_tok + padded_tok) // 2  # padded > chunk >= de-padded
    assert depadded_tok <= chunk < padded_tok

    calls = _run_capturing_chunks(ctx, chunk)

    # Fix: de-padded body fits under `chunk` -> ONE single-shot SEG (not a split).
    assert len(calls) == 1, f"expected single-shot, got {len(calls)} chunks (padded size leaked into the split decision)"
    # And the content handed to the model is de-padded (no "     123\t" prefixes).
    assert not _PADDED.search(calls[0]), "model input still has width-padded line numbers"
    assert re.search(r"(?m)^\d+\t", calls[0]), "de-padded bare 'N\\t' line numbers missing"
