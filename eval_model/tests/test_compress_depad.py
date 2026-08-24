"""Regression tests for eval_model.compress line-number handling (the `depad` switch).
The model call is stubbed, so these assert the routing + line-number shape only.

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


def _run_capturing_chunks(full_context, chunk, depad=False):
    calls = []

    def fake_seg(chunk_text, intent, level, seg_id, url, model, num_ctx, timeout, stats=None):
        if stats is not None:
            stats["chunks"] = stats.get("chunks", 0) + 1
        calls.append(chunk_text)
        return "<c>"

    orig = ec._seg_compress
    ec._seg_compress = fake_seg
    try:
        ec.compress_context(full_context, "do the task", chunk=chunk, depad=depad)
    finally:
        ec._seg_compress = orig
    return calls


def test_depad_before_chunk_decision_keeps_single_shot():
    # depad=True: the production-parity path. Sizing on the de-padded token count keeps
    # a body that only overflows because of the padding as ONE single-shot SEG.
    ctx = _padded_body()
    body = ctx.split("\n", 1)[1]  # drop the "# File:" header line
    padded_tok = count_tokens(body, ENC)
    depadded_tok = count_tokens(ec._depad_line_numbers(body), ENC)
    # Sanity: the padding must actually straddle the threshold we pick, else the test
    # would pass trivially.
    assert depadded_tok < padded_tok
    chunk = (depadded_tok + padded_tok) // 2  # padded > chunk >= de-padded
    assert depadded_tok <= chunk < padded_tok

    calls = _run_capturing_chunks(ctx, chunk, depad=True)

    # de-padded body fits under `chunk` -> ONE single-shot SEG (not a split).
    assert len(calls) == 1, f"expected single-shot, got {len(calls)} chunks (padded size leaked into the split decision)"
    # And the content handed to the model is de-padded (no "     123\t" prefixes).
    assert not _PADDED.search(calls[0]), "model input still has width-padded line numbers"
    assert re.search(r"(?m)^\d+\t", calls[0]), "de-padded bare 'N\\t' line numbers missing"


def test_padded_is_the_default_model_input():
    """Default (depad=False) feeds the model PADDED cat -n numbers — the shape that
    compresses tighter and degenerates less on SWE-bench (same-batch GPU A/B: padded
    18.5% vs bare 23.7% global kept; the 08-17 padded run resolved 48/100 vs a bare
    run's 42/100 + 8 errors). A padded body over the threshold may split; what this
    locks is that the pad is NOT stripped from what the model actually sees."""
    ctx = _padded_body(40)                       # small -> single-shot SEG
    calls = _run_capturing_chunks(ctx, chunk=10_000)   # default depad=False
    assert len(calls) == 1
    # width-padded "     N\t" prefixes survive to the model (checked on the interior
    # lines; the whole body is .strip()ed so only line 1 loses its leading pad).
    assert _PADDED.search(calls[0]), "default path must keep the width-padded line numbers"


def test_no_leading_blank_line_reaches_the_model():
    """The `# File:` split leaves a leading '\\n' on each body ("\\n1\\t..."). Wrapped
    as "[SEG ...]\\n{body}", that opens the SEG with a blank line ("[SEG ...]\\n\\n1\\t..")
    — out-of-distribution for the 4B and it collapses compression (0.23 -> 0.60 on the
    GPU for the same file). compress_context must strip it so the content immediately
    follows the SEG tag."""
    ctx = _padded_body(40)  # small: one single-shot SEG, body carries the split's "\n"
    calls = _run_capturing_chunks(ctx, chunk=10_000)  # big threshold -> guaranteed single-shot
    assert len(calls) == 1
    got = calls[0]
    assert got[:1] != "\n" and not got[:1].isspace(), (
        f"model input starts with whitespace/blank line: {got[:20]!r} — the SEG would open '[SEG]\\n\\n'"
    )
    assert got == got.strip(), "model input has leading/trailing whitespace"
