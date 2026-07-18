"""Structural chunking for long compression inputs.

Ported verbatim from training/scripts/swe_bench_compress_local.py, the SEG/level
production path. The runtime MUST match the SWE-bench Verified benchmark exactly,
which was evaluated with:
    CHUNK_SIZE = 2000
    max_single_block = 2000
    level = L1
    _TOP_LEVEL_DEF = r'^(class |def )\\w+'

Sending inputs longer than ~2000 tokens in one shot drives the model out of its
trained distribution and produces structural hallucinations (repeated pseudo-doc
text). Splitting first, compressing per-chunk, then deduplicating recreates the
training-time chunk-granularity the model saw.
"""

from __future__ import annotations

import re

from paritok.token_counter import count_tokens

# The model runs in an 8192-token context (Modelfile num_ctx) and each call also
# carries a ~2.2-2.9k-token system prompt, so a chunk + the system prompt + room
# to generate the compressed output must all fit in 8192. At 3000 the prompt still
# fits comfortably (3000 + ~2.9k ≈ 5.9k < 8192); _call_ollama caps num_predict to
# whatever context is left, so oversized chunks no longer 400 (they just get a
# smaller generation budget). The training/benchmark value was 2000.
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 0
MAX_SINGLE_BLOCK = 3000

# Match a top-level `class`/`def`, tolerating a leading Read-tool line-number
# prefix (cat -n style: "   43\t"). The optional group matches zero-width on
# clean code, so benchmark reproduction (clean input) is unchanged; it only adds
# boundary detection for the line-numbered input the proxy actually receives —
# without it, numbered files find 0 boundaries and never chunk (any size = 1 SEG).
_TOP_LEVEL_DEF = re.compile(r"^(?:\s*\d+\t)?(class |def )\w+", re.MULTILINE)
_DEF_NAME = re.compile(r"^(class\s+\w+|def\s+\w+)")
_HEADER_OR_DEF = re.compile(r"^(class\s|def\s|# Lines \d)")


def _find_structural_boundaries(text: str) -> list[int]:
    boundaries = []
    for m in _TOP_LEVEL_DEF.finditer(text):
        line_idx = text[: m.start()].count("\n")
        boundaries.append(line_idx)
    return boundaries


def _token_split_block(lines: list[str], chunk_size: int) -> list[list[str]]:
    pieces: list[list[str]] = []
    cur: list[str] = []
    cur_tok = 0
    for ln in lines:
        t = count_tokens(ln)
        if cur_tok + t > chunk_size and cur:
            pieces.append(cur)
            cur = [ln]
            cur_tok = t
        else:
            cur.append(ln)
            cur_tok += t
    if cur:
        pieces.append(cur)
    return pieces


def split_into_chunks_structural(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    max_single_block: int = MAX_SINGLE_BLOCK,
) -> list[tuple[str, int, int, int]]:
    """Split code by class/def boundaries. Returns (chunk_text, start_line, end_line, raw_tokens)."""
    lines = text.split("\n")
    boundaries = _find_structural_boundaries(text)

    if not boundaries:
        # No class/def boundaries at all — markdown, prose, directory listings,
        # logs, or line-numbered files whose structure the regex can't see. Do NOT
        # return the whole thing as one chunk: a large boundary-less input would be
        # sent as a single oversized SEG that overflows the model context (HTTP
        # 400). Hard-split by tokens so every chunk stays within chunk_size.
        chunks: list[tuple[str, int, int, int]] = []
        start = 0
        for piece in _token_split_block(lines, chunk_size):
            piece_text = "\n".join(piece)
            chunks.append((piece_text, start + 1, start + len(piece), count_tokens(piece_text)))
            start += len(piece)
        return chunks

    blocks: list[tuple[int, int]] = []
    if boundaries[0] > 0:
        blocks.append((0, boundaries[0]))
    for i, b in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(lines)
        blocks.append((b, end))

    chunks: list[tuple[str, int, int, int]] = []
    cur_lines: list[str] = []
    cur_start = 0
    cur_tokens = 0

    for blk_start, blk_end in blocks:
        blk_lines = lines[blk_start:blk_end]
        blk_text = "\n".join(blk_lines)
        blk_tok = count_tokens(blk_text)

        if blk_tok > max_single_block:
            if cur_lines:
                chunk_text = "\n".join(cur_lines)
                chunks.append(
                    (chunk_text, cur_start + 1, cur_start + len(cur_lines), count_tokens(chunk_text))
                )
                cur_lines = []
                cur_tokens = 0
            for piece in _token_split_block(blk_lines, chunk_size):
                piece_text = "\n".join(piece)
                piece_tok = count_tokens(piece_text)
                chunks.append((piece_text, blk_start + 1, blk_start + len(piece), piece_tok))
                blk_start += len(piece)
            cur_start = blk_end
            continue

        if blk_tok <= chunk_size and cur_tokens + blk_tok <= chunk_size:
            if not cur_lines:
                cur_start = blk_start
            cur_lines.extend(blk_lines)
            cur_tokens += blk_tok
            continue

        if cur_lines:
            chunk_text = "\n".join(cur_lines)
            chunks.append(
                (chunk_text, cur_start + 1, cur_start + len(cur_lines), count_tokens(chunk_text))
            )

        cur_lines = list(blk_lines)
        cur_tokens = blk_tok
        cur_start = blk_start

    if cur_lines:
        chunk_text = "\n".join(cur_lines)
        chunks.append(
            (chunk_text, cur_start + 1, cur_start + len(cur_lines), count_tokens(chunk_text))
        )

    return chunks


def deduplicate_definitions(text: str) -> str:
    """Drop repeated `class Foo` / `def bar` blocks that chunked compression produced twice."""
    seen_defs: set[str] = set()
    output_lines: list[str] = []
    skip_until_next_def = False

    for line in text.split("\n"):
        match = _DEF_NAME.match(line)
        if match:
            def_name = match.group(1)
            if def_name in seen_defs:
                skip_until_next_def = True
                continue
            seen_defs.add(def_name)
            skip_until_next_def = False
        elif skip_until_next_def:
            if _HEADER_OR_DEF.match(line):
                skip_until_next_def = False
                match2 = _DEF_NAME.match(line)
                if match2:
                    def_name = match2.group(1)
                    if def_name in seen_defs:
                        skip_until_next_def = True
                        continue
                    seen_defs.add(def_name)
                    skip_until_next_def = False
            else:
                continue

        output_lines.append(line)

    return "\n".join(output_lines)
