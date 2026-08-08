"""Regression: a single line larger than CHUNK_SIZE defeated the boundary-less guard.

`split_into_chunks_structural` already refuses to emit a whole boundary-less input
as one SEG -- the comment there says so explicitly, because an oversized SEG
overflows num_ctx and Ollama answers HTTP 400. But the hard split it delegates to,
`_token_split_block`, only ever cuts *between* lines:

    for ln in lines:
        if cur_tok + t > chunk_size and cur:   # `cur` is empty on the first line
            ...
        else:
            cur.append(ln)                     # so a huge line is appended whole

With a one-line input the flush branch never fires, the line goes through intact,
and the guard is bypassed by exactly the shape it was written to catch. Coding
agents hit that shape constantly: a JSON tool result, minified JS, a base64 blob.

Observed before the fix, on the shipped q4 model with num_ctx=8192:

    25,279-char single-line JSON  ->  httpx.HTTPStatusError: 400 Bad Request
    the same JSON re-indented     ->  compressed fine, and it is *larger* (33 KB)

`_split_oversized_line` cuts inside such a line, preferring a nearby separator so
JSON breaks between structural units. It is a no-op whenever every line already
fits, so benchmark reproduction on clean code is untouched.
"""
import json

from paritok.strategies.chunking import (
    CHUNK_SIZE,
    _token_split_block,
    count_tokens,
    split_into_chunks_structural,
)


def _largest(pieces: list[list[str]]) -> int:
    return max(count_tokens("\n".join(p)) for p in pieces)


def test_single_oversized_line_is_split():
    """The hole itself: one line, ten times the budget."""
    line = "word " * (CHUNK_SIZE * 10)
    assert count_tokens(line) > CHUNK_SIZE * 9  # sanity: the input really is huge

    pieces = _token_split_block([line], CHUNK_SIZE)

    assert len(pieces) > 1
    assert _largest(pieces) <= CHUNK_SIZE


def test_single_line_json_tool_result_chunks():
    """The real-world shape: an API response arrives as one line."""
    blob = json.dumps([{"id": i, "name": f"item-{i}", "tags": ["a", "b", "c"]}
                       for i in range(1200)])
    assert "\n" not in blob

    chunks = split_into_chunks_structural(blob)

    assert max(count_tokens(text) for text, *_ in chunks) <= CHUNK_SIZE


def test_line_without_separators_still_splits():
    """base64 and minified payloads have nothing to cut on; slice anyway."""
    payload = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVph" * 900
    assert " " not in payload

    pieces = _token_split_block([payload], CHUNK_SIZE)

    assert _largest(pieces) <= CHUNK_SIZE


def test_splitting_loses_no_content():
    """Chunks are model input, but they still have to add up to the original."""
    blob = json.dumps({"rows": [{"k": i, "v": "x" * 40} for i in range(800)]})

    pieces = _token_split_block([blob], CHUNK_SIZE)

    assert "".join("".join(p) for p in pieces) == blob


def test_normal_input_is_untouched():
    """No behaviour change when every line already fits -- the benchmark path.

    Reproduces the pre-fix algorithm inline and demands an identical result, so a
    future refactor cannot quietly alter chunk boundaries on clean code.
    """
    lines = [f"def f{i}(a, b):" if i % 8 == 0 else f"    total += compute({i}, a, b)"
             for i in range(600)]
    assert max(count_tokens(l) for l in lines) < CHUNK_SIZE

    expected: list[list[str]] = []
    cur: list[str] = []
    cur_tok = 0
    for ln in lines:
        t = count_tokens(ln)
        if cur_tok + t > CHUNK_SIZE and cur:
            expected.append(cur)
            cur = [ln]
            cur_tok = t
        else:
            cur.append(ln)
            cur_tok += t
    if cur:
        expected.append(cur)

    assert _token_split_block(lines, CHUNK_SIZE) == expected
