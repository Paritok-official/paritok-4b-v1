"""Regression: codex frames file reads in a shell command-output header
(Exit code:/Wall time:/Total output lines:/Output:). Compressing that header
together with the body used to feed the model something no other agent sends,
skewing the kind sniff (a source file summarized away + hallucinated).

_split_codex_header peels the frame off so ONLY the body is compressed — the same
bytes every other agent hands the model — giving one uniform compression result
regardless of which agent produced the content. The header is re-attached verbatim
by the caller. Content without the frame is returned untouched.
"""
from paritok.proxy.server import _split_codex_header

_CODE = '''"""Step 3: Run LLM on QuALITY questions."""
import argparse
import json


def answer_question(client, article, question):
    prompt = f"Read: {article}"
    return client.create(prompt)
'''

_HEADER = ("Exit code: 0\n"
           "Wall time: 0.1 seconds\n"
           "Total output lines: 292\n"
           "Output:\n")


def test_body_matches_what_other_agents_send():
    # THE consistency guarantee: peeling the codex frame yields exactly the bytes
    # a Claude/OpenAI agent would hand the model for the same file.
    header, body = _split_codex_header(_HEADER + _CODE)
    assert body == _CODE
    assert header == _HEADER
    assert header + body == _HEADER + _CODE  # lossless: header re-attaches verbatim


def test_unwrapped_content_untouched():
    # no codex frame -> header empty, body is the original content unchanged
    header, body = _split_codex_header(_CODE)
    assert header == ""
    assert body == _CODE


def test_header_only_leaves_empty_body():
    header, body = _split_codex_header(_HEADER)
    assert header == _HEADER
    assert body.strip() == ""


def test_traceback_body_preserved():
    # a real command failure: frame split off, body kept intact for the server to
    # sniff (as log_output) — same as if any other agent had produced the log.
    log = "Traceback (most recent call last):\n  File 'x.py'\nValueError: boom\n"
    header, body = _split_codex_header(_HEADER + log)
    assert header == _HEADER
    assert body == log


def test_no_false_header_on_prose_starting_with_output_word():
    # only exact frame prefixes count; ordinary prose is not misread as a header
    text = "Outputting results now\nall good\n"
    header, body = _split_codex_header(text)
    assert header == "" and body == text


# ── _ensure_line_numbers: codex reads have no line numbers (OOD); number source ──
from paritok.proxy.server import _ensure_line_numbers  # noqa: E402


def test_unnumbered_source_gets_arrow_numbering():
    # Claude-Read style: `<num>→line` (the format the model was trained on)
    src = "import os\n\n\ndef f():\n    return os.getcwd()\n"
    out = _ensure_line_numbers(src)
    lines = out.splitlines()
    assert lines[0] == f"{1:6d}→import os"
    assert lines[3] == f"{4:6d}→def f():"
    # every original line is preserved, just prefixed
    assert "def f():" in out and "return os.getcwd()" in out


def test_already_arrow_numbered_source_left_alone():
    numbered = "".join(f"{i:6d}→{ln}\n" for i, ln in enumerate(
        ["import os", "def f():", "    return 1", "x = f()"], 1))
    assert _ensure_line_numbers(numbered) == numbered


def test_already_cat_n_numbered_source_left_alone():
    # tab-numbered content (e.g. an actual cat -n) is also recognized, not doubled
    numbered = "".join(f"{i:6d}\t{ln}\n" for i, ln in enumerate(
        ["import os", "def f():", "    return 1", "x = f()"], 1))
    assert _ensure_line_numbers(numbered) == numbered


def test_logs_and_prose_not_numbered():
    # no code signals -> untouched (numbering a log would corrupt it)
    log = "starting run\nprocessed 10 items\nprocessed 20 items\ndone in 3s\n"
    assert _ensure_line_numbers(log) == log


def test_short_snippet_not_numbered():
    assert _ensure_line_numbers("def f(): return 1\n") == "def f(): return 1\n"
