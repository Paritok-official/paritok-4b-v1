"""Regression for issue #48: classify_kind_from_content ran the Traceback/FAILED
heuristic BEFORE the cat -n -> file_read refinement AND scanned the whole body, so
any Read whose content merely *contained* the literal 'Traceback' or 'FAILED'
(extremely common in tests, error-handling code, and log fixtures) was misrouted to
log_output -> the other.txt prompt instead of file_read.txt, degrading compression
on the product's most common input.

Fix: run reclassify_tool_result first (so cat -n / code reads win), and keep the
error-marker check only as a fallback scoped to `head`, not the whole body.
"""
from paritok.strategies.prompts import system_prompt_for_kind
from paritok.strategies.tagger import classify_kind_from_content

_CAT_N_HEADER = "Here's the result of running `cat -n` on django/db/models/deletion.py:\n"
_CODE = "\n".join(
    f"{i}\t{ln}"
    for i, ln in enumerate(
        [
            "import logging",
            "from django.db import models",
            "",
            "class Collector:",
            "    def collect(self, objs):",
            "        # error handling below; keep it robust",
            "        try:",
            "            return self._collect(objs)",
            "        except Exception:",
            "            logging.exception('collect failed')",
            "            raise",
        ],
        1,
    )
)


def test_cat_n_read_with_traceback_in_body_stays_file_read():
    # The literal 'Traceback' appears deep in the body (a logged stack trace in an
    # error-handling branch). Before the fix this flipped the whole read to log_output.
    body = (
        _CAT_N_HEADER
        + _CODE
        + "\n99\t            logging.error('Traceback (most recent call last):')"
    )
    assert "Traceback" in body
    assert classify_kind_from_content(body) == "file_read"
    assert system_prompt_for_kind(classify_kind_from_content(body)) == system_prompt_for_kind(
        "file_read"
    )


def test_cat_n_read_with_failed_in_body_stays_file_read():
    body = _CAT_N_HEADER + _CODE + "\n99\t        assert ok, 'operation FAILED'"
    assert "FAILED" in body
    assert classify_kind_from_content(body) == "file_read"


def test_plain_code_read_with_failed_keyword_stays_file_read():
    # No cat -n header, but 2+ code indicators + many lines -> reclassify calls it
    # file_read; a 'FAILED' string in the body must not steal it to log_output.
    src = "\n".join(
        [
            "import os",
            "from typing import Any",
            "",
            "class Runner:",
            "    def run(self) -> None:",
            "        for i in range(20):",
            "            if not self._ok(i):",
            "                raise RuntimeError('step FAILED at %d' % i)",
            "    def _ok(self, i: int) -> bool:",
            "        return i % 2 == 0",
            "    def report(self):",
            "        return 'done'",
        ]
    )
    assert "FAILED" in src
    assert classify_kind_from_content(src) == "file_read"


def test_real_traceback_still_log_output():
    # The fallback still fires when a genuine stack trace leads the content.
    log = "Traceback (most recent call last):\n  File 'x.py', line 3\nValueError: boom\n"
    assert classify_kind_from_content(log) == "log_output"


def test_pytest_failed_summary_still_log_output():
    # A real test-run log that opens with FAILED lines must still be log_output.
    log = (
        "FAILED tests/test_a.py::test_x - AssertionError\n"
        "FAILED tests/test_b.py::test_y - ValueError\n"
        "===== 2 failed, 5 passed in 1.20s =====\n"
    )
    assert classify_kind_from_content(log) == "log_output"
