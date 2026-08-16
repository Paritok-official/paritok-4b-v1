"""Regression for issue #21 (§C4): shell/bash command output and structured logs
were mis-classified as `file_read`.

`classify_kind_from_content` defaults to `file_read`, and its only fallback to
`log_output` — "> 5 newlines in the first 200 chars" — never trips on log output
whose lines are long (a single 110-char structured-log line is one newline). So a
`$ ./script.sh` run producing `2026-07-01T.. level=info ..` lines slipped through
to `file_read` and got the code-tuned system prompt instead of the broader "other"
one. The independent CPU reproduction (BayramAnnakov/paritok-repro) measured its
two `bash_output` segments landing on `file_read`, and one of them compressed to
only 0.172 as a result.

The fix adds explicit markers — an echoed `$ ` / `>>> ` prompt, an ISO-8601
timestamp at a line head, or a `level=<sev>` field — checked AFTER the code
signals so genuine source is never pulled into log_output.
"""
from paritok.strategies.tagger import classify_kind_from_content

# The exact shape harness/corpus.py::_bash_output emits ($ cmd + structured log
# lines with an ISO timestamp and level= field).
_BASH_OUTPUT = (
    "$ ./scripts/billing_report.sh --since 2026-07-01 --verbose\n"
    + "\n".join(
        f"2026-07-{(i % 28) + 1:02d}T{i % 24:02d}:{i % 60:02d}:{(i * 7) % 60:02d}Z "
        f"level=info component=billing record_id=billing-{i:08x} "
        f"amount_cents={i * 37} latency_ms={i * 1.5:.2f}"
        for i in range(40)
    )
)


def test_shell_prompt_output_is_log_output():
    assert classify_kind_from_content(_BASH_OUTPUT) == "log_output"


def test_structured_log_without_prompt_is_log_output():
    # No echoed command — starts straight into timestamped level= lines.
    body = "\n".join(
        f"2026-08-15T09:{i:02d}:00Z level=warn component=scheduler ret--{i}"
        for i in range(30)
    )
    assert classify_kind_from_content(body) == "log_output"


def test_level_field_marks_log_output():
    # Long lines, no timestamp, but the level= field is an unambiguous log marker.
    body = (
        "component=payout level=error msg=\"settlement guard tripped\" "
        "record_id=payout-0000abcd amount_cents=-5000 attempt=3 backoff_ms=1200\n"
        "component=payout level=info msg=\"retrying settlement\" record_id=payout-0000abcd"
    )
    assert classify_kind_from_content(body) == "log_output"


def test_repl_prompt_is_log_output():
    body = ">>> import billing\n>>> billing.compute()\nTraceback would go here but no keyword"
    assert classify_kind_from_content(body) == "log_output"


# ── the fix must NOT steal genuine file reads ────────────────────────────────

def test_plain_source_still_file_read():
    src = (
        "from __future__ import annotations\n"
        "import logging\n\n"
        "def compute_billing(record):\n"
        "    return record.amount_cents * 3\n"
    )
    assert classify_kind_from_content(src) == "file_read"


def test_source_mentioning_a_timestamp_still_file_read():
    # A source file whose head contains a date literal must stay file_read: the
    # code signals are checked first, so the timestamp regex never gets to vote.
    src = (
        "import datetime\n\n"
        "def stamp():\n"
        '    return datetime.date(2026, 7, 1).isoformat()  # 2026-07-01\n'
    )
    assert classify_kind_from_content(src) == "file_read"


def test_cat_n_read_still_file_read():
    numbered = "Here's the result of running `cat -n` on the file:\n" + "\n".join(
        f"{i:6d}\tdef f_{i}(): return {i}" for i in range(1, 20)
    )
    assert classify_kind_from_content(numbered) == "file_read"


def test_traceback_still_log_output():
    # Unchanged path — the Traceback keyword short-circuits before the new markers.
    log = "Traceback (most recent call last):\n  File 'x.py'\nValueError: boom\n"
    assert classify_kind_from_content(log) == "log_output"
