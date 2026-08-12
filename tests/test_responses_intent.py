"""Compression intent for the Responses API (codex), mirroring the Anthropic adapter.

The 4B keeps what the intent NAMES and drops the rest. If the intent stays pinned to the
static user task ("fix the bug in answer_question") while the agent has moved on to digging
into `quick_eval`, the compressor drops the very `quick_eval` context codex just fetched --
so codex re-searches (the redundant second `Select-String` we observed). extract_query must
prefer the agent's LATEST narration/reasoning (current intent), carrying the task as context.
"""
from paritok.proxy.adapters.responses import (
    _extract_task,
    _latest_assistant_intent,
    extract_query,
)

TASK = "fix the bug in answer_question"
INTENT = "I'll inspect quick_eval and its call sites, then make the smallest targeted fix."


def _user(text):
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def _assistant(text):
    return {"type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": text}]}


def _reasoning(text):
    return {"type": "reasoning", "summary": [{"type": "summary_text", "text": text}]}


# ── turn 0: no assistant text yet → the task alone ──

def test_turn0_falls_back_to_task():
    inp = [_user(TASK)]
    assert extract_query(inp) == TASK
    assert _latest_assistant_intent(inp) is None
    assert _extract_task(inp) == TASK


def test_bare_string_input_is_the_task():
    assert extract_query("fix answer_question") == "fix answer_question"


# ── later turns: agent narration is the current intent, task trails as context ──

def test_prefers_assistant_narration_with_task_context():
    inp = [
        _user(TASK),
        _reasoning("thinking about the eval"),
        _assistant(INTENT),
        {"type": "function_call", "name": "shell", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "rg results ..."},
    ]
    q = extract_query(inp)
    assert q.startswith(INTENT)
    assert "(overall task: fix the bug in answer_question)" in q
    # the code the agent is actually working on is named in the intent
    assert "quick_eval" in q


def test_assistant_message_wins_over_same_turn_reasoning():
    # reasoning precedes the message in the turn; the visible message text is preferred.
    inp = [_user(TASK), _reasoning("some private reasoning"), _assistant(INTENT)]
    assert _latest_assistant_intent(inp) == INTENT


def test_reasoning_summary_used_when_no_assistant_message():
    inp = [_user(TASK), _reasoning("Looking into quick_eval's counter")]
    q = extract_query(inp)
    assert q.startswith("Looking into quick_eval's counter")
    assert "(overall task: fix the bug in answer_question)" in q


def test_latest_intent_wins_across_turns():
    inp = [
        _user(TASK),
        _assistant("First I'll read answer_question."),
        {"type": "function_call_output", "call_id": "c1", "output": "..."},
        _assistant(INTENT),  # newest narration
    ]
    assert _latest_assistant_intent(inp) == INTENT


def test_intent_truncated_to_budget():
    long_intent = "x" * 2000
    inp = [_user(TASK), _assistant(long_intent)]
    q = extract_query(inp)
    assert q.startswith("x" * 900)
    assert "x" * 901 not in q  # capped at 900
    assert "(overall task: " in q


def test_reasoning_without_summary_text_is_ignored():
    # encrypted-only reasoning (no plaintext summary) must not be used as intent.
    inp = [_user(TASK), {"type": "reasoning", "summary": [], "encrypted_content": "…"}]
    assert _latest_assistant_intent(inp) is None
    assert extract_query(inp) == TASK


def test_empty_input_returns_none():
    assert extract_query([]) is None
    assert extract_query(None) is None
