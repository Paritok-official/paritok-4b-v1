"""Regression tests for the Gemini synthetic-tool-call reshape (#22).

Gemini's OpenAI-compat endpoint 400s on assistant tool_calls it did not itself sign
("Function call is missing a thought_signature"), which breaks the synthetic-bootstrap
tool history agents use to hand Paritok a role="tool" result to compress on turn 1.
The proxy folds those UNSIGNED tool_call+result pairs into plain user messages (after
compression, so savings survive) and leaves real signed calls untouched.
"""
from paritok.proxy.server import (
    _is_gemini_openai_upstream,
    _assistant_tool_call_is_signed,
    _degemini_synthetic_tool_calls,
)


def test_is_gemini_openai_upstream():
    assert _is_gemini_openai_upstream(
        "https://generativelanguage.googleapis.com/v1beta/openai", "gemini-flash-lite-latest")
    assert _is_gemini_openai_upstream("", "gemini-flash-latest")            # model signal alone
    assert _is_gemini_openai_upstream("https://gw.example/v1beta/openai", "whatever")
    assert _is_gemini_openai_upstream("https://x", "google/gemini-2.5-pro")
    assert not _is_gemini_openai_upstream("https://api.openai.com", "gpt-4o")
    assert not _is_gemini_openai_upstream("https://api.groq.com/openai", "llama-3.1-70b")


def test_signed_detection():
    signed_msg = {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "f"}}],
                  "extra_content": {"google": {"thought_signature": "abc"}}}
    signed_call = {"role": "assistant", "tool_calls": [
        {"id": "c1", "function": {"name": "f"},
         "extra_content": {"google": {"thought_signature": "xyz"}}}]}
    unsigned = {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "f"}}]}
    assert _assistant_tool_call_is_signed(signed_msg)
    assert _assistant_tool_call_is_signed(signed_call)
    assert not _assistant_tool_call_is_signed(unsigned)


def test_fold_unsigned_synthetic_tool_call():
    msgs = [
        {"role": "user", "content": "find settlements"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "search_repo", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "settlement.py: class Settlement: ..."},
    ]
    out, n = _degemini_synthetic_tool_calls(msgs)
    assert n == 1
    # the synthetic assistant tool_call and its role="tool" message are both gone
    assert not any(m.get("tool_calls") for m in out)
    assert not any(m.get("role") == "tool" for m in out)
    # the tool body is now a user message, content preserved (+ the tool name)
    folded = [m for m in out if m["role"] == "user" and "Tool result" in m["content"]]
    assert len(folded) == 1
    assert "settlement.py: class Settlement" in folded[0]["content"]
    assert "search_repo" in folded[0]["content"]


def test_signed_tool_call_untouched():
    msgs = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "function": {"name": "f"}}],
         "extra_content": {"google": {"thought_signature": "sig"}}},
        {"role": "tool", "tool_call_id": "c1", "content": "body"},
    ]
    out, n = _degemini_synthetic_tool_calls(msgs)
    assert n == 0
    assert out == msgs  # a real signed call passes through byte-for-byte


def test_no_tool_calls_is_noop():
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    out, n = _degemini_synthetic_tool_calls(msgs)
    assert n == 0
    assert out == msgs


def test_read_original_not_duplicated_on_double_injection():
    # The expand tool's schema name is "read_original", so the dedup guard must key on that
    # (not the abstract "expand_context"); otherwise the OpenAI path's second injection adds
    # it twice, which Gemini rejects as "Duplicate function declaration: read_original" (#22).
    from paritok.middleware.wrapper import _inject_virtual_tools
    tools = [{"name": "search_repo", "description": "d",
              "input_schema": {"type": "object", "properties": {}}}]
    once = _inject_virtual_tools(tools, has_compressed=True, has_filtered=False)
    twice = _inject_virtual_tools(once, has_compressed=True, has_filtered=False)
    names = [t["name"] for t in twice]
    assert names.count("read_original") == 1, names


def test_assistant_text_preserved_when_folding():
    msgs = [
        {"role": "assistant", "content": "let me search", "tool_calls": [
            {"id": "c1", "function": {"name": "search_repo"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "RESULT BODY"},
    ]
    out, n = _degemini_synthetic_tool_calls(msgs)
    assert n == 1
    assert out[0] == {"role": "assistant", "content": "let me search"}  # text kept, calls dropped
    assert out[1]["role"] == "user" and "RESULT BODY" in out[1]["content"]
