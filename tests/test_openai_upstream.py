"""Tests for OpenAI-compatible upstream routing and error-body robustness."""
from paritok.proxy.server import _openai_chat_url
from paritok.proxy.adapters.openai import find_virtual_tool_calls


def test_openai_chat_url_appends_suffix_for_base_hosts():
    # OpenAI and Groq give a base host; the standard suffix is appended.
    assert _openai_chat_url("https://api.openai.com") == "https://api.openai.com/v1/chat/completions"
    assert _openai_chat_url("https://api.groq.com/openai") == \
        "https://api.groq.com/openai/v1/chat/completions"
    # A trailing slash must not double up.
    assert _openai_chat_url("https://api.openai.com/") == "https://api.openai.com/v1/chat/completions"


def test_openai_chat_url_uses_full_endpoint_verbatim():
    # Gemini's OpenAI-compat path isn't {base}/v1/...; a full endpoint is used as-is.
    full = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    assert _openai_chat_url(full) == full
    assert _openai_chat_url(full + "/") == full


def test_find_virtual_tool_calls_tolerates_non_dict_body():
    # Some providers (e.g. Gemini) return a top-level list for errors; must not crash.
    assert find_virtual_tool_calls([{"error": {"code": 429}}]) == []
    assert find_virtual_tool_calls("boom") == []
    assert find_virtual_tool_calls({}) == []


def test_find_virtual_tool_calls_finds_virtual_calls():
    body = {"choices": [{"message": {"tool_calls": [
        {"id": "c1", "function": {"name": "expand_context", "arguments": "{}"}},
        {"id": "c2", "function": {"name": "read_file", "arguments": "{}"}},
    ]}}]}
    names = [tc["function"]["name"] for tc in find_virtual_tool_calls(body)]
    assert names == ["expand_context"]  # only the virtual tool, not the real one
