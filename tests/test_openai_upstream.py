"""Tests for OpenAI-compatible upstream routing and error-body robustness."""
from paritok.proxy.server import _openai_chat_url, _extract_tool_text, _pick_responses_upstream
from paritok.proxy.adapters.openai import find_virtual_tool_calls
from paritok.config import CodexConfig
from paritok.proxy.codex_setup import render_codex_config


CHATGPT = "https://chatgpt.com/backend-api/codex/responses"


def test_responses_upstream_api_key_goes_to_openai():
    # A real OpenAI key (sk-) routes to the configured OpenAI base, not ChatGPT.
    assert _pick_responses_upstream("Bearer sk-proj-abc", "https://api.openai.com") == \
        "https://api.openai.com/v1/responses"
    # Honors a custom openai_base_url too.
    assert _pick_responses_upstream("Bearer sk-abc", "http://127.0.0.1:8080") == \
        "http://127.0.0.1:8080/v1/responses"


def test_responses_upstream_subscription_goes_to_chatgpt_backend():
    # A ChatGPT subscription OAuth token (not sk-) routes to the ChatGPT backend.
    assert _pick_responses_upstream("Bearer eyJhbGciOi.oauth.token", "https://api.openai.com") == CHATGPT
    # Case/prefix: anything that isn't a Bearer sk- key and is non-empty → ChatGPT.
    assert _pick_responses_upstream("Bearer gho_notanopenaikey", "https://api.openai.com") == CHATGPT


def test_responses_upstream_missing_or_malformed_auth_defaults_to_openai():
    # No token (or non-Bearer) → default OpenAI path, never the subscription backend.
    assert _pick_responses_upstream("", "https://api.openai.com") == "https://api.openai.com/v1/responses"
    assert _pick_responses_upstream("Basic xyz", "https://api.openai.com") == \
        "https://api.openai.com/v1/responses"


def test_codex_config_defaults_to_subscription():
    # No subscription/api_key given → subscription is the default (requires_openai_auth),
    # even if an api_key is present but subscription isn't explicitly disabled.
    assert "requires_openai_auth = true" in render_codex_config(CodexConfig(model="gpt-5"), "127.0.0.1", 8080)
    with_key = render_codex_config(CodexConfig(model="gpt-5", api_key="sk-proj-XXX"), "127.0.0.1", 8080)
    assert "requires_openai_auth = true" in with_key
    assert "experimental_bearer_token" not in with_key


def test_codex_config_subscription_uses_oauth_not_key():
    body = render_codex_config(CodexConfig(model="gpt-5", subscription=True), "127.0.0.1", 8080)
    assert "requires_openai_auth = true" in body
    assert "experimental_bearer_token" not in body
    assert "env_key" not in body


def test_codex_config_api_key_mode_needs_subscription_false():
    # API key path only when subscription is explicitly disabled.
    body = render_codex_config(
        CodexConfig(model="gpt-5", subscription=False, api_key="sk-proj-XXX"), "127.0.0.1", 8080)
    assert 'experimental_bearer_token = "sk-proj-XXX"' in body
    assert "requires_openai_auth" not in body


def test_codex_config_empty_model_omits_model_line():
    # Empty model → no `model =` line, so Codex chooses (needed for ChatGPT accounts).
    body = render_codex_config(CodexConfig(subscription=True), "127.0.0.1", 8080)
    assert "\nmodel = " not in ("\n" + body)
    assert 'model_provider = "paritok"' in body
    # Explicit model still pins it.
    pinned = render_codex_config(CodexConfig(model="o3", subscription=True), "127.0.0.1", 8080)
    assert 'model = "o3"' in pinned


def test_codex_config_subscription_false_no_key_uses_env():
    body = render_codex_config(CodexConfig(model="gpt-5", subscription=False), "127.0.0.1", 8080)
    assert 'env_key = "OPENAI_API_KEY"' in body
    assert "requires_openai_auth" not in body


def test_openai_chat_url_appends_suffix_for_base_hosts():
    # OpenAI and Groq give a base host; the standard suffix is appended.
    assert _openai_chat_url("https://api.openai.com") == "https://api.openai.com/v1/chat/completions"
    assert _openai_chat_url("https://api.groq.com/openai") == \
        "https://api.groq.com/openai/v1/chat/completions"
    # A trailing slash must not double up.
    assert _openai_chat_url("https://api.openai.com/") == "https://api.openai.com/v1/chat/completions"


def test_openai_chat_url_no_double_version_for_versioned_bases():
    # A base whose path already carries a version segment must NOT get a second
    # /v1 (that yields .../v1/v1/chat/completions -> 404 on OpenRouter etc.).
    assert _openai_chat_url("https://openrouter.ai/api/v1") == \
        "https://openrouter.ai/api/v1/chat/completions"
    assert _openai_chat_url("https://generativelanguage.googleapis.com/v1beta/openai") == \
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    # User who already appended /v1 to an OpenAI-style host: no doubling.
    assert _openai_chat_url("https://api.openai.com/v1") == \
        "https://api.openai.com/v1/chat/completions"


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


def test_extract_tool_text_string_content():
    # Plain-string tool content: returned verbatim, rewrap is identity.
    text, rewrap = _extract_tool_text("   1\tprint('hi')\n")
    assert text == "   1\tprint('hi')\n"
    assert rewrap("COMPRESSED") == "COMPRESSED"


def test_extract_tool_text_list_content_single_text_block():
    # gptme / Claude-Code-OpenAI send content as a list of parts; the file read used
    # to be silently skipped. Extract the text and rewrap back into the list shape.
    content = [{"type": "text", "text": "```f.py\n  1\tx = 1\n  2\ty = 2\n```"}]
    text, rewrap = _extract_tool_text(content)
    assert "x = 1" in text and "y = 2" in text
    assert rewrap("[REF:abc]") == [{"type": "text", "text": "[REF:abc]"}]


def test_extract_tool_text_list_preserves_non_text_parts():
    content = [
        {"type": "text", "text": "big file body"},
        {"type": "image_url", "image_url": {"url": "data:x"}},
    ]
    text, rewrap = _extract_tool_text(content)
    assert text == "big file body"
    out = rewrap("SMALL")
    assert out[0] == {"type": "text", "text": "SMALL"}
    assert out[1] == {"type": "image_url", "image_url": {"url": "data:x"}}  # untouched


def test_extract_tool_text_no_text_returns_none():
    assert _extract_tool_text([{"type": "image_url", "image_url": {"url": "u"}}]) == (None, None)
    assert _extract_tool_text(123) == (None, None)
