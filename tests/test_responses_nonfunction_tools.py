"""Regression: Codex (Responses API) sends built-in tools like `local_shell`
that have no `name`. The proxy must NOT rewrite them into a function tool with
`name: null` (upstream rejects: "tools[i].name expected a string, got null").
Only real function tools — incl. injected virtual ones — get the flat rebuild.
"""
from paritok.proxy.server import _to_responses_tool


def test_local_shell_builtin_passes_through_untouched():
    t = {"type": "local_shell"}
    assert _to_responses_tool(t) == {"type": "local_shell"}


def test_web_search_builtin_passes_through():
    t = {"type": "web_search"}
    assert _to_responses_tool(t) == {"type": "web_search"}


def test_function_tool_is_rebuilt_flat():
    t = {"type": "function", "name": "apply_patch",
         "description": "patch", "parameters": {"type": "object", "properties": {}}}
    out = _to_responses_tool(t)
    assert out["type"] == "function"
    assert out["name"] == "apply_patch"
    assert out["parameters"] == {"type": "object", "properties": {}}


def test_virtual_tool_anthropic_shape_is_rebuilt_with_params_from_input_schema():
    from paritok.pipelines.virtual import EXPAND_CONTEXT_SCHEMA
    out = _to_responses_tool(EXPAND_CONTEXT_SCHEMA)
    assert out["type"] == "function"
    assert out["name"] == "expand_context"
    # input_schema must be surfaced as `parameters`
    assert out["parameters"] == EXPAND_CONTEXT_SCHEMA["input_schema"]
    assert "shadow_id" in out["parameters"]["properties"]


def test_nameless_custom_tool_never_gets_null_name():
    # anything without a usable name is forwarded verbatim, never name: null
    for t in ({"type": "custom", "format": {}}, {"type": "mcp"}, {"type": "image_generation"}):
        out = _to_responses_tool(t)
        assert out.get("name") is not None or "name" not in out
        assert out == t
