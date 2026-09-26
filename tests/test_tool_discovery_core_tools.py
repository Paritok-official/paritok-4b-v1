"""Regression (issue #27 part 2): Claude Code core tools must survive embedding
tool discovery even when the first-turn query is non-code (WHITELIST not applied)
and the selector would otherwise drop them.

The floor lives in `_CORE_EXEC_TOOLS` inside the embedding filter — protected
names are force-appended after ranking and may make kept_count exceed k_max.
"""
from __future__ import annotations

from paritok.config import ParitokConfig, ToolDiscoveryConfig
from paritok.pipelines.tool_discovery import ToolDiscoveryPipeline
from paritok.tool_topk import _tool_name

# Non-code query: must NOT match tool_topk._CODE_HINT (else WHITELIST would
# already prepend Read/Write/… and hide the floor bug).
_NON_CODE_QUERY = "What is the capital of France?"

_CLAUDE_CODE_CORE = ("Read", "Write", "Edit", "Glob", "Grep", "Bash")
_LOWERCASE_CORE = ("bash", "exec", "apply_patch", "shell")


def _schema(name: str, desc: str = "utility action") -> dict:
    return {
        "name": name,
        "description": desc,
        "input_schema": {"type": "object", "properties": {}},
    }


def _claude_code_shaped_pool() -> list[dict]:
    """Large pool: Claude Code head + many ordinary tools the selector prefers."""
    core = [_schema(n, f"Claude Code {n}") for n in _CLAUDE_CODE_CORE]
    ordinary = [
        _schema(f"tool_{i:02d}", f"ordinary utility number {i}")
        for i in range(20)
    ]
    return core + ordinary


def _pipeline() -> ToolDiscoveryPipeline:
    cfg = ParitokConfig()
    cfg.tool_discovery = ToolDiscoveryConfig(strategy="embedding", k_max=8)
    return ToolDiscoveryPipeline(cfg)


def _stub_predict_topk_frozen(monkeypatch, keep_names: list[str]):
    """Selector returns only the given names — never the protected core set."""
    import paritok.tool_topk as tk

    def fake(session_id, user_message, tools, k_max=None):
        present = {_tool_name(t) for t in tools}
        return [n for n in keep_names if n in present]

    # _embedding_filter does `from paritok.tool_topk import predict_topk_frozen`
    # at call time, so patching the module attribute is enough.
    monkeypatch.setattr(tk, "predict_topk_frozen", fake)


def test_claude_code_core_survives_non_code_embedding_discovery(monkeypatch):
    """Observable: Read/Write/Edit/Glob/Grep/Bash stay FULL after embedding filter."""
    tools = _claude_code_shaped_pool()
    _stub_predict_topk_frozen(monkeypatch, [f"tool_{i:02d}" for i in range(8)])

    result = _pipeline().filter_tools(
        tools, _NON_CODE_QUERY, session_id="cc-core-noncode"
    )
    full_names = {_tool_name(t) for t in result.full_tools}
    for name in _CLAUDE_CODE_CORE:
        assert name in full_names, f"{name} must remain a full tool"


def test_lowercase_core_exec_tools_still_protected(monkeypatch):
    tools = [_schema(n) for n in _LOWERCASE_CORE] + [
        _schema(f"tool_{i:02d}") for i in range(20)
    ]
    _stub_predict_topk_frozen(monkeypatch, [f"tool_{i:02d}" for i in range(8)])

    result = _pipeline().filter_tools(
        tools, _NON_CODE_QUERY, session_id="lower-core"
    )
    full_names = {_tool_name(t) for t in result.full_tools}
    for name in _LOWERCASE_CORE:
        assert name in full_names, f"{name} must remain a full tool"


def test_ordinary_tools_are_still_droppable(monkeypatch):
    tools = _claude_code_shaped_pool()
    # Selector keeps only a subset of ordinary tools; others must remain droppable.
    _stub_predict_topk_frozen(monkeypatch, [f"tool_{i:02d}" for i in range(5)])

    result = _pipeline().filter_tools(
        tools, _NON_CODE_QUERY, session_id="ordinary-drop"
    )
    full_names = {_tool_name(t) for t in result.full_tools}
    assert "tool_19" not in full_names
    assert "tool_10" not in full_names


def test_protected_tools_may_exceed_k_max(monkeypatch):
    """Force-kept core tools are not clamped back under k_max."""
    tools = _claude_code_shaped_pool()
    _stub_predict_topk_frozen(monkeypatch, [f"tool_{i:02d}" for i in range(8)])

    result = _pipeline().filter_tools(
        tools, _NON_CODE_QUERY, session_id="exceed-kmax"
    )
    full_names = {_tool_name(t) for t in result.full_tools}
    for name in _CLAUDE_CODE_CORE:
        assert name in full_names
    # 8 ordinary + 6 Claude Code core > k_max
    assert result.kept_count > 8
    assert result.kept_count == len(full_names)
