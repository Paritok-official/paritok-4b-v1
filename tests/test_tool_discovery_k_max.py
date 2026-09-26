"""Regression (issue #27 part 1): tool_discovery.k_max must reach the frozen
embedding selector. Without it, SessionFrozenSelector always clamps at 8 even
when config asks for more.
"""
from __future__ import annotations

from paritok.config import ParitokConfig, ToolDiscoveryConfig
from paritok.pipelines.tool_discovery import ToolDiscoveryPipeline
from paritok.tool_topk import SessionFrozenSelector, _tool_name, predict_topk_frozen


def _tools(n: int) -> list[dict]:
    return [
        {"name": f"tool_{i:02d}", "description": f"utility action number {i}",
         "input_schema": {"type": "object", "properties": {}}}
        for i in range(n)
    ]


def _stub_select_dynamic(monkeypatch, selector: SessionFrozenSelector):
    """Deterministic scorer: return tools in name order, capped at the k_max
    argument the selector actually passes through. Proves the limit is honored
    without needing an embedding model."""
    def fake(user_message, tools, alpha=0.9, k_min=5, k_max=20):
        return [_tool_name(t) for t in tools[:k_max]]
    monkeypatch.setattr(selector._sel, "select_dynamic", fake)


def test_frozen_selector_default_k_max_caps_at_8(monkeypatch):
    sel = SessionFrozenSelector()
    _stub_select_dynamic(monkeypatch, sel)
    tools = _tools(25)
    kept = sel.select("sess-default", "do utilities", tools)
    assert len(kept) == 8
    assert kept == [f"tool_{i:02d}" for i in range(8)]


def test_frozen_selector_explicit_k_max_8_caps_at_8(monkeypatch):
    sel = SessionFrozenSelector()
    _stub_select_dynamic(monkeypatch, sel)
    tools = _tools(25)
    kept = sel.select("sess-k8", "do utilities", tools, k_max=8)
    assert len(kept) == 8


def test_frozen_selector_k_max_20_can_retain_more_than_8(monkeypatch):
    sel = SessionFrozenSelector()
    _stub_select_dynamic(monkeypatch, sel)
    tools = _tools(25)
    kept = sel.select("sess-k20", "do utilities", tools, k_max=20)
    assert len(kept) == 20
    assert kept == [f"tool_{i:02d}" for i in range(20)]


def test_frozen_selector_boundaries(monkeypatch):
    sel = SessionFrozenSelector()
    _stub_select_dynamic(monkeypatch, sel)
    # Fewer tools than k_max → keep all.
    assert len(sel.select("b-few", "q", _tools(6), k_max=20)) == 6
    # Exactly k_max → keep exactly that many.
    assert len(sel.select("b-eq", "q", _tools(12), k_max=12)) == 12
    # More than k_max → clamp.
    assert len(sel.select("b-more", "q", _tools(30), k_max=12)) == 12


def test_frozen_set_stable_across_second_call_even_if_k_max_differs(monkeypatch):
    """Freeze captures the first call's selection; a later different k_max must
    not rewrite the session's frozen set."""
    sel = SessionFrozenSelector()
    _stub_select_dynamic(monkeypatch, sel)
    tools = _tools(25)
    first = sel.select("freeze-sid", "do utilities", tools, k_max=20)
    second = sel.select("freeze-sid", "do utilities", tools, k_max=5)
    assert first == second
    assert len(second) == 20


def test_independent_sessions_do_not_leak_k_max(monkeypatch):
    """Session A with k_max=8 and session B with k_max=20 must not interfere."""
    sel = SessionFrozenSelector()
    _stub_select_dynamic(monkeypatch, sel)
    tools = _tools(25)
    a = sel.select("sess-A", "do utilities", tools, k_max=8)
    b = sel.select("sess-B", "do utilities", tools, k_max=20)
    assert len(a) == 8
    assert len(b) == 20
    # Re-select either session: each keeps its own frozen size.
    assert len(sel.select("sess-A", "do utilities", tools, k_max=20)) == 8
    assert len(sel.select("sess-B", "do utilities", tools, k_max=5)) == 20


def test_predict_topk_frozen_omitted_k_max_keeps_default_8(monkeypatch):
    import paritok.tool_topk as tk
    _stub_select_dynamic(monkeypatch, tk._frozen_default)
    tools = _tools(25)
    kept = predict_topk_frozen("pred-default", "do utilities", tools)
    assert len(kept) == 8


def test_predict_topk_frozen_honors_explicit_k_max(monkeypatch):
    import paritok.tool_topk as tk
    _stub_select_dynamic(monkeypatch, tk._frozen_default)
    tools = _tools(25)
    kept = predict_topk_frozen("pred-k20", "do utilities", tools, k_max=20)
    assert len(kept) == 20


def test_pipeline_embedding_honors_configured_k_max(monkeypatch):
    """Observable contract: embedding discovery with k_max=20 retains >8 tools."""
    import paritok.tool_topk as tk
    _stub_select_dynamic(monkeypatch, tk._frozen_default)

    cfg = ParitokConfig()
    cfg.tool_discovery = ToolDiscoveryConfig(strategy="embedding", k_max=20)
    pipe = ToolDiscoveryPipeline(cfg)
    tools = _tools(25)

    result = pipe.filter_tools(tools, "do utilities", session_id="pipe-k20")
    assert result.kept_count == 20
    assert len(result.full_tools) == 20


def test_pipeline_embedding_default_k_max_still_8(monkeypatch):
    import paritok.tool_topk as tk
    _stub_select_dynamic(monkeypatch, tk._frozen_default)

    cfg = ParitokConfig()
    cfg.tool_discovery = ToolDiscoveryConfig(strategy="embedding")  # k_max defaults to 8
    pipe = ToolDiscoveryPipeline(cfg)
    tools = _tools(25)

    result = pipe.filter_tools(tools, "do utilities", session_id="pipe-default")
    assert result.kept_count == 8
    assert len(result.full_tools) == 8
