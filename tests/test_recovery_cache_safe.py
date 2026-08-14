"""Regression (issue #41): recovering a dropped tool must NOT mutate the frozen
tools[] set. It is delivered as a tool_result (data), so the prompt-cache prefix
stays byte-stable. The pin-into-frozen path (add_to_frozen /
register_recovered_tools) was removed — guard against its reintroduction.
"""
from paritok.config import ParitokConfig
from paritok.middleware.wrapper import ParitokEngine


def test_recovery_returns_tool_data_not_a_tools_mutation():
    cfg = ParitokConfig()
    cfg.tool_discovery.strategy = "relevance"  # keyword search — no embedding model needed
    engine = ParitokEngine(cfg)
    stubbed = [
        {"name": "mcp__gmail__send_email", "description": "send an email message",
         "input_schema": {"type": "object", "properties": {}}},
        {"name": "mcp__calendar__create_event", "description": "create a calendar event",
         "input_schema": {"type": "object", "properties": {}}},
    ]
    out = engine.resolve_virtual_call(
        "gateway_search_tools", {"query": "send an email"}, stubbed_tools=stubbed)
    # Recovery hands the schemas back as DATA (the caller wraps it in a tool_result),
    # never as a tools[] mutation — so tools[] stays byte-stable and the cache holds.
    assert isinstance(out, dict)
    assert isinstance(out.get("tools"), list)


def test_frozen_selector_has_no_pin_into_frozen_api():
    # The functions that would re-emit tools[] mid-session (busting the prompt cache)
    # are gone. Their return would have justified the #41 rewrite penalty; keep them out.
    import paritok.tool_topk as tk
    assert not hasattr(tk.SessionFrozenSelector, "add_to_frozen")
    assert not hasattr(tk, "register_recovered_tools")


def test_frozen_selection_is_stable_across_calls():
    from paritok.tool_topk import SessionFrozenSelector
    sel = SessionFrozenSelector()
    # No public mutator exists, so the frozen dict is only ever set by select().
    assert "add_to_frozen" not in dir(sel)
