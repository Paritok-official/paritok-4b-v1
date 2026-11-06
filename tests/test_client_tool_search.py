"""Client-side tool search (Claude Code's deferred tool loading) vs paritok's own
tool discovery.

With tool search on, Claude Code sends a small tools[] (core tools + ToolSearch +
a `defer_loading` placeholder) and loads the rest on demand. Paritok then never
DROPS a tool — a dropped tool makes the API 400 when ToolSearch references it
("Tool reference 'X' not found in available tools", seen live) — it marks the
ones its semantic selection leaves out `defer_loading: true`, so they cost nothing
until ToolSearch loads them. Savings count only schemas the API actually bills.
"""
import json

import pytest

from paritok.config import ParitokConfig, ToolDiscoveryConfig
from paritok.middleware.wrapper import ParitokEngine
from paritok.pipelines.tool_discovery import (
    ToolDiscoveryPipeline,
    client_tool_search_active,
)


def _tool(name, **extra):
    return {"name": name, "description": f"{name} tool",
            "input_schema": {"type": "object", "properties": {}}, **extra}


# The shape Claude Code 2.1.x sends with tool search on (captured from a real run).
CLAUDE_TOOL_SEARCH = [_tool(n) for n in (
    "Agent", "Bash", "Edit", "Glob", "Grep", "ListAgents", "PowerShell", "Read",
    "ReportFindings", "ScheduleWakeup", "Skill", "ToolSearch", "Workflow")] + [
    _tool("DeferredToolPlaceholder", defer_loading=True), _tool("Write")]

TASK = [{"role": "user", "content": "fix the .delete() bug in django/db/models/deletion.py"}]
SELECTED = ["Read", "Edit", "Grep", "Agent"]  # what the (stubbed) semantic ranker picks


@pytest.fixture
def ranker(monkeypatch):
    """Deterministic stand-in for the bge-small ranker (no model download in CI)."""
    import paritok.tool_topk as tk
    monkeypatch.setattr(tk, "predict_topk_frozen",
                        lambda sid, q, tools, k_max=None: [n for n in SELECTED
                                                           if n in {t["name"] for t in tools}])


def _engine(defer=True):
    cfg = ParitokConfig()
    cfg.tool_discovery = ToolDiscoveryConfig(strategy="embedding", k_max=8,
                                             defer_under_tool_search=defer)
    return ParitokEngine(cfg)


def _deferred(tools):
    return {t["name"] for t in tools if t.get("defer_loading")}


def test_detects_client_tool_search():
    assert client_tool_search_active(CLAUDE_TOOL_SEARCH)
    assert client_tool_search_active([_tool("Read"), _tool("mcp__x", defer_loading=True)])
    assert client_tool_search_active([{"type": "tool_search_tool_regex_20251119",
                                       "name": "tool_search_tool_regex"}])
    history = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1",
                                             "content": [{"type": "tool_reference",
                                                          "tool_name": "mcp__figma__x"}]}]}]
    assert client_tool_search_active([_tool("Read")], history)
    assert not client_tool_search_active([_tool(n) for n in ("Read", "Edit", "mcp__a__b")], TASK)


def test_unselected_tools_are_deferred_never_dropped(ranker):
    _, fwd, stats, stubbed = _engine().process_request(TASK, list(CLAUDE_TOOL_SEARCH))
    names = [t["name"] for t in fwd]
    # every client tool is still there, in the client's order (cache-stable)
    assert names[:len(CLAUDE_TOOL_SEARCH)] == [t["name"] for t in CLAUDE_TOOL_SEARCH]
    assert "read_original" in names and "gateway_search_tools" not in names
    assert stubbed == []
    # unselected → deferred; selected, core tools and ToolSearch stay loaded
    assert _deferred(fwd) == {"ListAgents", "PowerShell", "ReportFindings", "ScheduleWakeup",
                              "Skill", "Workflow", "DeferredToolPlaceholder"}
    assert set(stats.tools_deferred) == _deferred(fwd) - {"DeferredToolPlaceholder"}
    for keep in ("ToolSearch", "Bash", "Read", "Edit", "Write", "Glob", "Grep", "Agent"):
        assert keep not in _deferred(fwd)
    # schemas otherwise unchanged
    by = {t["name"]: t for t in fwd}
    assert {k: v for k, v in by["Workflow"].items() if k != "defer_loading"} == _tool("Workflow")


def test_tools_loaded_mid_session_are_not_deferred(ranker):
    """A tool the agent loaded through ToolSearch shows up in a LATER request; it was
    not on the session's first request, so it must stay loaded."""
    eng = _engine()
    eng.process_request(TASK, list(CLAUDE_TOOL_SEARCH))
    later = [*CLAUDE_TOOL_SEARCH, _tool("mcp__figma__get_screenshot")]
    _, fwd, _, _ = eng.process_request(TASK, later)
    assert "mcp__figma__get_screenshot" in [t["name"] for t in fwd]
    assert "mcp__figma__get_screenshot" not in _deferred(fwd)


def test_powershell_kept_when_it_is_the_only_shell(ranker):
    no_bash = [t for t in CLAUDE_TOOL_SEARCH if t["name"] != "Bash"]
    _, fwd, _, _ = _engine().process_request(TASK, no_bash)
    assert "PowerShell" not in _deferred(fwd)


def test_defer_off_passes_tools_through(ranker):
    _, fwd, stats, _ = _engine(defer=False).process_request(TASK, list(CLAUDE_TOOL_SEARCH))
    assert fwd[:len(CLAUDE_TOOL_SEARCH)] == CLAUDE_TOOL_SEARCH
    assert stats.tools_deferred == []


def test_server_and_deferred_tools_are_never_filtered(monkeypatch):
    """Without tool search, discovery still runs — but a server tool (a `type` the API
    executes) and a deferred tool are kept as-is, never dropped or stubbed."""
    import paritok.tool_topk as tk
    monkeypatch.setattr(tk, "predict_topk_frozen",
                        lambda sid, q, tools, k_max=None: ["tool_00", "tool_01"])
    web = {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}
    tools = [_tool(f"tool_{i:02d}") for i in range(20)] + [web]
    cfg = ParitokConfig()
    cfg.tool_discovery = ToolDiscoveryConfig(strategy="embedding", k_max=8)
    r = ToolDiscoveryPipeline(cfg).filter_tools(tools, "unrelated question", session_id="s")
    assert web in r.tools
    assert "tool_19" not in [t["name"] for t in r.tools]
    assert r.original_count == 21


class _Resp:
    status_code, headers = 200, {}

    def json(self):
        return {"id": "m", "type": "message", "role": "assistant",
                "stop_reason": "end_turn", "content": [{"type": "text", "text": "ok"}]}


class _Up:
    def __init__(self):
        self.bodies = []

    async def post(self, url, headers=None, json=None):
        self.bodies.append(json)
        return _Resp()

    async def aclose(self):
        pass


def _post(tools):
    pytest.importorskip("starlette")
    from starlette.testclient import TestClient
    from paritok.proxy import server
    up = _Up()
    app = server.create_app(anthropic_base_url="http://upstream.test", http_client=up)
    with TestClient(app) as client:
        client.post("/v1/messages", json={"model": "claude-sonnet-4-5-20250929",
                                          "max_tokens": 16, "messages": TASK, "tools": tools})
        snap = client.get("/stats").json()
    return up.bodies[0], snap


def test_proxy_names_deferred_tools_and_counts_real_savings(ranker):
    big = [{**t, "description": "x " * 800} if t["name"] in ("Workflow", "ScheduleWakeup") else t
           for t in CLAUDE_TOOL_SEARCH]
    body, snap = _post(big)
    system = json.dumps(body.get("system"))
    assert "ToolSearch" in system and "Workflow" in system and "ScheduleWakeup" in system
    assert _deferred(body["tools"]) >= {"Workflow", "ScheduleWakeup"}
    # The two big deferred schemas are no longer billed → a real, positive saving.
    assert snap["tool_filter_saved"] > 1000


def test_stats_do_not_count_client_deferred_schemas():
    """A schema the CLIENT already deferred adds 0 to the tool tokens on both sides."""
    big = {"name": "mcp__big", "description": "x " * 3000, "defer_loading": True,
           "input_schema": {"type": "object", "properties": {}}}
    _, snap = _post([_tool("Read"), _tool("ToolSearch"), big])
    assert snap["tool_orig"] < 500
    assert snap["tool_filter_saved"] <= 0  # only read_original's own schema was added
