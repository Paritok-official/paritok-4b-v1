"""Small tool sets must skip discovery entirely: the virtual gateway_search_tools
schema + per-tool stubs cost more than filtering a handful of tools can save (it can
go net-negative). Below _MIN_TOOLS_FOR_DISCOVERY (5) everything passes through."""
from paritok.config import ParitokConfig, ToolDiscoveryConfig
from paritok.pipelines.tool_discovery import ToolDiscoveryPipeline, _MIN_TOOLS_FOR_DISCOVERY


def _pipeline():
    cfg = ParitokConfig()
    # relevance + top_k=1 would normally stub everything past the first tool; the floor
    # must override that for small tool sets regardless of strategy/top_k.
    cfg.tool_discovery = ToolDiscoveryConfig(strategy="relevance", top_k=1)
    return ToolDiscoveryPipeline(cfg)


def _tools(n):
    return [{"name": f"t{i}", "description": "do a thing with files", "input_schema": {}}
            for i in range(n)]


def test_below_floor_passes_through_untouched():
    p = _pipeline()
    for n in range(1, _MIN_TOOLS_FOR_DISCOVERY):  # 1..4
        r = p.filter_tools(_tools(n), "work with files")
        assert r.stubbed_tools == [], f"{n} tools should not be stubbed"
        assert r.kept_count == n


def test_at_floor_discovery_applies():
    # Exactly _MIN_TOOLS_FOR_DISCOVERY tools: discovery is allowed to filter again.
    p = _pipeline()
    r = p.filter_tools(_tools(_MIN_TOOLS_FOR_DISCOVERY), "work with files")
    assert len(r.stubbed_tools) == _MIN_TOOLS_FOR_DISCOVERY - 1  # top_k=1 kept
    assert r.kept_count == 1
