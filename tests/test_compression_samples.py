"""Recent original→compressed pairs for the /stats dashboard (#42).

CompressionStats collects a few real before/after pairs per request; ProxyStats
keeps the biggest-saving one from each request in a small, truncated, newest-first
rolling window. The local dashboard shows one pair per page, fetched one at a time
via GET /stats?sample=N (N=0 = newest); the default /stats snapshot carries only
the count. The pairs are local-only by design — they hold real file content / tool
schemas, so they never leave the box (the hosted account dashboard sees only totals).
"""
from types import SimpleNamespace

from starlette.testclient import TestClient

from paritok.config import ParitokConfig, ToolDiscoveryConfig
from paritok.middleware.wrapper import CompressionStats, ParitokEngine
from paritok.proxy.server import (
    ProxyStats, create_app, _clip, _MAX_STATS_SAMPLES, _STATS_SAMPLE_CHARS,
)


def _cr(original_tokens, compressed_tokens, compressed="[REF:1] summary", skipped=False):
    """A stand-in CompressionResult with just the fields add_sample reads."""
    return SimpleNamespace(
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        compressed=compressed,
        saved_tokens=original_tokens - compressed_tokens,
        metadata={"skipped": skipped},
    )


def _record(ps, orig_tok, comp_tok, compressed="[REF] c", source="content", model="m"):
    s = CompressionStats(original_tokens=orig_tok, compressed_tokens=comp_tok)
    s.add_sample("x" * 50, _cr(orig_tok, comp_tok, compressed=compressed), source=source)
    ps.record(s, model=model)


# ── capture (CompressionStats.add_sample) ──

def test_add_sample_only_keeps_real_savings():
    s = CompressionStats()
    s.add_sample("x" * 100, _cr(500, 80), source="file_read")   # real saving → kept
    s.add_sample("y", _cr(500, 500), source="content")           # zero saving → dropped
    s.add_sample("z", _cr(500, 80, skipped=True), source="tool") # a skip → dropped
    assert len(s.samples) == 1
    assert s.samples[0]["source"] == "file_read"


def test_add_sample_is_capped_per_request():
    s = CompressionStats()
    for _ in range(10):
        s.add_sample("orig", _cr(500, 80), source="content")
    assert len(s.samples) == 4  # never holds a whole request's worth of text


# ── fold + paging (ProxyStats) ──

def test_record_folds_biggest_saving_and_truncates():
    ps = ProxyStats()
    s = CompressionStats(original_tokens=900, compressed_tokens=200, items_compressed=2)
    big_original = "L" * (_STATS_SAMPLE_CHARS + 500)
    s.add_sample("small original", _cr(300, 250, compressed="kept-a"), source="content")
    s.add_sample(big_original, _cr(600, 40, compressed="[REF:9]"), source="file_read")

    ps.record(s, model="claude-sonnet")
    page = ps.samples_page(0)

    assert page["samples_total"] == 1
    sample = page["sample"]
    # the 560-token saving wins over the 50-token one
    assert sample["source"] == "file_read"
    assert sample["original_tokens"] == 600
    assert sample["compressed_tokens"] == 40
    assert sample["tokens_saved"] == 560
    assert sample["kept_ratio"] == round(40 / 600, 3)
    assert sample["model"] == "claude-sonnet"
    # the long original is truncated; exact char count is still reported
    assert "more chars" in sample["original"]
    assert len(sample["original"]) < len(big_original)
    assert sample["original_chars"] == len(big_original)


def test_samples_page_is_newest_first_and_clamps():
    ps = ProxyStats()
    for i in range(4):
        _record(ps, 100, 10, compressed=f"comp-{i}")   # comp-0 oldest … comp-3 newest
    assert ps.samples_page(0)["sample"]["compressed"] == "comp-3"   # 0 = newest
    assert ps.samples_page(1)["sample"]["compressed"] == "comp-2"
    assert ps.samples_page(3)["sample"]["compressed"] == "comp-0"   # oldest
    # out of range clamps to the nearest valid page (never raises)
    hi = ps.samples_page(99)
    assert hi["sample_index"] == 3 and hi["sample"]["compressed"] == "comp-0"
    lo = ps.samples_page(-5)
    assert lo["sample_index"] == 0 and lo["sample"]["compressed"] == "comp-3"


def test_recent_samples_are_bounded():
    ps = ProxyStats()
    for i in range(_MAX_STATS_SAMPLES + 4):
        _record(ps, 100, 10, compressed=f"comp-{i}")
    assert ps.samples_page(0)["samples_total"] == _MAX_STATS_SAMPLES   # window bounded
    # newest is the last recorded; oldest kept is index MAX-1
    assert ps.samples_page(0)["sample"]["compressed"].endswith(str(_MAX_STATS_SAMPLES + 3))


def test_empty_window():
    ps = ProxyStats()
    assert ps.snapshot()["compression_samples_count"] == 0
    page = ps.samples_page(0)
    assert page == {"sample_index": 0, "samples_total": 0, "sample": None}


# ── HTTP route (GET /stats?sample=N) ──

def test_stats_route_serves_one_sample_at_a_time():
    app = create_app()
    _record(app.state.proxy_stats, 800, 120, compressed="[REF] newest", source="app/x.py")
    _record(app.state.proxy_stats, 800, 120, compressed="[REF] newer", source="app/y.py")
    client = TestClient(app)

    # default snapshot carries only the count, not the array
    snap = client.get("/stats", headers={"accept": "application/json"}).json()
    assert snap["compression_samples_count"] == 2
    assert "compression_samples" not in snap

    # one pair per request, newest first
    r0 = client.get("/stats?sample=0").json()
    assert r0["samples_total"] == 2
    assert r0["sample"]["source"] == "app/y.py"
    r1 = client.get("/stats?sample=1").json()
    assert r1["sample"]["source"] == "app/x.py"

    # a browser Accept header does not turn a ?sample fetch into HTML
    html_hdr = client.get("/stats?sample=0", headers={"accept": "text/html"})
    assert "application/json" in html_hdr.headers["content-type"]

    # out-of-range and garbage indexes are tolerated
    assert client.get("/stats?sample=99").json()["sample_index"] == 1
    assert client.get("/stats?sample=abc").json()["sample"]["source"] == "app/y.py"


def test_stats_route_empty_sample():
    client = TestClient(create_app())
    r = client.get("/stats?sample=0").json()
    assert r == {"sample_index": 0, "samples_total": 0, "sample": None}


# ── tool-schema filter samples (its own before/after row) ──

def test_tool_filter_sample_folds_as_its_own_row():
    ps = ProxyStats()
    s = CompressionStats(original_tokens=500, compressed_tokens=100)
    s.add_sample("file original", _cr(500, 100, compressed="[REF] file"), source="app/x.py")
    s.tool_filter_sample = {
        "original_names": ["Read", "Write", "Edit", "Bash", "Grep", "mcp__a__x", "mcp__b__y"],
        "forwarded_names": ["Read", "Edit", "expand_context"],
    }
    ps.record(s, model="claude", tools_original_tokens=900, tools_compressed_tokens=300)

    total = ps.samples_page(0)["samples_total"]
    assert total == 2  # one content pair + one tool-schema pair
    by_src = {ps.samples_page(i)["sample"]["source"]: ps.samples_page(i)["sample"]
              for i in range(total)}
    assert set(by_src) == {"app/x.py", "tool schemas"}
    # content is folded last, so it reads as the newest of the pair
    assert ps.samples_page(0)["sample"]["source"] == "app/x.py"

    tool = by_src["tool schemas"]
    assert tool["original_tokens"] == 900 and tool["compressed_tokens"] == 300
    assert tool["tokens_saved"] == 600
    assert "Write" in tool["original"] and "expand_context" in tool["compressed"]
    assert "stubbed" in tool["compressed"]  # the annotation line


def test_tool_filter_sample_skipped_when_no_net_saving():
    ps = ProxyStats()
    s = CompressionStats(original_tokens=500, compressed_tokens=100)
    s.add_sample("file", _cr(500, 100), source="content")
    # virtual tools made the tool block net-larger → nothing to show for tools
    s.tool_filter_sample = {"original_names": ["A", "B"],
                            "forwarded_names": ["A", "B", "expand_context"]}
    ps.record(s, model="m", tools_original_tokens=100, tools_compressed_tokens=140)
    assert ps.samples_page(0)["samples_total"] == 1
    assert ps.samples_page(0)["sample"]["source"] == "content"


def test_process_request_captures_tool_filter_before_after():
    """process_request records the offered→forwarded tool names when discovery
    actually drops tools (relevance strategy keeps it offline/deterministic)."""
    cfg = ParitokConfig()
    cfg.tool_discovery = ToolDiscoveryConfig(strategy="relevance", top_k=1)
    engine = ParitokEngine(cfg)
    tools = [{"name": f"tool_{i}", "description": "work with files and data",
              "input_schema": {}} for i in range(6)]
    msgs = [{"role": "user", "content": "please work with files"}]

    _m, out_tools, stats, _stub = engine.process_request(msgs, tools, query="work with files")

    assert stats.tools_kept < stats.tools_original          # discovery dropped some
    tfs = stats.tool_filter_sample
    assert tfs is not None
    assert set(tfs["original_names"]) == {f"tool_{i}" for i in range(6)}
    assert len(tfs["forwarded_names"]) == len(out_tools)
    assert any(n.startswith("tool_") for n in tfs["forwarded_names"])       # a kept real tool
    offered = set(tfs["original_names"])
    assert any(n not in offered for n in tfs["forwarded_names"])            # an injected virtual


# ── _clip truncation marker (Windows/GBK ellipsis is easy to corrupt) ──

def test_clip_leaves_short_text_and_marks_truncation_exactly():
    assert _clip("abc", 10) == "abc"                 # under the cap, unchanged
    out = _clip("L" * 2500, 1000)
    assert out.startswith("L" * 1000)
    assert out.endswith("… (+1,500 more chars)")     # exact U+2026 marker + thousands sep
