"""Tool Discovery Pipeline: filter tool schemas to reduce token usage.

AI agents expose 70+ tools to the LLM on every turn. Each tool's JSON schema
costs tokens. This pipeline keeps only the top-K most relevant tools (full schema)
and stubs the rest with minimal placeholders.

Stubbed tools can be recovered via the gateway_search_tools virtual tool.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from paritok.config import ParitokConfig

# Common English stopwords to ignore during keyword matching
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "about", "between",
    "through", "during", "before", "after", "and", "or", "but", "not",
    "no", "if", "then", "else", "when", "where", "how", "what", "which",
    "who", "that", "this", "it", "i", "me", "my", "we", "our", "you",
    "your", "he", "she", "they", "them", "its", "all", "each", "any",
    "some", "just", "also", "very", "so", "too", "up", "out", "get",
})


@dataclass
class DiscoveryResult:
    """Result of tool discovery filtering.

    tools: Full list to send to LLM (top-K full schemas + stubs for the rest)
    full_tools: The top-K tools with complete schemas
    stubbed_tools: Original schemas of tools that were replaced with stubs
    """
    tools: list[dict]
    full_tools: list[dict]
    stubbed_tools: list[dict]
    original_count: int
    kept_count: int

    @property
    def filtered_count(self) -> int:
        return self.original_count - self.kept_count


def _extract_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from text, lowercased, stopwords removed."""
    # Extract words before lowering (to preserve camelCase boundaries)
    raw_words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text)
    expanded = set()
    for w in raw_words:
        # snake_case split
        parts = w.split("_")
        # camelCase split (on original casing)
        for part in parts:
            sub = re.findall(r"[a-z]+|[A-Z][a-z]*", part)
            expanded.update(s.lower() for s in sub if len(s) > 1)
        expanded.add(w.lower())
    return expanded - _STOPWORDS


def _tool_text(tool: dict) -> str:
    """Extract searchable text from a tool schema."""
    parts = [tool.get("name", "")]
    desc = tool.get("description", "")
    if desc:
        parts.append(desc)
    # Include parameter names from input_schema
    schema = tool.get("input_schema", {})
    props = schema.get("properties", {})
    for prop_name, prop_def in props.items():
        parts.append(prop_name)
        if isinstance(prop_def, dict) and "description" in prop_def:
            parts.append(prop_def["description"])
    return " ".join(parts)


def _score_tool(tool: dict, query_keywords: set[str]) -> float:
    """Score a tool's relevance to the query. Higher = more relevant."""
    tool_keywords = _extract_keywords(_tool_text(tool))
    if not query_keywords or not tool_keywords:
        return 0.0

    # Exact matches
    exact = query_keywords & tool_keywords
    score = len(exact) * 2.0

    # Partial matches (query keyword is substring of tool keyword or vice versa)
    for qk in query_keywords:
        for tk in tool_keywords:
            if qk != tk and (qk in tk or tk in qk):
                score += 0.5

    # Bonus for tool name match (strongest signal)
    name_keywords = _extract_keywords(tool.get("name", ""))
    name_matches = query_keywords & name_keywords
    score += len(name_matches) * 3.0

    return score


def _make_stub(tool: dict) -> dict:
    """Create a minimal stub for a filtered-out tool."""
    return {
        "name": tool.get("name", "unknown"),
        "description": f"[Filtered] {tool.get('description', '')[:80]}. Use gateway_search_tools to access.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    }


def _search_tools(tools: list[dict], query: str) -> list[dict]:
    """Search filtered-out tools by query. Used by gateway_search_tools virtual tool."""
    query_keywords = _extract_keywords(query)
    scored = [(tool, _score_tool(tool, query_keywords)) for tool in tools]
    scored.sort(key=lambda x: x[1], reverse=True)
    # Return top 5 matches with score > 0
    return [tool for tool, score in scored[:5] if score > 0]


class ToolDiscoveryPipeline:
    """Filter tool schemas to keep only the most relevant ones."""

    def __init__(self, config: ParitokConfig | None = None):
        self.config = config or ParitokConfig()

    def filter_tools(self, tools: list[dict], query: str,
                     session_id: str | None = None) -> DiscoveryResult:
        """Filter tool schemas to keep only the most relevant ones.

        Args:
            tools: Full list of tool schemas
            query: The user's current query
            session_id: Stable per-conversation id (used by the "embedding" strategy to
                freeze its selection across turns; ignored by relevance/passthrough)

        Returns:
            DiscoveryResult with tools (full + stubs), full_tools, and stubbed_tools
        """
        cfg = self.config.tool_discovery
        limit = cfg.k_max if cfg.strategy == "embedding" else cfg.top_k

        if cfg.strategy == "passthrough" or len(tools) <= limit:
            return DiscoveryResult(
                tools=list(tools),
                full_tools=list(tools),
                stubbed_tools=[],
                original_count=len(tools),
                kept_count=len(tools),
            )

        if cfg.strategy == "relevance":
            return self._relevance_filter(tools, query, cfg.top_k)
        elif cfg.strategy == "embedding":
            return self._embedding_filter(tools, query, session_id, cfg)
        else:
            raise ValueError(f"Unknown tool discovery strategy: {cfg.strategy}")

    def _embedding_filter(self, tools: list[dict], query: str,
                          session_id: str | None, cfg) -> DiscoveryResult:
        """Semantic top-k selection with session freeze + rank-weighted adaptive apply.
        Requires the optional dependency: pip install "paritok[toolselect]"."""
        try:
            from paritok.tool_topk import (
                predict_topk_frozen, apply_selection_adaptive, apply_selection, _tool_name)
        except ImportError as e:  # sentence-transformers not installed
            raise RuntimeError(
                "tool_discovery.strategy='embedding' needs the optional dependency.\n"
                '  Install it with:  pip install "paritok[toolselect]"'
            ) from e

        sid = session_id or query or "default"
        keep_ordered = predict_topk_frozen(sid, query, tools)
        keep_names = set(keep_ordered)
        if getattr(cfg, "adaptive", True):
            new_tools = apply_selection_adaptive(
                tools, keep_ordered, wire="anthropic",
                mcp_signal_threshold=getattr(cfg, "mcp_signal_threshold", 1.0))
        else:
            new_tools = apply_selection(tools, keep_names, wire="anthropic")

        full = [t for t in tools if _tool_name(t) in keep_names]
        stubbed = [t for t in tools if _tool_name(t) not in keep_names]
        return DiscoveryResult(
            tools=new_tools, full_tools=full, stubbed_tools=stubbed,
            original_count=len(tools), kept_count=len(keep_names),
        )

    def search_filtered_tools(self, query: str, filtered_tools: list[dict]) -> list[dict]:
        """Search among filtered-out tools. Called by gateway_search_tools.

        Args:
            query: Search query
            filtered_tools: The stubbed_tools from DiscoveryResult
        """
        return _search_tools(filtered_tools, query)

    def _relevance_filter(
        self, tools: list[dict], query: str, top_k: int
    ) -> DiscoveryResult:
        query_keywords = _extract_keywords(query)

        # Score all tools
        scored = [(tool, _score_tool(tool, query_keywords)) for tool in tools]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Keep top_k with full schemas
        kept = [tool for tool, _ in scored[:top_k]]
        filtered = [tool for tool, _ in scored[top_k:]]

        # Create stubs for filtered tools
        stubs = [_make_stub(tool) for tool in filtered]

        return DiscoveryResult(
            tools=kept + stubs,
            full_tools=kept,
            stubbed_tools=filtered,
            original_count=len(tools),
            kept_count=top_k,
        )
