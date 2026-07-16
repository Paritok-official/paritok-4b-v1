"""SDK middleware + shared compression logic.

ParitokClient: wraps an Anthropic/OpenAI client for transparent compression.
ParitokEngine: core compression logic, used by both SDK and proxy.

Usage (SDK mode — secondary option):
    import anthropic
    import paritok

    client = paritok.ParitokClient(anthropic.Anthropic())
    response = client.messages.create(model="...", messages=[...])
    print(response._paritok_savings)

Usage (proxy mode — primary, recommended):
    paritok proxy --port 8080
    export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from paritok.config import ParitokConfig
from paritok.pipelines.compress import CompressionPipeline
from paritok.pipelines.virtual import (
    EXPAND_CONTEXT_SCHEMA,
    GATEWAY_SEARCH_TOOLS_SCHEMA,
    is_virtual_tool_call,
)
from paritok.pipelines.tool_discovery import ToolDiscoveryPipeline
from paritok.storage import MemoryShadowStorage, ShadowStorage


@dataclass
class CompressionStats:
    """Compression savings metadata."""

    original_tokens: int = 0
    compressed_tokens: int = 0
    items_compressed: int = 0
    items_skipped: int = 0
    cache_hits: int = 0
    tools_original: int = 0
    tools_kept: int = 0
    history_turns_compressed: int = 0

    @property
    def saved_tokens(self) -> int:
        return self.original_tokens - self.compressed_tokens

    @property
    def ratio(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return round(1 - self.compressed_tokens / self.original_tokens, 3)

    @property
    def tools_filtered(self) -> int:
        return self.tools_original - self.tools_kept


# ── Core Engine (shared between SDK and proxy) ──

class ParitokEngine:
    """Core compression engine. Used by both ParitokClient (SDK) and proxy server.

    Handles:
    - Tool discovery filtering
    - Content compression via Ollama
    - Virtual tool injection
    - Shadow storage for expand_context
    """

    def __init__(
        self,
        config: ParitokConfig | None = None,
        storage: ShadowStorage | None = None,
    ):
        self.config = config or ParitokConfig()
        self.storage = storage or MemoryShadowStorage()
        self.pipeline = CompressionPipeline(self.config, self.storage)
        self.discovery = ToolDiscoveryPipeline(self.config)

    def process_request(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> tuple[list[dict], list[dict] | None, CompressionStats, list[dict]]:
        """Process a request: compress context, filter tools, inject virtuals.

        Compresses both tool outputs AND old conversation history via the same
        local Ollama model, using different prompts for each.

        Args:
            messages: Conversation messages (Anthropic or OpenAI format)
            tools: Tool schemas (optional)

        Returns:
            (compressed_messages, modified_tools, stats, stubbed_tools)
            stubbed_tools: original schemas of filtered-out tools, needed for
            resolve_virtual_call. Caller must store this per-request.
        """
        stats = CompressionStats()
        query = _extract_query(messages)
        stubbed_tools: list[dict] = []

        # 1. Tool Discovery — filter tool schemas
        if tools and query and len(tools) > self.config.tool_discovery.top_k:
            result = self.discovery.filter_tools(tools, query)
            tools = result.tools
            stubbed_tools = result.stubbed_tools
            stats.tools_original = result.original_count
            stats.tools_kept = result.kept_count
        else:
            stats.tools_original = len(tools) if tools else 0
            stats.tools_kept = stats.tools_original

        # 2. Compress tool_result content blocks
        messages = _compress_messages(messages, self.pipeline, stats, query=query)

        # 3. Compress old conversation history if over threshold
        history_cfg = self.config.history
        if history_cfg.enabled:
            messages = _compress_history(
                messages, self.pipeline, stats,
                query=query,
                keep_recent_turns=history_cfg.keep_recent_turns,
                context_threshold=history_cfg.context_threshold,
                context_window=history_cfg.context_window,
            )

        # 4. Inject virtual tools
        if tools is not None:
            tools = _inject_virtual_tools(
                tools,
                has_compressed=stats.items_compressed > 0,
                has_filtered=stats.tools_kept < stats.tools_original,
            )

        return messages, tools, stats, stubbed_tools

    def resolve_virtual_call(
        self,
        tool_name: str,
        tool_input: dict,
        stubbed_tools: list[dict] | None = None,
    ) -> dict | None:
        """Resolve a virtual tool call. Returns result or None if not a virtual tool.

        Args:
            tool_name: Name of the tool called
            tool_input: Input dict for the tool
            stubbed_tools: From process_request return value. Required for
                gateway_search_tools. Pass the per-request stubbed_tools here.
        """
        if not is_virtual_tool_call(tool_name):
            return None

        if tool_name == "expand_context":
            # Be tolerant about the key ("shadow_id" per schema, but accept "id")
            # and the value ("abc123", "abc123 src=foo.py", or "[REF:abc123 ...]").
            raw = (tool_input.get("shadow_id") or tool_input.get("id") or "")
            shadow_id = raw.strip()
            if shadow_id.startswith("[REF:"):
                shadow_id = shadow_id[5:]
            shadow_id = shadow_id.split()[0].rstrip("]") if shadow_id else ""
            original = self.storage.retrieve(shadow_id)
            if original is None:
                # Non-fatal: nudge the model to keep going from the summary it
                # already holds instead of failing the turn.
                return {"content": (
                    f"[Reference '{shadow_id}' can no longer be expanded (it may have "
                    f"aged out of the shadow store). Its summary is still in your "
                    f"context above — carry on using that.]"
                )}
            return {"content": original}

        if tool_name == "gateway_search_tools":
            query = tool_input.get("query", "")
            found = self.discovery.search_filtered_tools(query, stubbed_tools or [])
            return {"tools": found}

        return None


# ── SDK Client (wraps Anthropic/OpenAI client) ──

class ParitokClient:
    """Wraps an Anthropic or OpenAI client, compressing messages transparently.

    This is the SDK mode (secondary option). For most users, proxy mode is recommended:
        paritok proxy --port 8080
        export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
    """

    def __init__(
        self,
        client,
        config: ParitokConfig | None = None,
        storage: ShadowStorage | None = None,
    ):
        self._client = client
        self._engine = ParitokEngine(config, storage)
        self._disabled = os.environ.get("PARITOK_DISABLE", "").strip() in ("1", "true", "yes")
        self.messages = _MessagesProxy(self)

    def __getattr__(self, name):
        return getattr(self._client, name)


class _MessagesProxy:
    """Intercepts messages.create() to apply compression."""

    def __init__(self, parent: ParitokClient):
        self._parent = parent

    def create(self, **kwargs):
        if self._parent._disabled:
            return self._parent._client.messages.create(**kwargs)

        messages = kwargs.get("messages", [])
        tools = kwargs.get("tools")

        # Use shared engine
        messages, tools, stats, stubbed_tools = self._parent._engine.process_request(messages, tools)
        kwargs["messages"] = messages
        if tools is not None:
            kwargs["tools"] = tools

        # Forward to underlying client
        response = self._parent._client.messages.create(**kwargs)
        response._paritok_savings = stats

        # Handle virtual tool calls in response.
        # NOTE: virtual tool calls are resolved here but NOT automatically fed back
        # into the conversation. The caller must check block._paritok_resolved and
        # construct a tool_result message for the next turn if needed.
        if hasattr(response, "content"):
            for block in response.content:
                if hasattr(block, "type") and block.type == "tool_use":
                    result = self._parent._engine.resolve_virtual_call(
                        block.name, block.input, stubbed_tools=stubbed_tools
                    )
                    if result is not None:
                        block._paritok_resolved = result

        return response

    def __getattr__(self, name):
        return getattr(self._parent._client.messages, name)


# ── Shared helpers ──

# Claude Code (and similar agents) inject <system-reminder>…</system-reminder>
# blocks into user turns — email, date, tool hints, etc. These are NOT the task
# and must not be handed to the compressor as the "intent", or it compresses
# generically (poor, high-retention) instead of toward the real task.
import re as _re  # noqa: E402 — colocated with the helper below

_SYSTEM_REMINDER = _re.compile(r"<system-reminder>.*?</system-reminder>", _re.DOTALL)


def _clean_intent(text: str | None) -> str | None:
    """Strip injected system-reminder blocks; return the remaining task text, or
    None if nothing meaningful is left."""
    if not text:
        return None
    stripped = _SYSTEM_REMINDER.sub("", text).strip()
    return stripped or None


def _extract_query(messages: list[dict]) -> str | None:
    """Extract the user's actual task/intent.

    Walks user turns newest-first and returns the first real text, ignoring
    injected <system-reminder> blocks and tool_result-only turns (which carry no
    task text). For an agent like Claude Code this skips the reminder/tool-result
    turns and lands on the original instruction (e.g. "fix the bug in ...").
    """
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            cleaned = _clean_intent(content)
            if cleaned:
                return cleaned
            continue
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    cleaned = _clean_intent(block.get("text", ""))
                    if cleaned:
                        return cleaned
    return None


def _build_tool_use_index(messages: list[dict]) -> dict[str, dict]:
    """Map tool_use_id → {name, input} by walking assistant messages.

    Lets _compress_messages associate each tool_result with the file_path
    (or similar source identifier) from its originating tool_use, so the
    pipeline can use path-keyed short-circuiting.
    """
    index: dict[str, dict] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use":
                continue
            tool_id = block.get("id")
            if not tool_id:
                continue
            index[tool_id] = {
                "name": block.get("name", ""),
                "input": block.get("input", {}) or {},
            }
    return index


def _source_from_tool_use(tool_use: dict | None) -> str | None:
    """Extract a stable source identifier from a tool_use record, if any.

    Today we only key on file_path (Read / NotebookRead / Edit etc.), which
    is the dominant case for re-read churn. Returns None when no usable
    source is available.
    """
    if not tool_use:
        return None
    inp = tool_use.get("input") or {}
    if not isinstance(inp, dict):
        return None
    path = inp.get("file_path") or inp.get("notebook_path")
    if isinstance(path, str) and path.strip():
        return path.strip()
    return None


def _compress_messages(
    messages: list[dict],
    pipeline: CompressionPipeline,
    stats: CompressionStats,
    *,
    query: str | None = None,
) -> list[dict]:
    """Walk messages and compress tool_result content blocks."""
    tool_index = _build_tool_use_index(messages)
    result = []
    for msg in messages:
        content = msg.get("content")
        if msg.get("role") == "user" and isinstance(content, list):
            new_blocks = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    source = _source_from_tool_use(
                        tool_index.get(block.get("tool_use_id", ""))
                    )
                    block = _compress_block(
                        block, pipeline, stats, query=query, source=source
                    )
                new_blocks.append(block)
            result.append({**msg, "content": new_blocks})
        else:
            result.append(msg)
    return result


def _compress_block(
    block: dict,
    pipeline: CompressionPipeline,
    stats: CompressionStats,
    *,
    query: str | None = None,
    source: str | None = None,
) -> dict:
    """Compress the content inside a tool_result block."""
    content = block.get("content", "")

    if isinstance(content, str) and content.strip():
        cr = pipeline.compress(content, query=query, source=source)
        stats.original_tokens += cr.original_tokens
        stats.compressed_tokens += cr.compressed_tokens
        if cr.metadata.get("skipped"):
            stats.items_skipped += 1
        else:
            stats.items_compressed += 1
        if cr.metadata.get("cache_hit") or cr.metadata.get("path_shortcircuit"):
            stats.cache_hits += 1
        return {**block, "content": cr.compressed}

    if isinstance(content, list):
        new_content = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                if text.strip():
                    cr = pipeline.compress(text, query=query, source=source)
                    stats.original_tokens += cr.original_tokens
                    stats.compressed_tokens += cr.compressed_tokens
                    if cr.metadata.get("skipped"):
                        stats.items_skipped += 1
                    else:
                        stats.items_compressed += 1
                    if cr.metadata.get("cache_hit") or cr.metadata.get("path_shortcircuit"):
                        stats.cache_hits += 1
                    new_content.append({**item, "text": cr.compressed})
                else:
                    new_content.append(item)
            else:
                new_content.append(item)
        return {**block, "content": new_content}

    return block


def _compress_history(
    messages: list[dict],
    pipeline: CompressionPipeline,
    stats: CompressionStats,
    *,
    query: str | None = None,
    keep_recent_turns: int = 4,
    context_threshold: float = 0.8,
    context_window: int = 200_000,
) -> list[dict]:
    """Compress old conversation turns if total context exceeds threshold.

    Uses the same Ollama model with HISTORY_SUMMARY_PROMPT to summarize
    old user/assistant messages. Question-aware: preserves info relevant to query.
    Recent turns are kept intact.
    """
    from paritok.token_counter import count_tokens

    # Count total tokens
    total_tokens = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_tokens += count_tokens(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text", "") or block.get("content", "")
                    if isinstance(text, str):
                        total_tokens += count_tokens(text)

    # Not over threshold — no compression needed
    threshold = int(context_window * context_threshold)
    if total_tokens <= threshold:
        return messages

    # Identify turn boundaries (user message starts a new turn)
    turns: list[list[int]] = []
    current_turn: list[int] = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "user" and current_turn:
            turns.append(current_turn)
            current_turn = [i]
        else:
            current_turn.append(i)
    if current_turn:
        turns.append(current_turn)

    # Not enough turns to compress
    if len(turns) <= keep_recent_turns:
        return messages

    # Split: old turns (to compress) vs recent turns (to keep)
    old_turn_groups = turns[:-keep_recent_turns]
    recent_turn_groups = turns[-keep_recent_turns:]

    old_indices = {i for group in old_turn_groups for i in group}
    old_messages = [messages[i] for i in sorted(old_indices)]
    recent_messages = [messages[i] for i in sorted(
        i for group in recent_turn_groups for i in group
    )]

    # Build text from old messages for summarization
    old_text_parts = []
    for msg in old_messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, str):
            old_text_parts.append(f"[{role}]: {content}")
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    old_text_parts.append(f"[{role}]: {block.get('text', '')}")

    old_text = "\n".join(old_text_parts)
    if not old_text.strip():
        return messages

    # Compress old conversation turns aggressively (level L3) with the "other"
    # (non-file_read) system prompt, which covers assistant_thinking/tool_result.
    from paritok.strategies.prompts import HISTORY_SUMMARY_PROMPT

    try:
        summary = pipeline._model.compress(
            old_text,
            query=query,
            level="L3",
            kind="assistant_thinking",
            system_prompt=HISTORY_SUMMARY_PROMPT,
        )
    except (ConnectionError, TimeoutError, ValueError) as e:
        import logging
        logging.getLogger("paritok").warning("History compression failed: %s", e)
        return messages

    stats.history_turns_compressed = len(old_messages)

    # Replace old turns with summary + keep recent
    compressed_messages = [
        {"role": "user", "content": f"[Conversation Summary]\n{summary}"},
        {"role": "assistant", "content": "Understood, I have the conversation context."},
    ] + recent_messages

    return compressed_messages


def _inject_virtual_tools(
    tools: list[dict],
    *,
    has_compressed: bool = False,
    has_filtered: bool = False,
) -> list[dict]:
    names = {t.get("name") for t in tools}
    result = list(tools)
    if has_compressed and "expand_context" not in names:
        result.append(EXPAND_CONTEXT_SCHEMA)
    if has_filtered and "gateway_search_tools" not in names:
        result.append(GATEWAY_SEARCH_TOOLS_SCHEMA)
    return result
