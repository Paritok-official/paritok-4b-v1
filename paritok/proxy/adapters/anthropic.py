"""Anthropic API format adapter.

Parses and reconstructs Anthropic Messages API requests/responses.
Handles both /v1/messages and streaming /v1/messages with stream=true.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ParsedRequest:
    """Parsed Anthropic Messages API request."""
    model: str = ""
    messages: list[dict] = field(default_factory=list)
    tools: list[dict] | None = None
    system: str | list | None = None
    max_tokens: int = 4096
    stream: bool = False
    # Pass through everything else unchanged
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Reconstruct the API request body."""
        body = {
            "model": self.model,
            "messages": self.messages,
            "max_tokens": self.max_tokens,
        }
        if self.tools is not None:
            body["tools"] = self.tools
        if self.system is not None:
            body["system"] = self.system
        if self.stream:
            body["stream"] = True
        body.update(self.extra)
        return body


def parse_request(body: dict) -> ParsedRequest:
    """Parse an Anthropic Messages API request body."""
    known_keys = {"model", "messages", "tools", "system", "max_tokens", "stream"}
    extra = {k: v for k, v in body.items() if k not in known_keys}

    return ParsedRequest(
        model=body.get("model", ""),
        messages=body.get("messages", []),
        tools=body.get("tools"),
        system=body.get("system"),
        max_tokens=body.get("max_tokens", 4096),
        stream=body.get("stream", False),
        extra=extra,
    )


import re as _re

_SYSTEM_REMINDER = _re.compile(r"<system-reminder>.*?</system-reminder>", _re.DOTALL)


def _clean_intent(text: str | None) -> str | None:
    """Drop injected <system-reminder> blocks; return remaining text or None."""
    if not text:
        return None
    stripped = _SYSTEM_REMINDER.sub("", text).strip()
    return stripped or None


def extract_query(messages: list[dict]) -> str | None:
    """Extract the user's real task text from Anthropic messages, ignoring
    injected <system-reminder> blocks and tool_result-only turns."""
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


def extract_tool_results(messages: list[dict]) -> list[tuple[int, int, dict]]:
    """Find all tool_result blocks in messages.

    Returns list of (message_index, block_index, block) tuples.
    """
    results = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for j, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                results.append((i, j, block))
    return results


def replace_tool_result_content(
    messages: list[dict], msg_idx: int, block_idx: int, new_content: str
) -> list[dict]:
    """Replace the content of a tool_result block (immutable — returns new list)."""
    messages = [dict(m) for m in messages]  # shallow copy
    msg = dict(messages[msg_idx])
    content = list(msg["content"])
    block = dict(content[block_idx])

    old_content = block.get("content", "")
    if isinstance(old_content, str):
        block["content"] = new_content
    elif isinstance(old_content, list):
        # Only replace the first text block — subsequent text blocks are left unchanged.
        # This matches the expectation that tool_result has one primary text output.
        new_items = []
        replaced = False
        for item in old_content:
            if isinstance(item, dict) and item.get("type") == "text" and not replaced:
                new_items.append({**item, "text": new_content})
                replaced = True
            else:
                new_items.append(item)
        block["content"] = new_items

    content[block_idx] = block
    msg["content"] = content
    messages[msg_idx] = msg
    return messages


def find_virtual_tool_uses(response_body: dict) -> list[dict]:
    """Find virtual tool_use blocks in an Anthropic response."""
    # Local import to avoid circular dependency with pipelines.virtual
    from paritok.pipelines.virtual import is_virtual_tool_call

    results = []
    for block in response_body.get("content", []):
        if block.get("type") == "tool_use" and is_virtual_tool_call(block.get("name", "")):
            results.append(block)
    return results
