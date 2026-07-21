"""OpenAI API format adapter.

Parses and reconstructs OpenAI Chat Completions API requests/responses.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ParsedRequest:
    """Parsed OpenAI Chat Completions API request."""
    model: str = ""
    messages: list[dict] = field(default_factory=list)
    tools: list[dict] | None = None
    stream: bool = False
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        body = {
            "model": self.model,
            "messages": self.messages,
        }
        if self.tools is not None:
            body["tools"] = self.tools
        if self.stream:
            body["stream"] = True
        body.update(self.extra)
        return body


def parse_request(body: dict) -> ParsedRequest:
    known_keys = {"model", "messages", "tools", "stream"}
    extra = {k: v for k, v in body.items() if k not in known_keys}

    return ParsedRequest(
        model=body.get("model", ""),
        messages=body.get("messages", []),
        tools=body.get("tools"),
        stream=body.get("stream", False),
        extra=extra,
    )


def extract_query(messages: list[dict]) -> str | None:
    """Extract user's latest query from OpenAI messages."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    return part.get("text", "")
    return None


def extract_tool_results(messages: list[dict]) -> list[tuple[int, dict]]:
    """Find all tool-role messages (OpenAI's tool result format).

    In OpenAI format, tool results are separate messages with role="tool",
    unlike Anthropic where they are blocks inside user messages.

    Returns list of (message_index, message) tuples.
    Note: Anthropic adapter returns (msg_idx, block_idx, block) — different shape.
    Server.py handles each adapter's format separately.
    """
    results = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "tool":
            results.append((i, msg))
    return results


def replace_tool_message_content(
    messages: list[dict], msg_idx: int, new_content: str
) -> list[dict]:
    """Replace content of a tool message."""
    messages = [dict(m) for m in messages]
    msg = dict(messages[msg_idx])
    msg["content"] = new_content
    messages[msg_idx] = msg
    return messages


def find_virtual_tool_calls(response_body: dict) -> list[dict]:
    """Find virtual tool calls in an OpenAI response."""
    # Local import to avoid circular dependency with pipelines.virtual
    from paritok.pipelines.virtual import is_virtual_tool_call

    # NOTE: tc["function"]["arguments"] is a JSON string, not a dict.
    # Caller must json.loads() it before passing to resolve_virtual_call().
    results = []
    if not isinstance(response_body, dict):  # some providers return non-dict error bodies
        return results
    for choice in response_body.get("choices", []):
        message = choice.get("message", {})
        for tc in message.get("tool_calls", []):
            fn = tc.get("function", {})
            if is_virtual_tool_call(fn.get("name", "")):
                results.append(tc)
    return results
