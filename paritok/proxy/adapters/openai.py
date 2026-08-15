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


def _message_text(content) -> str | None:
    """Text of an OpenAI message `content` (a plain string or a list of
    {type:'text', text} parts). Returns None when there is no non-blank text."""
    if isinstance(content, str):
        return content.strip() or None
    if isinstance(content, list):
        texts = [p.get("text", "") for p in content
                 if isinstance(p, dict) and p.get("type") == "text" and p.get("text", "").strip()]
        joined = " ".join(texts).strip()
        return joined or None
    return None


def _latest_assistant_reasoning(messages: list[dict]) -> str | None:
    """The agent's most recent narration/reasoning (the text it emitted alongside its
    latest tool call). This is the CURRENT intent — what the agent is doing right now —
    which motivated the newest tool result we're about to compress. Assistant messages
    that are pure tool calls carry no text and are skipped."""
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        text = _message_text(msg.get("content"))
        if text:
            return text
    return None


def _extract_task(messages: list[dict]) -> str | None:
    """The user's task instruction (latest real user text). Tool outputs are separate
    role='tool' messages, so this skips them and lands on the actual instruction."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        text = _message_text(msg.get("content"))
        if text:
            return text
    return None


def extract_query(messages: list[dict]) -> str | None:
    """Compression intent (#4). The 4B keeps what the intent NAMES and drops the rest, so
    a static task prompt ("fix the bug in answer_question") drops the exact code the agent
    is digging into right now — forcing a re-search. Prefer the agent's LATEST reasoning
    (current intent) so the compressor keeps the code in play, carrying the overall task as
    trailing context. Falls back to the task alone on turn 0 (no assistant text yet).
    Mirrors the Anthropic (Claude Code) and Responses (Codex) adapters."""
    reasoning = _latest_assistant_reasoning(messages)
    task = _extract_task(messages)
    if reasoning:
        return f"{reasoning[:900]}\n(overall task: {task[:200]})" if task else reasoning[:900]
    return task


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
