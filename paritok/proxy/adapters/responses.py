"""OpenAI Responses API (`/v1/responses`) format adapter.

Codex CLI speaks the Responses API by default (not Chat Completions). It differs
in shape from Chat Completions:
  - request: top-level `input` (a string or a list of items), `instructions`
    (system), and `tools` are flat `{"type":"function","name",...,"parameters"}`.
  - response: `output` is a list of items; assistant text is
    `{"type":"message","content":[{"type":"output_text","text":...}]}` and a tool
    call is `{"type":"function_call","call_id":...,"name":...,"arguments":"<json>"}`.
  - a tool result is fed back as `{"type":"function_call_output","call_id":...,
    "output":"..."}` appended to `input`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ParsedRequest:
    """Parsed OpenAI Responses API request."""
    model: str = ""
    input: list | str = field(default_factory=list)
    tools: list | None = None
    instructions: str | None = None
    stream: bool = False
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        body: dict = {"model": self.model, "input": self.input}
        if self.tools is not None:
            body["tools"] = self.tools
        if self.instructions is not None:
            body["instructions"] = self.instructions
        if self.stream:
            body["stream"] = True
        body.update(self.extra)
        return body


def parse_request(body: dict) -> ParsedRequest:
    known = {"model", "input", "tools", "instructions", "stream"}
    extra = {k: v for k, v in body.items() if k not in known}
    return ParsedRequest(
        model=body.get("model", ""),
        input=body.get("input", []),
        tools=body.get("tools"),
        instructions=body.get("instructions"),
        stream=body.get("stream", False),
        extra=extra,
    )


def normalize_input(inp) -> list:
    """`input` may be a bare string or a list of items. Always return a list."""
    if isinstance(inp, str):
        return [{"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": inp}]}]
    return list(inp or [])


def _message_text(content, part_type: str) -> str | None:
    """Join the `part_type` text parts of a message `content` (a bare string or a list of
    parts). Returns the joined text or None if there is none."""
    if isinstance(content, str):
        return content.strip() or None
    if isinstance(content, list):
        texts = [p.get("text", "") for p in content
                 if isinstance(p, dict) and p.get("type") in (part_type, "text")
                 and p.get("text", "").strip()]
        if texts:
            return " ".join(texts).strip() or None
    return None


def _extract_task(inp) -> str | None:
    """The user's task text — the latest user message (`input_text`)."""
    for item in reversed(normalize_input(inp)):
        if item.get("type", "message") == "message" and item.get("role") == "user":
            t = _message_text(item.get("content"), "input_text")
            if t:
                return t
    return None


def _latest_assistant_intent(inp) -> str | None:
    """The agent's most recent narration/reasoning — its CURRENT intent, what motivated the
    newest tool output we're about to compress. Prefer the assistant's visible message text
    (`output_text`); fall back to a `reasoning` item's summary. Distinguishes assistant from
    user by content-part type (`output_text` vs `input_text`), so a missing `role` is fine."""
    for item in reversed(normalize_input(inp)):
        itype = item.get("type", "message")
        if itype == "message" and item.get("role") != "user":
            t = _message_text(item.get("content"), "output_text")
            if t:
                return t
        elif itype == "reasoning":
            summary = item.get("summary")
            if isinstance(summary, list):
                texts = [s.get("text", "") for s in summary
                         if isinstance(s, dict) and s.get("type") == "summary_text"
                         and s.get("text", "").strip()]
                if texts:
                    return " ".join(texts).strip()
    return None


def extract_query(inp) -> str | None:
    """Compression intent, mirroring the Anthropic (Claude Code) adapter. The 4B keeps what
    the intent NAMES and drops the rest, so a static task prompt ("fix the bug in
    answer_question") drops the exact code the agent is digging into right now (`quick_eval`)
    -- forcing it to re-search. Prefer the agent's LATEST narration/reasoning (current
    intent) so the compressor keeps the code in play, carrying the overall task as trailing
    context. Falls back to the task alone on turn 0 (no assistant text yet)."""
    reasoning = _latest_assistant_intent(inp)
    task = _extract_task(inp)
    if reasoning:
        return f"{reasoning[:900]}\n(overall task: {task[:200]})" if task else reasoning[:900]
    return task


def find_virtual_function_calls(response_body: dict) -> list[dict]:
    """Find `function_call` output items whose name is a virtual tool."""
    from paritok.pipelines.virtual import is_virtual_tool_call
    out = []
    for item in response_body.get("output", []) or []:
        if item.get("type") == "function_call" and is_virtual_tool_call(item.get("name", "")):
            out.append(item)
    return out


def has_real_function_call(response_body: dict) -> bool:
    """True if the output contains a non-virtual function_call (client must run it)."""
    from paritok.pipelines.virtual import is_virtual_tool_call
    for item in response_body.get("output", []) or []:
        if item.get("type") == "function_call" and not is_virtual_tool_call(item.get("name", "")):
            return True
    return False


def conceal_virtual_calls(response_body: dict) -> dict:
    """Drop virtual function_call items the client cannot run."""
    from paritok.pipelines.virtual import is_virtual_tool_call
    output = response_body.get("output")
    if isinstance(output, list):
        response_body["output"] = [
            it for it in output
            if not (it.get("type") == "function_call"
                    and is_virtual_tool_call(it.get("name", "")))
        ]
    return response_body
