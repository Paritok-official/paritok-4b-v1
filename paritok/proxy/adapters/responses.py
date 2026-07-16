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


def extract_query(inp) -> str | None:
    """Latest user text from a Responses `input`."""
    for item in reversed(normalize_input(inp)):
        if item.get("type", "message") == "message" and item.get("role") == "user":
            content = item.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") in ("input_text", "text"):
                        return part.get("text", "")
    return None


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
