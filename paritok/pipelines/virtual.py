"""Virtual tools: injected into tool lists, intercepted by the middleware/proxy."""

from __future__ import annotations

EXPAND_CONTEXT_SCHEMA = {
    "name": "expand_context",
    "description": (
        "Retrieve the full original content for a compressed reference tag "
        "of the form [REF:id] or [REF:id src=path]. "
        "If the short summary next to the [REF:id] tag already answers what you "
        "need, do NOT call this — just use the summary. Call it only when you need "
        "the exact, full original: e.g. to read the code closely or to edit it. "
        "ALWAYS prefer this tool over re-reading the file with Read/Bash/Grep "
        "when you need the full text of content you have already seen in this "
        "conversation: it is instant, local, exact, and avoids re-running the "
        "compression model. The src=path hint tells you which file the ref "
        "corresponds to — if the user asks to view that file again, expand the "
        "ref instead of issuing a fresh Read."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "shadow_id": {
                "type": "string",
                "description": (
                    "The hex reference ID — the token immediately after 'REF:' "
                    "and before any space or ']'. Do NOT include the 'src=...' "
                    "portion. Example: for [REF:abc123 src=foo.py], pass 'abc123'."
                ),
            }
        },
        "required": ["shadow_id"],
    },
}

GATEWAY_SEARCH_TOOLS_SCHEMA = {
    "name": "gateway_search_tools",
    "description": (
        "Search for additional tools not shown in the current tool list. "
        "Use when you need a tool that is not available in your current tools."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query describing the tool you need",
            }
        },
        "required": ["query"],
    },
}

VIRTUAL_TOOL_NAMES = {"expand_context", "gateway_search_tools"}


def is_virtual_tool_call(tool_name: str) -> bool:
    return tool_name in VIRTUAL_TOOL_NAMES
