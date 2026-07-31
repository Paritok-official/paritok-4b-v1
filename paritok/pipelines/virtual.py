"""Virtual tools: injected into tool lists, intercepted by the middleware/proxy."""

from __future__ import annotations

# The tool the model calls to get back the exact original of content it saw
# shortened to a [REF:id] stub. Named `read_original` (not the abstract
# "expand_context") because the model reaches for it exactly when it thinks "I need
# to read the real file" — the old name didn't map to that intent, so the model
# dumped the file via Bash cat/sed instead. The old name stays accepted below.
EXPAND_TOOL_NAME = "read_original"

EXPAND_CONTEXT_SCHEMA = {
    "name": EXPAND_TOOL_NAME,
    "description": (
        "Return the EXACT, complete, verbatim original of a file or output that was "
        "shortened to a `[REF:id src=path]` stub. The text next to a [REF:...] is a "
        "LOSSY summary — reformatted, with blank lines / docstrings / some lines "
        "dropped — so it does NOT match the real file byte-for-byte. "
        "Whenever you need the precise contents of a [REF:...] file — especially to "
        "EDIT it — call this with the id. "
        "Do NOT try to recover the exact file by dumping it with Bash (cat / sed / "
        "head / tail) or a fresh Read: those outputs get shortened again, so you get "
        "the same lossy view. `read_original` is the ONLY way to get the true bytes — "
        "it is instant, local, and exact."
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

# Both the new and legacy names resolve to the same expand behavior, so a client
# (or a mid-session rename) never breaks.
EXPAND_TOOL_ALIASES = {EXPAND_TOOL_NAME, "expand_context"}
VIRTUAL_TOOL_NAMES = EXPAND_TOOL_ALIASES | {"gateway_search_tools"}


def is_virtual_tool_call(tool_name: str) -> bool:
    return tool_name in VIRTUAL_TOOL_NAMES


def is_expand_call(tool_name: str) -> bool:
    """True for the expand tool under either its current or legacy name."""
    return tool_name in EXPAND_TOOL_ALIASES
