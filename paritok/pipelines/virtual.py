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
    # Neutral wording: calling the views "LOSSY" and a fresh Read "the same lossy view"
    # made agents stop trusting Read altogether (see _REF_GUIDANCE in proxy/server.py),
    # and "especially to EDIT it" contradicted the guidance that edits need no original.
    "description": (
        "Return the complete original of a tool output shown condensed (it opens with "
        "`[REF:id src=path]`; the condensed view keeps the parts that matter for the "
        "current task). Use it when you need the whole original at once — for a few "
        "lines, a Read of that range is enough. Edits don't need it: they are matched "
        "to the file automatically. By default the original is shown for this step "
        "only; set keep to true if you will need it again in later steps."
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
            },
            # loopbench r16: an agent writing notes about one file expanded it 18 turns
            # in a row (the original is gone again the next turn). Kept originals are
            # re-attached after the turn that asked (proxy/kept_originals.py).
            "keep": {
                "type": "boolean",
                "description": (
                    "true if you will need this original again in later steps — e.g. "
                    "you will quote it, document it or work through it over several "
                    "turns: it then stays in the conversation from your next step on. "
                    "false (default) if you only need it for this step: it is not "
                    "kept, which keeps the context smaller."
                ),
            },
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
