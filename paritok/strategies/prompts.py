"""System prompts for the SEG-based local compression model (SFT checkpoint-2000).

These MUST match the system prompts used in training verbatim. The trained model
learned this exact text distribution; drifting from it degrades compression
quality. The two prompts are shipped as data files under
`paritok/strategies/system_prompts/` (file_read.txt and other.txt), copied
verbatim from the SFT training system prompts.

Runtime protocol (see `local_model.py`):
    SYSTEM: file_read.txt  (kind == "file_read")  OR  other.txt  (all other kinds)
    USER:
        USER INTENT:
        {intent}

        Compress the following segment under the rules in your system prompt.
        Output only the compressed [SEG]...[/SEG] block (or an empty one to drop):

        [SEG id={seg_id} kind={kind} level={level}]
        {content}
        [/SEG]

The model replies with a single [SEG ...]<body>[/SEG] block; an empty body means
"drop this segment". Levels L0-L3 set the target compression ratio
(L0 ≤ 0.50, L1 ≤ 0.35, L2 ≤ 0.25, L3 ≤ 0.20).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_PROMPT_DIR = Path(__file__).parent / "system_prompts"

# Kinds the file_read system prompt was trained on. Everything else (log_output,
# file_operation, assistant_thinking, bash_command, tool_result,
# directory_listing, meta_action, ...) uses the "other" prompt.
_FILE_READ_KINDS = {"file_read"}


@lru_cache(maxsize=None)
def _load(name: str) -> str:
    return (_PROMPT_DIR / name).read_text(encoding="utf-8")


def system_prompt_for_kind(kind: str | None) -> str:
    """Return the verbatim training system prompt for a SEG kind.

    file_read → the file_read prompt; any other kind → the "other" prompt.
    Unknown/None defaults to the "other" prompt (its decision flow is broader).
    """
    if kind in _FILE_READ_KINDS:
        return _load("file_read.txt")
    return _load("other.txt")


# Convenience accessors (the two distinct prompts).
SYSTEM_PROMPT_FILE_READ = _load("file_read.txt")
SYSTEM_PROMPT_OTHER = _load("other.txt")

# Backwards-compat aliases. The pre-SEG code split prompts into CODE / TOOL /
# HISTORY; map those onto the new two-prompt scheme. CODE == file_read; the
# generic tool/history buckets go to the broader "other" prompt.
SYSTEM_PROMPT_CODE = SYSTEM_PROMPT_FILE_READ
SYSTEM_PROMPT_TOOL = SYSTEM_PROMPT_OTHER
SYSTEM_PROMPT_HISTORY = SYSTEM_PROMPT_OTHER
HISTORY_SUMMARY_PROMPT = SYSTEM_PROMPT_OTHER
QUERY_SPECIFIC_PROMPT = SYSTEM_PROMPT_FILE_READ
QUERY_AGNOSTIC_PROMPT = SYSTEM_PROMPT_OTHER
