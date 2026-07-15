"""Rule-based SEG tagger — the first half of the "tagger + model" pipeline.

The compression model was trained on segments that a *rule-based* tagger had
already labelled with a `kind` (file_read, log_output, file_operation, ...) and a
`level` (L0-L3, the target compression ratio). Production must reproduce that
same labelling so train/inference distributions match — the model never guesses
kind/level itself.

Ported verbatim (minus training-only deps: orjson/tqdm/must_keep spans) from the
training repo's `02_parse_trajectories.py::classify_segment_kind` and
`04_label.py::{reclassify_tool_result, detect_stale_files, assign_level}`.

Two entry points:
  - tag_messages(messages): full-conversation tagging (kind + level per message),
    the way training labelled data — needs the whole message list for recency /
    staleness. Use this from the middleware.
  - classify_kind_from_content(content): best-effort kind from a lone content
    string, for callers that compress one blob without conversation context.
"""

from __future__ import annotations

import re

# Kinds that are always kept lightly (L0) — never aggressively compressed.
PROTECTED_KINDS = {"system_prompt", "user_turn_current", "user_turn_history"}
# Kinds treated as "tool results" for recency-based level assignment.
TOOL_RESULT_KINDS = {"tool_result", "log_output", "file_read"}

_PATH_RE = re.compile(r'"path":\s*"([^"]+)"')


# ── kind classification ─────────────────────────────────────────────────────

def classify_segment_kind(msg: dict, position_idx: int) -> str:
    """Initial kind from a raw chat message (02_parse_trajectories.py)."""
    role = msg.get("role")
    if role == "system":
        return "system_prompt"
    if role == "user":
        return "user_turn_current" if position_idx == 0 else "user_turn_history"
    if role == "assistant":
        if msg.get("tool_calls"):
            tc = msg["tool_calls"][0]
            name = tc.get("function", {}).get("name", "")
            if name in ("think",):
                return "assistant_thinking"
            if name in ("task_tracker", "finish"):
                return "meta_action"
            if name in ("str_replace_editor",):
                return "file_operation"
            if name in ("execute_bash", "bash"):
                return "bash_command"
            return "tool_call"
        return "assistant_thinking"
    if role == "tool":
        content = msg.get("content", "") or ""
        head = content[:200]
        if "Traceback" in content or "FAILED" in content or "Error:" in head:
            return "log_output"
        if head.strip().startswith(("/", ".", "#!")) or "@@" in head:
            return "file_read"
        if "\n" in head and head.count("\n") > 5:
            return "log_output"
        return "tool_result"
    return "tool_result"


def reclassify_tool_result(kind: str, content: str) -> str:
    """Refine a generic tool_result by its content patterns (04_label.py)."""
    if kind != "tool_result":
        return kind
    head = content[:300]
    if "Here's the result of running `cat -n`" in head:
        return "file_read"
    if head.startswith("Here's the result of running") and "cat" in head:
        return "file_read"
    if re.match(r"^The file .+ has been (created|edited|saved)", head):
        return "file_edit_confirm"
    if "Here's the files and directories" in head:
        return "directory_listing"
    code_indicators = sum(
        1 for kw in ("import ", "def ", "class ", "from ", "function ", "package ")
        if kw in head
    )
    if code_indicators >= 2 and content.count("\n") > 10:
        return "file_read"
    return "tool_result"


def classify_kind_from_content(content: str) -> str:
    """Best-effort kind for a lone content string (no conversation context).

    Mirrors the `role == "tool"` branch of classify_segment_kind plus the
    reclassify_tool_result refinements. Defaults to file_read (the product's
    most common input).
    """
    head = content[:300]
    if "[tool_calls]:" in head or '"str_replace_editor"' in head:
        return "file_operation"
    if "Traceback" in content or "FAILED" in content or "Error:" in head[:200]:
        return "log_output"
    kind = reclassify_tool_result("tool_result", content)
    if kind != "tool_result":
        return kind
    if "@@" in head[:200] or head.lstrip().startswith(("/", ".", "#!")):
        return "file_read"
    # Code-like content (def/class/import present) → file_read. Without a
    # conversation to place it, a Read result is the product's default; only
    # fall back to log_output for non-code multi-line blobs.
    if any(kw in head for kw in ("import ", "def ", "class ", "from ", "function ")):
        return "file_read"
    if content[:200].count("\n") > 5:
        return "log_output"
    return "file_read"


# ── staleness + level assignment ────────────────────────────────────────────

def detect_stale_files(segments: list[dict]) -> set[int]:
    """Indices of file accesses superseded by a later access (04_label.py)."""
    fop_positions: dict[str, list[int]] = {}
    for i, seg in enumerate(segments):
        if seg.get("kind") != "file_operation":
            continue
        for m in _PATH_RE.finditer(seg.get("content", "")):
            path = m.group(1)
            if not path.startswith("/"):
                continue
            if "." not in path.rsplit("/", 1)[-1]:
                continue
            fop_positions.setdefault(path, []).append(i)

    stale: set[int] = set()
    for positions in fop_positions.values():
        if len(positions) <= 1:
            continue
        for pos in positions[:-1]:
            stale.add(pos)
            if pos + 1 < len(segments) and segments[pos + 1].get("kind") == "tool_result":
                stale.add(pos + 1)
    return stale


def assign_level(seg: dict, seg_idx: int, total_segs: int,
                 stale_indices: set[int]) -> tuple[str, str]:
    """Assign L0/L1/L2/L3 from position + absolute distance + kind (04_label.py).

    Returns (level, reason). Verbatim port of the training labeller.
    """
    kind = seg.get("kind")
    is_current = seg.get("is_current_turn", False)
    relative_pos = seg_idx / max(1, total_segs - 1)
    turns_from_end = total_segs - 1 - seg_idx

    is_very_recent = (relative_pos > 0.85) or (turns_from_end <= 1)
    is_recent = (relative_pos > 0.7) or (turns_from_end <= 4)
    is_ancient = (relative_pos < 0.25) and (turns_from_end > 8)

    # L0
    if kind in PROTECTED_KINDS:
        return "L0", f"protected={kind}"
    if is_current:
        return "L0", "is_current_turn"
    if is_very_recent and kind in TOOL_RESULT_KINDS:
        return "L0", "very_recent_tool_result"

    # L3
    if seg_idx in stale_indices:
        return "L3", "stale_file_read"
    if is_ancient and kind not in ("file_operation", "bash_command"):
        return "L3", f"ancient(pos={relative_pos:.2f})"

    # L1: recent file/result
    if is_recent:
        if kind == "file_read":
            return "L1", "recent_file_read"
        if kind in TOOL_RESULT_KINDS:
            return "L1", f"recent_{kind}"

    # L1: action
    if kind in ("file_operation", "bash_command", "tool_call"):
        return "L1", f"action_{kind}"

    # L2: mid-conversation
    if kind == "file_read":
        return "L2", "mid_file_read"
    if kind == "assistant_thinking":
        return "L2", "thinking"

    if kind in TOOL_RESULT_KINDS:
        if relative_pos < 0.4:
            return "L3", f"old_{kind}"
        return "L2", f"mid_{kind}"

    return "L2", f"default_{kind}"


def tag_messages(messages: list[dict]) -> list[dict]:
    """Tag a full conversation with (seg_id, kind, level) per message.

    `messages` is a list of chat dicts (role/content, optional tool_calls). The
    last message is treated as the current turn. Returns a parallel list of
    {seg_id, kind, level, reason} — the SEG labels to feed the model.
    """
    segments = []
    n = len(messages)
    for i, msg in enumerate(messages):
        kind = classify_segment_kind(msg, i)
        content = msg.get("content", "") or ""
        kind = reclassify_tool_result(kind, content)
        segments.append({
            "seg_id": f"s{i}",
            "kind": kind,
            "content": content,
            "is_current_turn": (i == n - 1),
        })

    stale = detect_stale_files(segments)
    labels = []
    for i, seg in enumerate(segments):
        level, reason = assign_level(seg, i, n, stale)
        labels.append({
            "seg_id": seg["seg_id"],
            "kind": seg["kind"],
            "level": level,
            "reason": reason,
        })
    return labels
