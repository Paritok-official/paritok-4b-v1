"""What the agent saw and wrote for each file, replayed from the client's own messages.

edit_recovery maps an Edit the agent authored against a compressed view back onto the
file as the gateway last saw it: the original of a compressed Read. But the agent keeps
editing the file after that read, and its next Edit targets the file as it is NOW.
Re-mapping that onto the stale read turns a correct old_string into text that no longer
exists, or, when it matches a substring, splices the change into the wrong place.
loopbench r11: the agent added `# type:` comments, then converted them to annotations;
recovery took its own comments for text the summary had hidden and rewrote the edit 9
times, writing `self.data : '...' = {}  # type: ...` into deletion.py.

The client re-sends the whole conversation uncompressed, so the gateway can replay it:

  - `saw_verbatim`: the target is text the agent read uncompressed or wrote itself, so
    it was not authored from a summary; leave the edit alone.
  - `current`: a read's text with the agent's later successful writes applied (the file
    as it is now), or None when that can't be known exactly; that read must then not be
    used to rewrite anything.
"""

from __future__ import annotations

import re
from collections import defaultdict

from paritok.pipelines.edit_recovery import strip_read_line_numbers

_READ_TOOLS = ("Read", "NotebookRead")
_SHELL_TOOLS = ("Bash", "PowerShell")
# A shell command that may write a file (sed -i, redirection, a script's write/replace,
# PowerShell Set-Content, git checkout, ...). Its effect can't be replayed, so any file
# it names is treated as changed in an unknown way. Redirection excludes the usual
# `2>&1` / `>/dev/null` of test and grep commands.
_SHELL_WRITE = re.compile(
    r"sed\s+(?:-\w*\s+)*-i|perl\s+-\w*i|(?<![\d&>=<-])>>?(?![&=]|\s*/dev/null|\s*\$null|\s*nul\b)|"
    r"\.write\(|write_text|write_bytes|Set-Content|"
    r"Out-File|Add-Content|-replace|\.replace\(|\btee\b|\bmv\b|\bcp\b|\brm\b|\bpatch\b|"
    r"git\s+(?:checkout|restore|stash|apply|reset|am|cherry-pick|revert)", re.I)


def norm_path(path: str | None) -> str:
    return (path or "").strip().replace("\\", "/").lower()


def _result_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content
                         if isinstance(p, dict) and p.get("type") == "text")
    return ""


def _write_ops(name: str, inp: dict) -> list[tuple] | None:
    """The file changes a successful write tool call made, in order:
    ("edit", old, new, replace_all) | ("write", content) | ("unknown",).
    None when the call isn't a file write."""
    if name == "Edit":
        return [("edit", inp.get("old_string"), inp.get("new_string"), bool(inp.get("replace_all")))]
    if name == "MultiEdit":
        return [("edit", e.get("old_string"), e.get("new_string"), bool(e.get("replace_all")))
                for e in inp.get("edits") or [] if isinstance(e, dict)]
    if name == "Write":
        return [("write", inp.get("content"))]
    if name == "NotebookEdit":
        return [("unknown",)]
    if name in ("str_replace_editor", "str_replace_based_edit_tool"):
        cmd = inp.get("command")
        if cmd == "str_replace":
            return [("edit", inp.get("old_str"), inp.get("new_str"), False)]
        if cmd == "create":
            return [("write", inp.get("file_text"))]
        return None if cmd == "view" else [("unknown",)]
    if name == "str_replace":
        return [("edit", inp.get("old_string") or inp.get("old_str"),
                 inp.get("new_string") or inp.get("new_str"), False)]
    return None


class FileHistory:
    """Per-file reads and writes of one conversation, in order.

    `messages` are the CLIENT's messages (tool results uncompressed); `compressed` maps
    the tool_use_id of every tool result the gateway forwarded compressed to its
    shadow_id. A Read not in it was shown to the agent verbatim."""

    def __init__(self, messages: list[dict] | None, compressed: dict[str, str] | None = None):
        compressed = compressed or {}
        self._pos: dict[str, int] = {}
        self._writes: dict[str, list[tuple[int, tuple]]] = defaultdict(list)
        self._seen: dict[str, list[str]] = defaultdict(list)
        self._read_of: dict[str, str] = {}  # shadow_id -> latest tool_use_id that showed it
        uses, results, shell, known = [], {}, [], set()
        for msg in messages or []:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for b in content:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_use" and msg.get("role") == "assistant" and b.get("id"):
                    self._pos[b["id"]] = len(self._pos)
                    uses.append(b)
                elif b.get("type") == "tool_result":
                    results[b.get("tool_use_id")] = (bool(b.get("is_error")),
                                                     _result_text(b.get("content")))
        for b in uses:
            tid, name, inp = b["id"], b.get("name"), b.get("input") or {}
            res = results.get(tid)
            if res is None or res[0]:
                continue  # not run yet, or failed: changed nothing
            pos = self._pos[tid]
            path = norm_path(inp.get("file_path") or inp.get("path") or inp.get("notebook_path"))
            if path:
                known.add(path)
            if name in _READ_TOOLS:
                if not path:
                    continue
                if tid in compressed:
                    self._read_of[compressed[tid]] = tid
                else:
                    self._seen[path].append(strip_read_line_numbers(res[1]))
                continue
            if name in _SHELL_TOOLS:
                cmd = inp.get("command")
                if isinstance(cmd, str) and _SHELL_WRITE.search(cmd):
                    shell.append((pos, cmd.replace("\\", "/").lower()))
                continue
            ops = _write_ops(name, inp) if path else None
            if not ops:
                continue
            for op in ops:
                self._writes[path].append((pos, op))
                written = op[2] if op[0] == "edit" else op[1] if op[0] == "write" else None
                if isinstance(written, str) and written:
                    self._seen[path].append(written)
        # Files a shell command may have written: every known file it names.
        for pos, cmd in shell:
            for path in known:
                base = path.rsplit("/", 1)[-1]
                if base and base in cmd:
                    self._writes[path].append((pos, ("unknown",)))
        for lst in self._writes.values():
            lst.sort(key=lambda w: w[0])

    def saw_verbatim(self, path: str, text: str) -> bool:
        """Is `text` in something the agent read uncompressed, or wrote, for this file?"""
        return bool(text) and any(text in s for s in self._seen.get(norm_path(path), ()))

    def read_of(self, shadow_id: str | None) -> str | None:
        """tool_use_id of the latest compressed Read in this conversation that showed
        this shadow (None if it isn't in the conversation)."""
        return self._read_of.get(shadow_id) if shadow_id else None

    def wrote(self, path: str) -> bool:
        return bool(self._writes.get(norm_path(path)))

    def current(self, path: str, text: str, after: str | None) -> str | None:
        """`text`, what the read `after` returned, with the agent's later successful writes
        to this file applied: the file as it is now. None when that can't be known
        exactly: the read isn't in this conversation but the file was written, or a later
        write doesn't apply cleanly (it touched lines outside this read, or the file
        changed some other way)."""
        writes = self._writes.get(norm_path(path), [])
        start = self._pos.get(after) if after else None
        if start is None:
            return None if writes else text
        for pos, op in writes:
            if pos <= start:
                continue
            if op[0] == "write":
                if not isinstance(op[1], str):
                    return None
                text = op[1]
            elif op[0] == "edit":
                _, old, new, every = op
                if not (isinstance(old, str) and old and isinstance(new, str)):
                    return None
                n = text.count(old)
                if n == 0 or (n > 1 and not every):
                    return None
                text = text.replace(old, new) if every else text.replace(old, new, 1)
            else:
                return None
        return text
