"""Recover codex shell-authored edits against lossily-compressed file content.

Claude Code edits through a structured `Edit(old_string, new_string)` tool, so
`edit_recovery` can rewrite a lossy `old_string` back onto the true file. Codex has no
such tool: it edits by running a **shell command** that does a literal string replace —

    python:      s = s.replace('OLD', 'NEW')
    PowerShell:  $old = 'OLD'; $new = 'NEW'; $c = $c.Replace($old, $new)
    PowerShell:  ... -replace 'OLD', 'NEW'

When paritok compressed the file read codex saw, the 4B summary reflowed multi-line code
onto one line (e.g. inlined a `total = sum(...)` temp away). Codex copies that reflowed
form into its `OLD`, which no longer matches the real file, so the replace is a silent
no-op / "pattern not found" and codex re-reads and thrashes.

This module is the codex analog of `edit_recovery`:

  1. `parse_shell_replaces(command)` -> the literal (old -> new) replacements a shell
     command performs, with the source spans of each quoted literal.
  2. `recover_shell_replace(old, new, original)` -> (real_old, real_new) that apply to the
     true file, or None when it already matches / can't be uniquely & safely recovered.
  3. `rewrite_shell_command(command, resolve)` -> (new_command, n): splice the recovered
     literals back into the command so codex's replace lands on the first try.

Safety: a recovery is only spliced in when it is UNIQUE and the recovered literals contain
no quote/backslash characters (so re-quoting can't corrupt the command). Anything else is
left untouched — codex just falls back to its own re-read, exactly as today.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from paritok.pipelines.edit_recovery import _LINE_COMMENT, recover_edit

# Keys under a Responses `function_call.arguments` object that hold a shell command OR an
# apply_patch body (codex's `apply_patch` tool passes the patch under `input`/`patch`).
_COMMAND_KEYS = ("command", "cmd", "script", "input", "patch")

# Characters that make re-quoting a spliced literal ambiguous across shells/quote styles.
# If a recovered literal contains any, we abstain rather than risk a broken command.
_UNSAFE_SPLICE = set("'\"\\`")


def _trim_common(old: str, new: str) -> tuple[str, str]:
    """Return (old_core, new_core): `old`/`new` stripped of their common character prefix
    and suffix, isolating just the span that actually changed. E.g.
    ('a / (n + 1) b', 'a / n b') -> ('(n + 1)', 'n')."""
    p = 0
    while p < len(old) and p < len(new) and old[p] == new[p]:
        p += 1
    s = 0
    while s < len(old) - p and s < len(new) - p and old[-1 - s] == new[-1 - s]:
        s += 1
    return old[p:len(old) - s], new[p:len(new) - s]


def recover_shell_replace(old: str, new: str, original: str) -> tuple[str, str] | None:
    """Map a codex literal replace (`old` -> `new`), authored against a lossy/reflowed
    summary, onto the true `original` file.

    Returns (real_old, real_new) the shell replace can apply verbatim to the real file, or
    None when `old` already matches the file, nothing changed, or the region can't be
    located uniquely (caller then leaves codex's command untouched).
    """
    if not old or old == new:
        return None
    if old in original:
        return None  # codex's OLD already matches the file — no rewrite needed

    # 1. Whole-string recovery. Handles a whitespace-only reflow (multi-line signature or
    #    body collapsed onto one line, blank lines / docstrings dropped): edit_recovery's
    #    canonical match is whitespace/comment-insensitive, so the whole OLD still maps.
    rec = recover_edit(old, new, original)
    if rec is not None:
        return rec

    # 2. Structural reflow. The 4B didn't just re-space the code, it re-expressed it (a
    #    `total = sum(...)` temp inlined into a ternary), so the whole OLD no longer maps.
    #    But the *minimal changed span* — what codex actually meant to edit — usually still
    #    lives in the file verbatim. Recover just that.
    old_core, new_core = _trim_common(old, new)
    if not old_core or old_core == old:
        return None  # nothing to narrow to (pure insertion, or no common context)
    return recover_edit(old_core, new_core, original)


@dataclass
class ShellReplace:
    """One literal replacement a shell command performs. `old_span`/`new_span` are the
    (start, end) character spans in the command of the quoted literal to splice (INCLUDING
    the surrounding quotes), so a rewrite can substitute a recovered value in place."""
    old: str
    new: str
    old_span: tuple[int, int]
    new_span: tuple[int, int]
    quote_old: str
    quote_new: str


def _scan_quoted(s: str, i: int) -> tuple[str | None, int]:
    """Read a quoted string whose opening quote is at s[i]. Returns (value, end) where end
    is the index just past the closing quote, or (None, len(s)) if unterminated.

    Handles both escape conventions conservatively: a backslash escapes only a following
    quote or backslash (`\\'`, `\\\\`) — so Windows paths like 'C:\\x' keep the backslash —
    and a doubled quote ('' / "") is one literal quote (PowerShell / SQL)."""
    q = s[i]
    j = i + 1
    buf: list[str] = []
    while j < len(s):
        c = s[j]
        if c == "\\" and j + 1 < len(s) and s[j + 1] in (q, "\\"):
            buf.append(s[j + 1])
            j += 2
            continue
        if c == q:
            if j + 1 < len(s) and s[j + 1] == q:  # doubled-quote escape
                buf.append(q)
                j += 2
                continue
            return "".join(buf), j + 1
        buf.append(c)
        j += 1
    return None, len(s)


# `.replace(` / `.Replace(` — Python str.replace and PowerShell/.NET .Replace.
_REPLACE_CALL = re.compile(r"\.replace\s*\(\s*", re.IGNORECASE)
# `-replace` — PowerShell operator (pattern is a regex, but we still try; a literal-only
# pattern will match the file, a regex-metachar one simply won't and we abstain).
_REPLACE_OP = re.compile(r"-replace\s+")
# `$name = 'literal'` PowerShell assignment (value scanned separately).
_VAR_ASSIGN = re.compile(r"\$(\w+)\s*=\s*(?=['\"])")
# `.Replace($a, $b)` / `-replace $a, $b` using variables.
_VAR_REPLACE = re.compile(
    r"(?:\.replace\s*\(|-replace)\s*\$(\w+)\s*,\s*\$(\w+)", re.IGNORECASE)


def _scan_pair(s: str, i: int) -> tuple[ShellReplace | None, int]:
    """From index i (just after an opening `(` or `-replace `), read `Q1 , Q2` two quoted
    literals. Returns (ShellReplace-without-old/new-semantics, end) or (None, i)."""
    n = len(s)
    while i < n and s[i] in " \t\r\n":
        i += 1
    if i >= n or s[i] not in "'\"":
        return None, i
    q1 = s[i]
    old_start = i
    old, i = _scan_quoted(s, i)
    old_end = i
    if old is None:
        return None, i
    while i < n and s[i] in " \t\r\n":
        i += 1
    if i >= n or s[i] != ",":
        return None, i
    i += 1
    while i < n and s[i] in " \t\r\n":
        i += 1
    if i >= n or s[i] not in "'\"":
        return None, i
    q2 = s[i]
    new_start = i
    new, i = _scan_quoted(s, i)
    new_end = i
    if new is None:
        return None, i
    return ShellReplace(old, new, (old_start, old_end), (new_start, new_end), q1, q2), i


def parse_shell_replaces(command: str) -> list[ShellReplace]:
    """Find the literal (old -> new) replacements a shell command performs.

    Covers inline `.replace('a','b')` / `.Replace("a","b")` / `-replace 'a','b'`, and the
    variable form `$old='a'; $new='b'; ....Replace($old,$new)`. Returns them in source
    order; overlapping/duplicate spans are de-duplicated."""
    out: list[ShellReplace] = []
    seen: set[tuple[int, int]] = set()

    # Inline literal calls: `.replace(` and `-replace `.
    for rx in (_REPLACE_CALL, _REPLACE_OP):
        for m in rx.finditer(command):
            sr, _ = _scan_pair(command, m.end())
            if sr and sr.old_span not in seen:
                seen.add(sr.old_span)
                out.append(sr)

    # Variable form: collect `$name = 'literal'`, then `.Replace($a,$b)` / `-replace $a,$b`.
    varmap: dict[str, tuple[str, tuple[int, int], str]] = {}
    for m in _VAR_ASSIGN.finditer(command):
        val, end = _scan_quoted(command, m.end())
        if val is not None:
            varmap[m.group(1)] = (val, (m.end(), end), command[m.end()])
    for m in _VAR_REPLACE.finditer(command):
        a, b = m.group(1), m.group(2)
        if a in varmap and b in varmap:
            ov, ospan, oq = varmap[a]
            nv, nspan, nq = varmap[b]
            if ospan not in seen:
                seen.add(ospan)
                out.append(ShellReplace(ov, nv, ospan, nspan, oq, nq))

    out.sort(key=lambda sr: sr.old_span[0])
    return out


def _safe_to_splice(real_old: str, real_new: str) -> bool:
    """Only splice recovered literals that carry no quote/backslash/backtick, so wrapping
    them in the original quote char can't corrupt the command."""
    return not (_UNSAFE_SPLICE & set(real_old)) and not (_UNSAFE_SPLICE & set(real_new))


def rewrite_shell_command(command: str, resolve):
    """Rewrite literal file-replace(s) in `command` so their OLD matches the true file.

    `resolve` is a callable returning candidate original file contents to recover against:
    `resolve(command) -> Iterable[str]` (the caller wires it to the shadow store, using the
    file paths mentioned in the command). Returns (new_command, n) where n is how many
    replacements were rewritten; the command is unchanged when n == 0.
    """
    replaces = parse_shell_replaces(command)
    if not replaces:
        return command, 0
    originals = [o for o in (resolve(command) or []) if o]
    if not originals:
        return command, 0

    edits: list[tuple[tuple[int, int], str]] = []  # (span, replacement-with-quotes)
    n = 0
    for sr in replaces:
        rec = None
        for original in originals:
            rec = recover_shell_replace(sr.old, sr.new, original)
            if rec is not None:
                break
        if rec is None:
            continue
        real_old, real_new = rec
        if real_old == sr.old and real_new == sr.new:
            continue
        if not _safe_to_splice(real_old, real_new):
            continue
        edits.append((sr.old_span, f"{sr.quote_old}{real_old}{sr.quote_old}"))
        edits.append((sr.new_span, f"{sr.quote_new}{real_new}{sr.quote_new}"))
        n += 1

    if not edits:
        return command, 0
    # Splice right-to-left so earlier spans keep their indices.
    for (start, end), rep in sorted(edits, key=lambda e: e[0][0], reverse=True):
        command = command[:start] + rep + command[end:]
    return command, n


# ── apply_patch recovery ──────────────────────────────────────────────────────
#
# Codex's other edit path is the `apply_patch` tool, whose body is a V4A patch:
#
#     *** Begin Patch
#     *** Update File: path/to/file.py
#     @@ def answer_question(...):
#         context line
#     -    max_retries = -1
#     +    max_retries = 3
#     *** End Patch
#
# apply_patch locates each hunk by its context + removed (`-`) lines EXACTLY. When the
# file read codex saw was compressed, those old-side lines dropped a trailing comment /
# reflowed, so apply_patch fails with "Failed to find expected lines" and codex thrashes.
# We map each hunk's old-side block back onto the true file (comment/whitespace-insensitive,
# unique-match only) and rewrite those lines in place — the `+` added lines are codex's
# intended change and are never touched.

_PATCH_MARKER = "*** Begin Patch"


def _canon_line(line: str) -> str:
    """Canonicalize one code line the way the compression lint does: drop a trailing
    `# ...` comment, then all whitespace. So `    max_retries = -1  ##__o|o__` and the
    summary's `max_retries = -1` collapse to the same key."""
    return "".join(_LINE_COMMENT.sub("", line).split())


def _match_line_window(old_lines: list[str], original: str) -> list[str] | None:
    """Return the real file's lines (FULL, with their true indentation) that canonically
    match `old_lines`, or None unless exactly one contiguous window matches. Line-based (not
    `match_region`) so indentation is preserved verbatim — apply_patch is indent-sensitive."""
    co = [_canon_line(ln) for ln in old_lines]
    if not co or all(c == "" for c in co):
        return None  # nothing distinctive to anchor on
    orig_lines = original.split("\n")
    n = len(co)
    hits = [
        s for s in range(len(orig_lines) - n + 1)
        if [_canon_line(orig_lines[s + k]) for k in range(n)] == co
    ]
    if len(hits) != 1:
        return None
    s = hits[0]
    return orig_lines[s:s + n]


def rewrite_apply_patch(patch_text: str, originals) -> tuple[str, int]:
    """Rewrite the old-side (context + removed) lines of each hunk in an apply_patch body so
    they match the true file, recovered from `originals` (candidate real file contents).

    Returns (new_patch, n_hunks_rewritten); the text is returned unchanged when it is not a
    patch, no original matches, or a hunk can't be mapped 1:1 (codex then re-reads)."""
    originals = [o for o in (originals or []) if o]
    if not originals or not patch_text or _PATCH_MARKER not in patch_text:
        return patch_text, 0

    lines = patch_text.split("\n")
    # Group old-side line indices (context ` ` / removed `-`) per hunk. A hunk ends at a
    # `@@` anchor, a `*** ...` header, an added `+` line does NOT end it, and any other line
    # (blank/heredoc wrapper) ends it.
    hunks: list[list[int]] = []
    cur: list[int] = []

    def _flush() -> None:
        nonlocal cur
        if cur:
            hunks.append(cur)
            cur = []

    for idx, ln in enumerate(lines):
        if ln.startswith("*** ") or ln.startswith("@@"):
            _flush()
            continue
        c = ln[:1]
        if c in (" ", "-"):
            cur.append(idx)
        elif c == "+":
            continue  # added line: part of the hunk, but not old-side
        else:
            _flush()
    _flush()

    n = 0
    for hunk in hunks:
        old_lines = [lines[i][1:] for i in hunk]  # strip the 1-char ` `/`-` prefix
        real_lines = None
        for original in originals:
            real_lines = _match_line_window(old_lines, original)
            if real_lines is not None:
                break
        if real_lines is None or real_lines == old_lines:
            continue  # can't locate uniquely, or already matches — leave it
        for j, i in enumerate(hunk):
            lines[i] = lines[i][:1] + real_lines[j]  # keep the ` `/`-` prefix
        n += 1

    if not n:
        return patch_text, 0
    return "\n".join(lines), n


def rewrite_edit_command(value: str, resolve) -> tuple[str, int]:
    """Rewrite one command/patch string: try shell literal-replaces first, then, if that
    changed nothing and it looks like an apply_patch, rewrite the patch hunks. `resolve` is
    the shell callable `resolve(command) -> Iterable[str]` of candidate originals."""
    new_val, n = rewrite_shell_command(value, resolve)
    if n:
        return new_val, n
    if _PATCH_MARKER in value:
        return rewrite_apply_patch(value, resolve(value))
    return value, 0


def rewrite_function_call_arguments(arguments_json: str, originals):
    """Rewrite the shell command inside a codex Responses `function_call.arguments` JSON
    string so any literal file-replace matches the true file. Handles a command held as a
    string or a list of strings under `command`/`cmd`/`script`/`input`.

    `originals` are candidate real file contents (the read bodies from the request).
    Returns (new_arguments_json, n_rewritten); the input is returned unchanged when nothing
    parses or nothing is rewritten."""
    originals = [o for o in (originals or []) if o]
    if not arguments_json or not arguments_json.strip() or not originals:
        return arguments_json, 0
    try:
        args = json.loads(arguments_json)
    except (json.JSONDecodeError, ValueError):
        return arguments_json, 0
    if not isinstance(args, dict):
        return arguments_json, 0

    resolve = lambda _cmd: originals  # noqa: E731 — strict recovery filters candidates
    total = 0
    for k in _COMMAND_KEYS:
        v = args.get(k)
        if isinstance(v, str) and v.strip():
            nv, n = rewrite_edit_command(v, resolve)
            if n:
                args[k] = nv
                total += n
        elif isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            new_list, hits = [], 0
            for el in v:
                ne, n = rewrite_edit_command(el, resolve)
                new_list.append(ne)
                hits += n
            if hits:
                args[k] = new_list
                total += hits
    if not total:
        return arguments_json, 0
    return json.dumps(args), total


def rewrite_tool_call_input(text: str, originals):
    """Rewrite a codex `custom_tool_call.input` — a RAW command/patch string (codex's newer
    freeform tools), or occasionally a JSON args object. Returns (new_text, n_rewritten).

    Delegates JSON objects to `rewrite_function_call_arguments`; a raw string is run through
    both the shell-replace and apply_patch recovery paths."""
    originals = [o for o in (originals or []) if o]
    if not text or not text.strip() or not originals:
        return text, 0
    stripped = text.lstrip()
    if stripped[:1] == "{":  # looks like a JSON args object
        try:
            if isinstance(json.loads(text), dict):
                return rewrite_function_call_arguments(text, originals)
        except (json.JSONDecodeError, ValueError):
            pass
    return rewrite_edit_command(text, lambda _cmd: originals)
