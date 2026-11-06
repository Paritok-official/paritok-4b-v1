"""Recover exact-match edits against lossily-compressed file content.

When paritok compresses a file read to a `[REF]` summary, the model authors its
edit `old_string` against that summary — which drops docstrings, collapses
multi-line signatures onto one line, removes blank lines, strips trailing commas,
etc. (the deterministic training-lint transforms). The resulting `old_string`
no longer matches the real file byte-for-byte, so the agent's exact-match Edit
fails and it re-reads, burning turns.

This module maps a model-authored `old_string` back onto the true original file:

    recovered = match_region(old_string, original_file_text)
        -> the exact original substring old_string canonically corresponds to,
           or None if there is not exactly one canonical match (caller should then
           fall back to forcing an expand_context so the model re-authors the edit).

    new_old, new_new = recover_edit(old_string, new_string, original_file_text)
        -> a rewritten (old_string, new_string) pair the client can apply verbatim:
           old_string becomes the exact original region; new_string carries the
           model's change applied onto that region, preserving the docstrings /
           comments / blank lines / multi-line formatting the summary had dropped.

The canonical form is whitespace-, trailing-comma-, docstring- and comment-
insensitive, mirroring the lint the compression model was trained to emit. A
match is only accepted when it is UNIQUE, so a wrong region is never silently
rewritten — ambiguity returns None and the caller falls back to expand.
"""
from __future__ import annotations

import re

_TRAILING_COMMA = re.compile(r',(\s*)([)\]}])')
_TRIPLE_STR = re.compile(r'"""(?:.|\n)*?"""|\'\'\'(?:.|\n)*?\'\'\'')
_LINE_COMMENT = re.compile(r'#[^\n]*')
# Claude Code's Read returns "<n>\t<line>" (and, padded, "  <n>  <line>"). The shadow
# store holds that line-numbered form, but the Edit tool matches the raw file. Strip the
# numbering back to the raw file content before matching an old_string against it.
_READ_LINENO_TAB = re.compile(r'^[ \t]*\d+\t', re.M)


def strip_read_line_numbers(text: str) -> str:
    """Turn a Read tool_result ("1\\tcode\\n2\\t...") back into raw file text."""
    return _READ_LINENO_TAB.sub('', text)


def _canon_with_map(s: str) -> tuple[str, list[int]]:
    """Canonicalize `s` to the lint-equivalent form and return (canonical, offset_map)
    where offset_map[i] is the index in `s` of the i-th kept (canonical) character.

    Dropped (ignored) for matching: all whitespace, a trailing comma right before a
    closing bracket, triple-quoted docstrings, and `# ...` line comments — exactly the
    things the compression lint removes or reflows, so the model's summary-derived
    string and the true original collapse to the same canonical string."""
    drop: set[int] = set()
    for m in _TRAILING_COMMA.finditer(s):
        drop.add(m.start())
    for m in _TRIPLE_STR.finditer(s):
        drop.update(range(m.start(), m.end()))
    for m in _LINE_COMMENT.finditer(s):
        drop.update(range(m.start(), m.end()))
    out: list[str] = []
    omap: list[int] = []
    for i, ch in enumerate(s):
        if ch.isspace() or i in drop:
            continue
        out.append(ch)
        omap.append(i)
    return "".join(out), omap


def _match_span(old_string: str, original: str) -> tuple[int, int] | None:
    """(start, end) of the unique region of `original` whose canonical form equals
    `old_string`'s, or None unless there is exactly one such match."""
    co, _ = _canon_with_map(old_string)
    corig, omap = _canon_with_map(original)
    if not co:
        return None
    starts: list[int] = []
    i = corig.find(co)
    while i != -1:
        starts.append(i)
        i = corig.find(co, i + 1)
    if len(starts) != 1:
        return None
    st = starts[0]
    en = st + len(co) - 1
    return omap[st], omap[en] + 1


def match_region(old_string: str, original: str) -> str | None:
    """Return the exact substring of `original` whose canonical form equals
    `old_string`'s, or None unless there is exactly one such match."""
    span = _match_span(old_string, original)
    return None if span is None else original[span[0]:span[1]]


# ── indentation: the summary of a mid-file slice is often shifted left a level ──

def _lead(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _reindent(text: str, delta: int, skip_first: bool) -> str | None:
    """Shift every non-blank line of `text` by `delta` spaces (the first line too unless
    `skip_first`). None if a line can't be shifted (tabs, or too little indentation)."""
    if delta == 0:
        return text
    out = []
    for i, ln in enumerate(text.split("\n")):
        if (i == 0 and skip_first) or not ln.strip():
            out.append(ln)
            continue
        if ln[:1] == "\t":
            return None
        if delta > 0:
            out.append(" " * delta + ln)
        elif _lead(ln) >= -delta:
            out.append(ln[-delta:])
        else:
            return None
    return "\n".join(out)


def _indent_delta(old: str, real_lines: list[str], first_real_indent: int | None) -> int | None:
    """Indentation offset (real − model) of the lines of `old` that occur verbatim (modulo
    indentation), in order, in the real region. The compressed view shifts a whole slice
    by one constant, so one offset must cover >=80% of the matched lines; 0 when nothing
    can be compared, None when the offsets disagree (don't guess)."""
    deltas: dict[int, int] = {}
    j = 1  # real_lines[0] is the (possibly partial) first line, handled via first_real_indent
    for k, ol in enumerate(old.split("\n")):
        s = ol.strip()
        if not s:
            continue
        if k == 0:
            if ol[:1] == " " and first_real_indent is not None:
                d = first_real_indent - _lead(ol)
                deltas[d] = deltas.get(d, 0) + 1
            continue
        while j < len(real_lines) and real_lines[j].strip() != s:
            j += 1
        if j >= len(real_lines):
            break
        d = _lead(real_lines[j]) - _lead(ol)
        deltas[d] = deltas.get(d, 0) + 1
        j += 1
    if not deltas:
        return 0
    best, n = max(deltas.items(), key=lambda kv: kv[1])
    return best if n >= 0.8 * sum(deltas.values()) else None


# ── content the model never saw: keep it ──

def _code_part(line: str) -> str:
    """The line without a trailing `# comment` (a `#` outside string quotes), stripped."""
    q = None
    for i, ch in enumerate(line):
        if q:
            if ch == q:
                q = None
        elif ch in "'\"":
            q = ch
        elif ch == "#":
            return line[:i].strip()
    return line.strip()


def _canon_line(line: str) -> str:
    return "".join(_code_part(line).split())


def _reinsert_docstrings(recovered: str, result: str) -> str:
    """Put back any docstring that lived in `recovered` but was dropped from `result`
    (its location fell inside the model's changed span). Insert after the nearest
    preceding signature-closing `:` line, matching the following line's indent.
    No longer used by recover_edit (see `_restore_hidden`); eval_model/reanchor.py
    still re-anchors training patches with it."""
    for d in _TRIPLE_STR.findall(recovered):
        if d.strip() in result:
            continue
        lines = result.split("\n")
        ins = None
        for i, ln in enumerate(lines):
            if ln.rstrip().endswith(":"):
                ins = i
        if ins is None:
            continue
        indent = re.match(r"\s*", lines[ins + 1]).group(0) if ins + 1 < len(lines) else "    "
        lines.insert(ins + 1, indent + d.strip())
        result = "\n".join(lines)
    return result


def _restore_hidden(recovered: str, seen_old: str, result: str) -> str | None:
    """Put back what the real region holds but the model never saw, wherever the model's
    rewrite lost it. By construction of the canonical match every CODE character of the
    region was in the model's old_string (maybe reflowed); what it never saw is exactly
    what canonicalization ignores — docstrings and `#` comments. A model-written
    docstring standing where a hidden one lives yields to the original (the model
    couldn't know one existed). None if a hidden comment can't be placed: the rewrite is
    then abandoned rather than silently deleting the user's text."""
    # 1. hidden docstrings, as whole blocks
    for m in _TRIPLE_STR.finditer(recovered):
        block = m.group(0)
        if block in seen_old or block in result:
            continue
        mine = [d for d in _TRIPLE_STR.findall(result) if d not in recovered]
        if mine:  # the model wrote its own docstring there → keep the original instead
            result = result.replace(mine[0], block, 1)
            continue
        lines = result.split("\n")
        ins = max((i for i, ln in enumerate(lines) if ln.rstrip().endswith(":")), default=None)
        if ins is None:
            return None
        indent = re.match(r"[ \t]*", lines[ins + 1]).group(0) if ins + 1 < len(lines) else "    "
        lines.insert(ins + 1, indent + block.strip())
        result = "\n".join(lines)

    # 2. hidden `#` comments (outside docstrings): full-line ones and trailing ones
    in_doc = {i for m in _TRIPLE_STR.finditer(recovered) for i in range(m.start(), m.end())}
    res_lines = result.split("\n")
    rec_lines = recovered.split("\n")
    offs, pos = [], 0
    for rl in rec_lines:
        offs.append(pos)
        pos += len(rl) + 1

    def _is_doc(i):
        rl = rec_lines[i]
        return offs[i] + len(rl) - len(rl.lstrip()) in in_doc

    # Map each real code line onto the result line that carries it now, in order: same
    # canonical code, else the same head (an assignment target / `def name` / `if cond`
    # the model kept while e.g. adding a type annotation).
    mapping: dict[int, int] = {}
    j = 0
    for idx, rl in enumerate(rec_lines):
        if not _code_part(rl) or _is_doc(idx):
            continue
        for test in (_canon_line, _line_head):
            want = test(rl)
            hit = next((i for i in range(j, len(res_lines)) if want and test(res_lines[i]) == want),
                       None)
            if hit is not None:
                mapping[idx] = hit
                j = hit + 1
                break

    inserts: list[tuple[int, str]] = []  # (result line index to insert BEFORE, text)
    for idx, rl in enumerate(rec_lines):
        code = _code_part(rl)
        comment = rl.strip()[len(code):].strip() if code else rl.strip()
        if not comment.startswith("#") or _is_doc(idx):
            continue
        if comment in seen_old or any(comment in ln for ln in res_lines):
            continue  # the model saw it (its call), or it survived
        if code and idx in mapping:  # trailing comment: back onto its line
            k = mapping[idx]
            res_lines[k] = res_lines[k].rstrip() + "  " + comment
            continue
        # A full-line comment describes the code below it: put it before the line that
        # now carries the next real code line (else after the previous one).
        nxt = next((mapping[k] for k in range(idx + 1, len(rec_lines)) if k in mapping), None)
        prv = next((mapping[k] for k in range(idx - 1, -1, -1) if k in mapping), None)
        if nxt is not None:
            at = nxt
        elif prv is not None:
            at = prv + 1
        elif not any(_code_part(x) for x in rec_lines):
            at = 0
        else:
            return None  # can't place it: abandon the rewrite, never drop it
        inserts.append((at, rl[:len(rl) - len(rl.lstrip())] + comment))
    # Apply in position order, keeping the original order of comments sharing a spot.
    shift = 0
    for _, (at, text) in sorted(enumerate(inserts), key=lambda e: (e[1][0], e[0])):
        res_lines.insert(at + shift, text)
        shift += 1
    return "\n".join(res_lines)


def _line_head(line: str) -> str:
    """The leading part of a code line up to its first `=`, `:` or `(` (canonical):
    `self.data: Dict = {}` and `self.data = {}` share `self.data`."""
    c = _canon_line(line)
    cut = min((c.find(ch) for ch in "=:(" if ch in c), default=len(c))
    return c[:cut] if cut > 0 else ""


def recover_edit(old_string: str, new_string: str, original: str) -> tuple[str, str] | None:
    """Map a model-authored (old_string -> new_string) edit onto the true original.

    Returns (client_old, client_new): client_old is the exact original region (matches
    the file), client_new is that region with the model's change applied, preserving the
    docstrings / comments / blank lines / multi-line formatting the summary had dropped,
    and shifted to the file's real indentation when the summary had moved it. Returns
    None when the region can't be uniquely located, or the rewrite isn't safe (caller
    then leaves the model's edit untouched)."""
    span = _match_span(old_string, original)
    if span is None:
        return None
    start, end = span
    line_start = original.rfind("\n", 0, start) + 1
    prefix = original[line_start:start]
    first_real_indent = len(prefix) if prefix.strip() == "" and "\t" not in prefix else None
    delta = _indent_delta(old_string, original[start:end].split("\n"), first_real_indent)
    if delta is None:
        return None
    if delta:
        # The model wrote at the summary's indentation; write at the file's instead.
        skip_first = not old_string[:1].isspace()
        old2 = _reindent(old_string, delta, skip_first)
        new2 = _reindent(new_string, delta, skip_first)
        if old2 is None or new2 is None:
            return None
        old_string, new_string = old2, new2
        if original.count(old_string) == 1:
            return old_string, new_string
        span = _match_span(old_string, original)
        if span is None:
            return None
        start, end = span
    recovered = original[start:end]

    # Char-level common prefix/suffix of the model's own before/after: this is exactly
    # what the model changed (everything outside it is unchanged context).
    p = 0
    while p < len(old_string) and p < len(new_string) and old_string[p] == new_string[p]:
        p += 1
    s = 0
    while s < len(old_string) - p and s < len(new_string) - p and old_string[-1 - s] == new_string[-1 - s]:
        s += 1
    new_mid = new_string[p:len(new_string) - s]

    cr, omap = _canon_with_map(recovered)
    cpre, _ = _canon_with_map(old_string[:p])
    csuf, _ = _canon_with_map(old_string[len(old_string) - s:] if s > 0 else "")
    pre_byte = omap[len(cpre) - 1] if cpre else -1
    suf_byte = omap[len(cr) - len(csuf)] if csuf else len(recovered)

    # The canonical map drops whitespace + comments, so the boundary bytes land on the
    # nearest KEPT char — which would glue tokens together ("+5") and swallow a trailing
    # comment on the edited line. Reclaim that dropped formatting at both boundaries.
    comment_idx: set[int] = set()
    for m in _LINE_COMMENT.finditer(recovered):
        comment_idx.update(range(m.start(), m.end()))
    if not new_mid[:1].isspace():
        # Reclaim ALL dropped whitespace before the change, including a newline +
        # indent at a line boundary — matching the suffix side (which uses isspace()).
        # Restricting this to " \t" glued tokens across a dropped newline, e.g.
        # `...max_width` + `height =` -> `...max_widthheight =`.
        while pre_byte + 1 < suf_byte and recovered[pre_byte + 1].isspace():
            pre_byte += 1
    if not new_mid[-1:].isspace():
        while suf_byte > pre_byte + 1 and (
                recovered[suf_byte - 1].isspace() or (suf_byte - 1) in comment_idx):
            suf_byte -= 1

    result = recovered[:pre_byte + 1] + new_mid + recovered[suf_byte:]
    result = _restore_hidden(recovered, old_string, result)
    if result is None or (result == recovered and old_string != new_string):
        return None  # unsafe, or the change vanished in the mapping → don't rewrite
    return recovered, result


_TRAILING_DOCSTRING = re.compile(
    r'(?P<head>[^\n]*:[ \t]*\n)(?P<doc>[ \t]*("""|\'\'\')(?:.|\n)*?\3[ \t]*)\n?\s*\Z')
_LEADING_DOCSTRING = re.compile(r'\s*("""|\'\'\')(?P<body>(?:.|\n)*?)\1')


def drop_shadowed_docstring(old_string: str, new_string: str, following: str,
                            views: list[str] | None = None) -> str | None:
    """The model's edit adds a docstring right after a def/class header, but the real
    file already has one right after the edited region that the model never saw (the
    summary dropped it). Applying the edit as-is leaves two docstrings. Return
    new_string without the model's docstring so the original one stays — or None when
    the case doesn't apply (no inserted docstring, none following, or the model could
    see the existing one in `views`, so its docstring is deliberate)."""
    m = _TRAILING_DOCSTRING.search(new_string)
    if not m or "\"\"\"" in old_string or "'''" in old_string:
        return None
    fm = _LEADING_DOCSTRING.match(following)
    if not fm:
        return None
    first = next((ln.strip() for ln in fm.group("body").splitlines() if ln.strip()), "")
    if not first or any(first in v for v in (views or [])):
        return None
    out = new_string[:m.start("doc")]
    return out if old_string.endswith("\n") else out.rstrip("\n")


def reconcile_edit(old_string: str, new_string: str, original: str,
                   views: list[str] | None = None) -> tuple[str, str] | None:
    """Everything edit recovery does for one Edit against one stored read of the file.

    `original` is the real file text (line numbers stripped); `views` are the
    compressed texts the model saw for it. Returns the (old, new) the client should
    apply, or None to leave the model's edit untouched.
      - old_string matches the file exactly: fix only what the summary still breaks —
        a docstring the model duplicated, or a multi-line new_string written at the
        summary's shallower indentation while old_string matched as a line suffix.
      - otherwise: recover_edit (map onto the real region, re-indent, keep hidden lines).
    """
    if not old_string or old_string == new_string:
        return None
    if original.count(old_string) != 1:
        if old_string in original:
            return None  # ambiguous exact match — the client will say so itself
        return recover_edit(old_string, new_string, original)
    idx = original.find(old_string)
    new2 = new_string
    # The summary dedented this slice: old_string matched only as the tail of a deeper-
    # indented line, so its extra lines would land too shallow. Shift them.
    prefix = original[original.rfind("\n", 0, idx) + 1:idx]
    first = old_string.split("\n", 1)[0]
    if (prefix and not prefix.strip() and "\t" not in prefix and "\n" in new_string
            and first[:1] == " " and views
            and any(ln.rstrip() == first.rstrip() for v in views for ln in v.split("\n"))):
        shifted = _reindent(new_string, len(prefix), skip_first=True)
        if shifted is not None:
            new2 = shifted
    dropped = drop_shadowed_docstring(old_string, new2, original[idx + len(old_string):], views)
    if dropped is not None:
        new2 = dropped
    return None if new2 == new_string else (old_string, new2)
