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


def match_region(old_string: str, original: str) -> str | None:
    """Return the exact substring of `original` whose canonical form equals
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
    return original[omap[st]:omap[en] + 1]


def _reinsert_docstrings(recovered: str, result: str) -> str:
    """Put back any docstring that lived in `recovered` but was dropped from `result`
    (its location fell inside the model's changed span). Insert after the nearest
    preceding signature-closing `:` line, matching the following line's indent."""
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


def recover_edit(old_string: str, new_string: str, original: str) -> tuple[str, str] | None:
    """Map a model-authored (old_string -> new_string) edit onto the true original.

    Returns (client_old, client_new): client_old is the exact original region (matches
    the file), client_new is that region with the model's change applied, preserving the
    docstrings / comments / blank lines / multi-line formatting the summary had dropped.
    Returns None when the region can't be uniquely located (caller falls back to expand)."""
    recovered = match_region(old_string, original)
    if recovered is None:
        return None

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
    result = _reinsert_docstrings(recovered, result)
    return recovered, result
