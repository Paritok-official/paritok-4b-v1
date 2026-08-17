"""Recompute unified-diff hunk headers from the actual body (equivalent to
`git apply --recount`), and ensure a trailing newline. LLM-authored diffs
routinely mis-count the `@@ -a,b +c,d @@` line counts, which a strict `patch`
rejects as malformed; this fixes ONLY that metadata — never a line of the fix.
"""
from __future__ import annotations

import re

_HUNK = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$')


def recount(patch: str) -> str:
    if not patch:
        return patch
    lines = patch.split('\n')
    trailing = patch.endswith('\n')
    if trailing and lines and lines[-1] == '':
        lines = lines[:-1]

    out, i, n = [], 0, len(lines)
    while i < n:
        m = _HUNK.match(lines[i])
        if not m:
            out.append(lines[i]); i += 1; continue
        old_start, _oc, new_start, _nc, tail = m.groups()
        body, j = [], i + 1
        while j < n:
            bl = lines[j]
            if bl.startswith('@@ ') or bl.startswith('--- ') or bl.startswith('diff --git') or bl.startswith('Index: '):
                break
            body.append(bl); j += 1
        old_c = new_c = 0
        fixed = []
        for bl in body:
            if bl.startswith('\\'):        # "\ No newline at end of file"
                fixed.append(bl); continue
            if bl == '':                   # blank context line missing its space
                bl = ' '
            c = bl[0]
            if c == ' ':
                old_c += 1; new_c += 1
            elif c == '-':
                old_c += 1
            elif c == '+':
                new_c += 1
            else:                          # stray line -> treat as context
                bl = ' ' + bl; old_c += 1; new_c += 1
            fixed.append(bl)
        out.append(f'@@ -{old_start},{old_c} +{new_start},{new_c} @@{tail}')
        out.extend(fixed)
        i = j

    result = '\n'.join(out)
    if not result.endswith('\n'):
        result += '\n'
    return result
