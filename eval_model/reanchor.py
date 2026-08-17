"""Re-anchor an LLM-authored unified-diff patch onto the true source at
base_commit, using Paritok's OWN edit_recovery logic — the same recovery the
gateway applies to every Edit tool call in production. Running it here means the
eval measures the shipped gateway (compress + recover), not a compressor in
isolation. It does the real work on the compressed arm (undo reflow); the baseline
arm runs the same pass only for fairness — a no-op for recovery, it just re-emits a
clean git-appliable diff — so the only variable stays the compression.

Production intercepts Edit tool calls (old_string / new_string). A one-shot
SWE-bench answer is a unified diff instead, so this module supplies the one
eval-specific piece: decompose the diff into change blocks and feed each to the
recovery core. The canonical form + graft are imported from
``paritok.pipelines.edit_recovery`` — NOT re-implemented.

Two enhancements the eval exercises (candidates for upstreaming into
edit_recovery, so production benefits too):
  * a whitespace-only fallback matcher (``_canon_ws_map``) that keeps the
    docstring / comment text the docstring-dropping canonical erases, so an edit
    whose anchor lives inside a docstring still re-anchors;
  * growing the anchor with neighbouring context, then a nearest-line tiebreak,
    to disambiguate a block that matches in more than one place.
Plus ``fix_truncated_paths`` for the known compression bug where the ``# File:``
header is shortened to a basename and the model copies the short path into the
diff (git-apply then can't find the file).
"""
from __future__ import annotations

import difflib
import re

# Reuse the shipped recovery core (canonical map + comment regex + docstring
# reinsertion). We do NOT duplicate the matching logic; we only add a
# diff→edit adapter and the two fallbacks noted above.
from paritok.pipelines.edit_recovery import (
    _canon_with_map,
    _LINE_COMMENT,
    _reinsert_docstrings,
)

_WS = re.compile(r"\s+")
_FILE_A = re.compile(r"^--- a/(.+)$")
_HUNK = re.compile(r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@")
_PATCH_LINENO = re.compile(r"^([ +-])[ \t]*\d+\t")


def strip_patch_line_numbers(patch: str) -> str:
    """Drop the Read '<n>\\t' prefix the model copies from a line-numbered context
    into diff body lines (the +/-/space marker is preserved), so the hunk content
    matches the true (unnumbered) source before re-anchoring. A no-op on diffs that
    carry no line numbers (the marker + digits + TAB shape rarely occurs in code)."""
    out = []
    for ln in patch.split("\n"):
        m = _PATCH_LINENO.match(ln)
        out.append((ln[0] + ln[m.end():]) if m else ln)
    return "\n".join(out)


def _canon_ws_map(s: str):
    """Whitespace-only canonical: keeps docstrings/comments/strings that the
    docstring-dropping ``_canon_with_map`` removes. Because it is a per-char
    filter, a window's canonical is always a substring of the whole file's, so it
    natively handles reflow (a multi-line construct folded onto one line)."""
    out, omap = [], []
    for i, ch in enumerate(s):
        if not ch.isspace():
            out.append(ch)
            omap.append(i)
    return "".join(out), omap


def _graft(recovered: str, old_string: str, new_string: str, mapfn=_canon_with_map):
    """Apply the model's change onto the exact original region `recovered`, using the
    SAME canonical map that located it (`mapfn`).

    This is the graft half of ``edit_recovery.recover_edit`` (that function bundles
    match_region + graft; the eval supplies its own enhanced locate above, then reuses
    the identical graft here). Passing `mapfn` keeps locate and graft consistent: a
    span found only by the whitespace-only matcher must NOT be re-canonicalized with the
    docstring-dropping canon — that empties the offset map (`omap`) for a docstring
    anchor and would index out of range. Returns None if the boundaries can't be placed
    (caller then drops the whole patch to the recount fallback)."""
    p = 0
    while p < len(old_string) and p < len(new_string) and old_string[p] == new_string[p]:
        p += 1
    s = 0
    while s < len(old_string) - p and s < len(new_string) - p and old_string[-1 - s] == new_string[-1 - s]:
        s += 1
    new_mid = new_string[p:len(new_string) - s]
    cr, omap = mapfn(recovered)
    cpre, _ = mapfn(old_string[:p])
    csuf, _ = mapfn(old_string[len(old_string) - s:] if s > 0 else "")
    suf_idx = len(cr) - len(csuf)
    if (cpre and len(cpre) - 1 >= len(omap)) or (csuf and not (0 <= suf_idx < len(omap))):
        return None                                          # offset map inconsistent -> can't place
    pre_byte = omap[len(cpre) - 1] if cpre else -1
    suf_byte = omap[suf_idx] if csuf else len(recovered)
    # Comment/docstring reclaim only applies to the docstring-dropping canon; under the
    # whitespace-only matcher nothing was dropped, so those spans are already verbatim.
    docstring_drop = mapfn is _canon_with_map
    cidx: set[int] = set()
    if docstring_drop:
        for m in _LINE_COMMENT.finditer(recovered):
            cidx.update(range(m.start(), m.end()))
    if not new_mid[:1].isspace():
        while pre_byte + 1 < suf_byte and recovered[pre_byte + 1].isspace():
            pre_byte += 1
    if not new_mid[-1:].isspace():
        while suf_byte > pre_byte + 1 and (recovered[suf_byte - 1].isspace() or (suf_byte - 1) in cidx):
            suf_byte -= 1
    result = recovered[:pre_byte + 1] + new_mid + recovered[suf_byte:]
    return _reinsert_docstrings(recovered, result) if docstring_drop else result


def _spans(loc_lines: list[str], text: str, mapfn):
    cq, _ = mapfn("\n".join(loc_lines))
    ct, omap = mapfn(text)
    if not cq:
        return []
    out, i = [], ct.find(cq)
    while i != -1:
        out.append((omap[i], omap[i + len(cq) - 1] + 1))
        i = ct.find(cq, i + 1)
    return out


def _try_block(pre, rem, add, post, patched, a):
    """Locate the original byte-span this change block belongs to. Try the minimal
    anchor first (the removed lines); if ambiguous, grow with neighbouring context;
    if still ambiguous, take the occurrence nearest the hunk's `@@ -a` line. Each
    matcher is tried under the docstring-dropping canonical then the whitespace-only
    one. Returns ((start, end), old_str, new_str) or None."""
    cands = []
    if rem:
        cands.append((rem, rem, add))
        if pre is not None:
            cands.append(([pre] + rem, [pre] + rem, [pre] + add))
        if post is not None:
            cands.append((rem + [post], rem + [post], add + [post]))
        if pre is not None and post is not None:
            cands.append(([pre] + rem + [post], [pre] + rem + [post], [pre] + add + [post]))
    else:  # pure insertion: anchor on the surrounding context
        if pre is not None and post is not None:
            cands.append(([pre, post], [pre, post], [pre] + add + [post]))
        if pre is not None:
            cands.append(([pre], [pre], [pre] + add))
        if post is not None:
            cands.append(([post], [post], add + [post]))
    if not cands:
        return None
    for loc, o, n in cands:                                     # first UNIQUE (minimal first)
        for mapfn in (_canon_with_map, _canon_ws_map):
            sp = _spans(loc, patched, mapfn)
            if len(sp) == 1:
                return sp[0], "\n".join(o), "\n".join(n), mapfn
    loc, o, n = cands[0]                                        # nearest-line tiebreak on the minimal anchor
    for mapfn in (_canon_with_map, _canon_ws_map):
        sp = _spans(loc, patched, mapfn)
        if len(sp) > 1:
            best = min(sp, key=lambda s: abs(patched.count("\n", 0, s[0]) + 1 - a))
            return best, "\n".join(o), "\n".join(n), mapfn
    return None


def _parse(patch: str) -> dict[str, list]:
    lines = patch.split("\n")
    i, path, files = 0, None, {}
    while i < len(lines):
        m = _FILE_A.match(lines[i])
        if m:
            path = m.group(1)
            files.setdefault(path, [])
            i += 1
            continue
        m = _HUNK.match(lines[i])
        if m and path is not None:
            a = int(m.group(1))
            j = i + 1
            body = []
            while j < len(lines) and not lines[j].startswith("@@") and not lines[j].startswith("--- "):
                if lines[j][:1] in (" ", "-", "+"):
                    body.append(lines[j])
                elif lines[j][:1] == "\\":
                    pass
                else:
                    break
                j += 1
            files[path].append((a, body))
            i = j
            continue
        i += 1
    return files


def _blocks(body: list[str], a: int) -> list:
    out, i, n = [], 0, len(body)
    while i < n:
        if body[i][:1] in ("-", "+"):
            j, rem, add = i, [], []
            while j < n and body[j][:1] in ("-", "+"):
                (rem if body[j][0] == "-" else add).append(body[j][1:])
                j += 1
            pre = body[i - 1][1:] if i - 1 >= 0 and body[i - 1][:1] == " " else None
            post = body[j][1:] if j < n and body[j][:1] == " " else None
            out.append((a, pre, rem, add, post))
            i = j
        else:
            i += 1
    return out


def fix_truncated_paths(patch: str, full_context: str, fetch, repo: str, commit: str) -> str:
    """Restore a full repo-relative path onto any `--- a/<basename>` header the
    compressor truncated, using the uncompressed full_context's `# File:` headers."""
    # Map a truncated basename back to its full path ONLY when unambiguous. If two
    # context files share a basename (e.g. two `models.py`), don't guess — leave the
    # path truncated so the fetch fails and the patch drops to recount, rather than
    # silently re-anchoring the hunk against the wrong file.
    by_base: dict[str, set] = {}
    for fp in re.findall(r"^# File: (.+)$", full_context or "", re.M):
        by_base.setdefault(fp.strip().split("/")[-1], set()).add(fp.strip())
    bmap = {b: next(iter(s)) for b, s in by_base.items() if len(s) == 1}

    def repl(m):
        pre, x = m.group(1), m.group(2).strip()
        if fetch(repo, commit, x) is not None:      # already a valid full path
            return pre + x
        return pre + bmap.get(x.split("/")[-1], x)

    patch = re.sub(r"^(--- a/)(.+)$", repl, patch, flags=re.M)
    patch = re.sub(r"^(\+\+\+ b/)(.+)$", repl, patch, flags=re.M)
    return patch


def reanchor(patch: str, repo: str, commit: str, fetch, full_context: str | None = None,
             strip_line_numbers: bool = False) -> str | None:
    """Re-anchor `patch` onto the true source at base_commit and return a clean
    unified diff that is guaranteed to apply (it is a difflib diff of the real file
    vs the patched file). Returns None if any change block can't be uniquely placed
    or a target file can't be fetched — the caller then keeps the original patch.

    `fetch(repo, commit, path) -> str | None` supplies the true file bytes (e.g.
    ``eval_model.dataset._fetch``); wrap it in a cache for a full run.
    """
    if not patch:
        return None
    if strip_line_numbers:                       # --line-numbers regime: undo the Read <n>\t
        patch = strip_patch_line_numbers(patch)
    if full_context:
        patch = fix_truncated_paths(patch, full_context, fetch, repo, commit)
    try:
        out = []
        for path, hunks in _parse(patch).items():
            real = fetch(repo, commit, path)
            if real is None:
                return None
            patched = real
            blocks = []
            for a, body in hunks:
                blocks += _blocks(body, a)
            for a, pre, rem, add, post in blocks:
                r = _try_block(pre, rem, add, post, patched, a)
                if r is None:
                    return None
                (s0, s1), old_s, new_s, mapfn = r
                grafted = _graft(patched[s0:s1], old_s, new_s, mapfn)
                if grafted is None:
                    return None
                patched = patched[:s0] + grafted + patched[s1:]
            if patched != real:
                out.append("\n".join(difflib.unified_diff(
                    real.split("\n"), patched.split("\n"),
                    fromfile="a/" + path, tofile="b/" + path, lineterm="")))
        return ("\n".join(out) + "\n") if out else None
    except Exception:
        return None
