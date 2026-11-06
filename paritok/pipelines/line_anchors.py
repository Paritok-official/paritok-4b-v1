"""Line-range anchors on the definitions a compressed view keeps.

The 4B drops line numbers, so an agent holding a summary can't tell where a function
lives or that its body was cut, and spends turns on `grep -n` / re-reads to find out.
After compression the gateway (which holds the line-numbered original) appends to every
kept def/class/function header the range that block covers in the REAL file:

    def collect(self, objs, source=None):  # L184-262 (partial)

  - `L184-262`  the whole block in the original, however the body was cut;
  - `+`         the block runs past the end of what the agent read (`L57-80+`);
  - `(partial)` lines inside the block are missing from the view.

Exact, not guessed: headers are mapped back to their original line by text, spans come
from indentation (Python, YAML-like) or brace matching (C-like languages). Costs ~5% of
the compressed view. `strip_anchors` removes them from anything the agent writes back.
"""

from __future__ import annotations

import re

_NUMBERED = re.compile(r"^[ \t]*(\d+)(?:\t|→)(.*)$")
_PY_HEADER = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+\w+")
_BRACE_HEADER = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:(?:public|private|protected|internal|static|final|"
    r"abstract|async|override|virtual|sealed|open|suspend|inline)\s+)*"
    r"(?:function\b|class\b|interface\b|struct\b|enum\b|impl\b|trait\b|fn\b|func\b|"
    r"[\w<>\[\],.?]+\s+\w+\s*\([^;]*$)")
_BRACE_EXT = (".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".go", ".rs", ".c", ".cc", ".cpp",
              ".h", ".hpp", ".cs", ".php", ".swift", ".scala", ".dart")
ANCHOR_RE = re.compile(r"[ \t]+(?:#|//) L\d+-\d+\+?(?: \(partial\))?[ \t]*$", re.M)


def strip_anchors(text: str) -> str:
    """Remove anchors from text the agent writes back (an Edit / Write it copied from
    the view), so they never land in the user's file."""
    return ANCHOR_RE.sub("", text) if isinstance(text, str) else text


def _numbered(original: str) -> dict[int, str]:
    lines = {}
    for ln in original.splitlines():
        m = _NUMBERED.match(ln)
        if m:
            lines[int(m.group(1))] = m.group(2)
    return lines


def _indent(s: str) -> int:
    return len(s) - len(s.lstrip())


def _indent_end(orig: dict[int, str], start: int) -> tuple[int, bool]:
    """(last non-blank line of the indented block opened at `start`, closed?). Closed only
    when a following line at the header's indentation or left of it was actually read —
    a read that just runs out (even on blank lines) leaves the block open."""
    ind = _indent(orig[start])
    end = start
    for n in sorted(k for k in orig if k > start):
        s = orig[n]
        if s.strip() and _indent(s) <= ind and not s.lstrip().startswith((")", "]", "}")):
            return end, True
        if s.strip():
            end = n
    return end, False


def _brace_end(orig: dict[int, str], start: int) -> tuple[int, bool] | None:
    """(line where the brace block opened at/after `start` closes, closed?) — strings and
    // /* */ comments skipped; closed=False when the read ends inside the block. None when
    no block opens within a few lines (a declaration)."""
    depth, opened, in_block, quote = 0, False, False, None
    for n in sorted(k for k in orig if k >= start):
        s, i = orig[n], 0
        while i < len(s):
            c, nxt = s[i], s[i + 1:i + 2]
            if in_block:
                if c == "*" and nxt == "/":
                    in_block, i = False, i + 1
            elif quote:
                if c == "\\":
                    i += 1
                elif c == quote:
                    quote = None
            elif c == "/" and nxt == "/":
                break
            elif c == "/" and nxt == "*":
                in_block, i = True, i + 1
            elif c in "\"'`":
                quote = c
            elif c == "{":
                depth, opened = depth + 1, True
            elif c == "}":
                depth -= 1
                if opened and depth == 0:
                    return n, True
            i += 1
        if not opened and n - start > 3:
            return None
    return (max(orig), False) if opened else None


def _map_lines(lines: list[str], orig: dict[int, str], min_len: int = 1) -> list[int | None]:
    """The original line number of each view line: the next original line, in order, with
    the same stripped text. None for blank, rewritten or placeholder lines, and for lines
    shorter than `min_len` (which are not matched at all)."""
    hits: list[int | None] = []
    j, hi = min(orig), max(orig)
    for v in lines:
        s = v.strip()
        if len(s) < min_len:
            hits.append(None)
            continue
        hit = next((n for n in range(j, hi + 1) if n in orig and orig[n].strip() == s), None)
        hits.append(hit)
        if hit is not None:
            j = hit + 1
    return hits


def restore_indent(view: str, original: str) -> str:
    """Give the lines a compressed view keeps their indentation in the real file.

    The 4B shifts a slice read from the middle of a file to column 0 (61% of mid-file
    reads in loopbench, by the slice's own indent; learned from distillation targets that
    dedent 40% of them). An agent editing through a script copies that indentation and
    misses the file. Matched lines take the original's leading whitespace; rewritten
    lines, and short ones (`)`, `else:`, `pass` match anywhere), move with the nearest
    matched line. Costs ~0.1% tokens: runs of spaces tokenize together."""
    orig = _numbered(original)
    if len(orig) < 3 or not view:
        return view
    lines = view.split("\n")
    hits = _map_lines(lines, orig, min_len=8)
    # A line the in-order pass skipped (the summary reordered it) still has one right
    # answer when its text occurs once in the original.
    seen: dict[str, int | None] = {}
    for n, t in orig.items():
        s = t.strip()
        seen[s] = None if s in seen else n
    for i, v in enumerate(lines):
        s = v.strip()
        if hits[i] is None and len(s) >= 8 and seen.get(s) is not None:
            hits[i] = seen[s]
    shift: list[int | None] = [None] * len(lines)
    out = list(lines)
    for i, (v, n) in enumerate(zip(lines, hits)):
        if n is not None:
            o = orig[n]
            shift[i] = _indent(o) - _indent(v)
            out[i] = o[:_indent(o)] + v.lstrip()
    if all(s is None for s in shift):
        return view
    near: list[int | None] = [None] * len(lines)
    last = None
    for i, s in enumerate(shift):
        last = s if s is not None else last
        near[i] = last
    last = None
    for i in range(len(lines) - 1, -1, -1):
        last = shift[i] if shift[i] is not None else last
        if near[i] is None:
            near[i] = last
    for i, v in enumerate(lines):
        d = near[i]
        if shift[i] is not None or not v.strip() or not d:
            continue
        if d > 0:
            out[i] = " " * d + v
        else:
            cut = min(-d, _indent(v))
            out[i] = v[cut:]
    return "\n".join(out)


def annotate(view: str, original: str, source: str | None = None) -> str:
    """Append line-range anchors to the def/class headers kept in `view`, computed from the
    line-numbered `original`. Returns `view` unchanged when it can't be done exactly
    (no line numbers, unknown header kind, header not found)."""
    orig = _numbered(original)
    if len(orig) < 3 or not view:
        return view
    src = (source or "").lower()
    brace = src.endswith(_BRACE_EXT)
    header = _BRACE_HEADER if brace else _PY_HEADER
    comment = "//" if brace else "#"
    out = view.split("\n")
    kept: set[int] = set()
    headers: list[tuple[int, int]] = []  # (view line index, original line)
    for vi, hit in enumerate(_map_lines(out, orig)):
        if hit is None:
            continue
        kept.add(hit)
        if header.match(orig[hit]) and ANCHOR_RE.search(out[vi]) is None:
            headers.append((vi, hit))
    for vi, n in headers:
        if brace:
            found = _brace_end(orig, n)
            if found is None:
                continue
            end, closed = found
        else:
            end, closed = _indent_end(orig, n)
        if end <= n:
            continue
        body = [k for k in range(n + 1, end + 1) if k in orig and orig[k].strip()]
        partial = any(k not in kept for k in body)
        more = "" if closed else "+"
        out[vi] = f"{out[vi].rstrip()}  {comment} L{n}-{end}{more}" + (" (partial)" if partial else "")
    return "\n".join(out)
