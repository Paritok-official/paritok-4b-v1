"""Deterministic post-processor for model-emitted compressed SEG bodies.

Handles the lint/format-level rules that LLMs follow inconsistently:
  - strip stdlib + test framework imports
  - strip orphan trailing decorators (e.g. `@value.setter` at last line)
  - strip "Here's the result of running `cat -n` on /path/file.py:" framing
  - strip per-line number prefixes ("  280  ")
  - collapse multiple blank lines to one (or zero between top-level defs)
  - strip trailing whitespace
  - chain same-RHS assignments: `x = 0\n y = 0` → `x = y = 0`
  - collapse trivially-foldable multi-line calls

Pure-text passes — no AST parsing, no execution. Idempotent.

Public API:
  lint(body: str | None) -> str | None
      None → None  (drop signal preserved)
      ""   → ""    (also preserved)
"""
from __future__ import annotations
import re
from typing import Optional

# ---- stdlib import filter ------------------------------------------------

STDLIB_PREFIXES = (
    "math", "os", "sys", "time", "json", "re", "typing", "collections",
    "itertools", "functools", "logging", "abc", "copy", "io", "pathlib",
    "subprocess", "warnings", "datetime", "string", "tempfile", "shutil",
    "argparse", "enum", "dataclasses", "weakref", "inspect", "hashlib",
    "base64", "uuid", "random", "struct", "threading", "queue", "asyncio",
)
TEST_FW_PREFIXES = ("pytest", "unittest", "mock", "socket")


def _is_stdlib_import(line: str) -> bool:
    """Return True for stdlib / test-framework / __future__ imports."""
    s = line.strip()
    if not s:
        return False
    # `from X import ...`
    m = re.match(r"^from\s+([\w.]+)\s+import\s", s)
    if m:
        root = m.group(1).split(".")[0]
        if root == "__future__":
            return True
        if root in STDLIB_PREFIXES + TEST_FW_PREFIXES:
            return True
        # `from collections.abc import ...` covered by collections root.
        return False
    # `import X` / `import X, Y` / `import X as Y`
    m = re.match(r"^import\s+(.+)$", s)
    if m:
        # comma-separated: drop only if EVERY module is stdlib
        mods = [x.strip().split(" as ")[0].split(".")[0] for x in m.group(1).split(",")]
        return all(m_ in STDLIB_PREFIXES + TEST_FW_PREFIXES for m_ in mods)
    return False


# ---- framing / line-number prefix ---------------------------------------

CAT_N_FRAMING = re.compile(
    r"^Here'?s the result of running `cat -n` on (?P<path>\S+):\s*$"
)
LINE_NUM_PREFIX = re.compile(r"^(\s*)(\d+)\t(.*)$")
# Also handle pad-with-spaces form ("   280  content")
LINE_NUM_PREFIX_SPACE = re.compile(r"^\s*(\d{1,6})\s{2,}(.*)$")


def _strip_framing_and_lineno(lines: list[str]) -> list[str]:
    out = []
    for ln in lines:
        m = CAT_N_FRAMING.match(ln)
        if m:
            base = m.group("path").rsplit("/", 1)[-1]
            out.append(f"[file: {base}]")
            continue
        m = LINE_NUM_PREFIX.match(ln)
        if m:
            indent, _, rest = m.groups()
            out.append(indent + rest)
            continue
        m = LINE_NUM_PREFIX_SPACE.match(ln)
        if m:
            num_str, rest = m.groups()
            # only strip if num is plausibly a source line number (≤ 99999)
            # and the rest does not begin with another number (avoid eating
            # genuine numeric content like "   123    actual = 5")
            if int(num_str) <= 99999 and not re.match(r"^\d", rest):
                # preserve original indentation of the rest by computing
                # the offset of `rest` in the original line.
                idx = ln.find(rest)
                if idx > 0:
                    out.append(" " * (idx - len(num_str) - 2) + rest if idx - len(num_str) - 2 >= 0 else rest)
                else:
                    out.append(rest)
                continue
        out.append(ln)
    return out


# ---- orphan trailing decorator ------------------------------------------

DECORATOR_RE = re.compile(r"^\s*@\w[\w.]*(\(.*\))?\s*$")


def _strip_orphan_trailing_decorator(lines: list[str]) -> list[str]:
    """If the LAST non-empty line is a bare decorator (next SEG continues),
    drop it — it has no associated def in this SEG."""
    if not lines:
        return lines
    # Find last non-empty line
    i = len(lines) - 1
    while i >= 0 and not lines[i].strip():
        i -= 1
    if i < 0:
        return lines
    if DECORATOR_RE.match(lines[i]):
        # Drop the orphan decorator and any preceding blank line.
        return lines[:i]
    return lines


# ---- chain same-RHS assignments -----------------------------------------

ASSIGN_RE = re.compile(r"^(\s*)([\w\.\[\]]+)\s*=\s*(.+?)\s*$")


def _chain_same_rhs(lines: list[str]) -> list[str]:
    """`x = 0` / `y = 0` (same indent + same RHS) → `x = y = 0`. Cross-line."""
    out: list[str] = []
    i = 0
    while i < len(lines):
        m = ASSIGN_RE.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        indent, lhs0, rhs0 = m.groups()
        # Skip non-simple RHS to avoid joining e.g. function calls
        if any(ch in rhs0 for ch in "(["):
            out.append(lines[i])
            i += 1
            continue
        names = [lhs0]
        j = i + 1
        while j < len(lines):
            m2 = ASSIGN_RE.match(lines[j])
            if not m2:
                break
            indent2, lhs2, rhs2 = m2.groups()
            if indent2 == indent and rhs2 == rhs0 and not any(ch in rhs2 for ch in "(["):
                names.append(lhs2)
                j += 1
            else:
                break
        if len(names) >= 2:
            out.append(f"{indent}{' = '.join(names)} = {rhs0}")
        else:
            out.append(lines[i])
        i = j if len(names) >= 2 else i + 1
    return out


# ---- collapse multi-line call (small fits) ------------------------------

OPEN_CALL_RE = re.compile(r"^(\s*)(.+?\w)\(\s*$")


def _collapse_multiline_calls(lines: list[str], max_width: int = 140) -> list[str]:
    """`f(\n  a,\n  b,\n)<tail>` → `f(a, b)<tail>` when total fits in max_width.

    Handles:
      - `def fn(\n  arg,\n  arg,\n) -> T:`   (preserves `-> T:` suffix)
      - `raise X(\n  "foo".format(...)\n)`  (single-arg)
      - `result = f(\n  a,\n  b,\n)`          (assignment LHS)
      - `f(\n  a,\n  b,\n).method()`          (preserves chained call)
    Skips collapse if any inner line has a comment.
    """
    out: list[str] = []
    i = 0
    while i < len(lines):
        m = OPEN_CALL_RE.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        indent, head = m.groups()
        # gather until matching `)`. Track paren depth across (, [, {.
        depth_paren = 1
        depth_other = 0
        args: list[str] = []
        trailing = ""
        j = i + 1
        ok = True
        had_comment = False
        while j < len(lines):
            ln = lines[j]
            # comment with # is hard to inline safely
            if "#" in ln and not ln.lstrip().startswith("#"):
                had_comment = True
            stripped = ln.strip()
            # Update bracket depths (rough — adequate for typical code)
            for ch in ln:
                if ch == "(":
                    depth_paren += 1
                elif ch == ")":
                    depth_paren -= 1
                elif ch in "[{":
                    depth_other += 1
                elif ch in "]}":
                    depth_other -= 1
            if depth_paren == 0:
                # Closing `)` line; capture content before final `)`.
                # Split at the last `)` — content before goes into args,
                # content after (including `:`, `-> T`, `.x`, etc.) goes
                # to trailing.
                close_idx = ln.rfind(")")
                pre = ln[:close_idx]
                trailing = ln[close_idx + 1 :].rstrip()
                pre_stripped = pre.strip().rstrip(",").strip()
                if pre_stripped:
                    args.append(pre_stripped)
                j += 1
                break
            else:
                args.append(stripped.rstrip(","))
                j += 1
        else:
            ok = False
        if not ok or depth_paren != 0 or had_comment:
            out.append(lines[i])
            i += 1
            continue
        # Filter empty fragments
        non_empty = [a for a in args if a]
        joined = f"{indent}{head}({', '.join(non_empty)}){trailing}"
        if len(joined) <= max_width:
            out.append(joined)
            i = j
        else:
            out.append(lines[i])
            i += 1
    return out


# ---- blank line / whitespace ---------------------------------------------

DEF_OR_CLASS_RE = re.compile(r"^\s*(def |async def |class |@\w)")


def _collapse_blank_lines(lines: list[str]) -> list[str]:
    """Drop all blank lines; allow at most ONE blank line directly between
    a closing line and the next top-level def/class (none between adjacent
    defs)."""
    out: list[str] = []
    for ln in lines:
        if not ln.strip():
            # skip blank lines entirely; the structure is implied by indent
            continue
        out.append(ln.rstrip())
    return out


# ---- pipeline ------------------------------------------------------------

def lint(body: Optional[str]) -> Optional[str]:
    """Apply the deterministic lint pipeline. None / "" pass through."""
    if body is None:
        return None
    if not body.strip():
        return body

    lines = body.split("\n")
    lines = _strip_framing_and_lineno(lines)
    # Drop stdlib/test-framework imports
    lines = [ln for ln in lines if not _is_stdlib_import(ln)]
    # Orphan decorator at end
    lines = _strip_orphan_trailing_decorator(lines)
    # Collapse multi-line calls (one pass; conservative)
    lines = _collapse_multiline_calls(lines)
    # Chain same-RHS assignments
    lines = _chain_same_rhs(lines)
    # Whitespace lint
    lines = _collapse_blank_lines(lines)

    out = "\n".join(lines)
    return out


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("usage: python lint_compressed.py <gt_vN_samples.jsonl>")
        sys.exit(1)
    src = sys.argv[1]
    entries = [json.loads(l) for l in open(src, "r", encoding="utf-8") if l.strip()]
    changed = 0
    for e in entries:
        before = e.get("v_compressed")
        after = lint(before)
        if after != before:
            changed += 1
        for prefix in ("v5", "v"):
            e[f"{prefix}_compressed"] = after
            e[f"{prefix}_chars"] = len(after) if after else 0
            e[f"{prefix}_ratio"] = (
                round(e[f"{prefix}_chars"] / e["seg_original_chars"], 3)
                if e["seg_original_chars"] else 0.0
            )
            e[f"{prefix}_dropped"] = after is None or not (after or "").strip()
    out = src.replace(".jsonl", "_linted.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"linted {changed}/{len(entries)} entries → {out}")
