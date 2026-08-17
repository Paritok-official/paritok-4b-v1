"""Pull SWE-bench Lite straight from the source (Hugging Face) and build the
oracle file context for each instance — nothing about the dataset is vendored in
this repo.

Oracle context = the exact source files the gold patch touches, fetched at the
instance's base_commit from GitHub raw. That is the same context both the
baseline arm and the compressed arm receive; the only difference downstream is
whether it is compressed first.
"""
from __future__ import annotations

import re
import time
import urllib.request
import urllib.error

_DIFF_GIT = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.M)
_MINUS = re.compile(r"^--- a/(.+)$", re.M)
_FILE_HDR = re.compile(r"(?m)^# File: (.+)$")


def line_number_context(full_context: str) -> str:
    """`cat -n` each file body ('<6d>\\t<line>', which paritok's edit_recovery
    strip regex understands), leaving the '# File:' headers unnumbered. This
    simulates a real coding agent's Read output (line-numbered) — the compression
    model's in-distribution input — vs the raw source used by default.
    """
    parts = _FILE_HDR.split(full_context)
    out = []
    for i in range(1, len(parts), 2):
        path = parts[i].strip()
        body = parts[i + 1].lstrip("\n").rstrip("\n")
        numbered = "\n".join(f"{n:6d}\t{ln}" for n, ln in enumerate(body.split("\n"), 1))
        out.append(f"# File: {path}\n{numbered}")
    return "\n\n".join(out)


def _modified_files(patch: str) -> list[str]:
    """File paths a unified diff touches (deduped, order-preserved)."""
    files: list[str] = []
    for a, _b in _DIFF_GIT.findall(patch or ""):
        if a not in files:
            files.append(a)
    if not files:  # fall back to `--- a/...` headers
        for a in _MINUS.findall(patch or ""):
            if a != "/dev/null" and a not in files:
                files.append(a)
    return files


def _fetch(repo: str, commit: str, path: str, retries: int = 3) -> str | None:
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "paritok-eval"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    return None


def build_oracle_context(record: dict) -> tuple[str, list[str]]:
    """Return (full_context, files) for one SWE-bench record.

    full_context is `# File: <path>\n<content>` blocks concatenated — the same
    framing the compression model and the agent are given.
    """
    repo = record["repo"]
    commit = record["base_commit"]
    files = _modified_files(record.get("patch", ""))
    parts = []
    got = []
    for path in files:
        content = _fetch(repo, commit, path)
        if content is not None:
            parts.append(f"# File: {path}\n{content}\n")
            got.append(path)
    return "\n".join(parts), got


def load_instances(n: int = 300):
    """Yield SWE-bench Lite records (from Hugging Face) with oracle context added.

    Requires `datasets` (pip install datasets). The dataset itself is pulled from
    princeton-nlp/SWE-bench_Lite — never stored in this repo.
    """
    from datasets import load_dataset

    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    for i, record in enumerate(ds):
        if i >= n:
            break
        rec = dict(record)
        rec["full_context"], rec["context_files"] = build_oracle_context(rec)
        yield rec
