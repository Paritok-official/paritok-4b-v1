"""Coding agent: given a GitHub issue and (baseline or compressed) source context,
ask a frontier model for a unified-diff fix. One shot, no tools — the same call for
both arms, so the only variable is whether the context was compressed.

Model is configurable (``--agent-model`` / ``PARITOK_EVAL_AGENT_MODEL``); the key
is read from ``ANTHROPIC_API_KEY``.
"""
from __future__ import annotations

import os
import re
import time

# ~150k tokens. Deliberately large so the BASELINE (uncompressed) arm is not
# truncated more than the compressed arm — truncating the bigger baseline would
# handicap it and inflate the quality-retained ratio. Fits a 200k-context agent
# with room for the reply. Override with generate_patch(max_chars=...).
MAX_CONTEXT_CHARS = 600_000

_SYSTEM = (
    "You are an expert software engineer fixing a bug in an open-source Python project. "
    "You will be shown the relevant source files and a GitHub issue describing the bug. "
    "Produce a minimal unified diff that resolves the issue. You may reason briefly before "
    "the patch, but you MUST end your reply with a single fenced code block tagged ```diff "
    "containing the full patch in `git apply` format (with `--- a/<path>` and `+++ b/<path>` "
    "headers and `@@` hunks). If the issue cannot be fixed with the information provided, still "
    "emit a best-effort ```diff block — never refuse, never return an empty block."
)


def _extract_patch(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"```(?:diff|patch)\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    for m in re.finditer(r"```[a-zA-Z]*\s*\n(.*?)```", text, re.DOTALL):
        body = m.group(1)
        if "--- a/" in body or body.lstrip().startswith(("---", "diff --git")):
            return body.strip()
    m = re.search(r"(^|\n)(diff --git .+|--- [^\n]+\n\+\+\+ [^\n]+)", text)
    if m:
        return text[m.start():].strip().strip("`").strip()
    return text.strip()


def _client():
    import anthropic
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def generate_patch(client, model: str, context: str, problem: str) -> str:
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n\n[... context truncated ...]\n"
    user = (
        f"## GitHub Issue\n{problem}\n\n## Source Code\n```\n{context}\n```\n\n"
        "Return your fix as a unified diff inside a single ```diff code block at the end. "
        "Keep the patch minimal and focused on the bug."
    )
    for attempt in range(4):
        try:
            r = client.messages.create(
                model=model, max_tokens=4096, temperature=0, system=_SYSTEM,
                messages=[{"role": "user", "content": user}],
            )
            text = "\n".join(
                b.text for b in r.content
                if getattr(b, "type", None) == "text" and getattr(b, "text", None)
            ).strip()
            patch = _extract_patch(text)
            if patch:
                return patch
        except Exception:
            if attempt == 3:
                return ""
            time.sleep(2 ** attempt * 3)
    return ""
