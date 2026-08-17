"""Compress a SWE-bench file context with the Paritok SEG model, direct to a
local Ollama endpoint (no gateway, no hosted server).

This mirrors the production SEG/level path: split the context on `# File:`
markers (one stream per file), chunk each file at class/def boundaries, compress
each chunk as a `[SEG ... kind=file_read level=L1]` block with the shipped
file_read system prompt, re-inject the true `# File:` header, recombine + dedup.

The compression model and prompt come from the installed `paritok` package, so
this stays in lockstep with what the gateway ships.
"""
from __future__ import annotations

import json
import re
import time
import urllib.request

from paritok.token_counter import count_tokens
from paritok.strategies.prompts import system_prompt_for_kind
from paritok.strategies.chunking import (
    split_into_chunks_structural,
    deduplicate_definitions,
)

ENC = "cl100k_base"
_FILE_RE = re.compile(r"(?m)^# File: (.+)$")
_MODEL_FILE_MARKER = re.compile(r"(?m)^\s*\[file:[^\]]*\]\s*")
_SYS_FILE_READ = system_prompt_for_kind("file_read")


def _seg_compress(chunk: str, intent: str, level: str, seg_id: str,
                  ollama_url: str, model: str, num_ctx: int, timeout: float,
                  stats: dict | None = None) -> str:
    if stats is not None:
        stats["chunks"] = stats.get("chunks", 0) + 1
    user = (
        "USER INTENT:\n" + intent.strip() + "\n\n"
        "Compress the following segment under the rules in your system prompt. "
        "Output only the compressed [SEG]...[/SEG] block (or an empty one to drop):\n\n"
        f"[SEG id={seg_id} kind=file_read level={level}]\n{chunk}\n[/SEG]"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYS_FILE_READ},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0, "top_p": 1.0, "num_ctx": num_ctx, "num_predict": 5000},
    }
    data = json.dumps(payload).encode()
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                ollama_url, data=data, headers={"Content-Type": "application/json"})
            raw = json.loads(urllib.request.urlopen(req, timeout=timeout).read())["message"]["content"]
            body = re.sub(r"\[/?SEG[^\]]*\]", "", raw)      # drop the SEG wrapper
            body = _MODEL_FILE_MARKER.sub("", body)          # drop model-emitted [file: x]
            return body.strip()
        except Exception:
            if attempt == 3:
                # Ollama failed after all retries. Pass this chunk through UNCOMPRESSED
                # so one flaky request doesn't abort a long run — but COUNT it: a
                # passed-through chunk is byte-identical to the baseline, so a run with
                # many pass-throughs is not measuring compression and must not be trusted.
                # (run.py prints the pass-through total; check it is ~0 before reading QR.)
                if stats is not None:
                    stats["passthrough"] = stats.get("passthrough", 0) + 1
                return chunk.strip()
            time.sleep(2 ** attempt)


def compress_context(full_context: str, intent: str, *, chunk: int = 3000, level: str = "L1",
                     ollama_url: str = "http://localhost:11434/api/chat",
                     model: str = "paritok-4b-v1:latest", num_ctx: int = 16384,
                     timeout: float = 1800.0, stats: dict | None = None) -> str:
    """Compress a full `# File:`-framed context; returns the compressed context.

    Pass a mutable `stats` dict to accumulate {"chunks", "passthrough"} across a run
    (a chunk that Ollama failed to compress is passed through uncompressed and counted).
    """
    parts = _FILE_RE.split(full_context)
    files = [(parts[i].strip(), parts[i + 1]) for i in range(1, len(parts), 2)]
    if not files:
        files = [(None, full_context)]

    sections = []
    for path, body in files:
        if count_tokens(body, ENC) <= chunk:
            section_body = _seg_compress(body, intent, level, "s1", ollama_url, model, num_ctx, timeout, stats)
        else:
            pieces = []
            for i, (ctext, sl, el, _rt) in enumerate(
                    split_into_chunks_structural(body, chunk_size=chunk, max_single_block=chunk), 1):
                c = _seg_compress(ctext, intent, level, f"s{i}", ollama_url, model, num_ctx, timeout, stats)
                if c:
                    pieces.append(f"# Lines {sl}-{el}:\n{c}")
            section_body = deduplicate_definitions("\n\n".join(pieces))
        header = f"# File: {path}" if path else ""
        sections.append((header + "\n" + section_body).strip())
    return "\n\n".join(sections)
