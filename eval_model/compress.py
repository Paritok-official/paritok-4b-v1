"""Compress a SWE-bench file context with the Paritok SEG model, direct to a
local Ollama endpoint (no gateway, no hosted server).

This mirrors the production SEG/level path: split the context on `# File:`
markers (one stream per file), chunk each file at class/def boundaries, compress
each chunk as a `[SEG ... kind=file_read level=L1]` block with the shipped
file_read system prompt, re-inject the true `# File:` header, recombine + dedup.

The compression model and prompt come from the installed `paritok` package, so
this stays in lockstep with what the gateway ships.

REPRODUCING THE ~0.24 RETENTION LOCALLY needs BOTH of these (both verified — see
_seg_compress and run.py's version check):
  1. Ollama **0.32.1** (the version the RunPod pod pins in gpu-serverless/Dockerfile).
     Newer Ollama (e.g. 0.32.15) silently under-compresses this model to ~0.60.
  2. The **/v1/chat/completions** endpoint (used here), NOT native /api/chat — /api/chat
     reuses the cached system-prompt KV across requests and flips this knife-edge model
     to ~0.60 after the first request.
"""
from __future__ import annotations

import json
import re
import time
import urllib.request

from paritok.token_counter import count_tokens
from paritok.pipelines.compress import _depad_line_numbers
from paritok.strategies.prompts import system_prompt_for_kind
from paritok.strategies.chunking import (
    split_into_chunks_structural,
    deduplicate_definitions,
)

ENC = "cl100k_base"
_FILE_RE = re.compile(r"(?m)^# File: (.+)$")
_MODEL_FILE_MARKER = re.compile(r"(?m)^\s*\[file:[^\]]*\]\s*")
_SYS_FILE_READ = system_prompt_for_kind("file_read")


# Match the gateway's Ollama request (paritok LocalModelStrategy): tokenizer-slack
# and ctx-safety margin used to cap generation so prompt + max_tokens fits num_ctx.
_TOKENIZER_SLACK = 1.15
_CTX_SAFETY_MARGIN = 512
_MAX_OUTPUT = 3000  # == paritok CHUNK_SIZE; the compressed body is always < input


def _seg_compress(chunk: str, intent: str, level: str, seg_id: str,
                  endpoint: str, model: str, num_ctx: int, timeout: float,
                  stats: dict | None = None) -> str:
    if stats is not None:
        stats["chunks"] = stats.get("chunks", 0) + 1
    # `chunk` is already de-padded by compress_context (matching production, which
    # de-pads the whole content before chunking) — see the note there.
    #
    # strip() is load-bearing: the 4B was trained on the body IMMEDIATELY following
    # "[SEG ...]\n". A single leading blank line (which the `# File:` split leaves on
    # each body — "\n1\t...") makes the SEG open "[SEG ...]\n\n1\t...", which is
    # out-of-distribution and collapses compression (measured 0.23 -> 0.60 on the GPU
    # for the SAME file). Strip both ends so no blank line ever precedes the content.
    content = chunk.strip()
    user = (
        "USER INTENT:\n" + intent.strip() + "\n\n"
        "Compress the following segment under the rules in your system prompt. "
        "Output only the compressed [SEG]...[/SEG] block (or an empty one to drop):\n\n"
        f"[SEG id={seg_id} kind=file_read level={level}]\n{content}\n[/SEG]\n"
    )
    # ENDPOINT = /v1/chat/completions (OpenAI-compat), NOT Ollama's native /api/chat.
    # This is load-bearing. /api/chat reuses the cached system-prompt KV across requests;
    # on this knife-edge 4B that flips greedy output — only the FIRST request after a
    # fresh model load compresses to ~0.24, every request after it collapses to ~0.60
    # (measured, deterministic). /v1 re-prefills cleanly each request → stable ~0.24 on
    # EVERY request, which is exactly what the gateway (LocalModelStrategy) and the
    # RunPod pod (gpu-serverless/compressor.py) use. NOTE: also requires Ollama 0.32.1
    # (the pod's pinned version); newer Ollama (e.g. 0.32.15) gives ~0.60 regardless of
    # endpoint. run.py checks the version. See eval_model/README.
    #
    # OpenAI-compat takes `max_tokens` (not num_ctx/num_predict). Cap it so prompt +
    # max_tokens fits the model's context window, or Ollama 400s the request.
    prompt_tokens = count_tokens(_SYS_FILE_READ, ENC) + count_tokens(user, ENC)
    max_tokens = min(count_tokens(content, ENC) + 256, _MAX_OUTPUT)
    ctx_budget = num_ctx - int(prompt_tokens * _TOKENIZER_SLACK) - _CTX_SAFETY_MARGIN
    if ctx_budget < max_tokens:
        max_tokens = max(ctx_budget, 256)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYS_FILE_READ},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1.0,
        "stream": False,
    }
    data = json.dumps(payload).encode()
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                endpoint, data=data, headers={"Content-Type": "application/json"})
            resp = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
            raw = resp["choices"][0]["message"]["content"]  # OpenAI-compat response shape
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
                return content
            time.sleep(2 ** attempt)


def compress_context(full_context: str, intent: str, *, chunk: int = 3000, level: str = "L1",
                     endpoint: str = "http://localhost:11434/v1/chat/completions",
                     model: str = "paritok-4b-v1:latest", num_ctx: int = 8192,
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
        # Match production (paritok/pipelines/compress.py): de-pad width-padded Read
        # line numbers ("     123\t" -> "123\t") on the WHOLE file body BEFORE the
        # chunk-size decision and chunking. dataset.py emits cat -n padding (as a real
        # Claude Code Read does), and the 4B was tuned on the bare "N\t" shape, so the
        # pad is out-of-distribution AND inflates the token count ~20%. That inflation
        # pushed quality_agent.py (2,878 de-padded) over the 3,000 chunk threshold
        # (3,460 padded), splitting one file into 2 SEGs — which nearly halved the
        # compression vs a single-shot pass (measured kept 0.39 vs 0.19 on the GPU).
        # The gateway de-pads once, then sizes/chunks against the real token count.
        # .strip() drops the leading blank line the `# File:` split leaves on each
        # body (see the SEG-strip note in _seg_compress — that blank line alone
        # collapses compression 0.23 -> 0.60 on the same file).
        body = _depad_line_numbers(body).strip()
        if count_tokens(body, ENC) <= chunk:
            section_body = _seg_compress(body, intent, level, "s1", endpoint, model, num_ctx, timeout, stats)
        else:
            pieces = []
            for i, (ctext, sl, el, _rt) in enumerate(
                    split_into_chunks_structural(body, chunk_size=chunk, max_single_block=chunk), 1):
                c = _seg_compress(ctext, intent, level, f"s{i}", endpoint, model, num_ctx, timeout, stats)
                if c:
                    pieces.append(f"# Lines {sl}-{el}:\n{c}")
            section_body = deduplicate_definitions("\n\n".join(pieces))
        header = f"# File: {path}" if path else ""
        sections.append((header + "\n" + section_body).strip())
    return "\n\n".join(sections)
