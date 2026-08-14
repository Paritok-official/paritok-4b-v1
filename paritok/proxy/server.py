"""Paritok HTTP Proxy Server.

Sits between AI agents and LLM APIs. Uses ParitokEngine (shared with SDK)
for all compression logic. The proxy is a thin HTTP layer on top.

Usage:
    paritok proxy --port 8080
    # Then set ANTHROPIC_BASE_URL=http://localhost:8080

Supports:
    - Anthropic Messages API (/v1/messages)
    - OpenAI Chat Completions API (/v1/chat/completions)
    - Streaming (SSE passthrough) and non-streaming
"""

from __future__ import annotations
import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit

logger = logging.getLogger("paritok.proxy")


@dataclass
class ProxyStats:
    """Running statistics for the proxy."""
    requests_processed: int = 0
    total_original_tokens: int = 0
    total_compressed_tokens: int = 0
    total_tools_filtered: int = 0
    total_items_compressed: int = 0
    # How many times the model called expand_context (a [REF] was re-delivered in
    # full), and how many lossy Edit calls edit_recovery rewrote to match the file.
    total_expansions: int = 0
    total_edits_recovered: int = 0
    # Per-model buckets for the parts THIS middleware touches — the content it
    # compresses (tool results / file reads / old history) PLUS the tool schemas
    # it stubs. {model: {"orig", "comp"}}: `orig` = what those parts would be
    # without paritok; `comp` = what we actually forward. Everything paritok
    # can't affect (system prompt, model output, ...) is deliberately excluded.
    by_model: dict = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)

    @staticmethod
    def _new_bucket() -> dict:
        # content_* = compressed body (tool results / file reads / old history),
        #   priced at the base input rate (it's genuinely new input each turn).
        # The frozen tool-schema block is byte-stable across a conversation, so its
        #   FIRST tool-bearing turn is a cache WRITE (~1.25x) and every turn after is
        #   a cache READ (~0.1x). We split it so each side gets its true multiplier:
        #   tools_first_* = that first turn; tools_rest_* = all subsequent turns.
        return {"content_first_orig": 0, "content_first_comp": 0,
                "content_rest_orig": 0, "content_rest_comp": 0,
                "tools_first_orig": 0, "tools_first_comp": 0,
                "tools_rest_orig": 0, "tools_rest_comp": 0}

    def record(self, stats, model: str = "", *,
               tools_original_tokens: int = 0, tools_compressed_tokens: int = 0) -> None:
        """Fold one request into the totals. Counts only what paritok touches:
        the content it compressed (from `stats`) plus the tool-schema tokens it
        stubbed away (schema size before vs after discovery + virtual injection)."""
        self.requests_processed += 1
        self.total_tools_filtered += stats.tools_filtered
        self.total_items_compressed += stats.items_compressed
        orig = stats.original_tokens + tools_original_tokens
        comp = stats.compressed_tokens + tools_compressed_tokens
        self.total_original_tokens += orig
        self.total_compressed_tokens += comp
        bucket = self.by_model.setdefault(model or "unknown", self._new_bucket())
        # Content (tool results / file reads / old history) is re-sent inside the
        # cacheable prefix every turn just like the tool block, so it's a cache WRITE
        # the first turn and a cache READ afterwards — price it the same way, not at
        # full list price. (First request per model = write; the rest = reads.)
        cslot = "first" if bucket["content_first_orig"] == 0 else "rest"
        bucket[f"content_{cslot}_orig"] += stats.original_tokens
        bucket[f"content_{cslot}_comp"] += stats.compressed_tokens
        if tools_original_tokens > 0:
            # First tool-bearing turn for this model = the cache write; rest = reads.
            slot = "first" if bucket["tools_first_orig"] == 0 else "rest"
            bucket[f"tools_{slot}_orig"] += tools_original_tokens
            bucket[f"tools_{slot}_comp"] += tools_compressed_tokens

    def record_expansion(self, expanded_tokens: int, model: str = "") -> None:
        """Fold a served expand_context back onto the compressed side.

        When the model calls expand_context, the proxy hands it the *full
        original* it had compressed down to a [REF:id] stub — so for that item
        the compression did not stick. Count the re-delivered original as
        compressed output (original is left untouched): tokens_saved and
        cost_saved fall, going negative on a turn that expands more than it
        compressed, and compression_ratio rises toward (or past) 1.0. The
        expanded bytes re-enter the cacheable prefix on later turns just like
        other content, so they land in the same content slot the request used.
        """
        self.total_expansions += 1
        if expanded_tokens <= 0:
            return
        self.total_compressed_tokens += expanded_tokens
        bucket = self.by_model.setdefault(model or "unknown", self._new_bucket())
        cslot = "first" if bucket["content_first_orig"] == 0 else "rest"
        bucket[f"content_{cslot}_comp"] += expanded_tokens

    def record_edit_recovery(self) -> None:
        """Count one lossy Edit that edit_recovery rewrote to match the real file."""
        self.total_edits_recovered += 1

    @property
    def total_saved_tokens(self) -> int:
        return self.total_original_tokens - self.total_compressed_tokens

    @property
    def estimated_cost_saved_usd(self) -> float:
        """Cache-aware $ saved on the parts paritok touches, at each model's own
        input rate (unknown → $3/M). The compressed *content* is new input, priced
        at the base rate. The frozen *tool-schema* block is byte-stable across a
        conversation: its first turn is a cache WRITE (1.25x base), every turn after
        is a cache READ (Claude 0.1x, GPT-5 0.1x, ...) — each priced at its true
        multiplier rather than full list price."""
        from paritok.proxy.pricing import (
            input_price_per_token, cache_read_multiplier, CACHE_WRITE_MULT)
        total = 0.0
        for m, b in self.by_model.items():
            rate = input_price_per_token(m)
            cr = cache_read_multiplier(m)
            content_write = (b["content_first_orig"] - b["content_first_comp"]) * rate * CACHE_WRITE_MULT
            content_read = (b["content_rest_orig"] - b["content_rest_comp"]) * rate * cr
            tools_write = (b["tools_first_orig"] - b["tools_first_comp"]) * rate * CACHE_WRITE_MULT
            tools_read = (b["tools_rest_orig"] - b["tools_rest_comp"]) * rate * cr
            total += content_write + content_read + tools_write + tools_read
        return round(total, 4)

    def snapshot(self) -> dict:
        """The /stats payload — scoped to what paritok actually intervenes in."""
        orig, comp = self.total_original_tokens, self.total_compressed_tokens
        # TEMP diagnostic: split saved tokens into file-compression vs tool-filter.
        c_orig = c_comp = t_orig = t_comp = 0
        for b in self.by_model.values():
            c_orig += b["content_first_orig"] + b["content_rest_orig"]
            c_comp += b["content_first_comp"] + b["content_rest_comp"]
            t_orig += b["tools_first_orig"] + b["tools_rest_orig"]
            t_comp += b["tools_first_comp"] + b["tools_rest_comp"]
        return {
            "total_requests": self.requests_processed,
            "input_tokens_original": orig,
            "input_tokens_compressed": comp,
            "compression_ratio": round(comp / orig, 3) if orig else 0.0,
            "tokens_saved": self.total_saved_tokens,
            "file_compression_saved": c_orig - c_comp,
            "file_orig": c_orig, "file_comp": c_comp,
            "tool_filter_saved": t_orig - t_comp,
            "tool_orig": t_orig, "tool_comp": t_comp,
            "tools_filtered": self.total_tools_filtered,
            "expansions": self.total_expansions,
            "edits_recovered": self.total_edits_recovered,
            "estimated_cost_saved_usd": f"${self.estimated_cost_saved_usd:.2f}",
        }


def _tool_params(t: dict) -> dict:
    """Function-tool parameter schema, tolerating the injected virtual tools'
    Anthropic-style `input_schema` or the OpenAI-style `parameters`."""
    return t.get("parameters") or t.get("input_schema") or {"type": "object", "properties": {}}


def _openai_chat_url(base: str) -> str:
    """Resolve the upstream Chat Completions URL from --openai-url.

    Three cases:
    - full endpoint already ending in `/chat/completions` → used verbatim.
    - base whose path already carries a version segment (`/v1`, `/v1beta/openai`,
      `/api/v1`) — OpenRouter, Gemini, or anyone who included `/v1` themselves →
      append only `/chat/completions` (adding `/v1` here doubles it, e.g.
      `https://openrouter.ai/api/v1/v1/chat/completions` → 404).
    - plain host with no version (OpenAI `https://api.openai.com`, Groq
      `https://api.groq.com/openai`) → append the standard `/v1/chat/completions`.
    """
    base = base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if re.search(r"/v\d", urlsplit(base).path):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _extract_tool_text(content):
    """Pull the compressible text out of an OpenAI ``role:tool`` message ``content``.

    ``content`` may be a plain string OR a list of content parts
    (``[{"type":"text","text":...}, ...]``) — Claude Code's OpenAI mode and gptme use
    the list form. Returns ``(text, rewrap)`` where ``rewrap(compressed)`` rebuilds
    ``content`` in its original shape (non-text parts preserved). Returns
    ``(None, None)`` when there is no text to compress.
    """
    if isinstance(content, str):
        return content, (lambda c: c)
    if isinstance(content, list):
        texts = [p.get("text", "") for p in content
                 if isinstance(p, dict) and p.get("type") == "text"
                 and isinstance(p.get("text"), str)]
        if not texts:
            return None, None
        joined = "\n".join(texts)

        def _rewrap(compressed, _parts=content):
            # Fold the (single) compressed text into the first text part; keep any
            # non-text parts (images, etc.) exactly where they were.
            out, inserted = [], False
            for p in _parts:
                if isinstance(p, dict) and p.get("type") == "text":
                    if not inserted:
                        out.append({**p, "text": compressed})
                        inserted = True
                else:
                    out.append(p)
            if not inserted:
                out.append({"type": "text", "text": compressed})
            return out

        return joined, _rewrap
    return None, None


def _pick_responses_upstream(authorization: str, openai_base_url: str) -> str:
    """Choose the upstream for a Codex `/v1/responses` request by auth type.

    Codex keeps its own credentials and sends them here:
      - API key (`Bearer sk-...`)              → {openai_base_url}/v1/responses
      - ChatGPT subscription (OAuth, not `sk-`) → chatgpt.com/backend-api/codex/responses

    The Authorization header (and chatgpt-account-id / originator) is relayed
    unchanged, so the subscription token is accepted by the ChatGPT backend just
    as if Codex had called it directly. Module-level for unit testing.
    """
    token = authorization[len("Bearer "):] if authorization.startswith("Bearer ") else ""
    if token and not token.startswith("sk-"):
        return "https://chatgpt.com/backend-api/codex/responses"
    return f"{openai_base_url}/v1/responses"


def _to_responses_tool(t: dict) -> dict:
    """Render one tool in the flat Responses shape.

    Codex sends more than function tools: the Responses API also carries built-in
    and custom tools like `{"type":"local_shell"}`, `web_search`, `custom`, `mcp`,
    etc., which have no `name` field. Only function tools (incl. our injected
    virtual ones, which use the Anthropic `name`/`input_schema` shape) get rebuilt
    as `{"type":"function","name",...,"parameters"}`; anything else is forwarded
    verbatim so we don't stamp a null `name` onto it.
    """
    ttype = t.get("type")
    if ttype not in (None, "function") or not t.get("name"):
        return t
    return {"type": "function", "name": t.get("name"),
            "description": t.get("description", ""), "parameters": _tool_params(t)}


# Header lines codex prepends when it reads a file through the shell: a
# command-output frame (Exit code / Wall time / Total output lines / Output:).
# Those extra lines tip classify_kind_from_content over its log_output line-count
# heuristic, so a *source file* gets the aggressive "other" prompt that drops the
# code and hallucinates. We only strip this exact frame; nothing else is touched.
_CODEX_OUTPUT_HEADER_PREFIXES = (
    "Exit code:", "Wall time:", "Total output lines:", "Output:",
)


def _split_codex_header(content: str) -> tuple[str, str]:
    """Split codex's shell command-output header from the real body it wraps.

    codex runs file reads (and commands) through the shell, prefixing the actual
    output with the frame above. Compressing that frame together with the body
    feeds the model something no other agent sends — Claude/OpenAI hand over the
    raw tool output — which skews the kind sniff and the compressed result (e.g.
    a source file mistaken for log_output and summarized away). We split the frame
    off so ONLY the body is compressed, identical to what every other agent feeds
    the model, and re-attach the header verbatim afterwards.

    Returns (header, body). header is "" when there is no codex frame, leaving the
    content untouched — so unwrapped content and the Claude/OpenAI paths are
    completely unaffected.
    """
    lines = content.splitlines(keepends=True)
    i, saw_header = 0, False
    while i < len(lines) and (
        not lines[i].strip() or lines[i].startswith(_CODEX_OUTPUT_HEADER_PREFIXES)
    ):
        if lines[i].startswith(_CODEX_OUTPUT_HEADER_PREFIXES):
            saw_header = True
        i += 1
    if not saw_header:
        return "", content
    return "".join(lines[:i]), "".join(lines[i:])


def _normalize_crlf(text: str) -> str:
    """codex reads files through PowerShell, whose output carries CRLF (`\\r\\n`)
    line endings; Claude Code and the other agents feed the compressor clean LF.
    Strip the `\\r` so codex's tool output is byte-identical to what every other
    agent sends — in-distribution for the compressor — instead of an
    `\\r`-littered variant it never trained on."""
    if "\r" not in text:
        return text
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _compress_responses_item(item: dict, compress_text) -> tuple[dict, bool]:
    """Apply `compress_text(str) -> str | None` to a Responses tool-result item's
    text, returning (new_item, changed). Handles both shapes codex uses:

      - `function_call_output`: `output` is a plain string.
      - `custom_tool_call_output` (newer custom tools like `exec`/shell): `output`
        is a list of `{"type":"input_text","text":...}` blocks — each block's text
        is compressed independently; non-text blocks pass through untouched.

    `compress_text` returns None to skip (small/uncompressible), leaving that text
    as-is. Pure (no I/O / stats) so it is unit-testable; the caller's closure owns
    the actual compression, stats and read-body capture. Any other item type is
    returned unchanged.
    """
    itype = item.get("type")
    if itype == "function_call_output" and isinstance(item.get("output"), str):
        new = compress_text(item["output"])
        return ({**item, "output": new}, True) if new is not None else (item, False)
    if itype == "custom_tool_call_output" and isinstance(item.get("output"), list):
        new_output: list = []
        changed = False
        for blk in item["output"]:
            if (isinstance(blk, dict) and blk.get("type") == "input_text"
                    and isinstance(blk.get("text"), str)):
                new_text = compress_text(blk["text"])
                if new_text is not None:
                    new_output.append({**blk, "text": new_text})
                    changed = True
                    continue
            new_output.append(blk)
        return ({**item, "output": new_output}, changed) if changed else (item, False)
    return item, False


def _has_code_signals(text: str) -> bool:
    """A few strong, code-specific tokens present at least twice — enough to tell a
    real source file from prose/log output."""
    return sum(text.count(k) for k in ("def ", "class ", "import ", "return ", "self.")) >= 2


def _is_numbered_line(line: str) -> bool:
    """True if `line` starts (after indent) with digits then a tab or Read's arrow —
    i.e. it already carries a cat -n / Read line-number prefix."""
    s = line.lstrip()
    j = 0
    while j < len(s) and s[j].isdigit():
        j += 1
    return j > 0 and j < len(s) and s[j] in ("\t", "→")


def _ensure_line_numbers(text: str) -> str:
    """Add line numbers (`N\\tline`) to unnumbered source code.

    codex reads files through the shell WITHOUT line numbers, which is
    out-of-distribution for the compressor: it barely compresses (~0.19) and keeps
    everything. Numbering it lets a codex file read compress like Claude's.

    Use the SAME shape Claude Code's Read tool emits — a bare line number then a tab
    (`120\\tcode`), NOT a `cat -n`-style width-padded number (`   120\\tcode`). That's
    what paritok-4B saw in training (real Claude Code trajectories), so it's the
    in-distribution format; the leading padding spaces would otherwise add ~2 tokens
    per line (~500 on a 300-line file) that the model has to read for nothing.

    It MUST be a tab (`\\d+\\t`), not the Read arrow (`\\d+→`): the compressor chunks
    long files by `class`/`def` boundaries, and that boundary regex only tolerates a
    tab-prefixed line number. Arrow-numbered lines match ZERO boundaries, so a big
    file never chunks — it's sent as one oversized block the model truncates
    mid-function (dropping code). Tabs keep the chunker working: a long file splits
    per function and every chunk survives intact (~0.63, whole).

    Left unchanged when the body isn't source code (logs/prose, no code signals) or
    is already numbered (tab OR arrow) — so we never corrupt non-file output.
    """
    lines = text.splitlines()
    if len(lines) < 4 or not _has_code_signals(text):
        return text
    sample = lines[:20]
    if sum(1 for ln in sample if _is_numbered_line(ln)) >= len(sample) * 0.6:
        return text  # already numbered (e.g. content that arrived Read-style)
    return "".join(f"{i}\t{ln}\n" for i, ln in enumerate(lines, 1))


def create_app(
    anthropic_base_url: str = "https://api.anthropic.com",
    openai_base_url: str = "https://api.openai.com",
    config_path: str | None = None,
    http_client=None,
):
    """Create the Starlette ASGI app.

    Requires: pip install paritok[proxy]
    """
    try:
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse, StreamingResponse, Response
        from starlette.routing import Route
        import httpx
    except ImportError:
        raise ImportError(
            "Proxy dependencies not installed. Run: pip install paritok[proxy]"
        )

    from paritok.config import ParitokConfig
    from paritok.middleware.wrapper import (
        CompressionStats,
        ParitokEngine,
        _inject_virtual_tools,
    )
    from paritok.pipelines.edit_recovery import recover_edit, strip_read_line_numbers
    from paritok.pipelines.codex_edit_recovery import (
        rewrite_function_call_arguments,
        rewrite_tool_call_input,
    )
    from paritok.pipelines.virtual import is_expand_call, is_virtual_tool_call
    from paritok.proxy.adapters import anthropic as anth_adapter
    from paritok.proxy.adapters import openai as oai_adapter
    from paritok.proxy.adapters import responses as resp_adapter
    from paritok.token_counter import count_tokens

    # Initialize
    config = ParitokConfig.load(config_path) if config_path else ParitokConfig()
    engine = ParitokEngine(config)
    proxy_stats = ProxyStats()
    if http_client is None:
        http_client = httpx.AsyncClient(timeout=120.0)

    def _tools_tokens(tools) -> int:
        """Token size of a tool-schema list (0 when there are none)."""
        return count_tokens(json.dumps(tools)) if tools else 0

    # Fire-and-forget background tasks (hosted tool-savings reports). Kept in a
    # set so they aren't garbage-collected before they finish.
    _bg_tasks: set = set()
    # Upstream models whose frozen tool block we've already reported as a cache
    # WRITE. The first tool-bearing turn per model is the write; every one after
    # is a cache read. Mirrors ProxyStats' tools_first/tools_rest split so the
    # hosted dashboard's tool cost matches self-hosted /stats.
    _tool_write_seen: set = set()

    async def _send_tool_meter(model: str, tools_orig: int, tools_comp: int,
                               cache_role: str) -> None:
        base = config.gpu_server.base_url.rstrip("/")
        headers = {"Content-Type": "application/json"}
        if config.gpu_server.api_key:
            headers["Authorization"] = f"Bearer {config.gpu_server.api_key}"
        try:
            await http_client.post(
                f"{base}/meter",
                json={"tokens_in": tools_orig, "tokens_out": tools_comp,
                      "upstream_model": model or "", "cache_role": cache_role},
                headers=headers, timeout=10.0,
            )
        except Exception:  # noqa: BLE001 — metering is best-effort
            pass

    def _report_tool_savings(model: str, tools_orig: int, tools_comp: int) -> None:
        """In hosted (GPU-server) mode, report the proxy's local tool-schema
        stubbing to the website so its dashboard matches self-hosted /stats
        (content + tools). Self-hosted mode already counts it in /stats locally.

        The tool block is frozen (byte-stable) across a conversation, so it's a
        prompt-cache WRITE on its first turn and a cache READ after — we tell the
        website which, so it prices the saving at the cache rate, not list price."""
        if not config.use_gpu_server or tools_orig <= 0 or tools_orig <= tools_comp:
            return
        m = model or ""
        if m in _tool_write_seen:
            cache_role = "read"
        else:
            _tool_write_seen.add(m)
            cache_role = "write"
        task = asyncio.create_task(_send_tool_meter(m, tools_orig, tools_comp, cache_role))
        _bg_tasks.add(task)
        task.add_done_callback(_bg_tasks.discard)

    # ── Anthropic handler ──

    async def handle_anthropic(request: Request) -> Response:
        body = json.loads(await request.body())
        parsed = anth_adapter.parse_request(body)

        import os as _reqos  # TEMP request-descriptor + passthrough diagnostics; not committed
        _reqp = _reqos.environ.get("PARITOK_REQ_LOG")
        if _reqp:
            _msgs = body.get("messages", []) or []
            _last = _msgs[-1] if _msgs else {}
            _lc = _last.get("content")
            _kinds = ([b.get("type") for b in _lc if isinstance(b, dict)]
                      if isinstance(_lc, list) else [type(_lc).__name__])
            _sys = body.get("system")
            _syslen = len(json.dumps(_sys)) if _sys is not None else 0
            with open(_reqp, "a", encoding="utf-8") as _fh:
                _fh.write(f"model={str(body.get('model',''))[:26]} nmsg={len(_msgs)} "
                          f"last_role={_last.get('role')} last_kinds={_kinds} "
                          f"max_tokens={body.get('max_tokens')} syslen={_syslen}\n")
        # True passthrough: forward the client's request byte-for-byte (no compression,
        # no tool injection, no guidance) — measures the paritok relay overhead alone.
        if _reqos.environ.get("PARITOK_PASSTHROUGH"):
            headers = _forward_headers(request)
            _capp = _reqos.environ.get("PARITOK_CAPTURE")
            if _capp and len(body.get("messages", []) or []) >= 4:
                import os as _cos
                if not _cos.path.exists(_capp + ".body.json"):
                    with open(_capp + ".body.json", "w", encoding="utf-8") as _fh:
                        json.dump(body, _fh)
                    with open(_capp + ".headers.json", "w", encoding="utf-8") as _fh:
                        json.dump(dict(request.headers), _fh)
                    with open(_capp + ".fwdheaders.json", "w", encoding="utf-8") as _fh:
                        json.dump(headers, _fh)
            # Surgical diagnostic: drop all but core coding tools from the forwarded
            # request (nothing else changed) — isolates whether the +16.8k/turn is tools.
            _strip = _reqos.environ.get("PARITOK_STRIP_TOOLS")
            if _strip and isinstance(body.get("tools"), list):
                _core = {"Read", "Edit", "Bash", "Glob", "Grep", "Write", "PowerShell",
                         "MultiEdit", "NotebookEdit", "TodoWrite", "Task"}
                body = {**body, "tools": [t for t in body["tools"] if t.get("name") in _core]}
            url = f"{anthropic_base_url}/v1/messages"
            if parsed.stream:
                if _reqos.environ.get("PARITOK_PASSTHROUGH") == "raw":
                    # PURE raw stream: never buffer/reconstruct — truest possible relay.
                    async def _raw_stream():
                        async with http_client.stream("POST", url, headers=headers, json=body) as r:
                            async for chunk in r.aiter_bytes():
                                yield chunk
                    return StreamingResponse(_raw_stream(), media_type="text/event-stream")
                return await _stream_anthropic(url, headers, body, [])
            resp = await http_client.post(url, headers=headers, json=body)
            return JSONResponse(content=resp.json(), status_code=resp.status_code,
                                headers=_relay_headers(resp))

        # Use shared engine for compression + tool discovery. process_request does
        # BLOCKING 4B inference (sync httpx), so run it in a worker thread — otherwise it
        # freezes the whole event loop for the entire compression, stalling health checks,
        # count_tokens, and every other in-flight session until it returns.
        client_tools = parsed.tools  # the caller's full tool schemas, pre-stub
        parsed.messages, parsed.tools, stats, stubbed = await asyncio.to_thread(
            engine.process_request, parsed.messages, parsed.tools,
            upstream_model=body.get("model", ""),
        )

        tools_orig_tok = _tools_tokens(client_tools)
        tools_comp_tok = _tools_tokens(parsed.tools)
        proxy_stats.record(stats, model=body.get("model", ""),
                           tools_original_tokens=tools_orig_tok,
                           tools_compressed_tokens=tools_comp_tok)
        _report_tool_savings(body.get("model", ""), tools_orig_tok, tools_comp_tok)

        query = anth_adapter.extract_query(parsed.messages)
        logger.info("Request #%d, saved %d tokens, query=%s",
                     proxy_stats.requests_processed, stats.saved_tokens, (query or "")[:50])

        # If expand_context was injected, tell the model (once, via system) about the
        # [REF:] compression convention and that it can pull originals back on demand.
        # The proxy resolves expand_context itself (server-side), so the client agent
        # never needs to own the tool — this works for Claude Code, etc.
        if parsed.tools and any(is_expand_call(t.get("name", "")) for t in parsed.tools):
            parsed.system = _prepend_ref_guidance(parsed.system)

        # Forward
        headers = _forward_headers(request)
        url = f"{anthropic_base_url}/v1/messages"
        forward_body = parsed.to_dict()

        import os as _szos  # TEMP in-vs-out size diagnostic; not committed
        _szp = _szos.environ.get("PARITOK_SIZE_LOG")
        if _szp:
            _m = body.get("model", "")
            _in = (count_tokens(json.dumps(body.get("messages", [])), _m)
                   + count_tokens(json.dumps(body.get("system", "")), _m)
                   + count_tokens(json.dumps(body.get("tools", [])), _m))
            _out = (count_tokens(json.dumps(forward_body.get("messages", [])), _m)
                    + count_tokens(json.dumps(forward_body.get("system", "")), _m)
                    + count_tokens(json.dumps(forward_body.get("tools", [])), _m))
            _cc_in = json.dumps(body).count("cache_control")
            _cc_out = json.dumps(forward_body).count("cache_control")
            with open(_szp, "a", encoding="utf-8") as _fh:
                _fh.write(f"req#{proxy_stats.requests_processed} IN={_in} OUT={_out} "
                          f"delta={_out - _in} cache_control_in={_cc_in} out={_cc_out}\n")

        import os as _tlos  # TEMP tools[] stability diagnostic; not committed
        _tlp = _tlos.environ.get("PARITOK_TOOLS_LOG")
        if _tlp:
            _tn = [t.get("name") for t in (parsed.tools or [])]
            _virt = [n for n in _tn if n in ("read_original", "expand_context", "gateway_search_tools")]
            _nmsg = len(parsed.messages or [])
            _ttok = _tools_tokens(parsed.tools) if parsed.tools else 0
            _systok = count_tokens(json.dumps(parsed.system)) if parsed.system else 0
            _msgtok = count_tokens(json.dumps(parsed.messages)) if parsed.messages else 0
            _names = {}
            for _n in _tn:
                _names[_n] = _names.get(_n, 0) + 1
            _dups = {k: v for k, v in _names.items() if v > 1}
            with open(_tlp, "a", encoding="utf-8") as _fh:
                _fh.write(f"msgs={_nmsg} n_tools={len(_tn)} tools_tok={_ttok} sys_tok={_systok} msg_tok={_msgtok} dups={_dups} virtual={_virt}\n")

        if parsed.stream:
            return await _stream_anthropic(url, headers, forward_body, stubbed)
        try:
            final_body, status_code, resp_headers = await _anthropic_resolve(
                url, headers, forward_body, stubbed
            )
        except httpx.ConnectError as e:
            return JSONResponse(content={"error": f"Cannot connect to {url}: {e}"}, status_code=502)
        except httpx.TimeoutException:
            return JSONResponse(content={"error": f"Upstream timed out: {url}"}, status_code=504)
        final_body = _recover_edits(final_body)
        return JSONResponse(content=final_body, status_code=status_code, headers=resp_headers)

    # ── OpenAI handler ──

    async def handle_openai(request: Request) -> Response:
        body = json.loads(await request.body())
        parsed = oai_adapter.parse_request(body)

        # OpenAI wraps tools in {"type": "function", "function": {...}}
        # Unwrap for engine, then re-wrap after
        client_tools = parsed.tools  # caller's full tool schemas, pre-stub
        raw_tools = None
        if parsed.tools:
            raw_tools = [t.get("function", t) for t in parsed.tools]

        processed_messages, processed_tools, stats, stubbed = await asyncio.to_thread(
            engine.process_request, parsed.messages, raw_tools
        )  # off the event loop — blocking 4B inference (see handle_anthropic)
        # Keep process_request's compressed messages (issue #40): step 3 folds old
        # conversation history into a summary and returns it here. The OpenAI path used
        # to drop this (`_,`) and forward the uncompressed originals, so long sessions
        # went upstream in full. Step 2 (Anthropic tool_result compression) is a no-op
        # for OpenAI's role:"tool" messages — the loop below compresses those — so
        # adopting processed_messages does not double-compress.
        parsed.messages = processed_messages

        # Compress tool messages (OpenAI uses role="tool" instead of tool_result blocks).
        # `content` may be a plain string OR a list of content parts — extract the text
        # either way, compress it, and put it back in the original shape (see
        # _extract_tool_text). gptme/Claude-Code-OpenAI use the list form; string-only
        # handling here silently skipped their file reads.
        query = oai_adapter.extract_query(parsed.messages)
        for i, msg in enumerate(parsed.messages):
            if msg.get("role") != "tool":
                continue
            text, rewrap = _extract_tool_text(msg.get("content", ""))
            if not (text and text.strip()):
                continue
            cr = engine.pipeline.compress(text, query=query,
                                          upstream_model=body.get("model", ""))
            if not cr.metadata.get("skipped"):
                parsed.messages[i] = {**msg, "content": rewrap(cr.compressed)}
                stats.original_tokens += cr.original_tokens
                stats.compressed_tokens += cr.compressed_tokens
                stats.items_compressed += 1

        # Inject virtual tools now — process_request runs its injection BEFORE we
        # compress the OpenAI `role:tool` messages above, so at that point
        # items_compressed was 0 and expand_context was not added. Re-inject with the
        # updated count (idempotent: _inject_virtual_tools skips names already present).
        if processed_tools is not None:
            processed_tools = _inject_virtual_tools(
                processed_tools,
                has_compressed=True,  # always inject read_original (stable tools[] => cache-friendly)
                has_filtered=stats.tools_kept < stats.tools_original,
            )
            # Re-wrap; convert the virtual tools' Anthropic `input_schema` to OpenAI
            # `parameters` (real client tools already carry `parameters`).
            parsed.tools = [
                {"type": "function", "function": {
                    "name": t.get("name"),
                    "description": t.get("description", ""),
                    "parameters": _tool_params(t)}}
                for t in processed_tools
            ]

        tools_orig_tok = _tools_tokens(client_tools)
        tools_comp_tok = _tools_tokens(parsed.tools)
        proxy_stats.record(stats, model=body.get("model", ""),
                           tools_original_tokens=tools_orig_tok,
                           tools_compressed_tokens=tools_comp_tok)
        _report_tool_savings(body.get("model", ""), tools_orig_tok, tools_comp_tok)

        # If expand_context was injected, tell the model (once, via system) about the
        # [REF:] convention. The proxy resolves expand_context server-side (same as the
        # Anthropic path), so Codex / any OpenAI Chat Completions client never owns it.
        if processed_tools and any(is_expand_call(t.get("name", "")) for t in processed_tools):
            parsed.messages = _prepend_ref_guidance_openai(parsed.messages)

        # Forward
        headers = _forward_headers(request)
        url = _openai_chat_url(openai_base_url)
        forward_body = parsed.to_dict()

        if parsed.stream:
            return await _stream_openai(url, headers, forward_body, stubbed)
        try:
            final_body, status_code, resp_headers = await _openai_resolve(
                url, headers, forward_body, stubbed
            )
        except httpx.ConnectError as e:
            return JSONResponse(content={"error": f"Cannot connect to {url}: {e}"}, status_code=502)
        except httpx.TimeoutException:
            return JSONResponse(content={"error": f"Upstream timed out: {url}"}, status_code=504)
        return JSONResponse(content=final_body, status_code=status_code, headers=resp_headers)

    # ── OpenAI Responses API handler (Codex) ──

    async def handle_responses(request: Request) -> Response:
        body = json.loads(await request.body())
        parsed = resp_adapter.parse_request(body)

        query = resp_adapter.extract_query(parsed.input)
        input_items = resp_adapter.normalize_input(parsed.input)
        stats = CompressionStats()

        # Tool discovery — Responses tools are flat ({"type":"function","name",...}).
        raw_tools = list(parsed.tools) if parsed.tools else None
        tools = raw_tools
        # Modern codex declares its tools in an `additional_tools` item and leaves
        # the top-level `tools` empty. To give it read_original we still add the
        # tool to the top-level `tools[]` (as a standard function tool) — the model
        # then emits a plain `function_call`, which the streaming resolve loop can
        # intercept and expand server-side (the client never sees it). Injecting
        # into additional_tools instead makes codex emit a custom_tool_call we
        # can't intercept, so the client rejects it as unsupported.
        is_codex_responses = any(
            isinstance(it, dict) and it.get("type") == "additional_tools" for it in input_items
        )
        if tools is None and is_codex_responses:
            tools = []
        stubbed: list[dict] = []
        if raw_tools and query and len(raw_tools) > config.tool_discovery.top_k:
            result = engine.discovery.filter_tools(raw_tools, query)
            tools = result.tools
            stubbed = result.stubbed_tools
            stats.tools_original = result.original_count
            stats.tools_kept = result.kept_count
        elif raw_tools:
            stats.tools_original = stats.tools_kept = len(raw_tools)

        import os as _tnos  # TEMP codex tool-filter diagnostic; not committed
        if _tnos.environ.get("PARITOK_CODEXTOOLS_LOG") and raw_tools:
            def _nm(t): return t.get("name") or (t.get("function") or {}).get("name")
            with open(_tnos.environ["PARITOK_CODEXTOOLS_LOG"], "a", encoding="utf-8") as _fh:
                _fh.write(f"IN({len(raw_tools)})={[_nm(t) for t in raw_tools]} "
                          f"KEPT({len(tools)})={[_nm(t) for t in tools]} "
                          f"STUBBED={[_nm(t) for t in stubbed]}\n")

        # Only compress when the model will have a way to EXPAND a [REF:...] stub
        # (read_original is injected into tools[] below) — otherwise it can only
        # re-read the file, which just gets re-compressed, so it loops.
        can_expand = tools is not None

        # Compress tool-result items (the file reads / shell output that grow large).
        # Also keep each real read body (before compression): the file codex is about to
        # edit was read earlier in the conversation, so its true bytes live here — that's
        # what codex_edit_recovery maps a reflowed shell `-replace` back onto.
        codex_read_bodies: list[str] = []

        def _compress_codex_text(text: str) -> str | None:
            """Compress one shell/tool-output string. Splits codex's command-output
            header off so only the body is compressed (byte-identical to what every
            other agent feeds the model), re-attaches it, records stats + the read
            body. Returns the new text, or None when compression is skipped."""
            if not isinstance(text, str) or not text.strip():
                return None
            # codex reads via PowerShell (CRLF); normalize to LF so the content is
            # in-distribution for the compressor, like every other agent's output.
            text = _normalize_crlf(text)
            header, seg_body = _split_codex_header(text)
            if len(seg_body) > 40:
                codex_read_bodies.append(strip_read_line_numbers(seg_body))
            # codex reads files without line numbers (out-of-distribution for the
            # compressor). Feed the model a line-numbered form so it compresses like
            # every other agent's Read output, but keep seg_body (the real, unnumbered
            # content) as `content` so ratio/stats/expand reflect what the agent sent.
            cr = engine.pipeline.compress(seg_body, query=query,
                                          model_input=_ensure_line_numbers(seg_body),
                                          upstream_model=parsed.model)
            if cr.metadata.get("skipped"):
                return None
            stats.original_tokens += cr.original_tokens
            stats.compressed_tokens += cr.compressed_tokens
            stats.items_compressed += 1
            return header + cr.compressed

        if can_expand:
            # Blocking 4B inference (sync httpx) — run off the event loop so a slow
            # compression can't freeze health checks / count_tokens / other sessions.
            def _compress_all_items() -> None:
                for i, item in enumerate(input_items):
                    new_item, changed = _compress_responses_item(item, _compress_codex_text)
                    if changed:
                        input_items[i] = new_item

            await asyncio.to_thread(_compress_all_items)

        # Inject virtual tools, then convert to the flat Responses tool shape.
        if tools is not None:
            tools = _inject_virtual_tools(
                tools,
                has_compressed=True,  # always inject read_original (stable tools[] => cache-friendly)
                has_filtered=stats.tools_kept < stats.tools_original,
            )
            parsed.tools = [_to_responses_tool(t) for t in tools]
        parsed.input = input_items

        tools_orig_tok = _tools_tokens(raw_tools)
        tools_comp_tok = _tools_tokens(parsed.tools)
        proxy_stats.record(stats, model=body.get("model", ""),
                           tools_original_tokens=tools_orig_tok,
                           tools_compressed_tokens=tools_comp_tok)
        _report_tool_savings(body.get("model", ""), tools_orig_tok, tools_comp_tok)

        if parsed.tools and any(is_expand_call(t.get("name", "")) for t in parsed.tools):
            parsed.instructions = _prepend_ref_guidance_responses(parsed.instructions)

        headers = _forward_headers(request)
        url = _pick_responses_upstream(request.headers.get("authorization", ""), openai_base_url)
        forward_body = parsed.to_dict()

        if parsed.stream:
            return await _stream_responses(url, headers, forward_body, stubbed, codex_read_bodies)
        try:
            final_body, status_code, resp_headers = await _responses_resolve(
                url, headers, forward_body, stubbed
            )
        except httpx.ConnectError as e:
            return JSONResponse(content={"error": f"Cannot connect to {url}: {e}"}, status_code=502)
        except httpx.TimeoutException:
            return JSONResponse(content={"error": f"Upstream timed out: {url}"}, status_code=504)
        final_body = _recover_codex_edits(final_body, codex_read_bodies)
        return JSONResponse(content=final_body, status_code=status_code, headers=resp_headers)

    # ── Anthropic token counting ──

    async def handle_count_tokens(request: Request) -> Response:
        """`/v1/messages/count_tokens`, computed LOCALLY. Claude Code calls this every turn
        to size its context and decide when to auto-compact. Left unimplemented it 404s, and
        Claude Code falls back to its own estimate over the FULL uncompressed conversation —
        so it auto-compacts early instead of letting paritok's compression extend the window.

        We compress the payload exactly as the real /v1/messages path would (in a worker
        thread; reusing the SHA256 compression cache, so the subsequent /v1/messages reuses
        it and the 4B fires at most once per payload) and count the COMPRESSED result. The
        meter then reflects what paritok actually sends upstream, so Claude Code sees the
        extended window and stops compacting prematurely. Counting is local tiktoken (a few
        percent off for Claude, per the 'local is fine' choice) and never fails the meter:
        any compression error falls back to counting the raw payload."""
        body = json.loads(await request.body())
        model = body.get("model", "")
        try:
            parsed = anth_adapter.parse_request(body)
            msgs, tools, _stats, _stubbed = await asyncio.to_thread(
                engine.process_request, parsed.messages, parsed.tools, upstream_model=model
            )
            system = parsed.system
        except Exception:
            logger.exception("count_tokens: compression failed; counting raw payload")
            msgs, tools, system = (body.get("messages", []), body.get("tools"),
                                   body.get("system"))
        n = (count_tokens(json.dumps(msgs), model)
             + (count_tokens(json.dumps(system), model) if system else 0)
             + (count_tokens(json.dumps(tools), model) if tools else 0))
        return JSONResponse({"input_tokens": n})

    # ── Stats / Health ──

    async def handle_stats(request: Request) -> JSONResponse:
        return JSONResponse(proxy_stats.snapshot())

    async def handle_health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "version": "1.0.0"})

    # ── Helpers ──

    def _forward_headers(request: Request) -> dict[str, str]:
        return {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}

    async def _forward_json(
        client: httpx.AsyncClient, url: str, headers: dict, body: dict,
        engine: ParitokEngine, stubbed_tools: list[dict],
    ) -> JSONResponse:
        try:
            resp = await client.post(url, headers=headers, json=body)
        except httpx.ConnectError as e:
            return JSONResponse(content={"error": f"Cannot connect to {url}: {e}"}, status_code=502)
        except httpx.TimeoutException:
            return JSONResponse(content={"error": f"Upstream timed out: {url}"}, status_code=504)

        try:
            resp_body = resp.json()
        except Exception:
            return JSONResponse(
                content={"error": f"Upstream returned invalid JSON. Status: {resp.status_code}"},
                status_code=502,
            )

        # Resolve virtual tool calls via engine (per-request stubbed_tools)
        # Anthropic format: resp_body.content[].type == "tool_use"
        for block in resp_body.get("content", []):
            if block.get("type") == "tool_use":
                result = engine.resolve_virtual_call(
                    block.get("name", ""), block.get("input", {}),
                    stubbed_tools=stubbed_tools,
                )
                if result is not None:
                    block["_paritok_resolved"] = result

        # OpenAI format: resp_body.choices[].message.tool_calls[].function
        for choice in resp_body.get("choices", []):
            message = choice.get("message", {})
            for tc in message.get("tool_calls", []):
                fn = tc.get("function", {})
                name = fn.get("name", "")
                # OpenAI arguments is a JSON string, must parse
                args_str = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_str)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                result = engine.resolve_virtual_call(
                    name, args, stubbed_tools=stubbed_tools,
                )
                if result is not None:
                    tc["_paritok_resolved"] = result

        return JSONResponse(
            content=resp_body,
            status_code=resp.status_code,
            headers={k: v for k, v in resp.headers.items()
                     if k.lower() not in ("content-length", "content-encoding", "transfer-encoding")},
        )

    async def _stream_response(
        client: httpx.AsyncClient, url: str, headers: dict, body: dict,
    ) -> StreamingResponse:
        async def event_generator():
            async with client.stream("POST", url, headers=headers, json=body) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"cache-control": "no-cache", "x-paritok-proxy": "true"},
        )

    # ── Proxy-executed virtual tools (Anthropic) ──
    #
    # expand_context / gateway_search_tools live only in the model's tool list. The
    # client agent (Claude Code, ...) has no local handler for them, so the proxy
    # answers them in a short loop and keeps them off the wire back to the client —
    # the agent only ever receives the finished turn. This is not agent-specific;
    # any Anthropic Messages caller gets it.

    _MAX_RESOLVE_ROUNDS = 5
    _STREAM_HEADERS = {"cache-control": "no-cache", "x-paritok-proxy": "true"}
    _VIRTUAL_MARKERS = (b"read_original", b"expand_context", b"gateway_search_tools")
    _REF_GUIDANCE = (
        "A few earlier tool outputs here were shrunk to save context. Shrunk text opens "
        "with a marker like `[REF:<id> src=<path>]` and is followed by a lossy summary "
        "(reformatted, with blank lines / docstrings / some lines dropped) that does NOT "
        "match the real file byte-for-byte. When the summary already covers the step, keep "
        "going with it. "
        "To EDIT such a file, just write the Edit straight from the summary — your "
        "`old_string` does NOT need to match the real bytes exactly. The system "
        "automatically reconciles your edit against the true file (restoring dropped "
        "docstrings, comments and exact whitespace), so it applies on the first try. "
        "Do NOT call read_original merely to make an edit. "
        "Only call `read_original` with the `<id>` when you must study the exact untouched "
        "bytes for their own sake (not to edit). "
        "Do NOT try to reconstruct the file by dumping it with Bash (cat / sed / head / "
        "tail) or a fresh Read — those outputs get shrunk again, so you just get the same "
        "lossy view."
    )

    def _prepend_ref_guidance(system):
        """Fold the [REF:] guidance into whatever system prompt the caller supplied."""
        hint = {"type": "text", "text": _REF_GUIDANCE}
        if isinstance(system, str):
            return f"{system}\n\n{_REF_GUIDANCE}"
        if isinstance(system, list):
            return [*system, hint]
        return [hint]  # None, or an unexpected shape

    def _relay_headers(resp) -> dict:
        drop = {"content-length", "content-encoding", "transfer-encoding"}
        if resp is None:
            return {}
        return {k: v for k, v in resp.headers.items() if k.lower() not in drop}

    def _virtual_call_output(resolved: dict | None) -> str:
        if not resolved:
            return ""
        payload = resolved.get("content", resolved.get("tools"))
        if payload is None:
            return ""
        return payload if isinstance(payload, str) else json.dumps(payload)

    def _is_virtual_use(block) -> bool:
        return (isinstance(block, dict) and block.get("type") == "tool_use"
                and is_virtual_tool_call(block.get("name", "")))

    def _conceal_virtual_calls(message: dict) -> dict:
        """Drop virtual tool_use blocks the client cannot run; if that leaves no
        runnable call, relax stop_reason so the client treats it as a plain turn."""
        blocks = message.get("content")
        if not isinstance(blocks, list):
            return message
        visible = [b for b in blocks if not _is_virtual_use(b)]
        if len(visible) == len(blocks):
            return message
        patched = {**message, "content": visible}
        still_calling = any(isinstance(b, dict) and b.get("type") == "tool_use" for b in visible)
        if not still_calling and patched.get("stop_reason") == "tool_use":
            patched["stop_reason"] = "end_turn"
        return patched

    def _shadow_ref(call_input: dict) -> str:
        return (call_input.get("shadow_id") or call_input.get("id") or "").strip()

    def _pin_expanded(ref: str) -> None:
        """After the model expands a ref, stop it from being re-compressed (and thus
        re-expanded) every turn. If the ref has a source PATH (Read/Edit), pin the path
        so future reads of that file pass through verbatim. Always also pin the CONTENT
        (shadow_id) itself, so path-less content (Bash / Grep / pytest output — which has
        no file_path and so can't be pinned by path) is covered too."""
        import os as _os  # TEMP A/B gate for pin testing; not committed
        if not ref or _os.environ.get("PARITOK_NO_PIN"):
            return
        src = engine.storage.get_path_for_shadow(ref)
        if src:
            engine.storage.pin_source(src)
        engine.storage.pin_shadow(ref)

    _EDIT_TOOLS = ("Edit", "str_replace", "str_replace_editor")

    def _recover_edits(message: dict) -> dict:
        """Fix Edit tool calls the model authored against a lossy [REF] summary.

        The model's `old_string` reflects the compressed view (collapsed signature,
        dropped docstring/blank lines) and won't match the real file, so the client's
        exact-match Edit would fail and it would re-read. Map it back onto the true
        original (from the shadow store) and rewrite old_string/new_string so the Edit
        applies first try, preserving the docstrings/comments/formatting the summary
        dropped. On any ambiguity we leave the call untouched (the model can expand)."""
        blocks = message.get("content")
        if not isinstance(blocks, list):
            return message
        for b in blocks:
            if not (isinstance(b, dict) and b.get("type") == "tool_use"
                    and b.get("name") in _EDIT_TOOLS):
                continue
            inp = b.get("input") or {}
            fp = inp.get("file_path") or inp.get("path")
            old, new = inp.get("old_string"), inp.get("new_string")
            if not (fp and isinstance(old, str) and isinstance(new, str) and old):
                continue
            # Try each read stored for this path, most recent first. The latest read may
            # be a tiny offset/limit slice (which alone can't contain a multi-line edit),
            # so fall back to an earlier, fuller read of the same file.
            sids = engine.storage.get_shadows_for_path(fp)
            if not sids:
                continue
            out = None
            already = False
            for sid in sids:
                original = engine.storage.retrieve(sid)
                if not original:
                    continue
                clean = strip_read_line_numbers(original)
                if old in clean:
                    already = True
                    break  # this read already matches the file; nothing to fix
                out = recover_edit(old, new, clean)
                if out is not None:
                    break
            if already or out is None:
                continue
            client_old, client_new = out
            b["input"] = {**inp, "old_string": client_old, "new_string": client_new}
            proxy_stats.record_edit_recovery()
            logger.info("edit_recovery: rewrote Edit old_string for %s to match the file", fp)
        return message

    def _recover_codex_edits(reply: dict, originals: list[str]) -> dict:
        """Codex analog of `_recover_edits` for the Responses API. Codex has no Edit tool;
        it edits by running a shell command that does a literal string replace. When the
        file read it saw was compressed, the 4B summary reflowed multi-line code onto one
        line, so codex's `old` no longer matches the real file and the replace is a no-op.

        Rewrite the literal replace(s) inside each shell `function_call` so `old` matches
        the true file (recovered from `originals` — the real read bodies from this request).
        Only unique, quote-safe recoveries are spliced; anything else is left as codex sent
        it (it falls back to its own re-read, exactly as today)."""
        if not originals:
            return reply
        output = reply.get("output")
        if not isinstance(output, list):
            return reply
        for item in output:
            if not isinstance(item, dict):
                continue
            itype = item.get("type")
            if itype == "function_call":
                args_raw = item.get("arguments")
                if not isinstance(args_raw, str):
                    continue
                new_args, n = rewrite_function_call_arguments(args_raw, originals)
                if n:
                    item["arguments"] = new_args
            elif itype == "custom_tool_call":
                # Codex's newer freeform tools (incl. apply_patch) carry a raw
                # command/patch string under `input` rather than JSON `arguments`.
                inp = item.get("input")
                if not isinstance(inp, str):
                    continue
                new_inp, n = rewrite_tool_call_input(inp, originals)
                if n:
                    item["input"] = new_inp
            else:
                continue
            if n:
                proxy_stats.record_edit_recovery()
                logger.info("codex edit_recovery: rewrote a %s edit to match the file", itype)
        return reply

    def _parse_sse_to_message(raw: bytes) -> dict:
        """Reassemble a finished Anthropic message from a buffered SSE byte stream."""
        msg: dict = {"content": []}
        blocks: dict = {}
        frag: dict = {}
        for chunk in raw.split(b"\n\n"):
            data = None
            for line in chunk.split(b"\n"):
                if line.startswith(b"data:"):
                    data = line[5:].strip()
            if not data:
                continue
            try:
                ev = json.loads(data)
            except Exception:
                continue
            t = ev.get("type")
            if t == "message_start":
                m = ev.get("message", {}) or {}
                for k in ("id", "model", "role"):
                    msg[k] = m.get(k)
                msg["usage"] = m.get("usage", {})
            elif t == "content_block_start":
                i = ev.get("index")
                blocks[i] = dict(ev.get("content_block", {}) or {})
                frag[i] = ""
            elif t == "content_block_delta":
                i = ev.get("index")
                d = ev.get("delta", {}) or {}
                dt = d.get("type")
                if dt == "text_delta":
                    blocks[i]["text"] = blocks[i].get("text", "") + d.get("text", "")
                elif dt == "input_json_delta":
                    frag[i] += d.get("partial_json", "")
                elif dt == "thinking_delta":
                    blocks[i]["thinking"] = blocks[i].get("thinking", "") + d.get("thinking", "")
                elif dt == "signature_delta":
                    blocks[i]["signature"] = d.get("signature", "")
            elif t == "content_block_stop":
                i = ev.get("index")
                if i in blocks and blocks[i].get("type") == "tool_use" and frag.get(i):
                    try:
                        blocks[i]["input"] = json.loads(frag[i])
                    except Exception:
                        pass
            elif t == "message_delta":
                d = ev.get("delta", {}) or {}
                msg["stop_reason"] = d.get("stop_reason")
                msg["stop_sequence"] = d.get("stop_sequence")
                u = ev.get("usage") or {}
                if u:
                    msg["usage"] = {**(msg.get("usage") or {}), **u}
        msg["content"] = [blocks[i] for i in sorted(blocks)]
        return msg

    async def _anthropic_resolve(url, headers, body, stubbed_tools):
        """Loop the upstream (non-streaming), answering virtual tool calls ourselves,
        until a plain turn returns. Yields (message, status_code, response_headers)."""
        body = {**body, "stream": False}
        model = body.get("model", "")
        thread = list(body.get("messages", []))
        served_refs: set[str] = set()
        reply, resp = {}, None

        for _round in range(_MAX_RESOLVE_ROUNDS):
            body["messages"] = thread
            resp = await http_client.post(url, headers=headers, json=body)
            try:
                reply = resp.json()
            except Exception:
                return {"error": f"Upstream returned invalid JSON. Status: {resp.status_code}"}, 502, {}

            pending = anth_adapter.find_virtual_tool_uses(reply)
            if not pending:
                return reply, resp.status_code, _relay_headers(resp)

            # A real, client-side tool call in the same turn: yield control so the
            # client can run it (with the virtual ones concealed).
            if any(isinstance(b, dict) and b.get("type") == "tool_use"
                   and not is_virtual_tool_call(b.get("name", ""))
                   for b in reply.get("content", [])):
                return _conceal_virtual_calls(reply), resp.status_code, _relay_headers(resp)

            thread = [*thread, {"role": "assistant", "content": reply.get("content", [])}]
            results, fresh = [], 0
            for call in pending:
                args = call.get("input", {}) or {}
                ref = _shadow_ref(args)
                if ref and ref in served_refs:
                    out = ("[That reference was already expanded in this turn; "
                           "use the content provided just above.]")
                else:
                    if ref:
                        served_refs.add(ref)
                    # Count ANY freshly-answered virtual call as progress, not only
                    # ref-expansions — a gateway_search_tools call carries a query, not a
                    # shadow ref, so gating on `ref` here dropped its results and returned
                    # before the model ever saw the recovered tools. Bounded by
                    # _MAX_RESOLVE_ROUNDS; an already-served ref hits the branch above and
                    # still does not count as fresh.
                    fresh += 1
                    if is_expand_call(call.get("name", "")):
                        args = {**args, "shadow_id": ref}
                    out = _virtual_call_output(
                        engine.resolve_virtual_call(call.get("name", ""), args,
                                                    stubbed_tools=stubbed_tools))
                    if is_expand_call(call.get("name", "")):
                        proxy_stats.record_expansion(count_tokens(out, model), model)
                        import os as _dbgos  # TEMP expand-diagnostic; not committed
                        _dbgp = _dbgos.environ.get("PARITOK_EXPAND_LOG")
                        if _dbgp:
                            _src = engine.storage.get_path_for_shadow(ref)
                            _why = " ".join(
                                b.get("text", "") for b in reply.get("content", [])
                                if isinstance(b, dict) and b.get("type") == "text"
                            ).strip().replace("\n", " ")
                            with open(_dbgp, "a", encoding="utf-8") as _fh:
                                _fh.write(f"ref={ref} src={_src} out_tok={count_tokens(out, model)}\n")
                                _fh.write(f"  WHY: {_why[:600]}\n")
                        _pin_expanded(ref)
                results.append({"type": "tool_result",
                                "tool_use_id": call.get("id", ""), "content": out})
            thread = [*thread, {"role": "user", "content": results}]

            if fresh == 0:  # only repeats of already-served refs — stop looping
                return _conceal_virtual_calls(reply), resp.status_code, _relay_headers(resp)

        return (_conceal_virtual_calls(reply),
                resp.status_code if resp is not None else 502, _relay_headers(resp))

    async def _emit_once(payload: bytes):
        yield payload

    def _sse_stream(payload: bytes, status: int = 200):
        return StreamingResponse(_emit_once(payload), media_type="text/event-stream",
                                 status_code=status, headers=_STREAM_HEADERS)

    async def _stream_anthropic(url, headers, body, stubbed_tools):
        """Pull the upstream stream into memory. With no virtual tool call present,
        hand the exact bytes to the client. Otherwise re-run non-streaming, resolve,
        and rebuild the event stream from the finished message."""
        collected = bytearray()
        status = 200
        try:
            async with http_client.stream("POST", url, headers=headers,
                                          json={**body, "stream": True}) as resp:
                status = resp.status_code
                async for piece in resp.aiter_bytes():
                    collected.extend(piece)
        except httpx.ConnectError as e:
            return JSONResponse(content={"error": f"Cannot connect to {url}: {e}"}, status_code=502)
        except httpx.TimeoutException:
            return JSONResponse(content={"error": f"Upstream timed out: {url}"}, status_code=504)

        raw = bytes(collected)
        import os as _uos  # TEMP per-request upstream-usage diagnostic; not committed
        _up = _uos.environ.get("PARITOK_USAGE_LOG")
        if _up:
            try:
                _u = (_parse_sse_to_message(raw).get("usage") or {})
                with open(_up, "a", encoding="utf-8") as _fh:
                    _fh.write(f"req#{proxy_stats.requests_processed} "
                              f"in={_u.get('input_tokens')} "
                              f"cache_cr={_u.get('cache_creation_input_tokens')} "
                              f"cache_rd={_u.get('cache_read_input_tokens')} "
                              f"out={_u.get('output_tokens')}\n")
            except Exception:
                pass
        has_virtual = any(marker in raw for marker in _VIRTUAL_MARKERS)
        has_edit = any(f'"{t}"'.encode() in raw for t in _EDIT_TOOLS)
        if not has_virtual and not has_edit:
            return _sse_stream(raw, status)  # untouched pass-through

        if has_virtual:
            # A virtual tool call is present — resolve it server-side (may re-POST).
            try:
                message, _st, _hd = await _anthropic_resolve(url, headers, body, stubbed_tools)
            except (httpx.ConnectError, httpx.TimeoutException):
                return _sse_stream(raw, status)
        else:
            # Only a client-side Edit is present: reassemble the message we already
            # buffered (no extra upstream call) and rewrite the Edit in place.
            message = _parse_sse_to_message(raw)
        if not isinstance(message.get("content"), list):
            return _sse_stream(raw, status)  # error payload — prefer the raw stream
        message = _recover_edits(message)
        return _sse_stream(_message_to_events(message))

    def _event(name: str, data: dict) -> str:
        return "".join(("event: ", name, "\ndata: ", json.dumps(data), "\n\n"))

    def _block_events(idx: int, block: dict) -> list[str]:
        """The start / delta(s) / stop events for a single content block."""
        kind = block.get("type")
        if kind == "text":
            opener = {"type": "text", "text": ""}
            deltas = [{"type": "text_delta", "text": block.get("text", "")}]
        elif kind == "thinking":
            opener = {"type": "thinking", "thinking": ""}
            deltas = [{"type": "thinking_delta", "thinking": block.get("thinking", "")}]
            if block.get("signature"):
                deltas.append({"type": "signature_delta", "signature": block["signature"]})
        elif kind == "tool_use":
            opener = {"type": "tool_use", "id": block.get("id", ""),
                      "name": block.get("name", ""), "input": {}}
            deltas = [{"type": "input_json_delta",
                       "partial_json": json.dumps(block.get("input", {}) or {})}]
        else:
            opener, deltas = block, []
        seq = [_event("content_block_start",
                      {"type": "content_block_start", "index": idx, "content_block": opener})]
        seq += [_event("content_block_delta",
                       {"type": "content_block_delta", "index": idx, "delta": d}) for d in deltas]
        seq.append(_event("content_block_stop", {"type": "content_block_stop", "index": idx}))
        return seq

    def _message_to_events(message: dict) -> bytes:
        """Re-express a finished Anthropic message as its streaming event sequence."""
        usage = message.get("usage") or {}
        tokens_in = int(usage.get("input_tokens", 0) or 0)
        tokens_out = int(usage.get("output_tokens", 0) or 0)
        events = [_event("message_start", {
            "type": "message_start",
            "message": {
                "id": message.get("id"), "type": "message", "role": "assistant",
                "model": message.get("model"), "stop_reason": None, "stop_sequence": None,
                "content": [],
                "usage": {
                    "input_tokens": tokens_in, "output_tokens": 0,
                    "cache_creation_input_tokens": int(usage.get("cache_creation_input_tokens", 0) or 0),
                    "cache_read_input_tokens": int(usage.get("cache_read_input_tokens", 0) or 0),
                },
            }})]
        for idx, block in enumerate(message.get("content") or []):
            if isinstance(block, dict):
                events.extend(_block_events(idx, block))
        events.append(_event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": message.get("stop_reason") or "end_turn",
                      "stop_sequence": message.get("stop_sequence")},
            "usage": {"output_tokens": tokens_out}}))
        events.append(_event("message_stop", {"type": "message_stop"}))
        return "".join(events).encode("utf-8")

    # ── Proxy-executed virtual tools (OpenAI / Codex) ──
    #
    # Mirror of the Anthropic path for the /v1/chat/completions API (Codex CLI and
    # any OpenAI Chat Completions caller). OpenAI differs in shape: tool calls live
    # in choices[].message.tool_calls[] with a JSON-string `arguments`, and each tool
    # result is its own `{"role": "tool", "tool_call_id": ...}` message.

    def _prepend_ref_guidance_openai(messages):
        """Fold the [REF:] guidance into the OpenAI system/developer message
        (or prepend one when the caller sent none)."""
        for i, msg in enumerate(messages):
            if msg.get("role") in ("system", "developer"):
                content = msg.get("content", "")
                if isinstance(content, str):
                    messages[i] = {**msg, "content": f"{content}\n\n{_REF_GUIDANCE}"}
                elif isinstance(content, list):
                    messages[i] = {**msg, "content": [*content,
                                   {"type": "text", "text": _REF_GUIDANCE}]}
                return messages
        return [{"role": "system", "content": _REF_GUIDANCE}, *messages]

    def _conceal_virtual_calls_openai(reply: dict) -> dict:
        """Drop virtual tool_calls the client can't run; relax finish_reason to
        'stop' when concealing leaves no runnable call."""
        for choice in reply.get("choices", []):
            message = choice.get("message", {})
            tcs = message.get("tool_calls")
            if not isinstance(tcs, list):
                continue
            visible = [tc for tc in tcs
                       if not is_virtual_tool_call(tc.get("function", {}).get("name", ""))]
            if len(visible) == len(tcs):
                continue
            if visible:
                message["tool_calls"] = visible
            else:
                message.pop("tool_calls", None)
                if choice.get("finish_reason") == "tool_calls":
                    choice["finish_reason"] = "stop"
        return reply

    async def _openai_resolve(url, headers, body, stubbed_tools):
        """Loop the upstream (non-streaming), answering virtual tool calls ourselves,
        until a plain turn returns. Returns (reply, status_code, response_headers)."""
        body = {**body, "stream": False}
        model = body.get("model", "")
        thread = list(body.get("messages", []))
        served_refs: set[str] = set()
        reply, resp = {}, None

        for _round in range(_MAX_RESOLVE_ROUNDS):
            body["messages"] = thread
            resp = await http_client.post(url, headers=headers, json=body)
            try:
                reply = resp.json()
            except Exception:
                return {"error": f"Upstream returned invalid JSON. Status: {resp.status_code}"}, 502, {}

            # Relay upstream errors untouched instead of trying to resolve virtual
            # tool calls on them. Some OpenAI-compatible providers return a non-dict
            # body for errors (e.g. Gemini sends a top-level `[{"error": ...}]`), which
            # would otherwise crash the parser below.
            if resp.status_code >= 400 or not isinstance(reply, dict):
                return reply, resp.status_code, _relay_headers(resp)

            pending = oai_adapter.find_virtual_tool_calls(reply)
            if not pending:
                return reply, resp.status_code, _relay_headers(resp)

            choices = reply.get("choices", [])
            message = choices[0].get("message", {}) if choices else {}
            all_calls = message.get("tool_calls", []) or []

            # A real, client-side tool call in the same turn: yield control so the
            # client can run it (with the virtual ones concealed).
            if any(not is_virtual_tool_call(tc.get("function", {}).get("name", ""))
                   for tc in all_calls):
                return _conceal_virtual_calls_openai(reply), resp.status_code, _relay_headers(resp)

            thread = [*thread, {"role": "assistant",
                                "content": message.get("content"),
                                "tool_calls": all_calls}]
            fresh = 0
            for tc in pending:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    args = {}
                ref = _shadow_ref(args)
                if ref and ref in served_refs:
                    out = ("[That reference was already expanded in this turn; "
                           "use the content provided just above.]")
                else:
                    if ref:
                        served_refs.add(ref)
                    # Count ANY freshly-answered virtual call as progress, not only
                    # ref-expansions — a gateway_search_tools call carries a query, not a
                    # shadow ref, so gating on `ref` here dropped its results and returned
                    # before the model ever saw the recovered tools. Bounded by
                    # _MAX_RESOLVE_ROUNDS; an already-served ref hits the branch above and
                    # still does not count as fresh.
                    fresh += 1
                    if is_expand_call(fn.get("name", "")):
                        args = {**args, "shadow_id": ref}
                    out = _virtual_call_output(
                        engine.resolve_virtual_call(fn.get("name", ""), args,
                                                    stubbed_tools=stubbed_tools))
                    if is_expand_call(fn.get("name", "")):
                        proxy_stats.record_expansion(count_tokens(out, model), model)
                        _pin_expanded(ref)
                thread = [*thread, {"role": "tool",
                                    "tool_call_id": tc.get("id", ""), "content": out}]

            if fresh == 0:  # only repeats of already-served refs — stop looping
                return _conceal_virtual_calls_openai(reply), resp.status_code, _relay_headers(resp)

        return (_conceal_virtual_calls_openai(reply),
                resp.status_code if resp is not None else 502, _relay_headers(resp))

    def _openai_message_to_sse(reply: dict) -> bytes:
        """Re-express a finished OpenAI chat completion as its streaming chunk
        sequence (one content delta + a finish chunk per choice, then [DONE])."""
        base = {"id": reply.get("id", "chatcmpl-paritok"),
                "object": "chat.completion.chunk",
                "created": reply.get("created", 0),
                "model": reply.get("model", "")}
        out = []
        choices = reply.get("choices") or [{"index": 0, "message": {}, "finish_reason": "stop"}]
        for choice in choices:
            idx = choice.get("index", 0)
            message = choice.get("message", {})
            delta = {"role": "assistant"}
            if message.get("content") is not None:
                delta["content"] = message.get("content")
            if message.get("tool_calls"):
                delta["tool_calls"] = [
                    {"index": i, "id": tc.get("id"), "type": tc.get("type", "function"),
                     "function": tc.get("function", {})}
                    for i, tc in enumerate(message.get("tool_calls", []))
                ]
            out.append("data: " + json.dumps(
                {**base, "choices": [{"index": idx, "delta": delta, "finish_reason": None}]}) + "\n\n")
            out.append("data: " + json.dumps(
                {**base, "choices": [{"index": idx, "delta": {},
                                      "finish_reason": choice.get("finish_reason", "stop")}]}) + "\n\n")
        out.append("data: [DONE]\n\n")
        return "".join(out).encode("utf-8")

    async def _stream_openai(url, headers, body, stubbed_tools):
        """Pull the upstream stream into memory. With no virtual tool call present,
        hand the exact bytes to the client. Otherwise re-run non-streaming, resolve,
        and rebuild the chunk stream from the finished completion."""
        collected = bytearray()
        status = 200
        try:
            async with http_client.stream("POST", url, headers=headers,
                                          json={**body, "stream": True}) as resp:
                status = resp.status_code
                async for piece in resp.aiter_bytes():
                    collected.extend(piece)
        except httpx.ConnectError as e:
            return JSONResponse(content={"error": f"Cannot connect to {url}: {e}"}, status_code=502)
        except httpx.TimeoutException:
            return JSONResponse(content={"error": f"Upstream timed out: {url}"}, status_code=504)

        raw = bytes(collected)
        if not any(marker in raw for marker in _VIRTUAL_MARKERS):
            return _sse_stream(raw, status)  # untouched pass-through

        try:
            reply, _st, _hd = await _openai_resolve(url, headers, body, stubbed_tools)
        except (httpx.ConnectError, httpx.TimeoutException):
            return _sse_stream(raw, status)
        if not reply.get("choices"):
            return _sse_stream(raw, status)  # error payload — prefer the raw stream
        return _sse_stream(_openai_message_to_sse(reply))

    # ── Proxy-executed virtual tools (OpenAI Responses API / Codex) ──
    #
    # Codex speaks `/v1/responses`. Tool calls are `function_call` items in the
    # response `output[]`; results are fed back as `function_call_output` items
    # appended to `input`. Otherwise the resolve loop mirrors the other two paths.

    def _prepend_ref_guidance_responses(instructions):
        """Fold the [REF:] guidance into the Responses `instructions` (system) field."""
        if isinstance(instructions, str) and instructions.strip():
            return f"{instructions}\n\n{_REF_GUIDANCE}"
        return _REF_GUIDANCE

    async def _responses_resolve(url, headers, body, stubbed_tools):
        """Loop the upstream (non-streaming), answering virtual function_calls
        ourselves, until a plain turn returns. Returns (reply, status, headers)."""
        body = {**body, "stream": False}
        model = body.get("model", "")
        conv = resp_adapter.normalize_input(body.get("input", []))
        served_refs: set[str] = set()
        reply, resp = {}, None

        for _round in range(_MAX_RESOLVE_ROUNDS):
            body["input"] = conv
            resp = await http_client.post(url, headers=headers, json=body)
            try:
                reply = resp.json()
            except Exception:
                return {"error": f"Upstream returned invalid JSON. Status: {resp.status_code}"}, 502, {}

            pending = resp_adapter.find_virtual_function_calls(reply)
            if not pending:
                return reply, resp.status_code, _relay_headers(resp)

            # A real, client-side function_call in the same turn: yield control.
            if resp_adapter.has_real_function_call(reply):
                return resp_adapter.conceal_virtual_calls(reply), resp.status_code, _relay_headers(resp)

            # Carry the model's output items forward, then answer each virtual call.
            conv = [*conv, *(reply.get("output") or [])]
            fresh = 0
            for call in pending:
                try:
                    args = json.loads(call.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    args = {}
                ref = _shadow_ref(args)
                if ref and ref in served_refs:
                    out = ("[That reference was already expanded in this turn; "
                           "use the content provided just above.]")
                else:
                    if ref:
                        served_refs.add(ref)
                    # Count ANY freshly-answered virtual call as progress, not only
                    # ref-expansions — a gateway_search_tools call carries a query, not a
                    # shadow ref, so gating on `ref` here dropped its results and returned
                    # before the model ever saw the recovered tools. Bounded by
                    # _MAX_RESOLVE_ROUNDS; an already-served ref hits the branch above and
                    # still does not count as fresh.
                    fresh += 1
                    if is_expand_call(call.get("name", "")):
                        args = {**args, "shadow_id": ref}
                    out = _virtual_call_output(
                        engine.resolve_virtual_call(call.get("name", ""), args,
                                                    stubbed_tools=stubbed_tools))
                    if is_expand_call(call.get("name", "")):
                        proxy_stats.record_expansion(count_tokens(out, model), model)
                        import os as _dbgos  # TEMP expand-diagnostic; not committed
                        _dbgp = _dbgos.environ.get("PARITOK_EXPAND_LOG")
                        if _dbgp:
                            _src = engine.storage.get_path_for_shadow(ref)
                            with open(_dbgp, "a", encoding="utf-8") as _fh:
                                _fh.write(f"ref={ref} src={_src} out_tok={count_tokens(out, model)}\n")
                        _pin_expanded(ref)
                conv = [*conv, {"type": "function_call_output",
                                "call_id": call.get("call_id", ""), "output": out}]

            if fresh == 0:
                return resp_adapter.conceal_virtual_calls(reply), resp.status_code, _relay_headers(resp)

        return (resp_adapter.conceal_virtual_calls(reply),
                resp.status_code if resp is not None else 502, _relay_headers(resp))

    def _responses_to_sse(reply: dict) -> bytes:
        """Re-express a finished Responses object as a typed-event stream.

        Must emit the full item lifecycle, not just text deltas: clients (Codex)
        drop an `output_text.delta` that arrives without a preceding
        `output_item.added` + `content_part.added` ("OutputTextDelta without
        active item"). Sequence: created → in_progress → per item
        (output_item.added → [content_part.added → output_text.delta →
        output_text.done → content_part.done] or function_call_arguments.* →
        output_item.done) → completed → [DONE]. function_call items are carried
        through too, so the client can still run real tool calls."""
        seq = 0

        def ev(name, data):
            nonlocal seq
            data = {"type": name, "sequence_number": seq, **data}
            seq += 1
            return f"event: {name}\ndata: {json.dumps(data)}\n\n"

        shell = {"id": reply.get("id", ""), "object": "response", "status": "in_progress"}
        out = [ev("response.created", {"response": shell}),
               ev("response.in_progress", {"response": shell})]

        for idx, item in enumerate(reply.get("output") or []):
            item_id = item.get("id") or f"item_{idx}"
            item = {**item, "id": item_id}
            itype = item.get("type")

            if itype == "message":
                # Announce the shell item (empty content), then fill each part.
                out.append(ev("response.output_item.added",
                              {"output_index": idx, "item": {**item, "content": []}}))
                for cidx, part in enumerate(item.get("content") or []):
                    if part.get("type") != "output_text":
                        continue
                    text = part.get("text", "")
                    empty_part = {"type": "output_text", "text": "", "annotations": []}
                    out.append(ev("response.content_part.added",
                                  {"item_id": item_id, "output_index": idx,
                                   "content_index": cidx, "part": empty_part}))
                    out.append(ev("response.output_text.delta",
                                  {"item_id": item_id, "output_index": idx,
                                   "content_index": cidx, "delta": text}))
                    out.append(ev("response.output_text.done",
                                  {"item_id": item_id, "output_index": idx,
                                   "content_index": cidx, "text": text}))
                    out.append(ev("response.content_part.done",
                                  {"item_id": item_id, "output_index": idx,
                                   "content_index": cidx, "part": part}))
            elif itype == "function_call":
                args = item.get("arguments", "")
                out.append(ev("response.output_item.added",
                              {"output_index": idx, "item": item}))
                out.append(ev("response.function_call_arguments.delta",
                              {"item_id": item_id, "output_index": idx, "delta": args}))
                out.append(ev("response.function_call_arguments.done",
                              {"item_id": item_id, "output_index": idx, "arguments": args}))
            else:
                # Reasoning or any other item type: announce it, no inner deltas.
                out.append(ev("response.output_item.added",
                              {"output_index": idx, "item": item}))

            out.append(ev("response.output_item.done",
                          {"output_index": idx, "item": item}))

        out.append(ev("response.completed", {"response": reply}))
        out.append("data: [DONE]\n\n")
        return "".join(out).encode("utf-8")

    def _final_response_from_sse(raw: bytes) -> dict | None:
        """Pull the final Responses object out of a buffered SSE byte stream."""
        final = None
        for line in raw.split(b"\n"):
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == b"[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(obj, dict) and obj.get("type") in (
                    "response.completed", "response.incomplete", "response.failed"):
                resp = obj.get("response")
                if isinstance(resp, dict):
                    final = resp
        return final

    async def _stream_responses(url, headers, body, stubbed_tools, codex_read_bodies=None):
        """Buffer the upstream stream; replay verbatim unless the final response
        actually contains a virtual tool call (resolve non-streaming and rebuild the
        typed-event stream) or a codex shell edit we can reconcile against the real file.

        The trigger is a parsed check for a virtual `function_call`, NOT a byte
        scan: the Responses object echoes the request `tools` (so an injected
        `expand_context`/`gateway_search_tools` name appears in the bytes even
        when nothing was called), which would send every request down the
        rebuild path."""
        collected = bytearray()
        status = 200
        try:
            async with http_client.stream("POST", url, headers=headers,
                                          json={**body, "stream": True}) as resp:
                status = resp.status_code
                async for piece in resp.aiter_bytes():
                    collected.extend(piece)
        except httpx.ConnectError as e:
            return JSONResponse(content={"error": f"Cannot connect to {url}: {e}"}, status_code=502)
        except httpx.TimeoutException:
            return JSONResponse(content={"error": f"Upstream timed out: {url}"}, status_code=504)

        raw = bytes(collected)
        final = _final_response_from_sse(raw)

        import os as _ruos  # TEMP per-request responses-usage diagnostic; not committed
        _rup = _ruos.environ.get("PARITOK_RESP_USAGE_LOG")
        if _rup and final:
            _u = final.get("usage") or {}
            _itd = (_u.get("input_tokens_details") or {})
            _cached = _itd.get("cached_tokens", 0)
            _ti = _u.get("input_tokens", 0)
            _ninp = len(body.get("input", []) or [])
            with open(_rup, "a", encoding="utf-8") as _fh:
                _fh.write(f"input_items={_ninp} in={_ti} cached={_cached} "
                          f"uncached={_ti - _cached} out={_u.get('output_tokens', 0)}\n")
        if final is None:
            return _sse_stream(raw, status)

        if not resp_adapter.find_virtual_function_calls(final):
            # No virtual call. Still reconcile any codex shell edit against the real file;
            # rebuild the stream only if we actually rewrote something, else pass through.
            before = json.dumps(final.get("output"))
            fixed = _recover_codex_edits(final, codex_read_bodies or [])
            if json.dumps(fixed.get("output")) != before:
                return _sse_stream(_responses_to_sse(fixed))
            return _sse_stream(raw, status)  # true pass-through

        try:
            reply, _st, _hd = await _responses_resolve(url, headers, body, stubbed_tools)
        except (httpx.ConnectError, httpx.TimeoutException):
            return _sse_stream(raw, status)
        if not reply.get("output"):
            return _sse_stream(raw, status)  # error payload — prefer the raw stream
        reply = _recover_codex_edits(reply, codex_read_bodies or [])
        return _sse_stream(_responses_to_sse(reply))

    # ── Lifespan: clean up http_client on shutdown ──

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(_app):
        yield
        await http_client.aclose()

    # ── Build app ──

    routes = [
        Route("/v1/messages/count_tokens", handle_count_tokens, methods=["POST"]),
        Route("/v1/messages", handle_anthropic, methods=["POST"]),
        Route("/v1/chat/completions", handle_openai, methods=["POST"]),
        Route("/v1/responses", handle_responses, methods=["POST"]),
        Route("/stats", handle_stats, methods=["GET"]),
        Route("/health", handle_health, methods=["GET"]),
    ]

    return Starlette(routes=routes, lifespan=lifespan)


def _preflight_backend(config_path: str | None) -> None:
    """Check the compression backend before serving and warn (never abort) if it
    can't compress right now, so the user isn't surprised by passthrough."""
    from paritok.config import ParitokConfig

    config = ParitokConfig.load(config_path) if config_path else ParitokConfig.load()

    if config.use_gpu_server:
        from paritok.strategies.gpu_server import GpuServerStrategy

        available, message = GpuServerStrategy(config.gpu_server).check()
        if available:
            print(f"[paritok] Hosted GPU server OK — API key accepted "
                  f"({config.gpu_server.base_url}).")
            return
        print("\n" + "=" * 70)
        # `available` is false either because the key was rejected or the endpoint
        # is unreachable; `message` (from check()) says which — surface it verbatim.
        print("[paritok] WARNING: hosted compression is not available.")
        print(f"  {message}")
        print("  If you keep going, requests will NOT be compressed (passed through as-is).")
        print("  To compress, self-host the open model instead:")
        print("      ollama pull paritok/paritok-4b-v1")
        print("      ollama cp   paritok/paritok-4b-v1 paritok-4b-v1")
        print("      # then set use_gpu_server: false in paritok.yaml (the default)")
        print("=" * 70 + "\n")
        return

    # Self-hosted path: make sure Ollama is up and the model is pulled.
    from paritok.strategies.local_model import LocalModelStrategy

    if not LocalModelStrategy(config.local_model).is_available():
        print("\n" + "=" * 70)
        print("[paritok] WARNING: local compression model is not reachable.")
        print(f"  Expected Ollama at {config.local_model.base_url} serving "
              f"'{config.local_model.model}'.")
        print("  Start it with:")
        print("      ollama serve                       # if not already running")
        print("      ollama pull paritok/paritok-4b-v1")
        print(f"      ollama cp   paritok/paritok-4b-v1 {config.local_model.model}")
        print("  Requests will pass through UNCOMPRESSED until it is available.")
        print("=" * 70 + "\n")
    else:
        print(f"[paritok] Local model OK ('{config.local_model.model}' via Ollama).")


def run_proxy(
    host: str = "127.0.0.1",
    port: int = 8080,
    anthropic_base_url: str = "https://api.anthropic.com",
    openai_base_url: str = "https://api.openai.com",
    config_path: str | None = None,
    log_level: str = "info",
):
    """Start the proxy server."""
    try:
        import uvicorn
    except ImportError:
        raise ImportError("Proxy dependencies not installed. Run: pip install paritok[proxy]")

    _preflight_backend(config_path)

    app = create_app(
        anthropic_base_url=anthropic_base_url,
        openai_base_url=openai_base_url,
        config_path=config_path,
    )

    print(f"Paritok proxy starting on {host}:{port}")
    print(f"  Anthropic: set ANTHROPIC_BASE_URL=http://{host}:{port}")
    print(f"  OpenAI:    set OPENAI_BASE_URL=http://{host}:{port}")
    print(f"  Stats:     http://{host}:{port}/stats")
    print(f"  Health:    http://{host}:{port}/health")
    print()

    uvicorn.run(app, host=host, port=port, log_level=log_level)
