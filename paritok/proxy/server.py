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

import json
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger("paritok.proxy")


@dataclass
class ProxyStats:
    """Running statistics for the proxy."""
    requests_processed: int = 0
    total_original_tokens: int = 0
    total_compressed_tokens: int = 0
    total_tools_filtered: int = 0
    total_items_compressed: int = 0
    start_time: float = field(default_factory=time.time)

    def record(self, stats) -> None:
        """Fold one request's CompressionStats into the running totals."""
        self.requests_processed += 1
        self.total_original_tokens += stats.original_tokens
        self.total_compressed_tokens += stats.compressed_tokens
        self.total_tools_filtered += stats.tools_filtered
        self.total_items_compressed += stats.items_compressed

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def total_saved_tokens(self) -> int:
        return self.total_original_tokens - self.total_compressed_tokens

    @property
    def saved_percent(self) -> float:
        """Share of compressible tokens removed, 0-100."""
        if self.total_original_tokens == 0:
            return 0.0
        return round(100.0 * self.total_saved_tokens / self.total_original_tokens, 1)

    def snapshot(self) -> dict:
        """The /stats payload: raw totals plus the human-friendly derived rates."""
        reqs = self.requests_processed or 1  # avoid div-by-zero for averages
        return {
            "requests_processed": self.requests_processed,
            "total_original_tokens": self.total_original_tokens,
            "total_compressed_tokens": self.total_compressed_tokens,
            "total_saved_tokens": self.total_saved_tokens,
            "saved_percent": self.saved_percent,
            "avg_saved_tokens_per_request": round(self.total_saved_tokens / reqs, 1),
            "items_compressed": self.total_items_compressed,
            "total_tools_filtered": self.total_tools_filtered,
            "uptime_seconds": round(self.uptime_seconds, 1),
        }


def _tool_params(t: dict) -> dict:
    """Function-tool parameter schema, tolerating the injected virtual tools'
    Anthropic-style `input_schema` or the OpenAI-style `parameters`."""
    return t.get("parameters") or t.get("input_schema") or {"type": "object", "properties": {}}


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
    from paritok.pipelines.virtual import is_virtual_tool_call
    from paritok.proxy.adapters import anthropic as anth_adapter
    from paritok.proxy.adapters import openai as oai_adapter
    from paritok.proxy.adapters import responses as resp_adapter

    # Initialize
    config = ParitokConfig.load(config_path) if config_path else ParitokConfig()
    engine = ParitokEngine(config)
    proxy_stats = ProxyStats()
    if http_client is None:
        http_client = httpx.AsyncClient(timeout=120.0)

    # ── Anthropic handler ──

    async def handle_anthropic(request: Request) -> Response:
        body = json.loads(await request.body())
        parsed = anth_adapter.parse_request(body)

        # Use shared engine for compression + tool discovery
        parsed.messages, parsed.tools, stats, stubbed = engine.process_request(
            parsed.messages, parsed.tools
        )

        proxy_stats.record(stats)

        query = anth_adapter.extract_query(parsed.messages)
        logger.info("Request #%d, saved %d tokens, query=%s",
                     proxy_stats.requests_processed, stats.saved_tokens, (query or "")[:50])

        # If expand_context was injected, tell the model (once, via system) about the
        # [REF:] compression convention and that it can pull originals back on demand.
        # The proxy resolves expand_context itself (server-side), so the client agent
        # never needs to own the tool — this works for Claude Code, etc.
        if parsed.tools and any(t.get("name") == "expand_context" for t in parsed.tools):
            parsed.system = _prepend_ref_guidance(parsed.system)

        # Forward
        headers = _forward_headers(request)
        url = f"{anthropic_base_url}/v1/messages"
        forward_body = parsed.to_dict()

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
        return JSONResponse(content=final_body, status_code=status_code, headers=resp_headers)

    # ── OpenAI handler ──

    async def handle_openai(request: Request) -> Response:
        body = json.loads(await request.body())
        parsed = oai_adapter.parse_request(body)

        # OpenAI wraps tools in {"type": "function", "function": {...}}
        # Unwrap for engine, then re-wrap after
        raw_tools = None
        if parsed.tools:
            raw_tools = [t.get("function", t) for t in parsed.tools]

        _, processed_tools, stats, stubbed = engine.process_request(parsed.messages, raw_tools)

        # Compress tool messages (OpenAI uses role="tool" instead of tool_result blocks)
        query = oai_adapter.extract_query(parsed.messages)
        for i, msg in enumerate(parsed.messages):
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    cr = engine.pipeline.compress(content, query=query)
                    if not cr.metadata.get("skipped"):
                        parsed.messages[i] = {**msg, "content": cr.compressed}
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
                has_compressed=stats.items_compressed > 0,
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

        proxy_stats.record(stats)

        # If expand_context was injected, tell the model (once, via system) about the
        # [REF:] convention. The proxy resolves expand_context server-side (same as the
        # Anthropic path), so Codex / any OpenAI Chat Completions client never owns it.
        if processed_tools and any(t.get("name") == "expand_context" for t in processed_tools):
            parsed.messages = _prepend_ref_guidance_openai(parsed.messages)

        # Forward
        headers = _forward_headers(request)
        url = f"{openai_base_url}/v1/chat/completions"
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
        stubbed: list[dict] = []
        if raw_tools and query and len(raw_tools) > config.tool_discovery.top_k:
            result = engine.discovery.filter_tools(raw_tools, query)
            tools = result.tools
            stubbed = result.stubbed_tools
            stats.tools_original = result.original_count
            stats.tools_kept = result.kept_count
        elif raw_tools:
            stats.tools_original = stats.tools_kept = len(raw_tools)

        # Compress function_call_output items (the tool results that grow large).
        for i, item in enumerate(input_items):
            if (item.get("type") == "function_call_output"
                    and isinstance(item.get("output"), str) and item["output"].strip()):
                cr = engine.pipeline.compress(item["output"], query=query)
                if not cr.metadata.get("skipped"):
                    input_items[i] = {**item, "output": cr.compressed}
                    stats.original_tokens += cr.original_tokens
                    stats.compressed_tokens += cr.compressed_tokens
                    stats.items_compressed += 1

        # Inject virtual tools, then convert to the flat Responses tool shape.
        if tools is not None:
            tools = _inject_virtual_tools(
                tools,
                has_compressed=stats.items_compressed > 0,
                has_filtered=stats.tools_kept < stats.tools_original,
            )
            parsed.tools = [
                {"type": "function", "name": t.get("name"),
                 "description": t.get("description", ""), "parameters": _tool_params(t)}
                for t in tools
            ]
        parsed.input = input_items

        proxy_stats.record(stats)

        if parsed.tools and any(t.get("name") == "expand_context" for t in parsed.tools):
            parsed.instructions = _prepend_ref_guidance_responses(parsed.instructions)

        headers = _forward_headers(request)
        url = f"{openai_base_url}/v1/responses"
        forward_body = parsed.to_dict()

        if parsed.stream:
            return await _stream_responses(url, headers, forward_body, stubbed)
        try:
            final_body, status_code, resp_headers = await _responses_resolve(
                url, headers, forward_body, stubbed
            )
        except httpx.ConnectError as e:
            return JSONResponse(content={"error": f"Cannot connect to {url}: {e}"}, status_code=502)
        except httpx.TimeoutException:
            return JSONResponse(content={"error": f"Upstream timed out: {url}"}, status_code=504)
        return JSONResponse(content=final_body, status_code=status_code, headers=resp_headers)

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
    _VIRTUAL_MARKERS = (b"expand_context", b"gateway_search_tools")
    _REF_GUIDANCE = (
        "A few earlier tool outputs here were shrunk to save context. Shrunk text opens "
        "with a marker like `[REF:<id> src=<path>]` and is followed by a brief summary. "
        "When that summary already covers the step, keep going with it. When you genuinely "
        "need the untouched original — say, to study the code line by line or to change it — "
        "call `expand_context` with the `<id>` and the exact original is returned to you. "
        "Reach for `expand_context` rather than re-reading the same file via Read/Bash/Grep."
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

    async def _anthropic_resolve(url, headers, body, stubbed_tools):
        """Loop the upstream (non-streaming), answering virtual tool calls ourselves,
        until a plain turn returns. Yields (message, status_code, response_headers)."""
        body = {**body, "stream": False}
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
                        fresh += 1
                    if call.get("name") == "expand_context":
                        args = {**args, "shadow_id": ref}
                    out = _virtual_call_output(
                        engine.resolve_virtual_call(call.get("name", ""), args,
                                                    stubbed_tools=stubbed_tools))
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
        if not any(marker in raw for marker in _VIRTUAL_MARKERS):
            return _sse_stream(raw, status)  # untouched pass-through

        try:
            message, _st, _hd = await _anthropic_resolve(url, headers, body, stubbed_tools)
        except (httpx.ConnectError, httpx.TimeoutException):
            return _sse_stream(raw, status)
        if not isinstance(message.get("content"), list):
            return _sse_stream(raw, status)  # error payload — prefer the raw stream
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
                        fresh += 1
                    if fn.get("name") == "expand_context":
                        args = {**args, "shadow_id": ref}
                    out = _virtual_call_output(
                        engine.resolve_virtual_call(fn.get("name", ""), args,
                                                    stubbed_tools=stubbed_tools))
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
                        fresh += 1
                    if call.get("name") == "expand_context":
                        args = {**args, "shadow_id": ref}
                    out = _virtual_call_output(
                        engine.resolve_virtual_call(call.get("name", ""), args,
                                                    stubbed_tools=stubbed_tools))
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

    async def _stream_responses(url, headers, body, stubbed_tools):
        """Buffer the upstream stream; replay verbatim unless the final response
        actually contains a virtual tool call, in which case resolve it
        non-streaming and rebuild the typed-event stream.

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
        if final is None or not resp_adapter.find_virtual_function_calls(final):
            return _sse_stream(raw, status)  # no real virtual call → true pass-through

        try:
            reply, _st, _hd = await _responses_resolve(url, headers, body, stubbed_tools)
        except (httpx.ConnectError, httpx.TimeoutException):
            return _sse_stream(raw, status)
        if not reply.get("output"):
            return _sse_stream(raw, status)  # error payload — prefer the raw stream
        return _sse_stream(_responses_to_sse(reply))

    # ── Lifespan: clean up http_client on shutdown ──

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(_app):
        yield
        await http_client.aclose()

    # ── Build app ──

    routes = [
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
            print(f"[paritok] Hosted GPU server OK ({config.gpu_server.base_url}).")
            return
        print("\n" + "=" * 70)
        print("[paritok] WARNING: the hosted compression server is not reachable.")
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
