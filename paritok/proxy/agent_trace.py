"""Agent-loop trace (opt-in, off by default).

Records what the gateway did on every request so a run can explain *where* a
coding agent spent extra turns when routed through paritok:

- what the model actually saw for each tool result (compressed / cache hit /
  path short-circuit / passthrough + reason), and when that view changed
  between turns;
- every upstream round, including the ones the client never sees (a streamed
  reply thrown away because it called a virtual tool, then the non-streaming
  read_original / gateway_search_tools resolve rounds), with their usage;
- virtual calls that were dropped unanswered because the same turn also made a
  real tool call;
- tool-schema filtering (kept / stubbed names) and history compression;
- the edit_recovery outcome of every Edit.

Enable with ``PARITOK_AGENT_TRACE=<path.jsonl>``. Each line is one JSON event
with an ``ev`` field; ``rid`` ties the events of one client request together
and ``msg_id`` / ``tool_use_id`` join them to the Claude Code transcript.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time

# Input keys worth keeping verbatim in a tool_use summary (small, and they say
# what was read / searched / run).
_BRIEF_KEYS = ("file_path", "path", "notebook_path", "offset", "limit", "pattern",
               "glob", "type", "output_mode", "-n", "-A", "-B", "-C", "head_limit",
               "shadow_id", "id", "query", "description", "replace_all", "url")
# Long string inputs: keep their size and a head, not the body.
_LONG_KEYS = ("old_string", "new_string", "content", "prompt")
_COMMAND_CHARS = 600
_HEAD_CHARS = 200
_TEXT_HEAD_CHARS = 600


def brief_input(inp) -> dict:
    """A compact, JSON-safe summary of a tool_use input."""
    if not isinstance(inp, dict):
        return {}
    out = {k: inp[k] for k in _BRIEF_KEYS if k in inp}
    if "command" in inp:
        out["command"] = str(inp["command"])[:_COMMAND_CHARS]
    for k in _LONG_KEYS:
        v = inp.get(k)
        if isinstance(v, str):
            out[f"{k}_len"] = len(v)
            out[f"{k}_head"] = v[:_HEAD_CHARS]
    if "edits" in inp and isinstance(inp["edits"], list):
        out["edits_n"] = len(inp["edits"])
    return out


def summarize_blocks(content) -> list[dict]:
    """One row per content block of an assistant message."""
    rows = []
    for b in content if isinstance(content, list) else []:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "text":
            text = b.get("text", "") or ""
            rows.append({"t": "text", "len": len(text), "head": text[:_TEXT_HEAD_CHARS]})
        elif t == "thinking":
            th = b.get("thinking", "") or ""
            rows.append({"t": "thinking", "len": len(th), "head": th[:_TEXT_HEAD_CHARS]})
        elif t == "redacted_thinking":
            rows.append({"t": "redacted_thinking"})
        elif t == "tool_use":
            rows.append({"t": "tool_use", "id": b.get("id"), "name": b.get("name"),
                         "input": brief_input(b.get("input"))})
        else:
            rows.append({"t": t})
    return rows


def content_tokens(messages, count) -> int:
    """Tokens of what the messages actually SAY: text, thinking text, tool inputs and
    tool-result text. Not json.dumps(messages): that also counts JSON escaping, block
    ids, and thinking `signature`s — opaque base64 the API doesn't bill as text, which
    on an Opus session was ~80% of the json.dumps count of a late request."""
    parts: list[str] = []
    for msg in messages or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, str):
            parts.append(content)
            continue
        for b in content if isinstance(content, list) else []:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text":
                parts.append(b.get("text") or "")
            elif t == "thinking":
                parts.append(b.get("thinking") or "")
            elif t == "tool_use":
                parts.append(json.dumps(b.get("input") or {}, ensure_ascii=False))
            elif t == "tool_result":
                inner = b.get("content")
                if isinstance(inner, str):
                    parts.append(inner)
                elif isinstance(inner, list):
                    parts.extend(x.get("text") or "" for x in inner
                                 if isinstance(x, dict) and x.get("type") == "text")
    return sum(count(p) for p in parts if p)


def usage_of(message: dict) -> dict:
    u = (message or {}).get("usage") or {}
    return {"in": u.get("input_tokens"), "cache_write": u.get("cache_creation_input_tokens"),
            "cache_read": u.get("cache_read_input_tokens"), "out": u.get("output_tokens")}


def _item_action(cr) -> str:
    md = cr.metadata or {}
    if md.get("skipped"):
        return "skipped"
    if md.get("path_shortcircuit"):
        return "path_shortcircuit"
    if md.get("cache_hit"):
        return "cache_hit"
    return "compressed"


def _text_sha(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", "replace")).hexdigest()[:16]


class AgentTrace:
    """Append-only JSONL event writer plus the little state needed to log
    changes (not repeats): each request re-sends the whole conversation, so
    per-item decisions, tool lists and compressed bodies are logged only when
    they first appear or change."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._rid = 0
        self._item_view: dict[tuple, tuple] = {}   # (conv, tool_use_id, part) -> (action, sha)
        self._tools_seen: dict[str, tuple] = {}    # conv -> (kept names, stubbed names)
        self._content_seen: set[str] = set()       # sha of forwarded bodies already logged
        self._history_seen: set[str] = set()

    @classmethod
    def from_env(cls) -> "AgentTrace | None":
        path = os.environ.get("PARITOK_AGENT_TRACE", "").strip()
        return cls(path) if path else None

    def next_rid(self) -> int:
        with self._lock:
            self._rid += 1
            return self._rid

    def emit(self, ev: str, **fields) -> None:
        rec = {"ev": ev, "ts": round(time.time(), 3), **fields}
        try:
            line = json.dumps(rec, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return
        with self._lock:
            try:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError:
                pass

    # ── request side ──

    def on_request(self, rid: int, *, conv: str, model: str, stream: bool,
                   messages: list, client_tools, forwarded_tools, stubbed, stats,
                   compress_s: float, msg_tokens_in: int | None = None,
                   msg_tokens_out: int | None = None) -> None:
        """Log one /v1/messages request after compression + tool discovery."""
        last = messages[-1] if messages else {}
        last_content = last.get("content")
        result_ids, last_text = [], ""
        if isinstance(last_content, list):
            for b in last_content:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    result_ids.append(b.get("tool_use_id"))
                elif isinstance(b, dict) and b.get("type") == "text" and not last_text:
                    last_text = b.get("text", "") or ""
        elif isinstance(last_content, str):
            last_text = last_content

        # Tool list: log names only when the kept/stubbed set changes for this conv.
        def _names(ts):
            return [t.get("name") or (t.get("function") or {}).get("name") or "?"
                    for t in (ts or [])]
        kept, stub = tuple(_names(forwarded_tools)), tuple(_names(stubbed))
        deferred = tuple(getattr(stats, "tools_deferred", None) or ())
        tools_changed = self._tools_seen.get(conv) != (kept, stub, deferred)
        if tools_changed:
            self._tools_seen[conv] = (kept, stub, deferred)

        items, counts = [], {}
        for it in getattr(stats, "items", None) or []:
            cr = it["cr"]
            action = _item_action(cr)
            counts[action] = counts.get(action, 0) + 1
            sha = _text_sha(cr.compressed)
            key = (conv, it.get("tool_use_id"), it.get("part", 0))
            prev = self._item_view.get(key)
            # Same forwarded bytes = same view, even if the label moved from
            # "compressed" to "cache_hit"/"path_shortcircuit" on a later turn.
            if prev is not None and prev[1] == sha:
                continue
            self._item_view[key] = (action, sha)
            items.append({
                "tool_use_id": it.get("tool_use_id"), "part": it.get("part", 0),
                "tool": it.get("tool"), "input": brief_input(it.get("input")),
                "action": action, "reason": (cr.metadata or {}).get("reason"),
                "orig_tok": cr.original_tokens, "fwd_tok": cr.compressed_tokens,
                "sid": cr.shadow_id, "fwd_sha": sha,
                # The model's view of an EARLIER result changed this turn (e.g. it
                # was passed through on a GPU miss and is compressed now).
                "view_changed": prev is not None,
                "prev_action": prev[0] if prev else None,
            })
            if action != "skipped" and sha not in self._content_seen:
                self._content_seen.add(sha)
                self.emit("content", rid=rid, conv=conv, sha=sha, sid=cr.shadow_id,
                          tool_use_id=it.get("tool_use_id"), tool=it.get("tool"),
                          source=(cr.metadata or {}).get("source") or it.get("source"),
                          action=action, orig_tok=cr.original_tokens,
                          fwd_tok=cr.compressed_tokens,
                          original=it.get("original", ""), forwarded=cr.compressed)

        hist = getattr(stats, "history", None)
        if hist and hist.get("summary"):
            hsha = _text_sha(hist["summary"])
            if hsha not in self._history_seen:
                self._history_seen.add(hsha)
                self.emit("history", rid=rid, conv=conv, turns=hist.get("turns"),
                          orig_tok=hist.get("orig_tok"), comp_tok=hist.get("comp_tok"),
                          summary=hist["summary"])

        self.emit(
            "request", rid=rid, conv=conv, model=model, stream=stream,
            n_msgs=len(messages), last_role=last.get("role"),
            last_tool_result_ids=result_ids, last_text_head=last_text[:_TEXT_HEAD_CHARS],
            n_tools_client=len(client_tools or []), n_tools_fwd=len(forwarded_tools or []),
            tools_kept=list(kept) if tools_changed else None,
            tools_stubbed=list(stub) if tools_changed else None,
            tools_deferred=list(deferred) if tools_changed else None,
            items_by_action=counts, items_changed=items,
            skipped_by_reason=getattr(stats, "skipped_by_reason", {}) or {},
            history_fired=bool(hist), compress_s=round(compress_s, 3),
            client_tool_search=bool(getattr(stats, "client_tool_search", False)),
            msg_tokens_in=msg_tokens_in, msg_tokens_out=msg_tokens_out,
            msg_tokens_basis="content",  # text/tool payloads only (see content_tokens)
        )

    # ── upstream side ──

    def on_upstream(self, rid: int, *, rnd: int, kind: str, status: int, message: dict,
                    elapsed_s: float, outcome: str, virtual: list | None = None,
                    concealed_unanswered: list | None = None) -> None:
        """One upstream POST. outcome: 'visible' (relayed to the client as-is),
        'discarded' (streamed reply thrown away and re-run non-streaming),
        'hidden' (a virtual-call round the client never sees), 'final' (the
        resolved turn the client gets)."""
        self.emit("upstream", rid=rid, round=rnd, kind=kind, status=status,
                  elapsed_s=round(elapsed_s, 3), outcome=outcome,
                  msg_id=(message or {}).get("id"),
                  stop_reason=(message or {}).get("stop_reason"),
                  usage=usage_of(message), blocks=summarize_blocks((message or {}).get("content")),
                  virtual=virtual or [], concealed_unanswered=concealed_unanswered or [],
                  error=(message or {}).get("error") if isinstance(message, dict) else None)

    def on_edit(self, rid: int, *, tool_use_id: str, file_path: str, outcome: str,
                model_old: str, model_new: str, n_reads: int,
                old_in_compressed: bool | None, client_old: str | None = None,
                client_new: str | None = None) -> None:
        """edit_recovery outcome for one Edit: no_shadow (file never compressed),
        matches_original, recovered, unrecoverable. For `recovered`, client_old /
        client_new are what the client actually applies (the rewrite)."""
        cap = 4000
        self.emit("edit", rid=rid, tool_use_id=tool_use_id, file_path=file_path,
                  outcome=outcome, n_reads=n_reads, old_in_compressed=old_in_compressed,
                  model_old=(model_old or "")[:cap], model_new=(model_new or "")[:cap],
                  client_old=None if client_old is None else client_old[:cap],
                  client_new=None if client_new is None else client_new[:cap])

    def on_final(self, rid: int, *, status: int, message: dict | None, total_s: float,
                 upstream_calls: int) -> None:
        msg = message or {}
        self.emit("final", rid=rid, status=status, msg_id=msg.get("id"),
                  stop_reason=msg.get("stop_reason"),
                  client_tool_uses=[{"id": b.get("id"), "name": b.get("name")}
                                    for b in msg.get("content") or []
                                    if isinstance(b, dict) and b.get("type") == "tool_use"],
                  total_s=round(total_s, 3), upstream_calls=upstream_calls)


class RequestTrace:
    """Per-request handle threaded through the forward/resolve helpers."""

    def __init__(self, trace: AgentTrace, rid: int):
        self.trace, self.rid, self.t0 = trace, rid, time.time()
        self.upstream_calls = 0

    def upstream(self, **kw) -> None:
        self.upstream_calls += 1
        self.trace.on_upstream(self.rid, **kw)

    def edit(self, **kw) -> None:
        self.trace.on_edit(self.rid, **kw)

    def final(self, status: int, message: dict | None) -> None:
        self.trace.on_final(self.rid, status=status, message=message,
                            total_s=time.time() - self.t0,
                            upstream_calls=self.upstream_calls)
