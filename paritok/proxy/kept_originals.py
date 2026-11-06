"""read_original(keep=true): originals the agent asked to keep for the rest of a conversation.

An expansion normally lives for one request: the next turn shows the summary again, and
an agent that needs the same original over several steps expands it every turn (loopbench
r16: one 1.4k-token file expanded 18 times in a row, 1.08M input tokens of hidden calls).
With keep=true the original stays — but it can't go back where the summary was: changing
content already sent rewrites the prompt cache from that point, and on Opus/Sonnet 5.5 it
invalidates the thinking blocks of every later assistant turn (requests replaying them
are rejected for new accounts).

So it is attached to the first content the client sends AFTER the turn that asked for
it — a message never sent upstream before — and re-attached there, byte-identical, on
every later request: nothing already sent changes, the cache prefix stays stable.

The target is the user message that answers the turn's first tool call (found by its
tool_use_id), or, if the turn ended without tool calls, the message right after it —
only when the assistant message before it is that same turn (its fingerprint matches).

The conversation key is the hash of the first user message, which two sessions opened
with the same prompt share. So a request whose history doesn't contain an entry's anchor
leaves the entry alone (it may belong to the other session, or this session's history
was compacted) instead of dropping it: dropping would remove an original from a message
the owning session already sent. Entries go away only by eviction — least recently used
conversation first, past MAX_CONVERSATIONS or MAX_TOTAL_CHARS.
"""

from __future__ import annotations

import copy
import hashlib
from collections import OrderedDict

MAX_CONVERSATIONS = 64
MAX_TOTAL_CHARS = 8_000_000  # ~2M tokens of kept originals across all conversations


def _header(ref: str, source: str | None) -> str:
    src = f" src={source}" if source else ""
    return (f"[read_original kept at your request: REF:{ref}{src}. This is the complete "
            f"original; it stays here for the rest of the conversation.]\n")


def turn_fingerprint(content) -> str | None:
    """Identity of an assistant turn as the client sends it back: its text, or failing
    that its thinking signatures. None when there is nothing to identify it by."""
    if isinstance(content, str):
        parts = ["t:" + content.strip()] if content.strip() else []
    else:
        blocks = [b for b in content or [] if isinstance(b, dict)]
        parts = ["t:" + b.get("text", "").strip() for b in blocks
                 if b.get("type") == "text" and b.get("text", "").strip()]
        if not parts:
            parts = ["s:" + str(b.get("signature") or b.get("data") or "") for b in blocks
                     if b.get("type") in ("thinking", "redacted_thinking")]
    if not parts:
        return None
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:24]


class KeptOriginals:
    def __init__(self, max_conversations: int = MAX_CONVERSATIONS,
                 max_total_chars: int = MAX_TOTAL_CHARS) -> None:
        self._by_conv: OrderedDict[str, list[dict]] = OrderedDict()
        self.max_conversations = max_conversations
        self.max_total_chars = max_total_chars

    def add(self, conv: str, ref: str, text: str, source: str | None,
            anchor_tool_use_id: str | None, anchor_index: int,
            anchor_fp: str | None = None) -> None:
        entries = self._by_conv.setdefault(conv, [])
        self._by_conv.move_to_end(conv)
        if any(e["ref"] == ref and e["anchor_id"] == anchor_tool_use_id
               and e["anchor_index"] == anchor_index for e in entries):
            return  # the same keep registered twice
        entries.append({"ref": ref, "text": _header(ref, source) + text,
                        "anchor_id": anchor_tool_use_id, "anchor_index": anchor_index,
                        "anchor_fp": anchor_fp})
        self._evict()

    def refs(self, conv: str) -> list[str]:
        return [e["ref"] for e in self._by_conv.get(conv, [])]

    def _evict(self) -> None:
        def total():
            return sum(len(e["text"]) for es in self._by_conv.values() for e in es)
        while len(self._by_conv) > self.max_conversations or (
                len(self._by_conv) > 1 and total() > self.max_total_chars):
            self._by_conv.popitem(last=False)

    @staticmethod
    def _target(messages: list[dict], entry: dict) -> int | None:
        if entry["anchor_id"]:
            for i, m in enumerate(messages):
                if m.get("role") == "user" and isinstance(m.get("content"), list) and any(
                        isinstance(b, dict) and b.get("type") == "tool_result"
                        and b.get("tool_use_id") == entry["anchor_id"] for b in m["content"]):
                    return i
            return None
        i = entry["anchor_index"]
        if not (1 <= i < len(messages)) or messages[i].get("role") != "user":
            return None
        prev = messages[i - 1]
        if prev.get("role") != "assistant" or entry.get("anchor_fp") is None:
            return None
        return i if turn_fingerprint(prev.get("content")) == entry["anchor_fp"] else None

    def attach(self, conv: str, messages: list[dict], count=None) -> tuple[list[dict], int]:
        """`messages` with every kept original of this conversation attached at its target
        (a copy; the input is untouched). Returns (messages, tokens attached)."""
        entries = self._by_conv.get(conv)
        if not entries:
            return messages, 0
        self._by_conv.move_to_end(conv)
        out = list(messages)
        added, attached_refs = 0, set()
        for e in entries:
            if len(out) <= e["anchor_index"] or e["ref"] in attached_refs:
                continue  # not reached yet (or a side request), or already attached earlier
            i = self._target(out, e)
            if i is None:
                continue  # another session under the same key, or compacted history
            attached_refs.add(e["ref"])
            m = copy.copy(out[i])
            content = m.get("content")
            blocks = [{"type": "text", "text": content}] if isinstance(content, str) else list(content or [])
            kept_block = {"type": "text", "text": e["text"]}
            # The client's cache breakpoint sits on the last block; appended after it, the
            # original would be billed uncached once before entering the cache. Moving the
            # marker onto it puts it inside the cached prefix from the first request.
            if blocks and isinstance(blocks[-1], dict) and "cache_control" in blocks[-1]:
                last = dict(blocks[-1])
                kept_block["cache_control"] = last.pop("cache_control")
                blocks[-1] = last
            blocks.append(kept_block)
            m["content"] = blocks
            out[i] = m
            added += count(e["text"]) if count else len(e["text"])
        return out, added
