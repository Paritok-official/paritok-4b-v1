"""Unified Compression Pipeline.

One pipeline, one model. All context compression goes through the local
Ollama model. No rules, no heuristics — the model decides what to keep.

Pipeline steps:
1. Already-compressed check ([REF:] prefix)
2. Token threshold gating (min/max)
3. SHA256 cache dedup
4. Call Ollama model for compression
5. Effectiveness check (refusal_threshold)
6. Store original in shadow storage, tag with [REF:id]
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field

from paritok.config import ParitokConfig
from paritok.storage import ShadowStorage, MemoryShadowStorage, content_hash
from paritok.strategies.local_model import LocalModelStrategy
from paritok.token_counter import count_tokens

_REF_PATTERN = re.compile(r"^\[REF:[a-f0-9]+(?:\s+src=[^\]]*)?\]")

# Matches the line-number prefix added by Claude Code's Read tool (cat -n style):
# "     1\tactual line text". We strip these for content-equality checks so that
# re-reading the same file (even with different offset/limit) maps to the same
# stored shadow instead of triggering a fresh compression.
_LINE_NUMBER_PREFIX = re.compile(r"^\s*\d+\t", re.MULTILINE)


def _normalize_for_match(text: str) -> str:
    """Strip Read-tool line-number prefixes and trailing whitespace for
    similarity comparison only. Does NOT affect what gets stored."""
    return _LINE_NUMBER_PREFIX.sub("", text).strip()


def _sanitize_source(source: str) -> str:
    """Make a path safe to embed inside a [REF:id src=...] tag."""
    # ']' would break the tag; newlines would break line parsing. Replace.
    return source.replace("]", "_").replace("\n", " ").strip()

# Debug trace: when enabled (via `trace.enabled` in paritok.yaml, or the
# PARITOK_DEBUG_DUMP env var as an override), every compression event is appended
# to the trace file as JSONL. Lets the user diff original vs compressed per
# tool_result to catch hallucinations or paraphrase. View: tools/view_trace.py.
_DEBUG_DUMP_LOCK = threading.Lock()


def _resolve_trace_path(config) -> str | None:
    """Trace file path, or None if disabled. Env var wins over the yaml toggle."""
    env_path = os.environ.get("PARITOK_DEBUG_DUMP", "").strip()
    if env_path:
        return env_path
    trace = getattr(config, "trace", None)
    if trace is not None and getattr(trace, "enabled", False):
        return trace.path
    return None


@dataclass
class CompressionResult:
    compressed: str
    original_tokens: int
    compressed_tokens: int
    shadow_id: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def ratio(self) -> float:
        """Compression ratio (0.0 = no savings, 1.0 = 100% savings).

        Note: compressed_tokens includes the [REF:id] tag overhead (~5 tokens),
        so the ratio is slightly lower than the pure compression ratio.
        """
        if self.original_tokens == 0:
            return 0.0
        return round(1 - self.compressed_tokens / self.original_tokens, 3)

    @property
    def saved_tokens(self) -> int:
        return self.original_tokens - self.compressed_tokens


class CompressionPipeline:
    """Unified compression pipeline. Compresses any content via local Ollama model."""

    def __init__(
        self,
        config: ParitokConfig | None = None,
        storage: ShadowStorage | None = None,
    ):
        self.config = config or ParitokConfig()
        self.storage = storage or MemoryShadowStorage()
        # Active backend: self-hosted local model (Ollama), or the Paritok GPU
        # server (hosted endpoint). The GPU-server backend degrades to a no-op
        # passthrough when the hosted endpoint / GPU is unavailable.
        if self.config.use_gpu_server:
            from paritok.strategies.gpu_server import GpuServerStrategy
            self._model = GpuServerStrategy(self.config.gpu_server)
        else:
            self._model = LocalModelStrategy(self.config.local_model)
        # Where per-compression traces go (None = disabled).
        self._trace_path = _resolve_trace_path(self.config)

    def _debug_dump(self, record: dict) -> None:
        if not self._trace_path:
            return
        try:
            line = json.dumps(record, ensure_ascii=False)
        except (TypeError, ValueError):
            return
        with _DEBUG_DUMP_LOCK:
            try:
                with open(self._trace_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                pass

    def compress(
        self,
        content: str,
        *,
        query: str | None = None,
        level: str | None = None,
        kind: str | None = None,
        target_ratio: str | None = None,
        source: str | None = None,
    ) -> CompressionResult:
        """Compress content via the local SEG model.

        Args:
            content: Text to compress (tool output, conversation history, etc.)
            query: USER INTENT — the agent's current task. Drives keep/drop.
            level: SEG level L0-L3 (target ratio). Defaults to the model default (L0).
            kind: SEG kind (file_read, log_output, ...). If None, sniffed from content.
            target_ratio: Legacy ratio knob ("30%"/"0.3"), mapped to a level when
                `level` is not given. Prefer `level`.
            source: Optional source identifier (e.g. a file_path from Read).
                When given, enables path-keyed short-circuit: re-reading the
                same file — even partially, with line-number prefixes, or
                different offsets — returns the existing [REF:id] without
                re-invoking the local model.
        """
        cfg = self.config.compression
        original_tokens = count_tokens(content)
        t0 = time.time()

        # 1. Already-compressed check
        if _REF_PATTERN.match(content.strip()):
            return self._skip(content, original_tokens, "already_compressed")

        # 1b. Path-keyed short-circuit (Read short-circuit). Bypasses the
        # min/max token gates: if we have a prior ref for this exact source
        # path and the new content is byte-equal or a normalized substring
        # of the stored original, reuse the existing tag.
        if source:
            prior_sid = self.storage.get_shadow_for_path(source)
            if prior_sid:
                prior_content = self.storage.retrieve(prior_sid)
                cached_tag = self.storage.get_cached_compressed(prior_sid)
                if prior_content is not None and cached_tag is not None:
                    norm_new = _normalize_for_match(content)
                    norm_prior = _normalize_for_match(prior_content)
                    if norm_new and (
                        norm_new == norm_prior or norm_new in norm_prior
                    ):
                        compressed_tokens = count_tokens(cached_tag)
                        return CompressionResult(
                            compressed=cached_tag,
                            original_tokens=original_tokens,
                            compressed_tokens=compressed_tokens,
                            shadow_id=prior_sid,
                            metadata={
                                "path_shortcircuit": True,
                                "source": source,
                            },
                        )

        # 2. Too small
        if original_tokens < cfg.min_tokens:
            return self._skip(content, original_tokens, "below_min_tokens")

        # 3. Too large
        if original_tokens > cfg.max_tokens:
            return self._skip(content, original_tokens, "above_max_tokens")

        # sid is deterministic (SHA256 of content), same value in cache check and store
        sid = content_hash(content)

        # 4. Cache check (idempotent: same content always gets same sid)
        cached = self.storage.get_cached_compressed(sid)
        if cached is not None:
            if source:
                self.storage.set_shadow_for_path(source, sid)
            compressed_tokens = count_tokens(cached)
            return CompressionResult(
                compressed=cached,
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                shadow_id=sid,
                metadata={"cache_hit": True},
            )

        # 5. Call model (SEG protocol: intent + kind + level)
        compressed = self._model.compress(
            content,
            query=query,
            level=level,
            kind=kind,
            target_ratio=target_ratio,
        )

        # 6. Effectiveness check
        compressed_tokens = count_tokens(compressed)
        savings_ratio = 1 - compressed_tokens / original_tokens if original_tokens > 0 else 0
        if savings_ratio < cfg.refusal_threshold:
            return self._skip(content, original_tokens, "below_refusal_threshold")

        # 7. Store original + cache tagged result
        # [REF:sid src=...] tag adds ~5–15 tokens overhead to compressed_tokens
        self.storage.store(content)
        if source:
            tagged = f"[REF:{sid} src={_sanitize_source(source)}] {compressed}"
            self.storage.set_shadow_for_path(source, sid)
        else:
            tagged = f"[REF:{sid}] {compressed}"
        self.storage.cache_compressed(sid, tagged)

        tagged_tokens = count_tokens(tagged)

        self._debug_dump({
            "ts": round(time.time(), 3),
            "elapsed_s": round(time.time() - t0, 3),
            "query": query,
            "original_tokens": original_tokens,
            "compressed_tokens": tagged_tokens,
            "ratio": round(1 - tagged_tokens / original_tokens, 3) if original_tokens else 0.0,
            "shadow_id": sid,
            "original": content,
            "compressed": compressed,
        })

        return CompressionResult(
            compressed=tagged,
            original_tokens=original_tokens,
            compressed_tokens=tagged_tokens,
            shadow_id=sid,
            metadata={"cache_hit": False},
        )

    def _skip(self, content: str, original_tokens: int, reason: str) -> CompressionResult:
        self._debug_dump({
            "ts": round(time.time(), 3),
            "skipped": True,
            "reason": reason,
            "original_tokens": original_tokens,
        })
        return CompressionResult(
            compressed=content,
            original_tokens=original_tokens,
            compressed_tokens=original_tokens,
            metadata={"skipped": True, "reason": reason},
        )
