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
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field

from paritok.config import ParitokConfig
from paritok.storage import ShadowStorage, build_shadow_storage, content_hash
from paritok.strategies.chunking import CHUNK_SIZE
from paritok.strategies.local_model import (
    LocalModelStrategy,
    # num_ctx-reservation constants — imported (not duplicated) so the intent
    # budget below stays in lockstep with local_model's own prompt sizing.
    _CTX_SAFETY_MARGIN,
    _MIN_NUM_PREDICT,
    _TOKENIZER_SLACK,
)
from paritok.strategies.prompts import system_prompt_for_kind
from paritok.strategies.tagger import classify_kind_from_content
from paritok.token_counter import _DEFAULT_ENCODING, _get_encoder, count_tokens

logger = logging.getLogger(__name__)

# --- Intent (query) context-budget guard ------------------------------------
# The intent is injected into the system+user prompt of EVERY compression request,
# right next to a content chunk (<= CHUNK_SIZE). Ollama rejects a request whose
# prompt + num_predict exceeds the model's num_ctx window up-front (HTTP 400), so a
# pathologically large query (e.g. a whole SWE-bench issue, ~7k tokens) makes the
# backend 400 and the read pass through UNCOMPRESSED. Cap the intent so the request
# always fits. The reservation mirrors local_model._call_ollama exactly.
_INTENT_WRAPPER = (
    "USER INTENT:\n\n\nCompress the following segment under the rules in your "
    "system prompt. Output only the compressed [SEG]...[/SEG] block (or an empty "
    "one to drop):\n\n[SEG id=s1 kind=file_read level=L1]\n\n[/SEG]\n"
)
_INTENT_WRAPPER_TOKENS = count_tokens(_INTENT_WRAPPER, _DEFAULT_ENCODING)


def _intent_budget(system_tokens: int, content_tokens: int, num_ctx: int) -> int:
    """Max intent (query) tokens, in cl100k, that keep system + intent + a content
    chunk + a minimal generation under the model's num_ctx window.

    The content in any single request is at most one chunk (<= CHUNK_SIZE), so we
    reserve min(content, CHUNK_SIZE): a small read leaves more room for the intent.
    Uses local_model's SLACK / CTX margin / MIN_NUM_PREDICT so both sides agree on
    what fits. Never negative (0 = no room for any intent -> drop it entirely).
    """
    usable = int((num_ctx - _MIN_NUM_PREDICT - _CTX_SAFETY_MARGIN) / _TOKENIZER_SLACK)
    reserve = system_tokens + min(content_tokens, CHUNK_SIZE) + _INTENT_WRAPPER_TOKENS
    return max(usable - reserve, 0)


def _truncate_intent(query: str, budget: int) -> tuple[str, int, bool]:
    """Truncate `query` to `budget` cl100k tokens. Returns (possibly-truncated query,
    original token count, was_truncated)."""
    enc = _get_encoder(_DEFAULT_ENCODING)
    toks = enc.encode(query)
    if len(toks) <= budget:
        return query, len(toks), False
    return enc.decode(toks[:budget]), len(toks), True

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
        self.storage = storage or build_shadow_storage(self.config)
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
        # num_ctx the backend will run under (local Ollama or the hosted GPU server —
        # both ship 8192), used to size the intent budget. Warn at most once per
        # distinct oversized query so a big task doesn't spam every tool_result.
        self._intent_num_ctx = getattr(
            getattr(self.config, "local_model", None), "num_ctx", 8192
        )
        self._warned_intents: set[str] = set()

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
        upstream_model: str | None = None,
        model_input: str | None = None,
    ) -> CompressionResult:
        """Compress content via the local SEG model.

        Args:
            content: Text to compress (tool output, conversation history, etc.)
                Used for token counting, caching, shadow storage and the ratio —
                i.e. what the agent actually sent.
            model_input: Alternate text to FEED THE MODEL, when it should see a
                different form than `content` — e.g. codex's unnumbered source
                line-numbered so it's in-distribution. `content` still drives
                original_tokens / hash / store / expand, so the ratio reflects the
                real saving and `expand_context` returns the true original.
                Defaults to `content`.
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
        # Count with the UPSTREAM model's tokenizer (o200k for gpt-5/4.1/o3, ...)
        # so original/compressed token counts — and the savings the dashboard bills
        # on — match the provider's actual billing tokenizer, not a cl100k estimate.
        enc = upstream_model or _DEFAULT_ENCODING
        original_tokens = count_tokens(content, enc)
        t0 = time.time()

        # 1. Already-compressed check
        if _REF_PATTERN.match(content.strip()):
            return self._skip(content, original_tokens, "already_compressed")

        # 1a. Pin-on-expand: once the model has called read_original on this file it needs
        # the exact bytes, so stop shrinking that file — pass the read through VERBATIM.
        # The model then sees the current file directly and no longer re-expands it every
        # turn. This passes through whatever the client currently sends (always current),
        # so editing the file stays safe. Checked before the path short-circuit so a pinned
        # file never gets handed back a [REF] stub.
        if source and self.storage.is_source_pinned(source):
            return self._skip(content, original_tokens, "pinned")

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
                        compressed_tokens = count_tokens(cached_tag, enc)
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

        # 3b. Pinned content: the model already expanded this exact content (which has no
        # source path — e.g. Bash/pytest output), so pass it through VERBATIM instead of
        # re-compressing it and forcing the model to re-expand it every turn.
        if self.storage.is_shadow_pinned(sid):
            return self._skip(content, original_tokens, "pinned_shadow")

        # 4. Cache check (idempotent: same content always gets same sid)
        cached = self.storage.get_cached_compressed(sid)
        if cached is not None:
            if source:
                self.storage.set_shadow_for_path(source, sid)
            compressed_tokens = count_tokens(cached, enc)
            return CompressionResult(
                compressed=cached,
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                shadow_id=sid,
                metadata={"cache_hit": True},
            )

        # 5. Call model (SEG protocol: intent + kind + level). Feed model_input when
        # given (e.g. line-numbered form) — content still governs everything else.
        model_text = model_input if model_input is not None else content
        # Classify kind centrally so every backend receives a real kind. Without
        # this only LocalModelStrategy sniffs it internally; GpuServerStrategy would
        # forward kind=None to the server. Classify on model_text (what the model
        # actually sees), matching the local fallback.
        if kind is None:
            kind = classify_kind_from_content(model_text)
        # Cap the intent so system + intent + content chunk always fit num_ctx. A huge
        # query otherwise overflows the window and the backend 400s -> uncompressed
        # passthrough. Trims only the compressor's keep/drop guidance, never the real
        # request to the LLM.
        if query:
            query = self._cap_intent(query, model_text, kind)
        try:
            compressed = self._model.compress(
                model_text,
                query=query,
                level=level,
                kind=kind,
                target_ratio=target_ratio,
                upstream_model=upstream_model,
            )
        except Exception as e:
            # Compression is an optimization, not a hard dependency. If the backend
            # fails (timeout / model still loading, connection error, bad response),
            # forward the ORIGINAL content uncompressed instead of 500-ing the whole
            # request — otherwise the agent gets stuck in a retry loop (issue #36).
            logger.warning(
                "compression backend failed (%s: %s); forwarding original uncompressed",
                type(e).__name__, e,
            )
            return self._skip(content, original_tokens, f"backend_error:{type(e).__name__}")

        # 5b. An empty / whitespace-only body is a FAILED compression, not a perfect
        # one. count_tokens("") == 0 makes savings_ratio ≈ 1.0, which would sail past
        # the refusal-threshold check below and forward an EMPTY prompt for non-empty
        # input (the hosted /api/compress could relay "" straight from the GPU worker).
        # Keep the original — never emit empty compressed output (issue #38).
        if content.strip() and not (isinstance(compressed, str) and compressed.strip()):
            return self._skip(content, original_tokens, "empty_compression")

        # 6. Effectiveness check
        compressed_tokens = count_tokens(compressed, enc)
        savings_ratio = 1 - compressed_tokens / original_tokens if original_tokens > 0 else 0
        if savings_ratio < cfg.refusal_threshold:
            # A backend that echoes the input VERBATIM is a passthrough, not a weak
            # compression (a real compression always reflows / drops something). For the
            # hosted GPU that means the worker was unavailable or throttled — it returns
            # the original on gpu_available:false (#30). Record that as the reason instead
            # of "below_refusal_threshold", which reads like the content "didn't compress"
            # and sends debuggers to tune a threshold that has nothing to do with it.
            if compressed == model_text:
                reason = "gpu_unavailable" if self.config.use_gpu_server else "backend_passthrough"
            else:
                reason = "below_refusal_threshold"
            return self._skip(content, original_tokens, reason)

        # 7. Store original + cache tagged result
        # [REF:sid src=...] tag adds ~5–15 tokens overhead to compressed_tokens
        self.storage.store(content)
        if source:
            tagged = f"[REF:{sid} src={_sanitize_source(source)}] {compressed}"
            self.storage.set_shadow_for_path(source, sid)
        else:
            tagged = f"[REF:{sid}] {compressed}"
        self.storage.cache_compressed(sid, tagged)

        tagged_tokens = count_tokens(tagged, enc)

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

    def _cap_intent(self, query: str, content_text: str, kind: str) -> str:
        """Truncate the intent (query) to the budget that keeps this request under
        num_ctx (see `_intent_budget`), warning once per distinct oversized query."""
        system_tokens = count_tokens(system_prompt_for_kind(kind), _DEFAULT_ENCODING)
        content_tokens = count_tokens(content_text, _DEFAULT_ENCODING)
        budget = _intent_budget(system_tokens, content_tokens, self._intent_num_ctx)
        capped, original_len, truncated = _truncate_intent(query, budget)
        if truncated:
            key = content_hash(query)
            if key not in self._warned_intents:
                self._warned_intents.add(key)
                logger.warning(
                    "paritok: intent/query is %d tokens, over the %d-token max intent "
                    "budget for this request (model context %d); truncating the intent "
                    "to %d tokens — the tail is dropped. This only trims the keep/drop "
                    "guidance sent to the compressor; the request to the LLM is "
                    "unaffected.",
                    original_len, budget, self._intent_num_ctx, budget,
                )
        return capped

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
