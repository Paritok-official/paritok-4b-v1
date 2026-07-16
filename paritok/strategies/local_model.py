"""Local model strategy: SEG-based compression via a locally-running Ollama model.

The model (SFT checkpoint-2000, Qwen3-4B) runs locally via Ollama, which exposes
an OpenAI-compatible API at http://localhost:11434/v1. No data leaves the machine.

The system/user message format below MUST match training verbatim — the model
learned the "[SEG ...]<body>[/SEG]" distribution and drifting from it degrades
compression quality. Each call compresses ONE segment:

    SYSTEM: file_read.txt (kind == file_read) or other.txt (any other kind)
    USER:
        USER INTENT:
        {intent}

        Compress the following segment under the rules in your system prompt.
        Output only the compressed [SEG]...[/SEG] block (or an empty one to drop):

        [SEG id={seg_id} kind={kind} level={level}]
        {content}
        [/SEG]

The reply is a single [SEG ...]<body>[/SEG]; an empty body means "drop". Levels
L0-L3 set the target ratio (L0 ≤ 0.50, L1 ≤ 0.35, L2 ≤ 0.25, L3 ≤ 0.20).

Setup:
    ollama pull paritok/paritok-4b-v1               # public registry name
    ollama cp   paritok/paritok-4b-v1 paritok-4b-v1  # tag as the runtime name

Config:
    local_model:
      base_url: http://localhost:11434/v1
      model: paritok-4b-v1
"""

from __future__ import annotations

import re

from paritok.config import LocalModelConfig
from paritok.strategies.chunking import (
    CHUNK_SIZE,
    deduplicate_definitions,
    split_into_chunks_structural,
)
from paritok.strategies.prompts import system_prompt_for_kind
from paritok.strategies.tagger import classify_kind_from_content
from paritok.token_counter import count_tokens

# Default level when the caller doesn't specify one. L1 (target ≤ 0.35) is the
# level the SWE-bench Verified benchmark was run at, so the runtime reproduces
# the reported compression rate / quality-retained numbers.
DEFAULT_LEVEL = "L1"

# Per-level target ratio ceilings (from the training system prompt). Used only to
# translate a legacy `target_ratio` into the nearest level for back-compat.
_LEVEL_CEILINGS = (("L3", 0.20), ("L2", 0.25), ("L1", 0.35), ("L0", 0.50))
_VALID_LEVELS = {"L0", "L1", "L2", "L3"}

# Training used tiktoken cl100k_base for token counts; keep that for parity.
_TOKEN_ENCODING = "cl100k_base"

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
# Unwrap the model's [SEG ...]<body>[/SEG] reply. DOTALL so bodies span lines.
_SEG_RE = re.compile(r"\[SEG\b[^\]]*\]\s*(.*?)\s*\[/SEG\]", re.DOTALL)


def _ratio_to_level(target_ratio: str) -> str:
    """Map a legacy ratio string ("30%", "0.3") to the nearest SEG level.

    Chooses the tightest level whose ceiling still covers the requested ratio,
    so "50%"→L0, "35%"→L1, "25%"→L2, "20%" or lower→L3.
    """
    s = target_ratio.strip()
    ratio = float(s[:-1]) / 100 if s.endswith("%") else float(s)
    for level, ceiling in _LEVEL_CEILINGS:      # L3, L2, L1, L0
        if ratio <= ceiling:
            return level
    return "L0"


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks (Qwen3 may emit one before the body)."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = _THINK_BLOCK_RE.sub("", text)
    return text.strip()


def _unwrap_seg(raw: str) -> str:
    """Extract the compressed body from a [SEG ...]<body>[/SEG] reply.

    Returns the inner body (empty string when the model dropped the segment).
    Falls back to the whole stripped text if no SEG wrapper is present.
    """
    text = _strip_thinking(raw)
    m = _SEG_RE.search(text)
    return (m.group(1) if m else text).strip()


class LocalModelStrategy:
    name = "local_model"

    def __init__(self, config: LocalModelConfig):
        self.config = config

    def compress(
        self,
        content: str,
        *,
        query: str | None = None,
        level: str | None = None,
        kind: str | None = None,
        target_ratio: str | None = None,
        system_prompt: str | None = None,
        **kwargs,
    ) -> str:
        """Compress content as one (or, for long inputs, several) [SEG] blocks.

        Args:
            content: Text to compress (a file read, tool output, etc.)
            query: USER INTENT — the agent's current task. Drives keep/drop.
            level: SEG level L0-L3 (target ratio). Defaults to DEFAULT_LEVEL.
            kind: SEG kind (file_read, log_output, file_operation, ...). If None,
                sniffed from content via _classify_kind. Selects the system prompt.
            target_ratio: Legacy knob ("30%"/"0.3"). If given and `level` is not,
                mapped to the nearest level. Prefer `level`.
            system_prompt: Override the kind-selected system prompt.

        For inputs larger than CHUNK_SIZE the content is split at top-level
        class/def boundaries and each chunk is compressed as its own SEG, then
        merged — one-shot calls on long inputs drive the q4 model out of
        distribution and produce structural hallucinations.
        """
        if level is None:
            level = _ratio_to_level(target_ratio) if target_ratio else DEFAULT_LEVEL
        if level not in _VALID_LEVELS:
            raise ValueError(f"level must be one of {sorted(_VALID_LEVELS)}, got {level!r}")

        if kind is None:
            kind = classify_kind_from_content(content)
        system = system_prompt if system_prompt is not None else system_prompt_for_kind(kind)

        input_tokens = count_tokens(content, _TOKEN_ENCODING)

        # Short / non-code inputs: single SEG.
        if input_tokens <= CHUNK_SIZE:
            return self._call_ollama(system, query, content, kind, level, seg_id="s1")

        # Long code inputs: one SEG per structural chunk, then merge + dedup.
        chunks = split_into_chunks_structural(content)
        compressed_parts: list[str] = []
        for i, (chunk_text, start_line, end_line, _raw_tok) in enumerate(chunks, start=1):
            body = self._call_ollama(
                system, query, chunk_text, kind, level, seg_id=f"s{i}"
            )
            if body:  # skip dropped chunks entirely
                compressed_parts.append(f"# Lines {start_line}-{end_line}:\n{body}")

        combined = "\n\n".join(compressed_parts)
        return deduplicate_definitions(combined)

    def _call_ollama(
        self,
        system: str,
        query: str | None,
        content: str,
        kind: str,
        level: str,
        *,
        seg_id: str,
    ) -> str:
        """Single Ollama chat completion in the training SEG message layout.

        Returns the unwrapped compressed body ("" if the model dropped the SEG).
        """
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for local_model strategy. "
                "Install with: pip install paritok[llm]"
            )

        intent = query.strip() if query else ""
        user_message = (
            f"USER INTENT:\n{intent}\n\n"
            "Compress the following segment under the rules in your system prompt. "
            "Output only the compressed [SEG]...[/SEG] block (or an empty one to drop):\n\n"
            f"[SEG id={seg_id} kind={kind} level={level}]\n{content}\n[/SEG]\n"
        )

        # The compressed body is always < input; cap generation just above input
        # size (bounded by the chunk budget) with headroom for the SEG wrapper.
        input_tokens = count_tokens(content, _TOKEN_ENCODING)
        max_tokens = min(input_tokens + 256, CHUNK_SIZE)

        # Bearer auth: empty for local Ollama, set for the Paritok GPU server
        # (or any self-hosted endpoint behind an API gateway).
        headers = {}
        api_key = getattr(self.config, "api_key", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            resp = httpx.post(
                f"{self.config.base_url}/chat/completions",
                json={
                    "model": self.config.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_message},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": self.config.temperature,
                    "stream": False,
                },
                headers=headers,
                timeout=self.config.timeout,
            )
            resp.raise_for_status()
            try:
                data = resp.json()
            except Exception:
                raise ValueError(
                    f"Ollama returned invalid JSON. Status: {resp.status_code}, "
                    f"Body: {resp.text[:200]}"
                )
            try:
                raw = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                raise ValueError(
                    f"Unexpected Ollama response format: {str(data)[:200]}"
                )
            return _unwrap_seg(raw)
        except httpx.ConnectError:
            raise ConnectionError(
                f"Cannot connect to the compression backend at {self.config.base_url}. "
                f"If self-hosting, is Ollama running (ollama serve)? "
                f"If using the GPU server, check use_gpu_server, base_url and api_key."
            )
        except httpx.TimeoutException:
            raise TimeoutError(
                f"Compression request timed out after {self.config.timeout}s. "
                f"The model may be loading. Try again or increase timeout."
            )

    def is_available(self) -> bool:
        """Check if Ollama is running and the configured model is available."""
        try:
            import httpx
        except ImportError:
            return False

        try:
            # Ollama's native tags endpoint (more reliable than /v1/models).
            base = self.config.base_url.rstrip("/")
            if base.endswith("/v1"):
                base = base[:-3]
            resp = httpx.get(f"{base}/api/tags", timeout=5.0)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                # Ollama appends ":latest"; prefix match.
                return any(name.startswith(self.config.model) for name in model_names)
            return False
        except Exception:
            return False
