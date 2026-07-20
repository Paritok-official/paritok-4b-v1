"""Configuration system for Paritok."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar


@dataclass
class LocalModelConfig:
    """Self-hosted compression model (Ollama / any OpenAI-compatible endpoint)."""

    base_url: str = "http://localhost:11434/v1"  # Ollama default
    model: str = "paritok-4b-v1"  # ollama pull paritok/paritok-4b-v1 && ollama cp paritok/paritok-4b-v1 paritok-4b-v1
    temperature: float = 0.0
    timeout: float = 120.0
    api_key: str = ""  # optional bearer token (empty for local Ollama)
    num_ctx: int = 8192  # model context window (matches the shipped Modelfile); output is capped so prompt+generation fit


@dataclass
class GpuServerConfig:
    """Paritok-hosted GPU compression server.

    Used when `use_gpu_server: true`. Point your compression at our managed
    endpoint instead of self-hosting — all you need is an API key created at
    https://paritok.com. The proxy POSTs each segment to `{base_url}/compress`
    and health-checks `{base_url}/test`. If the hosted server can't be reached,
    the proxy warns at startup and passes content through uncompressed.
    """

    base_url: str = "https://www.paritok.com/api"
    model: str = "paritok-4b-v1"
    temperature: float = 0.0
    timeout: float = 90.0
    api_key: str = ""  # from the paritok.com dashboard, or PARITOK_API_KEY


@dataclass
class CompressionConfig:
    min_tokens: int = 512
    max_tokens: int = 50000
    refusal_threshold: float = 0.05

    def __post_init__(self):
        assert self.min_tokens >= 0, f"min_tokens must be >= 0, got {self.min_tokens}"
        assert self.max_tokens > self.min_tokens, f"max_tokens ({self.max_tokens}) must be > min_tokens ({self.min_tokens})"
        assert 0.0 <= self.refusal_threshold <= 1.0, f"refusal_threshold must be 0.0-1.0, got {self.refusal_threshold}"


@dataclass
class ToolDiscoveryConfig:
    strategy: str = "embedding"  # "embedding" (default) | "relevance" | "passthrough"
    top_k: int = 5

    # --- "embedding" strategy (default): semantic top-k select + session freeze +
    #     adaptive apply. Ships with the [proxy] extra; standalone: pip install "paritok[toolselect]"
    k_max: int = 8                     # max tools kept in full schema
    adaptive: bool = True              # coding tasks drop unselected MCP; MCP tasks stub them
    mcp_signal_threshold: float = 1.0  # rank-weighted MCP signal needed to stub MCP tools

    _VALID_STRATEGIES: ClassVar[frozenset] = frozenset({"relevance", "passthrough", "embedding"})

    def __post_init__(self):
        assert self.top_k > 0, f"top_k must be > 0, got {self.top_k}"
        assert self.strategy in self._VALID_STRATEGIES, \
            f"strategy must be one of {self._VALID_STRATEGIES}, got '{self.strategy}'"


@dataclass
class HistoryConfig:
    enabled: bool = True
    keep_recent_turns: int = 4  # keep last N user/assistant turns intact
    context_threshold: float = 0.8  # compress when context exceeds this fraction of window
    context_window: int = 200_000  # model's context window in tokens

    def __post_init__(self):
        assert self.keep_recent_turns >= 1, f"keep_recent_turns must be >= 1, got {self.keep_recent_turns}"
        assert 0.0 < self.context_threshold <= 1.0, f"context_threshold must be (0, 1], got {self.context_threshold}"
        assert self.context_window > 0, f"context_window must be > 0, got {self.context_window}"


@dataclass
class CodexConfig:
    """Auto-configure the Codex CLI to route through this proxy.

    Codex ignores `OPENAI_BASE_URL` and only takes its endpoint from
    `~/.codex/config.toml`. When `enabled`, `paritok up`/`proxy` writes that file
    for the user (backing up any existing one) so everything lives here in
    paritok.yaml — flip the switch, paste your key, done.

    `api_key` is embedded into the generated config.toml as
    `experimental_bearer_token`; leave it empty to fall back to Codex reading
    `env_key` (OPENAI_API_KEY) from the environment instead. Codex custom
    providers only support the `responses` wire protocol, so that is fixed.
    """

    enabled: bool = False
    model: str = "gpt-5"
    api_key: str = ""  # OpenAI key, embedded into ~/.codex/config.toml (empty → use env OPENAI_API_KEY)
    config_path: str = ""  # override the generated config.toml location (empty → ~/.codex/config.toml)


@dataclass
class TraceConfig:
    """Per-compression debug trace. When enabled, every compression event
    (original + compressed body, tokens, ratio) is appended to `path` as JSONL.
    Inspect it with `python tools/view_trace.py`."""

    enabled: bool = False
    path: str = "compress_trace.jsonl"


@dataclass
class ParitokConfig:
    # Backend toggle. false → self-host the open model locally (default);
    # true → route compression to the Paritok GPU server (needs an api_key).
    use_gpu_server: bool = False

    compression: CompressionConfig = field(default_factory=CompressionConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    tool_discovery: ToolDiscoveryConfig = field(default_factory=ToolDiscoveryConfig)
    local_model: LocalModelConfig = field(default_factory=LocalModelConfig)
    gpu_server: GpuServerConfig = field(default_factory=GpuServerConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)
    codex: CodexConfig = field(default_factory=CodexConfig)
    shadow_storage: str = "memory"  # "memory" | "redis"

    _VALID_SHADOW_STORAGE: ClassVar[frozenset] = frozenset({"memory", "redis"})

    def __post_init__(self):
        assert self.shadow_storage in self._VALID_SHADOW_STORAGE, \
            f"shadow_storage must be one of {self._VALID_SHADOW_STORAGE}, got '{self.shadow_storage}'"

    @property
    def model(self) -> LocalModelConfig | GpuServerConfig:
        """The active compression backend config (GPU server or local model)."""
        return self.gpu_server if self.use_gpu_server else self.local_model

    @classmethod
    def _merge_dataclass(cls, target, overrides: dict):
        """Merge dict into dataclass, validating keys and re-triggering __post_init__."""
        valid_keys = set(target.__dataclass_fields__.keys())
        for k in overrides:
            if k not in valid_keys:
                raise ValueError(f"Unknown config key '{k}' in {type(target).__name__}. Valid: {valid_keys}")
        # Reconstruct to trigger __post_init__ validation
        merged = {**target.__dict__, **overrides}
        return type(target)(**merged)

    @classmethod
    def from_dict(cls, data: dict) -> ParitokConfig:
        config = cls()
        if "use_gpu_server" in data:
            config.use_gpu_server = bool(data["use_gpu_server"])
        if "compression" in data:
            config.compression = cls._merge_dataclass(config.compression, data["compression"])
        if "history" in data:
            config.history = cls._merge_dataclass(config.history, data["history"])
        if "tool_discovery" in data:
            config.tool_discovery = cls._merge_dataclass(config.tool_discovery, data["tool_discovery"])
        if "local_model" in data:
            config.local_model = cls._merge_dataclass(config.local_model, data["local_model"])
        if "gpu_server" in data:
            config.gpu_server = cls._merge_dataclass(config.gpu_server, data["gpu_server"])
        if "trace" in data:
            config.trace = cls._merge_dataclass(config.trace, data["trace"])
        if "codex" in data:
            config.codex = cls._merge_dataclass(config.codex, data["codex"])
        if "shadow_storage" in data:
            config.shadow_storage = data["shadow_storage"]
        config.__post_init__()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> ParitokConfig:
        import yaml

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def load(cls, path: str | Path | None = None) -> ParitokConfig:
        """Load config from file, then override with environment variables."""
        if path and Path(path).exists():
            config = cls.from_yaml(path)
        else:
            config = cls()

        # Environment overrides
        if (flag := os.environ.get("PARITOK_USE_GPU_SERVER")) is not None:
            config.use_gpu_server = flag.strip().lower() in ("1", "true", "yes")
        if model := os.environ.get("PARITOK_MODEL"):
            config.local_model.model = model
            config.gpu_server.model = model
        if base_url := os.environ.get("PARITOK_OLLAMA_URL"):
            config.local_model.base_url = base_url
        # A single API key env var feeds whichever backend needs auth. The GPU
        # server always needs it; a self-hosted endpoint behind a gateway may too.
        if api_key := os.environ.get("PARITOK_API_KEY"):
            config.gpu_server.api_key = api_key
            config.local_model.api_key = api_key

        return config
