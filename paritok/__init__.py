"""Paritok: Open-source agent context compression via local model."""

__version__ = "1.3.0"

from paritok.config import ParitokConfig
from paritok.middleware.wrapper import CompressionStats, ParitokClient, ParitokEngine
from paritok.pipelines.compress import CompressionPipeline, CompressionResult
from paritok.pipelines.tool_discovery import ToolDiscoveryPipeline
from paritok.storage import (
    MemoryShadowStorage,
    RedisShadowStorage,
    build_shadow_storage,
)

__all__ = [
    "CompressionPipeline",
    "CompressionResult",
    "CompressionStats",
    "MemoryShadowStorage",
    "RedisShadowStorage",
    "ParitokClient",
    "ParitokEngine",
    "ParitokConfig",
    "ToolDiscoveryPipeline",
    "build_shadow_storage",
]
