"""Paritok: Open-source agent context compression via local model."""

__version__ = "1.1.1"

from paritok.config import ParitokConfig
from paritok.middleware.wrapper import CompressionStats, ParitokClient, ParitokEngine
from paritok.pipelines.compress import CompressionPipeline, CompressionResult
from paritok.pipelines.tool_discovery import ToolDiscoveryPipeline
from paritok.storage import MemoryShadowStorage

__all__ = [
    "CompressionPipeline",
    "CompressionResult",
    "CompressionStats",
    "MemoryShadowStorage",
    "ParitokClient",
    "ParitokEngine",
    "ParitokConfig",
    "ToolDiscoveryPipeline",
]
