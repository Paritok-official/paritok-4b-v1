"""Shadow storage for original content, enabling expand-context."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod


# 16 hex chars = 64-bit hash. Collision probability negligible for typical
# agent workloads (<1M unique contexts). Increase if needed.
def content_hash(content: str) -> str:
    """SHA256 hash of content, used as shadow_id."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


class ShadowStorage(ABC):
    @abstractmethod
    def store(self, content: str) -> str:
        """Store content and return its shadow_id."""
        ...

    @abstractmethod
    def retrieve(self, shadow_id: str) -> str | None:
        """Retrieve content by shadow_id, or None if not found."""
        ...

    @abstractmethod
    def has(self, shadow_id: str) -> bool:
        ...

    @abstractmethod
    def cache_compressed(self, shadow_id: str, compressed: str):
        """Cache a compression result for deduplication."""
        ...

    @abstractmethod
    def get_cached_compressed(self, shadow_id: str) -> str | None:
        """Get cached compressed result, or None if not cached."""
        ...

    def set_shadow_for_path(self, path: str, shadow_id: str) -> None:
        """Associate a source path (e.g. a file_path from Read) with a shadow_id.
        Default no-op; backends that support it should override."""
        return None

    def get_shadow_for_path(self, path: str) -> str | None:
        """Return the most recent shadow_id stored for a source path, or None."""
        return None


class MemoryShadowStorage(ShadowStorage):
    """In-process dict storage. Fast, lost on restart."""

    def __init__(self):
        self._store: dict[str, str] = {}
        self._compressed_cache: dict[str, str] = {}
        self._path_to_shadow: dict[str, str] = {}

    def store(self, content: str) -> str:
        sid = content_hash(content)
        self._store[sid] = content
        return sid

    def retrieve(self, shadow_id: str) -> str | None:
        return self._store.get(shadow_id)

    def has(self, shadow_id: str) -> bool:
        return shadow_id in self._store

    def cache_compressed(self, shadow_id: str, compressed: str):
        self._compressed_cache[shadow_id] = compressed

    def get_cached_compressed(self, shadow_id: str) -> str | None:
        return self._compressed_cache.get(shadow_id)

    def set_shadow_for_path(self, path: str, shadow_id: str) -> None:
        if path:
            self._path_to_shadow[path] = shadow_id

    def get_shadow_for_path(self, path: str) -> str | None:
        return self._path_to_shadow.get(path) if path else None


# TODO: RedisShadowStorage — for persistent storage across proxy restarts.
# Config supports shadow_storage: "redis" but implementation is not yet built.
# When implemented, it should use the same interface as MemoryShadowStorage.
