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

    def get_shadows_for_path(self, path: str) -> list[str]:
        """All shadow_ids ever stored for a path, most recent first. Lets a consumer
        (edit_recovery) fall back to an earlier, fuller read when the latest read was a
        tiny offset/limit slice that would otherwise be the only thing keyed to the path.
        Default: just the latest (single-shadow backends)."""
        sid = self.get_shadow_for_path(path)
        return [sid] if sid else []


class MemoryShadowStorage(ShadowStorage):
    """In-process dict storage. Fast, lost on restart."""

    def __init__(self):
        self._store: dict[str, str] = {}
        self._compressed_cache: dict[str, str] = {}
        self._path_to_shadow: dict[str, str] = {}
        self._path_to_shadows: dict[str, list[str]] = {}

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
        if not path:
            return
        self._path_to_shadow[path] = shadow_id
        lst = self._path_to_shadows.setdefault(path, [])
        if shadow_id in lst:
            lst.remove(shadow_id)
        lst.append(shadow_id)  # most recent last
        del lst[:-8]  # keep the last few reads only

    def get_shadow_for_path(self, path: str) -> str | None:
        return self._path_to_shadow.get(path) if path else None

    def get_shadows_for_path(self, path: str) -> list[str]:
        return list(reversed(self._path_to_shadows.get(path, []))) if path else []


class RedisShadowStorage(ShadowStorage):
    """Redis-backed shadow store — persists across proxy restarts, so [REF:id]
    entries (and expand_context) survive a restart, unlike MemoryShadowStorage.

    Keys are namespaced `{prefix}:{kind}:{id}` for the three maps (shadow / cache
    / path). Values are stored decoded (str). An optional ttl expires entries.

    Construct with `from_url(...)` in production (lazy `redis` import + startup
    PING so a misconfigured/unreachable Redis fails loudly instead of silently
    degrading to memory), or inject a client directly for testing.
    """

    def __init__(self, client, *, key_prefix: str = "paritok", ttl_seconds: int | None = None):
        self._r = client
        self._prefix = key_prefix
        self._ttl = ttl_seconds

    @classmethod
    def from_url(cls, url: str, *, key_prefix: str = "paritok",
                 ttl_seconds: int | None = None) -> "RedisShadowStorage":
        try:
            import redis
        except ImportError as e:
            raise ImportError(
                "shadow_storage: 'redis' needs the redis package. "
                "Install with: pip install paritok[redis]"
            ) from e
        client = redis.Redis.from_url(url, decode_responses=True)
        try:
            client.ping()
        except Exception as e:  # noqa: BLE001 — surface any connection failure loudly
            raise ConnectionError(
                f"shadow_storage: 'redis' selected but the Redis at {url} is "
                f"unreachable ({e}). Fix the connection or set shadow_storage: "
                "memory. Not falling back silently — that would lose every "
                "[REF:id] on the next restart."
            ) from e
        return cls(client, key_prefix=key_prefix, ttl_seconds=ttl_seconds)

    def _key(self, kind: str, key: str) -> str:
        return f"{self._prefix}:{kind}:{key}"

    def store(self, content: str) -> str:
        sid = content_hash(content)
        self._r.set(self._key("shadow", sid), content, ex=self._ttl)
        return sid

    def retrieve(self, shadow_id: str) -> str | None:
        return self._r.get(self._key("shadow", shadow_id))

    def has(self, shadow_id: str) -> bool:
        return bool(self._r.exists(self._key("shadow", shadow_id)))

    def cache_compressed(self, shadow_id: str, compressed: str):
        self._r.set(self._key("cache", shadow_id), compressed, ex=self._ttl)

    def get_cached_compressed(self, shadow_id: str) -> str | None:
        return self._r.get(self._key("cache", shadow_id))

    def set_shadow_for_path(self, path: str, shadow_id: str) -> None:
        if path:
            self._r.set(self._key("path", path), shadow_id, ex=self._ttl)

    def get_shadow_for_path(self, path: str) -> str | None:
        return self._r.get(self._key("path", path)) if path else None


def build_shadow_storage(config) -> ShadowStorage:
    """Pick the shadow-storage backend from `config.shadow_storage`.

    "memory" (default) → in-process; "redis" → RedisShadowStorage from
    `config.redis`. Wiring this off the config is what makes the setting actually
    take effect (otherwise the pipeline always used memory regardless of config).

    If "redis" is selected but Redis is unavailable (package missing or server
    unreachable), print a loud startup WARNING and fall back to memory rather than
    refusing to start — so upgrading users who set redis aren't hard-blocked. The
    warning is explicit that [REF:id] won't persist; the fallback is never silent.
    """
    backend = getattr(config, "shadow_storage", "memory")
    if backend == "redis":
        rc = config.redis
        try:
            return RedisShadowStorage.from_url(
                rc.url, key_prefix=rc.key_prefix, ttl_seconds=rc.ttl_seconds
            )
        except Exception as e:  # noqa: BLE001 — missing pkg or unreachable server
            print(
                f"[paritok] WARNING: shadow_storage: 'redis' selected but Redis is "
                f"unavailable ({e}). Falling back to in-process MEMORY - [REF:id] "
                f"entries will NOT persist across restarts, so expand_context can "
                f"fail after a restart. Install 'paritok[redis]' and start a Redis "
                f"server, or set shadow_storage: memory to silence this warning.",
                flush=True,
            )
            return MemoryShadowStorage()
    return MemoryShadowStorage()
