"""shadow_storage config + backend selection.

"memory" (default) and "redis" are both valid. The setting must actually pick
the backend (regression: the pipeline used to ignore it and always use memory).
RedisShadowStorage is tested against an injected fake client so no live Redis is
needed.
"""
import pytest

from paritok.config import ParitokConfig
from paritok.storage import (
    MemoryShadowStorage,
    RedisShadowStorage,
    build_shadow_storage,
)


# ── config validation ────────────────────────────────────────────────────────

def test_memory_is_valid_and_default():
    assert ParitokConfig().shadow_storage == "memory"
    assert ParitokConfig(shadow_storage="memory").shadow_storage == "memory"


def test_redis_is_accepted():
    assert ParitokConfig(shadow_storage="redis").shadow_storage == "redis"


def test_redis_connection_config_loads_from_yaml_path():
    cfg = ParitokConfig.from_dict({
        "shadow_storage": "redis",
        "redis": {"url": "redis://cache:6379/2", "key_prefix": "pk", "ttl_seconds": 3600},
    })
    assert cfg.shadow_storage == "redis"
    assert cfg.redis.url == "redis://cache:6379/2"
    assert cfg.redis.key_prefix == "pk"
    assert cfg.redis.ttl_seconds == 3600


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="must be one of"):
        ParitokConfig(shadow_storage="postgres")


# ── backend selection ────────────────────────────────────────────────────────

def test_build_shadow_storage_memory():
    assert isinstance(build_shadow_storage(ParitokConfig()), MemoryShadowStorage)


def test_build_shadow_storage_redis_unavailable_warns_and_falls_back(monkeypatch, capsys):
    # Redis missing/unreachable → loud WARNING + fall back to memory (not a crash,
    # not silent). Covers upgrading users who had shadow_storage: redis set.
    def boom(*a, **k):
        raise ConnectionError("Connection refused")

    monkeypatch.setattr(RedisShadowStorage, "from_url", staticmethod(boom))
    cfg = ParitokConfig(shadow_storage="redis")
    storage = build_shadow_storage(cfg)

    assert isinstance(storage, MemoryShadowStorage)
    out = capsys.readouterr().out
    assert "WARNING" in out and "redis" in out.lower() and "memory" in out.lower()


def test_build_shadow_storage_redis_uses_config(monkeypatch):
    captured = {}

    def fake_from_url(url, *, key_prefix, ttl_seconds):
        captured.update(url=url, key_prefix=key_prefix, ttl_seconds=ttl_seconds)
        return "SENTINEL"

    monkeypatch.setattr(RedisShadowStorage, "from_url", staticmethod(fake_from_url))
    cfg = ParitokConfig.from_dict({
        "shadow_storage": "redis",
        "redis": {"url": "redis://h:6379/1", "key_prefix": "pk", "ttl_seconds": 60},
    })
    assert build_shadow_storage(cfg) == "SENTINEL"
    assert captured == {"url": "redis://h:6379/1", "key_prefix": "pk", "ttl_seconds": 60}


# ── RedisShadowStorage behaviour (injected fake client) ──────────────────────

class FakeRedis:
    def __init__(self):
        self.d = {}
        self.ex = {}

    def set(self, k, v, ex=None):
        self.d[k] = v
        self.ex[k] = ex

    def get(self, k):
        return self.d.get(k)

    def exists(self, k):
        return 1 if k in self.d else 0

    def ping(self):
        return True


def test_redis_storage_roundtrip():
    r = FakeRedis()
    s = RedisShadowStorage(r, key_prefix="pk", ttl_seconds=42)

    sid = s.store("hello world")
    assert s.retrieve(sid) == "hello world"
    assert s.has(sid) is True
    assert s.has("missing") is False

    s.cache_compressed(sid, "[REF:x] hi")
    assert s.get_cached_compressed(sid) == "[REF:x] hi"

    s.set_shadow_for_path("/a/b.py", sid)
    assert s.get_shadow_for_path("/a/b.py") == sid
    assert s.get_shadow_for_path("") is None

    # keys are namespaced under the prefix, and ttl is applied.
    assert any(k.startswith("pk:shadow:") for k in r.d)
    assert r.ex[f"pk:shadow:{sid}"] == 42


def test_redis_storage_id_matches_memory():
    # Same content hashes to the same shadow_id in both backends (stable [REF:id]).
    mem = MemoryShadowStorage()
    red = RedisShadowStorage(FakeRedis())
    assert mem.store("abc") == red.store("abc")
