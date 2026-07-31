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


def test_path_keeps_all_reads_most_recent_first():
    """A path must remember EVERY read, not just the latest. A tiny offset/limit slice
    read must not clobber the full-file read out of existence — edit_recovery falls back
    to the fuller earlier read when the latest can't contain a multi-line edit."""
    s = MemoryShadowStorage()
    full = s.store("full file\nline 2\nline 3\n")
    s.set_shadow_for_path("/f.py", full)
    slice_sid = s.store("line 2\n")  # a 1-line offset/limit slice
    s.set_shadow_for_path("/f.py", slice_sid)

    # latest is the slice (back-compat), but the full read is still reachable, first.
    assert s.get_shadow_for_path("/f.py") == slice_sid
    shadows = s.get_shadows_for_path("/f.py")
    assert shadows == [slice_sid, full]  # most recent first
    assert full in shadows  # the full read survived the slice read
    assert s.get_shadows_for_path("") == []
    assert s.get_shadows_for_path("/unknown") == []


def test_pin_source_and_reverse_lookup():
    """After a shadow is stored for a path, the path is reverse-lookable, and pinning
    the path is a per-path flag (used to pass a file through verbatim after expand)."""
    s = MemoryShadowStorage()
    sid = s.store("code")
    s.set_shadow_for_path("/f.py", sid)
    assert s.get_path_for_shadow(sid) == "/f.py"
    assert s.get_path_for_shadow("nope") is None
    assert s.is_source_pinned("/f.py") is False
    s.pin_source("/f.py")
    assert s.is_source_pinned("/f.py") is True
    assert s.is_source_pinned("/other.py") is False
    assert s.is_source_pinned("") is False


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
