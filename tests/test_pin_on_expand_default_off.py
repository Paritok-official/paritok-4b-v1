"""Pin-on-expand is opt-in (default OFF) as of 1.3.10.

A [REF] the model expands via read_original must NOT be pinned unless
PARITOK_PIN_ON_EXPAND is set. Pinning passes the file through VERBATIM for the rest of
the session (never re-compressed), which is net-negative on typical read/edit-heavy
coding tasks — the uncompressed re-send every turn costs more than the occasional
re-expand it avoids. This locks the default so it can't silently flip back on.
"""
from paritok.proxy.server import _maybe_pin_expanded
from paritok.storage import MemoryShadowStorage


def _seed(storage, content="def f():\n    return 1\n", path="/repo/a.py"):
    sid = storage.store(content)
    storage.set_shadow_for_path(path, sid)  # so get_path_for_shadow(sid) -> path
    return sid, path


def test_pin_on_expand_off_by_default(monkeypatch):
    monkeypatch.delenv("PARITOK_PIN_ON_EXPAND", raising=False)
    storage = MemoryShadowStorage()
    sid, path = _seed(storage)

    pinned = _maybe_pin_expanded(storage, sid)

    assert pinned is False
    assert not storage.is_shadow_pinned(sid)
    assert not storage.is_source_pinned(path)


def test_pin_on_expand_opt_in(monkeypatch):
    monkeypatch.setenv("PARITOK_PIN_ON_EXPAND", "1")
    storage = MemoryShadowStorage()
    sid, path = _seed(storage)

    pinned = _maybe_pin_expanded(storage, sid)

    assert pinned is True
    assert storage.is_shadow_pinned(sid)   # content pinned
    assert storage.is_source_pinned(path)  # and its source path


def test_pin_on_expand_ignores_empty_ref(monkeypatch):
    monkeypatch.setenv("PARITOK_PIN_ON_EXPAND", "1")
    storage = MemoryShadowStorage()
    assert _maybe_pin_expanded(storage, "") is False
