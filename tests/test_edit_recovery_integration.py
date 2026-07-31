"""End-to-end: the proxy rewrites a lossy Edit so the client's exact match succeeds.

Drives the real Starlette proxy (handle_anthropic -> _recover_edits) with a mock
upstream that returns an Edit whose old_string was authored against the compressed
(collapsed-signature, docstring-dropped) view. The shadow store is pre-seeded with
the line-numbered original mapped to the file path (standing in for the 4B compression
step). We assert the proxy rewrites old_string back to the true multi-line original so
the downstream Edit tool would match the real file — and that docstrings survive.
"""
import pytest

pytest.importorskip("starlette")
from starlette.testclient import TestClient  # noqa: E402

MODEL = "claude-sonnet-4-5-20250929"
FILE_PATH = "/proj/engine.py"

# The true file, as Claude Code's Read returns it (line-numbered). Multi-line
# signature + docstring — exactly what compression collapses/drops.
RAW = (
    "def process_records(\n"
    "    records,\n"
    "    threshold,\n"
    "    strict,\n"
    "):\n"
    '    """Filter, normalize, and score a batch of records."""\n'
    "    out = []\n"
    "    return out\n"
)
LINE_NUMBERED = "".join(f"{i}\t{ln}\n" for i, ln in enumerate(RAW.splitlines(), 1))

# What the model authored against the compressed view: one-line signature, no docstring.
COLLAPSED_OLD = (
    "def process_records(records, threshold, strict):\n"
    "    out = []\n"
    "    return out"
)
NEW = (
    "def process_records(records, threshold, strict, dedupe=False):\n"
    "    out = []\n"
    "    return out"
)


class _FakeResp:
    def __init__(self, payload):
        self._payload, self.status_code, self.headers = payload, 200, {}

    def json(self):
        return self._payload


class _FakeUpstream:
    def __init__(self):
        self.calls = []

    async def post(self, url, headers=None, json=None):
        self.calls.append(json)
        return _FakeResp({
            "type": "message", "role": "assistant", "stop_reason": "tool_use",
            "content": [{"type": "tool_use", "id": "tu1", "name": "Edit",
                         "input": {"file_path": FILE_PATH,
                                   "old_string": COLLAPSED_OLD, "new_string": NEW}}],
        })

    async def aclose(self):
        pass


def test_proxy_rewrites_lossy_edit(monkeypatch):
    from paritok.storage import MemoryShadowStorage
    import paritok.middleware.wrapper as wrapper
    from paritok.proxy import server

    # Pre-seed: the shadow store holds the line-numbered original, keyed by file path
    # (what real compression of the Read result would have produced).
    storage = MemoryShadowStorage()
    sid = storage.store(LINE_NUMBERED)
    storage.set_shadow_for_path(FILE_PATH, sid)
    monkeypatch.setattr(wrapper, "build_shadow_storage", lambda config: storage)

    upstream = _FakeUpstream()
    app = server.create_app(anthropic_base_url="http://upstream.test", http_client=upstream)

    body = {
        "model": MODEL,
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "add a dedupe param to process_records"}],
    }
    with TestClient(app) as client:
        r = client.post("/v1/messages", json=body)
        assert r.status_code == 200
        reply = r.json()

    edit = next(b for b in reply["content"] if b.get("type") == "tool_use")
    got_old = edit["input"]["old_string"]
    got_new = edit["input"]["new_string"]

    # old_string was rewritten to the exact multi-line original (matches the real file).
    assert got_old != COLLAPSED_OLD, "old_string should have been rewritten"
    assert got_old in RAW, "rewritten old_string must be an exact slice of the real file"
    assert "def process_records(\n" in got_old, "multi-line signature restored"
    # the dropped docstring is preserved in the replacement, and the edit landed.
    assert '"""Filter, normalize, and score a batch of records."""' in got_new
    assert "dedupe=False" in got_new


def test_proxy_leaves_matching_edit_untouched(monkeypatch):
    """If old_string already matches the file, the proxy must not touch it."""
    from paritok.storage import MemoryShadowStorage
    import paritok.middleware.wrapper as wrapper
    from paritok.proxy import server

    storage = MemoryShadowStorage()
    sid = storage.store(LINE_NUMBERED)
    storage.set_shadow_for_path(FILE_PATH, sid)
    monkeypatch.setattr(wrapper, "build_shadow_storage", lambda config: storage)

    good_old = "    out = []\n    return out"

    class _Up:
        def __init__(self):
            self.calls = []

        async def post(self, url, headers=None, json=None):
            self.calls.append(json)
            return _FakeResp({
                "type": "message", "role": "assistant", "stop_reason": "tool_use",
                "content": [{"type": "tool_use", "id": "tu1", "name": "Edit",
                             "input": {"file_path": FILE_PATH, "old_string": good_old,
                                       "new_string": "    out = []\n    return sorted(out)"}}],
            })

        async def aclose(self):
            pass

    app = server.create_app(anthropic_base_url="http://upstream.test", http_client=_Up())
    with TestClient(app) as client:
        r = client.post("/v1/messages", json={"model": MODEL, "max_tokens": 8,
                                              "messages": [{"role": "user", "content": "x"}]})
        edit = next(b for b in r.json()["content"] if b.get("type") == "tool_use")
    assert edit["input"]["old_string"] == good_old  # untouched
