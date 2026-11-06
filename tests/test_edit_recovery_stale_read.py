"""Edit recovery maps onto the file as it is NOW, not onto a read the agent has since edited.

loopbench r11 (task 5): the agent Read deletion.py (compressed), added `# type:` comments
with Edit, then converted them to annotations. Its old_string was exactly the current
file, but the gateway's stored read predated the first edit; recovery, which ignores
comments when matching, took the agent's own comments for text the summary had hidden
and rewrote the edit onto the stale read, 9 times. Failed edits, and once a substring
match that wrote `self.data : '...' = {}  # type: ...` into the file.
"""
import pytest

from paritok.pipelines.edit_recovery import recover_edit, strip_read_line_numbers
from paritok.pipelines.file_history import FileHistory

FP = "/proj/django/db/models/deletion.py"


def _file() -> str:
    lines = [
        "class Collector:",
        "    def __init__(self, using):",
        "        # Initially, {model: {instances}}, later values become lists.",
        "        self.data = {}",
        "        self.dependencies = {}",
        "",
        "    def can_fast_delete(",
        "        self,",
        "        objs,",
        "        from_field=None,",
        "    ):",
        "        return bool(objs)",
        "",
    ]
    for i in range(40):
        lines += [f"    def helper_{i}(self, value):",
                  f"        # Helper {i} normalises the value before it is collected for deletion.",
                  f"        # It keeps the original ordering of related objects number {i}.",
                  f"        return value + {i}",
                  ""]
    return "\n".join(lines) + "\n"


RAW = _file()
NUMBERED = "".join(f"{n:6d}\t{ln}\n" for n, ln in enumerate(RAW.splitlines(), 1))


class _SummaryModel:
    """Stands in for the 4B: drops comments, blank lines and line numbers, and collapses
    the multi-line signature onto one line."""

    def compress(self, text, **kwargs):
        out, sig = [], None
        for ln in strip_read_line_numbers(text).splitlines():
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            if sig is not None:
                sig.append(s)
                if s.startswith("):"):
                    out.append(sig[0] + ", ".join(p.rstrip(",") for p in sig[1:-1]) + "):")
                    sig = None
                continue
            if s.endswith("(") and s.startswith("def "):
                sig = [ln]
                continue
            out.append(ln)
        return "\n".join(out)


def _tool_use(tid, name, inp):
    return {"role": "assistant", "content": [{"type": "tool_use", "id": tid, "name": name, "input": inp}]}


def _result(tid, text, error=False):
    block = {"type": "tool_result", "tool_use_id": tid, "content": text}
    if error:
        block["is_error"] = True
    return {"role": "user", "content": [block]}


def _conversation(*steps):
    msgs = [{"role": "user", "content": "add type hints to Collector"},
            _tool_use("r1", "Read", {"file_path": FP}), _result("r1", NUMBERED)]
    for tid, name, inp, res in steps:
        msgs += [_tool_use(tid, name, inp), _result(tid, res)]
    return msgs


def _ok(path=FP):
    return f"The file {path} has been updated successfully."


def _proxy_edit(monkeypatch, messages, old, new):
    """Run one request through the real proxy (4B replaced by _SummaryModel) whose
    upstream answers with Edit(old -> new); return the Edit input the client gets."""
    pytest.importorskip("starlette")
    from starlette.testclient import TestClient
    import paritok.pipelines.compress as compress_mod
    import paritok.strategies.gpu_server as gpu_mod
    from paritok.proxy import server

    monkeypatch.setattr(compress_mod, "LocalModelStrategy", lambda cfg: _SummaryModel())
    monkeypatch.setattr(gpu_mod, "GpuServerStrategy", lambda cfg: _SummaryModel())

    class _Resp:
        status_code, headers = 200, {}

        def json(self):
            return {"id": "m", "type": "message", "role": "assistant", "stop_reason": "tool_use",
                    "content": [{"type": "tool_use", "id": "e9", "name": "Edit",
                                 "input": {"file_path": FP, "old_string": old, "new_string": new}}]}

    class _Up:
        async def post(self, url, headers=None, json=None):
            return _Resp()

        async def aclose(self):
            pass

    app = server.create_app(anthropic_base_url="http://upstream.test", http_client=_Up())
    with TestClient(app) as client:
        r = client.post("/v1/messages", json={"model": "claude-sonnet-4-5-20250929",
                                              "max_tokens": 64, "messages": messages})
    assert r.status_code == 200
    return r.json()["content"][0]["input"]


def test_edit_of_the_agents_own_text_is_left_alone(monkeypatch):
    first_new = "        self.data = {}  # type: Dict[str, int]"
    messages = _conversation(
        ("e1", "Edit", {"file_path": FP, "old_string": "        self.data = {}",
                        "new_string": first_new}, _ok()))
    old, new = first_new, "        self.data: 'Dict[str, int]' = {}"
    # The hazard: matched against the stale read, recovery WOULD rewrite this edit.
    assert recover_edit(old, new, RAW) is not None
    got = _proxy_edit(monkeypatch, messages, old, new)
    assert (got["old_string"], got["new_string"]) == (old, new)


def test_summary_edit_is_still_recovered_after_an_unrelated_edit(monkeypatch):
    messages = _conversation(
        ("e1", "Edit", {"file_path": FP, "old_string": "    def helper_39(self, value):",
                        "new_string": "    def helper_39(self, value, extra=0):"}, _ok()))
    now = RAW.replace("    def helper_39(self, value):", "    def helper_39(self, value, extra=0):")
    old = "    def can_fast_delete(self, objs, from_field=None):\n        return bool(objs)"
    new = "    def can_fast_delete(self, objs, from_field=None, strict=False):\n        return bool(objs)"
    got = _proxy_edit(monkeypatch, messages, old, new)
    assert got["old_string"] != old, "the collapsed signature should be mapped back"
    assert now.count(got["old_string"]) == 1, "must match the file as it is now"
    assert "strict=False" in got["new_string"]


def test_no_rewrite_when_the_file_changed_in_a_way_that_cant_be_replayed(monkeypatch):
    messages = _conversation(
        ("b1", "Bash", {"command": f"sed -i 's/self.dependencies = {{}}/self.deps = {{}}/' {FP}"}, ""))
    old = "    def can_fast_delete(self, objs, from_field=None):\n        return bool(objs)"
    new = "    def can_fast_delete(self, objs, from_field=None, strict=False):\n        return bool(objs)"
    got = _proxy_edit(monkeypatch, messages, old, new)
    assert (got["old_string"], got["new_string"]) == (old, new)


# ── FileHistory ──

def _history(*steps, compressed=None):
    return FileHistory(_conversation(*steps), {"r1": "sid1"} if compressed is None else compressed)


def test_current_replays_later_successful_edits_in_order():
    """Failed edits (is_error) changed nothing and are skipped."""
    msgs = _conversation()
    msgs += [_tool_use("e1", "Edit", {"file_path": FP, "old_string": "self.data = {}",
                                      "new_string": "self.data = []"}), _result("e1", _ok()),
             _tool_use("e2", "Edit", {"file_path": FP, "old_string": "nothing like this",
                                      "new_string": "x"}), _result("e2", "String not found", error=True),
             _tool_use("e3", "MultiEdit", {"file_path": FP, "edits": [
                 {"old_string": "self.data = []", "new_string": "self.data = ()"},
                 {"old_string": "value + ", "new_string": "value - ", "replace_all": True}]}),
             _result("e3", _ok())]
    h = FileHistory(msgs, {"r1": "sid1"})
    now = h.current(FP, RAW, h.read_of("sid1"))
    assert "self.data = ()" in now and "value + " not in now and "value - 39" in now


def test_current_is_none_when_a_later_edit_does_not_apply():
    h = _history(("e1", "Edit", {"file_path": FP, "old_string": "not in this read",
                                 "new_string": "y"}, _ok()))
    assert h.current(FP, RAW, "r1") is None
    # a read that isn't in this conversation can't be placed once the file was written
    assert h.current(FP, RAW, None) is None
    assert FileHistory(_conversation()).current(FP, RAW, None) == RAW


def test_write_replaces_the_text_and_counts_as_seen():
    h = _history(("w1", "Write", {"file_path": FP, "content": "x = 1\n"}, _ok()))
    assert h.current(FP, RAW, "r1") == "x = 1\n"
    assert h.saw_verbatim(FP, "x = 1")


def test_shell_writes_count_but_redirecting_stderr_does_not():
    grep = ("g1", "Bash", {"command": f"grep -n 'def ' {FP} 2>/dev/null | head; cat {FP} 2>&1"}, "1:def")
    assert _history(grep).current(FP, RAW, "r1") == RAW
    for cmd in (f"sed -i 's/a/b/' {FP}", f"python - <<'EOF'\np='{FP}'\ns=open(p).read()\n"
                f"open(p,'w').write(s.replace('a','b'))\nEOF", f"echo x > {FP}"):
        assert _history(("b1", "Bash", {"command": cmd}, "")).current(FP, RAW, "r1") is None, cmd


def test_saw_verbatim_counts_uncompressed_reads_and_own_writes_only():
    h = _history(("e1", "Edit", {"file_path": FP, "old_string": "self.data = {}",
                                 "new_string": "self.data = {}  # type: Dict"}, _ok()))
    assert h.saw_verbatim(FP, "self.data = {}  # type: Dict")   # its own write
    assert not h.saw_verbatim(FP, "class Collector:")          # r1 was shown compressed
    h2 = FileHistory(_conversation(), {})                     # r1 shown verbatim
    assert h2.saw_verbatim(FP, "class Collector:\n    def __init__(self, using):")
    assert h2.saw_verbatim(FP.upper().replace("/", "\\"), "class Collector:")  # path normalised
