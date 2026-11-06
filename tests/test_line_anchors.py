"""Line-range anchors on the definitions a compressed view keeps — computed exactly from the
line-numbered original — and never written back into the user's files."""
import pytest

from paritok.config import ParitokConfig
from paritok.pipelines.compress import CompressionPipeline
from paritok.pipelines.line_anchors import annotate, restore_indent, strip_anchors

PY = [
    "class Collector:",                                  # 10
    "    def __init__(self, using):",                     # 11
    "        self.using = using",                         # 12
    "        # comment the summary drops",                # 13
    "        self.data = {}",                             # 14
    "",                                                   # 15
    "    def add(self, objs):",                           # 16
    '        """Add objs."""',                            # 17
    "        if not objs:",                               # 18
    "            return []",                              # 19
    "        return objs",                                # 20
    "",                                                   # 21
    "def helper():",                                      # 22
    "    return 1",                                       # 23
    "def tail_func(x):",                                  # 24
    "    y = x + 1",                                      # 25
]
ORIG = "".join(f"{n}\t{ln}\n" for n, ln in enumerate(PY, 10))
VIEW = "\n".join([
    "class Collector:",
    "    def __init__(self, using):",
    "        self.using = using",
    "        self.data = {}",
    "    def add(self, objs):",
    "        if not objs:",
    "            return []",
    "        return objs",
    "def tail_func(x):",
    "    y = x + 1",
])


def test_python_ranges_partial_and_truncation():
    out = annotate(VIEW, ORIG, "/p/deletion.py").split("\n")
    assert out[0] == "class Collector:  # L10-20 (partial)"          # whole class, lines dropped
    assert out[1] == "    def __init__(self, using):  # L11-14 (partial)"  # the comment was dropped
    assert out[4] == "    def add(self, objs):  # L16-20 (partial)"   # docstring dropped
    assert out[8] == "def tail_func(x):  # L24-25+"                   # runs past the read
    assert out[2] == "        self.using = using"                     # body lines untouched


def test_read_ending_inside_a_block_marks_it_open_even_on_blank_lines():
    """Seen live: a Read of compiler.py lines 1-60 stopped inside get_group_by's docstring,
    with line 60 blank — the block must read as continuing (`+`), not as ending at 59."""
    lines = ["class SQLCompiler:", "    def get_group_by(self, select):", '        """',
             "        Return a list of 2-tuples.", ""]
    orig = "".join(f"{n}\t{ln}\n" for n, ln in enumerate(lines, 56))
    out = annotate("class SQLCompiler:\n    def get_group_by(self, select):", orig, "/p/c.py")
    assert out.split("\n") == ["class SQLCompiler:  # L56-59+ (partial)",
                               "    def get_group_by(self, select):  # L57-59+ (partial)"]


def test_complete_block_has_no_partial_flag():
    view = "def helper():\n    return 1"
    assert annotate(view, ORIG, "/p/x.py") == "def helper():  # L22-23\n    return 1"


def test_unnumbered_input_is_left_alone():
    assert annotate(VIEW, "\n".join(PY), "/p/x.py") == VIEW


def test_brace_language_skips_braces_in_strings_and_comments():
    js = [
        "export function render(items) {",       # 1
        "  const open = '{';  // not a block {",  # 2
        "  /* } still not */",                     # 3
        "  return items.map(i => {",               # 4
        "    return i;",                           # 5
        "  });",                                   # 6
        "}",                                       # 7
        "function next() {",                       # 8
        "  return 2;",                             # 9
        "}",                                       # 10
    ]
    orig = "".join(f"{n}\t{ln}\n" for n, ln in enumerate(js, 1))
    view = "export function render(items) {\n  return items.map(i => {\n}\nfunction next() {\n  return 2;\n}"
    out = annotate(view, orig, "/web/app.ts").split("\n")
    assert out[0] == "export function render(items) {  // L1-7 (partial)"
    assert out[3] == "function next() {  // L8-10"


def test_strip_anchors_only_removes_anchors():
    text = "def f(a):  # L12-40 (partial)\n    x = 1  # L1 cache line\nfunction g() {  // L3-9+\n"
    assert strip_anchors(text) == "def f(a):\n    x = 1  # L1 cache line\nfunction g() {\n"


class _ViewModel:
    def compress(self, text, **kwargs):
        return VIEW


class _EchoModel:
    def compress(self, text, **kwargs):
        return text


def test_pipeline_adds_anchors_and_keeps_echo_detection():
    big = PY * 1  # pad the read past min_tokens with more numbered lines
    lines = big + [f"    v{i} = compute_value_number_{i}(row, column, extra)" for i in range(200)]
    orig = "".join(f"{n}\t{ln}\n" for n, ln in enumerate(lines, 10))
    pipe = CompressionPipeline(ParitokConfig())
    pipe._model = _ViewModel()
    cr = pipe.compress(orig, query="add docstrings", source="/p/deletion.py")
    assert "class Collector:  # L10-" in cr.compressed
    # a GPU passthrough (echo) is still reported as such, not as a weak compression
    cfg = ParitokConfig()
    cfg.use_gpu_server = True
    pipe2 = CompressionPipeline(cfg)
    pipe2._model = _EchoModel()
    cr2 = pipe2.compress(orig, query="x", source="/p/deletion.py")
    assert cr2.metadata.get("reason") == "gpu_unavailable"


def test_proxy_strips_anchors_from_edits_and_writes(monkeypatch):
    pytest.importorskip("starlette")
    from starlette.testclient import TestClient
    from paritok.proxy import server

    class _Resp:
        status_code, headers = 200, {}

        def __init__(self, payload):
            self._p = payload

        def json(self):
            return self._p

    blocks = [
        {"type": "tool_use", "id": "e1", "name": "Edit",
         "input": {"file_path": "/never/read.py", "old_string": "def f(a):  # L12-40 (partial)",
                   "new_string": "def f(a, b):  # L12-40 (partial)"}},
        {"type": "tool_use", "id": "w1", "name": "Write",
         "input": {"file_path": "/x/new.py", "content": "class A:  # L1-9\n    pass\n"}},
    ]

    class _Up:
        async def post(self, url, headers=None, json=None):
            return _Resp({"id": "m", "type": "message", "role": "assistant",
                          "stop_reason": "tool_use", "content": blocks})

        async def aclose(self):
            pass

    app = server.create_app(anthropic_base_url="http://upstream.test", http_client=_Up())
    with TestClient(app) as client:
        r = client.post("/v1/messages", json={"model": "claude-sonnet-4-5-20250929", "max_tokens": 16,
                                              "messages": [{"role": "user", "content": "edit"}]})
    edit, write = r.json()["content"]
    assert edit["input"]["old_string"] == "def f(a):"
    assert edit["input"]["new_string"] == "def f(a, b):"
    assert write["input"]["content"] == "class A:\n    pass\n"


# ── restore_indent: the 4B shifts mid-file slices to column 0 ──

# loopbench r14: a Read of deletion.py from line 169; the view had `def collect` at
# column 0 and its continuation at 12 spaces (real: 4 and 16). Five tasks later the agent
# wrote a replace script with the 12-space continuation and missed the file.
MID = [
    "    def collect(self, objs, source=None, nullable=False, collect_related=True,",  # 169
    "                source_attr=None, reverse_dependency=False, keep_parents=False):",  # 170
    '        """',                                                                    # 171
    "        Add 'objs' to the collection of objects to be deleted.",                 # 172
    '        """',                                                                    # 173
    "        if self.can_fast_delete(objs):",                                         # 174
    "            self.fast_deletes.append(objs)",                                     # 175
    "            return",                                                             # 176
    "        new_objs = self.add(objs, source, nullable,",                            # 177
    "                            reverse_dependency=reverse_dependency)",             # 178
    "        if not new_objs:",                                                       # 179
    "            return",                                                             # 180
]
MID_ORIG = "".join(f"{n}\t{ln}\n" for n, ln in enumerate(MID, 169))


def test_dedented_mid_file_slice_gets_its_real_indentation_back():
    view = "\n".join([
        "def collect(self, objs, source=None, nullable=False, collect_related=True,",
        "            source_attr=None, reverse_dependency=False, keep_parents=False):",
        "    if self.can_fast_delete(objs):",
        "        self.fast_deletes.append(objs)",
        "        return",
        "    new_objs = self.add(objs, source, nullable, reverse_dependency=reverse_dependency)",
    ])
    out = restore_indent(view, MID_ORIG).split("\n")
    assert out[0] == MID[0]
    assert out[1] == MID[1]                       # the 16-space continuation
    assert out[2:5] == MID[5:8]
    # the reflowed call has no exact match: it moves with its neighbours (+4)
    assert out[5] == "        new_objs = self.add(objs, source, nullable, reverse_dependency=reverse_dependency)"


def test_short_lines_follow_their_neighbours_not_a_far_match():
    orig = "".join(f"{n}\t{ln}\n" for n, ln in enumerate([
        "    def a(self):",            # 10
        "        if self.ready_to_go:",  # 11
        "            pass",            # 12
        "        else:",               # 13
        "            go_somewhere()",  # 14
        "def top():",                  # 15
        "    pass",                    # 16
    ], 10))
    view = "def a(self):\n    if self.ready_to_go:\n        pass\n    else:\n        go_somewhere()"
    assert restore_indent(view, orig).split("\n") == [
        "    def a(self):", "        if self.ready_to_go:", "            pass",
        "        else:", "            go_somewhere()"]


def test_correct_or_unnumbered_views_are_unchanged():
    assert restore_indent(VIEW, ORIG) == VIEW               # already at real indentation
    assert restore_indent(VIEW, "\n".join(PY)) == VIEW      # no line numbers to map against
    assert restore_indent("(irrelevant to the current task)", MID_ORIG) == "(irrelevant to the current task)"


class _DedentModel:
    def compress(self, text, **kwargs):
        return ("def collect(self, objs, source=None, nullable=False, collect_related=True,\n"
                "            source_attr=None, reverse_dependency=False, keep_parents=False):\n"
                "    if self.can_fast_delete(objs):\n"
                "        return")


def test_pipeline_restores_indentation_before_anchoring():
    lines = MID + [f"        v{i} = compute_value_number_{i}(row, column, extra)" for i in range(200)]
    orig = "".join(f"{n}\t{ln}\n" for n, ln in enumerate(lines, 169))
    pipe = CompressionPipeline(ParitokConfig())
    pipe._model = _DedentModel()
    view = pipe.compress(orig, query="add type hints", source="/p/deletion.py").compressed
    assert "\n                source_attr=None, reverse_dependency=False" in view
    assert "    def collect(self, objs, source=None, nullable=False, collect_related=True,  # L169-" in view
    cfg = ParitokConfig()
    cfg.compression.restore_indent = False
    pipe_off = CompressionPipeline(cfg)
    pipe_off._model = _DedentModel()
    off = pipe_off.compress(orig, query="add type hints", source="/p/other.py").compressed
    assert "\n            source_attr=None" in off


def test_reordered_line_keeps_its_own_indentation():
    """Seen live (query.py): the summary moved a line, the in-order pass skipped it, and
    it took its neighbours' shift (8 -> 12 spaces). A line whose text occurs once in the
    original takes that line's indentation."""
    orig = "".join(f"{n}\t{ln}\n" for n, ln in enumerate([
        "    def values_list(self, *fields):",         # 40
        "        names = list(fields)",                # 41
        "        indexes = range(len(names))",         # 42
        "        return self._values_list(names)",     # 43
    ], 40))
    view = ("def values_list(self, *fields):\n"
            "    names = list(fields)\n"
            "    return self._values_list(names)\n"
            "        indexes = range(len(names))")       # moved down, already at 8
    assert restore_indent(view, orig).split("\n") == [
        "    def values_list(self, *fields):", "        names = list(fields)",
        "        return self._values_list(names)", "        indexes = range(len(names))"]
