"""Edit recovery vs the three ways a compressed view misleads an edit, each taken from
a real loopbench run (django, Claude Code + sonnet through the gateway):

1. Indentation: a Read of a mid-function slice comes back shifted one level left, so
   the model writes old/new 4 spaces too shallow. Recovery used to fix only old_string
   and spliced new_string in at the summary's indentation — corrupting the file.
2. Hidden comments: the summary dropped comments, the model rewrote the block, and
   recovery replaced the whole real region with new_string — deleting 8 lines of the
   user's comments the model never saw.
3. Hidden docstring: the summary dropped a docstring; the model "added" one after the
   (exactly matching) signature line, leaving two.
Plus the Bash route: the same indentation slip inside a `python - <<EOF ... replace`.

Recovery must fix all of these after the fact, without putting anything back into
the view the model sees (that would spend the tokens compression saved).
"""
import pytest

from paritok.pipelines.codex_edit_recovery import parse_shell_replaces, rewrite_shell_command
from paritok.pipelines.edit_recovery import reconcile_edit, recover_edit

# django/db/models/deletion.py @ 8180ffba21, Collector.collect (real indentation 20/24)
REAL = (
    "                batches = self.get_del_batches(new_objs, field)\n"
    "                for batch in batches:\n"
    "                    sub_objs = self.related_objects(related, batch)\n"
    "                    if self.can_fast_delete(sub_objs, from_field=field):\n"
    "                        self.fast_deletes.append(sub_objs)\n"
    "                    elif sub_objs:\n"
    "                        field.remote_field.on_delete(self, field, sub_objs, self.using)\n"
    "            for field in model._meta.private_fields:\n"
)
# What the model wrote from the summary, which showed this block 4 spaces shallower.
MODEL_OLD = ("                elif sub_objs:\n"
             "                    field.remote_field.on_delete(self, field, sub_objs, self.using)\n")
MODEL_NEW = ("                else:\n"
             "                    related_model = related.related_model\n"
             "                    if sub_objs:\n"
             "                        field.remote_field.on_delete(self, field, sub_objs, self.using)\n")
INTENDED = ("                    else:\n"
            "                        related_model = related.related_model\n"
            "                        if sub_objs:\n"
            "                            field.remote_field.on_delete(self, field, sub_objs, self.using)\n")


def _apply(original, pair):
    old, new = pair
    assert original.count(old) == 1, "the rewritten old_string must match the file once"
    return original.replace(old, new, 1)


def test_dedented_view_edit_lands_at_real_indentation():
    assert MODEL_OLD not in REAL  # the model's edit as written would fail (or worse)
    out = recover_edit(MODEL_OLD, MODEL_NEW, REAL)
    assert out is not None
    assert _apply(REAL, out) == REAL.replace(
        "                    elif sub_objs:\n"
        "                        field.remote_field.on_delete(self, field, sub_objs, self.using)\n",
        INTENDED)


def test_inconsistent_indentation_is_left_alone():
    """If the model's lines disagree on the offset, don't guess — leave the edit be."""
    old = ("                elif sub_objs:\n"
           "                        field.remote_field.on_delete(self, field, sub_objs, self.using)\n"
           "        for field in model._meta.private_fields:\n")
    assert recover_edit(old, old.replace("elif", "else"), REAL) is None


# Collector.__init__ — real file vs what the model saw and wrote (r4b, task 5).
INIT_REAL = (
    "    def __init__(self, using):\n"
    "        self.using = using\n"
    "        # Initially, {model: {instances}}, later values become lists.\n"
    "        self.data = {}\n"
    "        self.field_updates = {}  # {model: {(field, value): {instances}}}\n"
    "        # fast_deletes is a list of queryset-likes that can be deleted without\n"
    "        # fetching the objects into memory.\n"
    "        self.fast_deletes = []\n"
    "\n"
    "        # Tracks deletion-order dependency for databases without transactions\n"
    "        # or ability to defer constraint checks. Only concrete model classes\n"
    "        # should be included, as the dependencies exist only between actual\n"
    "        # database tables; proxy models are represented here by their concrete\n"
    "        # parent.\n"
    "        self.dependencies = {}  # {model: {models}}\n"
)
INIT_OLD = ("    def __init__(self, using):\n        self.using = using\n        self.data = {}\n"
            "        self.field_updates = {}\n        self.fast_deletes = []\n"
            "        self.dependencies = {}")
INIT_NEW = (
    "    def __init__(self, using: str) -> None:\n"
    '        """Gather the objects to delete for the database alias `using`."""\n'
    "        self.using = using\n"
    "        # Initially, {model: {instances}}, later values become lists.\n"
    "        self.data: Dict = {}\n"
    "        self.field_updates: Dict = {}\n"
    "        # Fast deletes are collected as a list of QuerySets (or instances).\n"
    "        self.fast_deletes: List[Any] = []\n"
    "        self.dependencies: Dict = {}")


def test_comments_the_model_never_saw_survive_its_rewrite():
    out = recover_edit(INIT_OLD, INIT_NEW, INIT_REAL)
    assert out is not None
    result = _apply(INIT_REAL, out)
    # each hidden comment is back, in order, right above the code it describes —
    # trailing ones on their (now annotated) line
    assert ("        # fast_deletes is a list of queryset-likes that can be deleted without\n"
            "        # fetching the objects into memory.\n"
            "        self.fast_deletes: List[Any] = []\n") in result
    assert ("        # Tracks deletion-order dependency for databases without transactions\n"
            "        # or ability to defer constraint checks. Only concrete model classes\n"
            "        # should be included, as the dependencies exist only between actual\n"
            "        # database tables; proxy models are represented here by their concrete\n"
            "        # parent.\n"
            "        self.dependencies: Dict = {}  # {model: {models}}") in result
    assert "self.field_updates: Dict = {}  # {model: {(field, value): {instances}}}" in result
    # the model's own changes land, and a comment it DID see isn't doubled
    assert "def __init__(self, using: str) -> None:" in result
    assert "self.fast_deletes: List[Any] = []" in result
    assert result.count("# Initially, {model: {instances}}") == 1
    compile(result.replace("    def", "def", 1).replace("\n    ", "\n"), "x", "exec")


ADD_REAL = (
    "    def add(self, objs, source=None, nullable=False, reverse_dependency=False):\n"
    '        """\n'
    "        Add 'objs' to the collection of objects to be deleted.  If the call is\n"
    "        the result of a cascade, 'source' should be the model that caused it,\n"
    "        and 'nullable' should be set to True if the relation can be null.\n"
    "\n"
    "        Return a list of all objects that were not already collected.\n"
    '        """\n'
    "        if not objs:\n"
    "            return []\n"
)
ADD_OLD = "    def add(self, objs, source=None, nullable=False, reverse_dependency=False):\n"
ADD_NEW = (
    "    def add(\n"
    "        self,\n"
    "        objs: Sequence['Model'],\n"
    "        source: Optional[Type['Model']] = None,\n"
    "        nullable: bool = False,\n"
    "        reverse_dependency: bool = False,\n"
    "    ) -> List['Model']:\n"
    '        """\n'
    "        Add `objs` to the collection of objects to delete.\n"
    '        """\n'
)
ADD_VIEW = (  # the summary kept the signature and body, dropped the docstring
    "    def add(self, objs, source=None, nullable=False, reverse_dependency=False):\n"
    "        if not objs:\n"
    "            return []\n"
)


def test_hidden_docstring_is_not_duplicated():
    out = reconcile_edit(ADD_OLD, ADD_NEW, ADD_REAL, views=[ADD_VIEW])
    assert out is not None
    result = _apply(ADD_REAL, out)
    assert result.count('"""') == 2  # exactly one docstring: the original
    assert "Add 'objs' to the collection of objects to be deleted." in result
    assert ") -> List['Model']:" in result  # the type hints still land


def test_docstring_the_model_could_see_is_its_call():
    """If the view showed the existing docstring, the model's new one is deliberate."""
    assert reconcile_edit(ADD_OLD, ADD_NEW, ADD_REAL, views=[ADD_REAL]) is None


def test_exact_suffix_match_with_dedented_new_lines_is_shifted():
    """old_string matched only as the tail of a deeper line (the view was dedented);
    its extra lines must move to the file's indentation."""
    view = REAL.replace("                    elif", "                elif").replace(
        "                        field.remote_field", "                    field.remote_field")
    old = "                elif sub_objs:"
    new = "                elif sub_objs and batch:\n                    pass"
    out = reconcile_edit(old, new, REAL, views=[view])
    assert out is not None
    result = _apply(REAL, out)
    assert "                    elif sub_objs and batch:\n                        pass" in result


def test_exact_match_needing_nothing_is_untouched():
    assert reconcile_edit("self.fast_deletes.append(sub_objs)",
                          "self.fast_deletes.append(sub_objs)  # fast", REAL, views=[REAL]) is None


# The Bash route (r4b, task 0): a python heredoc replace written at the summary's indent.
BASH = (
    "cd D:/I/x/django && python - <<'E'\n"
    "p='django/db/models/deletion.py'\n"
    "s=open(p).read()\n"
    'old="""' + MODEL_OLD + '"""\n'
    'new="""' + MODEL_NEW + '"""\n'
    "assert old in s\n"
    "open(p,'w').write(s.replace(old,new,1))\n"
    "E\n"
)


def test_bash_heredoc_replace_is_rewritten_to_the_real_file():
    (sr,) = parse_shell_replaces(BASH)
    assert sr.old == MODEL_OLD and sr.new == MODEL_NEW
    files = {"D:/I/x/django/django/db/models/deletion.py": REAL}
    new_cmd, n = rewrite_shell_command(
        BASH, lambda cmd: [b for p, b in files.items() if p.endswith("db/models/deletion.py")])
    assert n == 1
    (fixed,) = parse_shell_replaces(new_cmd)
    assert fixed.old in REAL  # the script's `assert old in s` now passes
    assert _apply(REAL, (fixed.old, fixed.new)).count("                    else:\n") == 1


class _Resp:
    def __init__(self, payload):
        self._payload, self.status_code, self.headers = payload, 200, {}

    def json(self):
        return self._payload


class _Up:
    def __init__(self, block):
        self.block = block

    async def post(self, url, headers=None, json=None):
        return _Resp({"id": "m", "type": "message", "role": "assistant",
                      "stop_reason": "tool_use", "content": [self.block]})

    async def aclose(self):
        pass


def _through_proxy(monkeypatch, block, messages, seed=None):
    pytest.importorskip("starlette")
    from starlette.testclient import TestClient
    from paritok.storage import MemoryShadowStorage
    import paritok.middleware.wrapper as wrapper
    from paritok.proxy import server
    storage = MemoryShadowStorage()
    if seed:
        path, original, view = seed
        sid = storage.store(original)
        storage.set_shadow_for_path(path, sid)
        storage.cache_compressed(sid, view)
    monkeypatch.setattr(wrapper, "build_shadow_storage", lambda config: storage)
    app = server.create_app(anthropic_base_url="http://upstream.test", http_client=_Up(block))
    with TestClient(app) as client:
        r = client.post("/v1/messages", json={"model": "claude-sonnet-4-5-20250929",
                                              "max_tokens": 64, "messages": messages})
    return r.json()["content"][0]["input"]


def test_proxy_rewrites_a_bash_heredoc_replace(monkeypatch):
    path = "D:\\I\\x\\django\\django\\db\\models\\deletion.py"
    numbered = "".join(f"{i}\t{ln}\n" for i, ln in enumerate(REAL.splitlines(), 1))
    messages = [
        {"role": "user", "content": "fix the delete optimization"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "r1", "name": "Read",
                                           "input": {"file_path": path}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "r1",
                                      "content": numbered}]},
    ]
    block = {"type": "tool_use", "id": "b1", "name": "Bash", "input": {"command": BASH}}
    cmd = _through_proxy(monkeypatch, block, messages)["command"]
    (fixed,) = parse_shell_replaces(cmd)
    assert fixed.old in REAL and "                    else:\n" in fixed.new


def test_proxy_drops_a_duplicate_docstring(monkeypatch):
    path = "/p/deletion.py"
    numbered = "".join(f"{i}\t{ln}\n" for i, ln in enumerate(ADD_REAL.splitlines(), 1))
    block = {"type": "tool_use", "id": "e1", "name": "Edit",
             "input": {"file_path": path, "old_string": ADD_OLD, "new_string": ADD_NEW}}
    inp = _through_proxy(monkeypatch, block, [{"role": "user", "content": "add type hints"}],
                         seed=(path, numbered, "[REF:x src=/p/deletion.py] " + ADD_VIEW))
    assert inp["old_string"] == ADD_OLD
    assert '"""' not in inp["new_string"] and ") -> List['Model']:" in inp["new_string"]


@pytest.mark.parametrize("literal", ['has """ inside', "back\\slash"])
def test_bash_literals_that_cannot_be_respliced_are_skipped(literal):
    cmd = 'python - <<E\nold="""' + literal + '"""\nnew="""x"""\ns.replace(old,new)\nE'
    assert all(sr.old != literal for sr in parse_shell_replaces(cmd))
