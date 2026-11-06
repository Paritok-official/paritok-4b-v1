"""PARITOK_POSTCOMPRESS_HOOK: an experiment hook run on each model output (the loopbench v2
simulation). It sees what produced the content, never breaks a request, and is skipped
for a GPU echo (so passthrough detection still works)."""
from paritok.config import ParitokConfig
from paritok.pipelines import compress as compress_mod
from paritok.pipelines.compress import CompressionPipeline

TEXT = "".join(f"{n}\tvalue_{n} = compute_something_long(row_{n}, column_{n}, extra)\n"
               for n in range(1, 120))
HOOK = '''
CALLS = []

def fix(compressed, *, original, query=None, source=None, kind=None, context=None):
    CALLS.append({"query": query, "source": source, "context": context})
    return "FIXED: " + compressed

def boom(compressed, **kwargs):
    raise RuntimeError("corrector down")
'''


class _Model:
    def __init__(self, out=None):
        self.out = out

    def compress(self, text, **kwargs):
        return self.out if self.out is not None else "summary: 119 assignments"


def _pipe(model, gpu=False):
    cfg = ParitokConfig()
    cfg.use_gpu_server = gpu
    pipe = CompressionPipeline(cfg)
    pipe._model = model
    return pipe


def test_hook_rewrites_output_and_sees_context(tmp_path, monkeypatch):
    f = tmp_path / "hook.py"
    f.write_text(HOOK, encoding="utf-8")
    monkeypatch.setenv("PARITOK_POSTCOMPRESS_HOOK", f"{f}:fix")
    compress_mod._HOOKS.clear()
    ctx = {"tool": "Bash", "input": {"command": "python -c 'print(1)'"}}
    cr = _pipe(_Model()).compress(TEXT, query="fix it", source=None, context=ctx)
    assert "FIXED: summary: 119 assignments" in cr.compressed
    calls = compress_mod._HOOKS[f"{f}:fix"].__globals__["CALLS"]
    assert calls == [{"query": "fix it", "source": None, "context": ctx}]


def test_failing_hook_falls_back_to_the_model_output(tmp_path, monkeypatch):
    f = tmp_path / "hook.py"
    f.write_text(HOOK, encoding="utf-8")
    monkeypatch.setenv("PARITOK_POSTCOMPRESS_HOOK", f"{f}:boom")
    compress_mod._HOOKS.clear()
    cr = _pipe(_Model()).compress(TEXT, query="x")
    assert "summary: 119 assignments" in cr.compressed and not cr.metadata.get("skipped")


def test_gpu_echo_skips_the_hook(tmp_path, monkeypatch):
    f = tmp_path / "hook.py"
    f.write_text(HOOK, encoding="utf-8")
    monkeypatch.setenv("PARITOK_POSTCOMPRESS_HOOK", f"{f}:fix")
    compress_mod._HOOKS.clear()
    cr = _pipe(_Model(out=TEXT), gpu=True).compress(TEXT, query="x")
    assert cr.metadata.get("reason") == "gpu_unavailable"


def test_no_hook_by_default(monkeypatch):
    monkeypatch.delenv("PARITOK_POSTCOMPRESS_HOOK", raising=False)
    assert compress_mod._post_compress_hook() is None
