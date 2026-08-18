"""Width-padded Read line numbers ("     123\\t") are stripped to the bare "123\\t"
the 4B was trained on BEFORE the model sees them — the padded (cat -n) form is
out-of-distribution and compresses markedly worse. Only the MODEL INPUT is de-padded;
`content` (token count / shadow store / expand_context) keeps the exact bytes the
agent sent, so accounting and read_original are unchanged.

Regression guard for the padded-line-number compression bug.
"""
from paritok.config import ParitokConfig
from paritok.pipelines.compress import (
    CompressionPipeline, _depad_line_numbers,
)
from paritok.token_counter import count_tokens


# ── the pure helper ──

def test_depad_strips_only_the_leading_pad():
    assert _depad_line_numbers("     123\tcode") == "123\tcode"
    assert _depad_line_numbers("\t\t7\tx") == "7\tx"
    # already bare / unnumbered / indented code are untouched
    assert _depad_line_numbers("123\tcode") == "123\tcode"
    assert _depad_line_numbers("def f():\n    return 1") == "def f():\n    return 1"
    # content after the tab (including its own leading spaces) is preserved verbatim
    assert _depad_line_numbers("     12\t    indented = 1") == "12\t    indented = 1"


def test_depad_is_per_line_and_idempotent():
    padded = "\n".join(f"{n:6d}\t line {n}" for n in range(1, 6))
    bare = "\n".join(f"{n}\t line {n}" for n in range(1, 6))
    assert _depad_line_numbers(padded) == bare
    assert _depad_line_numbers(bare) == bare               # idempotent
    assert _depad_line_numbers(_depad_line_numbers(padded)) == bare


# ── the pipeline wiring: model sees bare, content keeps the pad ──

class _FakeModel:
    """Records the text handed to the backend and returns a short compression."""
    def __init__(self):
        self.seen = None

    def compress(self, text, **kwargs):
        self.seen = text
        return "[compressed summary]"


def _pipeline_with_fake():
    cfg = ParitokConfig()
    cfg.compression.min_tokens = 1          # don't skip our small fixture
    cfg.use_gpu_server = False
    pipe = CompressionPipeline(cfg)
    fake = _FakeModel()
    pipe._model = fake
    return pipe, fake


def test_model_receives_depadded_input_but_accounting_keeps_the_original():
    pipe, fake = _pipeline_with_fake()
    padded = "\n".join(f"{n:6d}\tuser = record.get('user_{n}')  # noise line {n}"
                       for n in range(1, 40))

    cr = pipe.compress(padded, query="find the bug")

    # the model saw the BARE form — no width padding
    assert "     1\t" not in fake.seen
    assert "1\tuser = record.get('user_1')" in fake.seen
    assert fake.seen == _depad_line_numbers(padded)

    # but the reported original reflects what the agent ACTUALLY sent (padded) —
    # de-padding must not shrink the counted/stored original
    assert cr.original_tokens == count_tokens(padded)
    assert not cr.metadata.get("skipped")


def test_bare_input_is_unchanged_through_the_pipeline():
    pipe, fake = _pipeline_with_fake()
    bare = "\n".join(f"{n}\tconst x{n} = compute({n});" for n in range(1, 40))
    pipe.compress(bare, query="find the bug")
    assert fake.seen == bare                # already bare → untouched
