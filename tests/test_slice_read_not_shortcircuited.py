"""A Read of a slice (offset/limit) of a file read earlier must get THAT slice, not the
cached summary of the earlier whole-file read.

The path short-circuit used to accept any "normalized substring" of the earlier read
and hand back its whole summary — bypassing the token gates. Seen live (loopbench,
Opus): the agent read deletion.py lines 1-10 (71 tokens) and got the 1954-token
summary of the whole file, which didn't contain those lines, so it called
read_original next. An identical re-read still reuses the tag.
"""
from paritok.config import ParitokConfig
from paritok.pipelines.compress import CompressionPipeline

PATH = "/proj/django/db/models/deletion.py"
LINES = [f"    value_{i} = compute_something_for(row_{i}, column_{i})  # step {i}" for i in range(1, 301)]
FULL = "".join(f"{i:6d}\t{ln}\n" for i, ln in enumerate(LINES, 1))


class _CountingModel:
    def __init__(self):
        self.calls = []

    def compress(self, text, **kwargs):
        self.calls.append(text)
        return "summary of " + str(len(text))


def _pipe():
    pipe = CompressionPipeline(ParitokConfig())
    pipe._model = _CountingModel()
    return pipe


def _slice(start, n):
    return "".join(f"{i:6d}\t{LINES[i - 1]}\n" for i in range(start, start + n))


def test_small_slice_of_an_earlier_read_comes_back_verbatim():
    pipe = _pipe()
    full = pipe.compress(FULL, query="fix", source=PATH)
    assert full.compressed.startswith("[REF:")
    small = _slice(1, 10)
    cr = pipe.compress(small, query="fix", source=PATH)
    assert cr.compressed == small  # the lines the agent asked for, not the summary
    assert cr.metadata.get("reason") == "below_min_tokens"
    assert not cr.metadata.get("path_shortcircuit")


def test_large_slice_is_compressed_on_its_own():
    pipe = _pipe()
    pipe.compress(FULL, query="fix", source=PATH)
    big = _slice(100, 150)
    cr = pipe.compress(big, query="fix", source=PATH)
    assert not cr.metadata.get("path_shortcircuit")
    assert cr.shadow_id != pipe.compress(FULL, query="fix", source=PATH).shadow_id
    assert pipe._model.calls[-1].count("\n") >= 149  # the model saw the slice itself
    assert pipe.storage.retrieve(cr.shadow_id) == big  # read_original returns the slice


def test_identical_reread_still_reuses_the_tag():
    pipe = _pipe()
    first = pipe.compress(FULL, query="fix", source=PATH)
    n_calls = len(pipe._model.calls)
    # same file again, line numbers formatted differently (e.g. cat -n vs Read)
    again = "".join(f"{i}\t{ln}\n" for i, ln in enumerate(LINES, 1))
    cr = pipe.compress(again, query="fix", source=PATH)
    assert cr.metadata.get("path_shortcircuit")
    assert cr.compressed == first.compressed
    assert len(pipe._model.calls) == n_calls  # no second model call
