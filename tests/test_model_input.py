"""Regression: `model_input` lets the compressor feed the model a different form
(e.g. codex source with line numbers added) WITHOUT inflating the reported ratio.
The model sees model_input; original_tokens / hash / store all stay on `content`
— what the agent actually sent — so the ratio is honest and expand returns the
true original.
"""
from paritok.config import ParitokConfig
from paritok.pipelines.compress import CompressionPipeline
from paritok.token_counter import count_tokens


class _RecordingModel:
    def __init__(self):
        self.seen = None

    def compress(self, text, **kwargs):
        self.seen = text
        return "compressed"


def test_model_input_feeds_model_but_content_drives_metrics():
    pipe = CompressionPipeline(ParitokConfig())
    model = _RecordingModel()
    pipe._model = model

    content = "def f(x):\n    return x + 1\n" * 60          # unnumbered, > min_tokens
    model_input = "".join(f"{i:6d}→{ln}\n" for i, ln in enumerate(content.splitlines(), 1))
    assert count_tokens(model_input) > count_tokens(content)  # numbering inflates

    cr = pipe.compress(content, query="fix f", model_input=model_input)

    # the MODEL was fed the numbered form...
    assert model.seen == model_input
    # ...but the REPORTED original is the real (unnumbered) content, not inflated
    assert cr.original_tokens == count_tokens(content)
    # and expand returns the true original, never the numbered form
    assert pipe.storage.retrieve(cr.shadow_id) == content


def test_no_model_input_defaults_to_content():
    pipe = CompressionPipeline(ParitokConfig())
    model = _RecordingModel()
    pipe._model = model
    content = "def g(x):\n    return x + 42\n" * 80  # comfortably over min_tokens
    pipe.compress(content, query="x")
    assert model.seen == content  # unchanged behavior when model_input omitted
