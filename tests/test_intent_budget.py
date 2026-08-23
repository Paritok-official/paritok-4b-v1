"""Regression: a pathologically large intent/query (e.g. a whole SWE-bench issue,
~7k tokens) was injected verbatim into EVERY compression request next to a content
chunk, overflowing the model's num_ctx window (8192) so the backend 400'd and the
read passed through UNCOMPRESSED. The pipeline now caps the intent to a budget that
keeps system + intent + content chunk under num_ctx, warning once per oversized query.
"""
from paritok.config import ParitokConfig
from paritok.pipelines.compress import (
    CompressionPipeline,
    _INTENT_WRAPPER_TOKENS,
    _intent_budget,
    _truncate_intent,
)
from paritok.strategies.chunking import CHUNK_SIZE
from paritok.strategies.local_model import (
    _CTX_SAFETY_MARGIN,
    _MIN_NUM_PREDICT,
    _TOKENIZER_SLACK,
)
from paritok.strategies.prompts import system_prompt_for_kind
from paritok.token_counter import count_tokens

# Bare "N\t" line numbers (the shape the compressor de-pads to); the intent-budget
# math here is independent of the line-number format.
CONTENT = "\n".join(f"{i}\tdef f_{i}(): return {i}" for i in range(200))


class _RecordingModel:
    """Backend stand-in that records the intent it was handed."""
    def __init__(self):
        self.seen_query = "UNSET"

    def compress(self, content, *, query=None, **kwargs):
        self.seen_query = query
        return "compressed body"


def _pipeline():
    cfg = ParitokConfig()
    cfg.compression.min_tokens = 1        # force a compression attempt
    cfg.compression.refusal_threshold = 0.0
    pipe = CompressionPipeline(cfg)
    model = _RecordingModel()
    pipe._model = model
    return pipe, model


def test_intent_budget_keeps_request_under_num_ctx():
    sys_big = max(count_tokens(system_prompt_for_kind(k))
                  for k in ("file_read", "code", "edit", "tool_result"))
    b_full = _intent_budget(sys_big, CHUNK_SIZE, 8192)   # worst case: full chunk
    b_small = _intent_budget(sys_big, 100, 8192)         # tiny read
    assert b_full > 0
    assert b_small > b_full                              # small content -> more intent room
    usable = int((8192 - _MIN_NUM_PREDICT - _CTX_SAFETY_MARGIN) / _TOKENIZER_SLACK)
    # system + intent + chunk + wrapper must fit the usable prompt window
    assert sys_big + b_full + CHUNK_SIZE + _INTENT_WRAPPER_TOKENS <= usable
    # a bigger num_ctx grants a bigger budget (so raising num_ctx auto-relaxes the cap)
    assert _intent_budget(sys_big, CHUNK_SIZE, 16384) > b_full


def test_truncate_intent():
    out, orig, cut = _truncate_intent("word " * 5000, 100)
    assert cut and orig > 100
    assert count_tokens(out) <= 100
    out2, orig2, cut2 = _truncate_intent("small query", 100)
    assert not cut2 and out2 == "small query"


def test_oversized_intent_truncated_before_backend():
    pipe, model = _pipeline()
    huge_intent = "fix the bug in this function. " * 2000     # thousands of tokens
    pipe.compress(CONTENT, query=huge_intent, kind="file_read")
    # Use the pipeline's actual num_ctx (from config, not a hardcoded 8192) so this
    # tracks the configured default — the cap the pipeline applies is computed from it.
    budget = _intent_budget(count_tokens(system_prompt_for_kind("file_read")),
                            count_tokens(CONTENT), pipe._intent_num_ctx)
    assert model.seen_query is not None
    assert count_tokens(model.seen_query) <= budget                 # capped to the budget
    assert count_tokens(model.seen_query) < count_tokens(huge_intent)


def test_normal_intent_untouched():
    pipe, model = _pipeline()
    pipe.compress(CONTENT, query="fix the off-by-one in the loop", kind="file_read")
    assert model.seen_query == "fix the off-by-one in the loop"
