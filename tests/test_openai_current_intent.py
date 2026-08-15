"""Regression (issue #4): the OpenAI Chat Completions path compresses against the
agent's CURRENT intent (latest reasoning), not the stale original task.

- openai.extract_query() now prefers the latest assistant reasoning + carries the
  overall task as trailing context (mirrors the Anthropic/Codex adapters).
- process_request() accepts a query override so the proxy threads that current-intent
  query into both history and tool-output compression.
"""
from paritok.proxy.adapters import openai as oai
from paritok.config import ParitokConfig
from paritok.middleware.wrapper import ParitokEngine


def test_extract_query_uses_current_agent_intent():
    messages = [
        {"role": "user", "content": "fix the bug in answer_question"},
        {"role": "assistant", "content": "Now tracing the caller in quick_eval to see how it invokes it"},
        {"role": "tool", "tool_call_id": "1", "content": "<file contents>"},
    ]
    q = oai.extract_query(messages)
    assert q is not None
    assert "quick_eval" in q                       # current intent (latest reasoning)
    assert "fix the bug in answer_question" in q   # overall task carried as context


def test_extract_query_turn0_falls_back_to_task():
    messages = [{"role": "user", "content": "fix the bug in answer_question"}]
    assert oai.extract_query(messages) == "fix the bug in answer_question"


def test_extract_query_handles_list_content():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "do the task"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "inspecting module foo now"}]},
    ]
    q = oai.extract_query(messages)
    assert "inspecting module foo" in q and "do the task" in q


def test_extract_query_skips_pure_tool_call_assistant():
    # An assistant turn that is only a tool call (content null) carries no reasoning;
    # extraction must not crash and should fall back to the task.
    messages = [
        {"role": "user", "content": "the task"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "type": "function"}]},
    ]
    assert oai.extract_query(messages) == "the task"


def test_process_request_honors_query_override():
    cfg = ParitokConfig()
    cfg.compression.min_tokens = 1
    cfg.compression.refusal_threshold = 0.0
    engine = ParitokEngine(cfg)

    seen = {}
    from paritok.pipelines.compress import CompressionResult

    def fake_compress(content, **kw):
        seen["query"] = kw.get("query")
        return CompressionResult(compressed=content, original_tokens=1,
                                 compressed_tokens=1, metadata={"skipped": True})

    engine.pipeline.compress = fake_compress
    # Anthropic-format tool_result so step-2 compression runs and passes query through.
    msgs = [{"role": "user",
             "content": [{"type": "tool_result", "tool_use_id": "1", "content": "x" * 80}]}]
    engine.process_request(msgs, None, query="CURRENT_INTENT_XYZ")
    assert seen.get("query") == "CURRENT_INTENT_XYZ"


def test_process_request_defaults_to_task_when_no_override():
    cfg = ParitokConfig()
    cfg.compression.min_tokens = 1
    cfg.compression.refusal_threshold = 0.0
    engine = ParitokEngine(cfg)

    seen = {}
    from paritok.pipelines.compress import CompressionResult

    def fake_compress(content, **kw):
        seen["query"] = kw.get("query")
        return CompressionResult(compressed=content, original_tokens=1,
                                 compressed_tokens=1, metadata={"skipped": True})

    engine.pipeline.compress = fake_compress
    msgs = [
        {"role": "user", "content": "the real task"},
        {"role": "user",
         "content": [{"type": "tool_result", "tool_use_id": "1", "content": "y" * 80}]},
    ]
    engine.process_request(msgs, None)  # no override -> wrapper._extract_query (task)
    assert seen.get("query") == "the real task"
