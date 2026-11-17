"""Regression (issue #63): the eval's agent defaulted to claude-sonnet-4-5-20250929,
which Anthropic retires on 2026-11-30, so `python eval_model/run.py` would stop at the
agent step. The default is now claude-sonnet-5-5, and the request is shaped for it:
no temperature (Sonnet 5.5 rejects non-default sampling values), room in max_tokens for
the thinking it does by default, and only text blocks are read for the diff.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval_model import agent  # noqa: E402

DIFF = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a = 1\n+a = 2\n"


class _Client:
    def __init__(self, content):
        self.kwargs = None
        self._content = content
        self.messages = self

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(content=self._content)


def test_default_agent_model_is_current():
    assert agent.DEFAULT_AGENT_MODEL == "claude-sonnet-5-5"


def test_run_uses_the_agent_default(monkeypatch):
    monkeypatch.delenv("PARITOK_EVAL_AGENT_MODEL", raising=False)
    src = open(os.path.join(os.path.dirname(agent.__file__), "run.py"), encoding="utf-8").read()
    assert "claude-sonnet-4-5-20250929\"))" not in src           # no hard-coded retired default
    assert 'os.environ.get("PARITOK_EVAL_AGENT_MODEL", agent.DEFAULT_AGENT_MODEL)' in src


def test_request_has_no_sampling_params_and_room_for_thinking():
    client = _Client([SimpleNamespace(type="text", text=f"```diff\n{DIFF}```")])
    agent.generate_patch(client, agent.DEFAULT_AGENT_MODEL, "a = 1", "a should be 2")
    kw = client.kwargs
    assert kw["model"] == "claude-sonnet-5-5"
    assert not {"temperature", "top_p", "top_k"} & set(kw)
    assert kw["max_tokens"] >= 16_000


def test_thinking_blocks_are_skipped_when_reading_the_patch():
    client = _Client([SimpleNamespace(type="thinking", thinking="", text=None),
                      SimpleNamespace(type="text", text=f"Fix:\n```diff\n{DIFF}```")])
    assert agent.generate_patch(client, "claude-sonnet-5-5", "a = 1", "a should be 2") == DIFF.strip()
