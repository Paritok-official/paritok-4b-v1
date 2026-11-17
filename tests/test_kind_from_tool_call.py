"""The proxy decides each tool output's kind locally, from the tool call that produced it,
and sends it to the backend.

Before: every path called pipeline.compress() without a kind, so the kind was sniffed from
the text (paritok.strategies.tagger) and ~90% of real traffic went out as file_read -- grep
hits, test runs and the agent's own probe output all got the read-a-source-file prompt.

Now each path labels the output from its call (Anthropic tool_use, OpenAI tool_calls,
Responses function_call), the pipeline maps that label onto the names the deployed v1 model
was trained on (compression.kind_scheme: v1), and the GPU request carries both the mapped
`kind` and the precise `kind_detail`.
"""
import json

import httpx
import pytest

from paritok.config import GpuServerConfig, ParitokConfig
from paritok.middleware.wrapper import ParitokEngine
from paritok.pipelines.compress import CompressionPipeline, CompressionResult
from paritok.strategies.gpu_server import GpuServerStrategy

BIG = "line of output that is long enough to be worth compressing\n" * 60


def _skipped(content):
    return CompressionResult(compressed=content, original_tokens=1, compressed_tokens=1,
                             metadata={"skipped": True})


# ── Anthropic tool_result blocks (middleware) ──

def test_anthropic_tool_results_get_the_kind_of_their_tool_use():
    cfg = ParitokConfig()
    cfg.compression.min_tokens = 1
    engine = ParitokEngine(cfg)
    seen = []

    def fake_compress(content, **kw):
        seen.append(kw.get("kind_detail"))
        return _skipped(content)

    engine.pipeline.compress = fake_compress
    calls = [("r", "Read", {"file_path": "/w/a.py"}),
             ("g", "Grep", {"pattern": "def f"}),
             ("t", "Bash", {"command": "cd /w && pytest -q"}),
             ("p", "Bash", {"command": "python - <<'EOF'\nprint(1)\nEOF"}),
             ("d", "Bash", {"command": "git diff"})]
    msgs = [{"role": "user", "content": "fix the bug"}]
    for cid, name, inp in calls:
        msgs.append({"role": "assistant", "content": [{"type": "tool_use", "id": cid, "name": name, "input": inp}]})
        msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": cid, "content": BIG}]})
    engine.process_request(msgs, None, query="fix the bug")
    assert seen == ["file_read", "search_result", "test_output", "agent_generated_result", "vcs_output"]


def test_anthropic_tool_result_without_its_tool_use_falls_back_to_content():
    cfg = ParitokConfig()
    cfg.compression.min_tokens = 1
    engine = ParitokEngine(cfg)
    seen = []
    engine.pipeline.compress = lambda content, **kw: (seen.append(kw.get("kind_detail")), _skipped(content))[1]
    msgs = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "gone", "content": BIG}]}]
    engine.process_request(msgs, None, query="q")
    assert seen == [None]          # no call to go by: the pipeline sniffs the content


# ── the pipeline: map for the model, forward the detail ──

class _FakeModel:
    def __init__(self):
        self.kwargs = None

    def compress(self, text, **kwargs):
        self.kwargs = kwargs
        return "[compressed]"


def _pipeline(scheme="v1"):
    cfg = ParitokConfig()
    cfg.compression.min_tokens = 1
    cfg.compression.kind_scheme = scheme
    cfg.use_gpu_server = False
    pipe = CompressionPipeline(cfg)
    pipe._model = _FakeModel()
    return pipe


@pytest.mark.parametrize("detail, sent", [
    ("search_result", "tool_result"),
    ("agent_generated_result", "log_output"),
    ("test_output", "log_output"),
    ("file_read", "file_read"),
    ("directory_listing", "directory_listing"),
])
def test_pipeline_maps_kind_detail_for_the_v1_model(detail, sent):
    pipe = _pipeline("v1")
    pipe.compress(BIG, query="q", kind_detail=detail)
    assert pipe._model.kwargs["kind"] == sent
    assert pipe._model.kwargs["kind_detail"] == detail


def test_pipeline_v2_scheme_sends_the_kind_as_is():
    pipe = _pipeline("v2")
    pipe.compress(BIG, query="q", kind_detail="search_result")
    assert pipe._model.kwargs["kind"] == "search_result"


def test_explicit_kind_is_used_verbatim():
    pipe = _pipeline("v1")
    pipe.compress(BIG, query="q", kind="assistant_thinking")
    assert pipe._model.kwargs["kind"] == "assistant_thinking"


def test_no_call_sniffs_the_content():
    pipe = _pipeline("v1")
    numbered = "".join(f"{i}\tvalue_{i} = compute({i})\n" for i in range(1, 80))
    pipe.compress(numbered, query="q")
    assert pipe._model.kwargs["kind_detail"] == "file_read"
    assert pipe._model.kwargs["kind"] == "file_read"


def test_kind_scheme_is_validated():
    with pytest.raises(AssertionError):
        ParitokConfig.from_dict({"compression": {"kind_scheme": "v3"}})
    assert ParitokConfig.from_dict({"compression": {"kind_scheme": "v2"}}).compression.kind_scheme == "v2"
    assert ParitokConfig().compression.kind_scheme == "v1"


# ── the GPU server request carries both ──

def test_gpu_request_carries_kind_and_kind_detail(monkeypatch):
    seen = {}

    def fake_post(url, **kwargs):
        seen["json"] = kwargs.get("json")
        return httpx.Response(200, json={"compressed": "x", "gpu_available": True},
                              request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    GpuServerStrategy(GpuServerConfig(api_key="k")).compress(
        BIG, query="q", kind="tool_result", kind_detail="search_result")
    assert seen["json"]["kind"] == "tool_result"
    assert seen["json"]["kind_detail"] == "search_result"


def test_gpu_request_omits_kind_detail_when_unknown(monkeypatch):
    seen = {}
    monkeypatch.setattr(httpx, "post", lambda url, **kw: (seen.update(json=kw.get("json")), httpx.Response(
        200, json={"compressed": "x", "gpu_available": True}, request=httpx.Request("POST", url)))[1])
    GpuServerStrategy(GpuServerConfig(api_key="k")).compress(BIG, query="q", kind="file_read")
    assert "kind_detail" not in seen["json"]


# ── OpenAI chat + Codex Responses (the proxy handlers) ──

class _Upstream(httpx.AsyncBaseTransport):
    def __init__(self, body):
        self.body = body

    async def handle_async_request(self, request):
        return httpx.Response(200, headers={"content-type": "application/json"},
                              content=json.dumps(self.body).encode())


def _capture_pipeline(monkeypatch):
    seen = []

    def fake_compress(self, content, **kw):
        seen.append({"kind_detail": kw.get("kind_detail"), "model_input": kw.get("model_input"),
                     "content": content})
        return _skipped(content)

    monkeypatch.setattr(CompressionPipeline, "compress", fake_compress)
    return seen


def test_openai_chat_tool_messages_get_the_kind_of_their_tool_call(monkeypatch):
    starlette = pytest.importorskip("starlette.testclient")
    from paritok.proxy.server import create_app
    seen = _capture_pipeline(monkeypatch)
    up = _Upstream({"id": "c", "object": "chat.completion", "choices": [
        {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}]})
    app = create_app(http_client=httpx.AsyncClient(transport=up), openai_base_url="http://upstream.test")
    msgs = [{"role": "user", "content": "fix it"}]
    for cid, cmd in (("1", "grep -rn foo src"), ("2", "python -c 'print(1)'"), ("3", "cat src/a.py")):
        msgs.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": cmd})}}]})
        msgs.append({"role": "tool", "tool_call_id": cid, "content": BIG})
    with starlette.TestClient(app) as client:
        r = client.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": msgs})
    assert r.status_code == 200
    assert [s["kind_detail"] for s in seen] == ["search_result", "agent_generated_result", "file_read"]


def test_codex_items_get_the_kind_of_their_call_and_only_file_reads_are_numbered(monkeypatch):
    starlette = pytest.importorskip("starlette.testclient")
    from paritok.proxy.server import create_app
    seen = _capture_pipeline(monkeypatch)
    up = _Upstream({"id": "r", "object": "response", "status": "completed", "output": [
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}],
        "usage": {"input_tokens": 1, "output_tokens": 1}})
    app = create_app(http_client=httpx.AsyncClient(transport=up), openai_base_url="http://upstream.test")
    code = "".join(f"def f{i}(x):\n    return x + {i}\n" for i in range(40))
    items = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "fix it"}]}]
    for cid, cmd in (("a", "sed -n '1,200p' src/m.py"), ("b", "python - <<'EOF'\nimport m\nprint(m)\nEOF")):
        items.append({"type": "function_call", "call_id": cid, "name": "shell_command",
                      "arguments": json.dumps({"command": cmd})})
        items.append({"type": "function_call_output", "call_id": cid,
                      "output": "Exit code: 0\nWall time: 0.1 seconds\nOutput:\n" + code})
    req = {"model": "gpt-5", "input": items, "stream": False,
           "tools": [{"type": "function", "name": "shell_command", "parameters": {"type": "object"}}]}
    with starlette.TestClient(app) as client:
        r = client.post("/v1/responses", json=req)
    assert r.status_code == 200
    assert [s["kind_detail"] for s in seen] == ["file_read", "agent_generated_result"]
    # the file read is line-numbered for the model; the script's printout is not dressed up as a file
    assert seen[0]["model_input"].startswith("1\t")
    assert seen[1]["model_input"] == seen[1]["content"]
