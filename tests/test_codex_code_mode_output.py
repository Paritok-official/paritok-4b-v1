"""Codex shell output in the formats current Codex versions actually send.

Codex 0.169 with gpt-6.1-sol runs commands through code mode: the model writes
JavaScript that calls `tools.exec_command(...)`, and each result comes back as a
JSON object (codex-rs tools/context.rs `code_mode_result`). What gets printed is up
to the model's script: one result per block (`text(await tools.exec_command(...))`),
or nested (`text({i, ...results[i]})` after Promise.allSettled), or an array. The
proxy used to compress that JSON as-is: ONE line of `\\n`-escaped text, which the 4B
never trained on. On the hosted endpoint that either failed the no-gain guard (the
original came back) or produced code that isn't in the file -- in a live session the
agent then called read_original and the turn cost more than it saved. Now every
command output inside the JSON is compressed unwrapped and put back in place.

A direct exec_command call (Codex 0.149) frames its output with a new header,
`Chunk ID: / Wall time: / Process exited with code N / Original token count: / Output:`.
`_split_codex_header` only knew `Exit code:`, so the header was compressed with the
body (the model dropped it, losing the exit status) AND line-numbered.

Fixtures copy the exact shapes captured from Codex 0.169.0 / 0.149.1 sessions.
"""
import json

import httpx
import pytest

from paritok.pipelines.compress import CompressionPipeline, CompressionResult
from paritok.proxy.server import (
    _ensure_line_numbers,
    _normalize_crlf,
    _split_codex_envelopes,
    _split_codex_header,
)

SOURCE = "".join(
    f"def rule_{i}(total: float) -> float:\r\n"
    f'    """Pricing rule {i}: adjust the total."""\r\n'
    f"    if total > {100 + i}:\r\n"
    f"        return total - {i}\r\n"
    "    return total\r\n\r\n"
    for i in range(12)
)
GREP = "".join(f"def rule_{i}(total: float) -> float:\n" for i in range(12))


def envelope(output, chunk="8846d4", wall="0.0000194"):
    # serde_json's compact form, as codex emits it
    return ('{"chunk_id":"%s","wall_time_seconds":%s,"exit_code":0,"original_token_count":574,"output":%s}'
            % (chunk, wall, json.dumps(output, ensure_ascii=False)))


# ── the JSON envelope ──

def test_envelope_unwraps_to_the_real_output():
    outputs, rewrap = _split_codex_envelopes(envelope(SOURCE))
    assert outputs == [SOURCE]                 # CRLF and quotes decoded


def test_rewrap_is_lossless_and_keeps_every_other_byte():
    text = envelope(SOURCE)
    outputs, rewrap = _split_codex_envelopes(text)
    assert rewrap(outputs) == text             # round trip is byte-identical
    new = rewrap(["[REF:abc] def rule_0(total: float) -> float: ..."])
    assert json.loads(new)["output"] == "[REF:abc] def rule_0(total: float) -> float: ..."
    assert new.startswith('{"chunk_id":"8846d4","wall_time_seconds":0.0000194,"exit_code":0,')


def test_session_id_variant_and_surrounding_whitespace():
    text = "\n" + json.dumps({"chunk_id": "a1", "wall_time_seconds": 1.0, "session_id": 7,
                              "output": "line\nline"}, separators=(",", ":")) + "\n"
    outputs, rewrap = _split_codex_envelopes(text)
    assert outputs == ["line\nline"]
    assert rewrap(outputs) == text


def test_result_nested_by_the_models_own_script():
    # Seen live (Codex 0.169.0): Promise.allSettled + text({i, ...results[i]})
    text = '{"i":0,"status":"fulfilled","value":%s}' % envelope(SOURCE)
    outputs, rewrap = _split_codex_envelopes(text)
    assert outputs == [SOURCE]
    new = json.loads(rewrap(["SHORT"]))
    assert new["i"] == 0 and new["status"] == "fulfilled" and new["value"]["output"] == "SHORT"
    assert new["value"]["chunk_id"] == "8846d4"


def test_several_results_in_one_block_are_each_replaced_in_order():
    text = "[%s,%s]" % (envelope(SOURCE), envelope(GREP, chunk="35d19e"))
    outputs, rewrap = _split_codex_envelopes(text)
    assert outputs == [SOURCE, GREP]
    new = json.loads(rewrap(["A", "B"]))
    assert [r["output"] for r in new] == ["A", "B"] and new[1]["chunk_id"] == "35d19e"


def test_pretty_printed_result_keeps_its_layout():
    text = json.dumps({"chunk_id": "a", "wall_time_seconds": 0.1, "exit_code": 0, "output": GREP}, indent=2)
    outputs, rewrap = _split_codex_envelopes(text)
    new = rewrap(["SHORT"])
    assert json.loads(new)["output"] == "SHORT" and new.startswith('{\n  "chunk_id": "a",')


def test_rejected_promise_and_other_fields_pass_through():
    text = '[{"status":"rejected","reason":"boom"},{"status":"fulfilled","value":%s}]' % envelope(GREP)
    outputs, rewrap = _split_codex_envelopes(text)
    assert outputs == [GREP]
    new = json.loads(rewrap(["SHORT"]))
    assert new[0] == {"status": "rejected", "reason": "boom"} and new[1]["value"]["output"] == "SHORT"


def test_differently_escaped_envelope_still_rewraps_to_valid_json():
    text = '{"chunk_id":"a","wall_time_seconds":0.1,"exit_code":0,"output":"x \\u003e y"}'
    outputs, rewrap = _split_codex_envelopes(text)
    assert outputs == ["x > y"]
    assert rewrap(outputs) == text
    assert json.loads(rewrap(["short"]))["output"] == "short"


@pytest.mark.parametrize("text", [
    '{"name": "app", "version": "1.0.0", "output": "dist"}',            # a package.json
    '{"content": [{"type": "text", "text": "hi"}], "isError": false}',  # an MCP result
    '{"output": "x"}',                                                  # no codex keys
    '{"chunk_id": "a", "output": 3}',                                   # output not text
    '[{"status": "rejected", "reason": "x"}]',
    "[1, 2, 3]",
    "not json {",
    "def f():\n    return {}",
])
def test_other_text_is_not_an_envelope(text):
    assert _split_codex_envelopes(text) is None


# ── the command-output header ──

def test_direct_exec_command_header_is_split_and_lines_number_from_one():
    text = ("Chunk ID: 7b541d\nWall time: 0.3155 seconds\nProcess exited with code 0\n"
            "Original token count: 748\nOutput:\n") + SOURCE
    header, body = _split_codex_header(_normalize_crlf(text))
    assert header.startswith("Chunk ID: 7b541d\n") and header.endswith("Output:\n")
    assert body.startswith("def rule_0(")
    assert _ensure_line_numbers(body).startswith("1\tdef rule_0(")   # not "6\t"


def test_running_process_header():
    header, body = _split_codex_header(
        "Chunk ID: a\nWall time: 10.0 seconds\nProcess running with session ID 3\nOutput:\nbuilding...\n")
    assert body == "building...\n"


def test_code_mode_status_block_is_all_header():
    assert _split_codex_header("Script completed\nWall time 0.4 seconds\nOutput:\n") == \
        ("Script completed\nWall time 0.4 seconds\nOutput:\n", "")


def test_truncation_warning_is_header():
    header, body = _split_codex_header(
        "Warning: truncated output (original token count: 11919)\nTotal output lines: 1285\n\n# Title\ntext\n")
    assert body == "# Title\ntext\n"


def test_old_shell_command_header_unchanged():
    header, body = _split_codex_header("Exit code: 0\nWall time: 0.1 seconds\nOutput:\ndef f():\n    pass\n")
    assert header == "Exit code: 0\nWall time: 0.1 seconds\nOutput:\n" and body == "def f():\n    pass\n"


# ── the proxy handler, with real Codex request shapes ──

class _Upstream(httpx.AsyncBaseTransport):
    def __init__(self):
        self.forwarded = None

    async def handle_async_request(self, request):
        self.forwarded = json.loads(request.content.decode())
        return httpx.Response(200, headers={"content-type": "application/json"}, content=json.dumps({
            "id": "r", "object": "response", "status": "completed",
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1}}).encode())


def _fake_compress(seen):
    def compress(self, content, **kw):
        seen.append({"content": content, "model_input": kw.get("model_input")})
        return CompressionResult(compressed=f"[REF:x{len(seen)}] COMPRESSED", original_tokens=600,
                                 compressed_tokens=10)
    return compress


def test_handler_compresses_code_mode_results_inside_their_json(monkeypatch):
    starlette = pytest.importorskip("starlette.testclient")
    from paritok.proxy.server import create_app
    seen = []
    monkeypatch.setattr(CompressionPipeline, "compress", _fake_compress(seen))
    up = _Upstream()
    app = create_app(http_client=httpx.AsyncClient(transport=up), openai_base_url="http://upstream.test")
    status = "Script completed\nWall time 0.4 seconds\nOutput:\n"
    # block 1 as printed by text(await ...), block 2 as printed by the live allSettled script
    file_block = envelope(SOURCE)
    grep_block = '{"i":1,"status":"fulfilled","value":%s}' % envelope(GREP, chunk="35d19e", wall="0.0000029")
    req = {"model": "gpt-6.1-sol", "stream": False,
           "tools": [{"type": "custom", "name": "exec"}],
           "input": [
               {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "what does rule_3 do"}]},
               {"type": "custom_tool_call", "call_id": "c1", "name": "exec",
                "input": 'text(await tools.exec_command({cmd:"Get-Content src/calc.py"}));\n'
                         'text(await tools.exec_command({cmd:"rg \'def \' src/calc.py"}));\n'},
               {"type": "custom_tool_call_output", "call_id": "c1", "output": [
                   {"type": "input_text", "text": status},
                   {"type": "input_text", "text": file_block},
                   {"type": "input_text", "text": grep_block}]},
           ]}
    with starlette.TestClient(app) as client:
        assert client.post("/v1/responses", json=req).status_code == 200

    # the compressor saw the real, LF, unescaped outputs -- not JSON
    assert [s["content"] for s in seen] == [_normalize_crlf(SOURCE), GREP]
    assert seen[0]["model_input"].startswith("1\tdef rule_0(")
    # what went upstream: status block untouched, each result still the same JSON shape
    blocks = [it for it in up.forwarded["input"] if it.get("type") == "custom_tool_call_output"][0]["output"]
    assert blocks[0]["text"] == status
    for blk, chunk in ((blocks[1], "8846d4"), (blocks[2], "35d19e")):
        obj = json.loads(blk["text"])
        obj = obj.get("value", obj)
        assert obj["chunk_id"] == chunk and obj["exit_code"] == 0
        assert obj["output"].endswith("COMPRESSED")
    assert json.loads(blocks[2]["text"])["status"] == "fulfilled"


def test_handler_strips_the_direct_exec_command_header(monkeypatch):
    starlette = pytest.importorskip("starlette.testclient")
    from paritok.proxy.server import create_app
    seen = []
    monkeypatch.setattr(CompressionPipeline, "compress", _fake_compress(seen))
    up = _Upstream()
    app = create_app(http_client=httpx.AsyncClient(transport=up), openai_base_url="http://upstream.test")
    header = ("Chunk ID: 7b541d\nWall time: 0.3155 seconds\nProcess exited with code 0\n"
              "Original token count: 748\nOutput:\n")
    req = {"model": "gpt-6.1-sol", "stream": False,
           "tools": [{"type": "function", "name": "exec_command", "parameters": {"type": "object"}}],
           "input": [
               {"type": "function_call", "call_id": "c1", "name": "exec_command",
                "arguments": json.dumps({"cmd": "Get-Content src/calc.py"})},
               {"type": "function_call_output", "call_id": "c1", "output": header + SOURCE},
           ]}
    with starlette.TestClient(app) as client:
        assert client.post("/v1/responses", json=req).status_code == 200
    assert seen[0]["content"] == _normalize_crlf(SOURCE)        # header not compressed
    assert seen[0]["model_input"].startswith("1\tdef rule_0(")  # numbering starts at the file's line 1
    out = [it for it in up.forwarded["input"] if it.get("type") == "function_call_output"][0]["output"]
    assert out.startswith(header) and out.endswith("COMPRESSED")
