"""paritok.strategies.kind_label: the kind of a tool output comes from the tool call that
produced it (tool name + shell command), and only falls back to the text when there is no
call. The old content sniffer labelled ~90% of production traffic file_read (grep results,
test runs, the agent's own probe scripts all got the read-a-source-file prompt); on the
SWE-rebench trajectories the v1 training set was cut from, the call-based kind is exact.
"""
import json

import pytest

from paritok.strategies.kind_label import (
    V1_KIND,
    code_mode_calls,
    code_mode_kinds,
    code_mode_kinds_by_position,
    kind_from_command,
    kind_from_content,
    kinds_by_position,
    label,
    label_conversation,
    wire_kind,
)


# ── from the tool name ──

@pytest.mark.parametrize("tool, inp, want", [
    ("Read", {"file_path": "a.py"}, "file_read"),
    ("Grep", {"pattern": "def f"}, "search_result"),
    ("Glob", {"pattern": "**/*.py"}, "search_result"),
    ("mcp__fs__read_file", {"path": "a"}, "file_read"),
    ("WebFetch", {"url": "https://x"}, "tool_result"),
    ("Edit", {"file_path": "a.py"}, "tool_result"),
    ("execute_ipython_cell", {"code": "1"}, "agent_generated_result"),
    ("str_replace_editor", {"command": "view", "path": "/w/a.py"}, "file_read"),
    ("str_replace_editor", {"command": "view", "path": "/w/src"}, "directory_listing"),
    ("str_replace_editor", {"command": "view", "path": "/workspace/moto__4.1"}, "directory_listing"),
    ("str_replace_editor", {"command": "view", "path": "/w/Makefile"}, "file_read"),
    ("str_replace_editor", {"command": "str_replace", "path": "/w/a.py"}, "tool_result"),
    ("exec_command", {"cmd": "sed -n '1,200p' src/App.tsx"}, "file_read"),
    ("shell_command", {"command": "rg -n useState src"}, "search_result"),
])
def test_kind_from_tool(tool, inp, want):
    assert label("x", tool, inp) == (want, "tool")


def test_editor_view_of_a_versioned_repo_dir_uses_the_output_header():
    out = ("Here's the files and directories up to 2 levels deep in /workspace/neogeny__TatSu__4.3, "
           "excluding hidden items:\n/workspace/neogeny__TatSu__4.3/")
    assert label(out, "str_replace_editor", {"command": "view", "path": "/workspace/neogeny__TatSu__4.3"})[0] \
        == "directory_listing"


# ── from the shell command ──

@pytest.mark.parametrize("cmd, files, want", [
    ("python -c 'print(1)'", None, "agent_generated_result"),
    ("cd /w && python3 - <<'EOF'\nimport os\nprint(os.getcwd())\nEOF", None, "agent_generated_result"),
    ("cat <<'EOF' | python\nprint(1)\nEOF", None, "agent_generated_result"),
    ("cat > t.py <<'EOF'\nimport pytest\nEOF\npython t.py", None, "agent_generated_result"),
    ("python probe.py 2>&1 | head -40", {"probe.py"}, "agent_generated_result"),
    ("python probe.py 2>&1 | head -40", None, "command_output"),
    ("g++ t.cpp -o t && ./t", None, "agent_generated_result"),
    ("python -m pytest test_repro.py -q", {"test_repro.py"}, "test_output"),
    ("python tests/runtests.py delete --verbosity 2", None, "test_output"),
    ("cd /w && pytest -x 2>&1 | tail -30", None, "test_output"),
    ("uv run pytest", None, "test_output"),
    ("npx tsc --noEmit", None, "test_output"),
    (["bash", "-lc", "npm test -- --run"], None, "test_output"),
    ("pip install pytest", None, "command_output"),
    ("git --no-pager diff HEAD~1", None, "vcs_output"),
    ("cd /w && git status && git diff", None, "vcs_output"),
    ("rg -n foo src/", None, "search_result"),
    ("find . -name '*.py' | xargs grep -l foo", None, "search_result"),
    ('grep -A5 -B5 "if a.n > 2 and b == 1:" skrf/network.py', None, "search_result"),
    ("sed -n '10,40p' a.py", None, "file_read"),
    ("cat -n a.py | sed -n 10,40p", None, "file_read"),
    ("if [ -f a.py ]; then cat a.py; fi", None, "file_read"),
    ('powershell -NoProfile -Command "Get-Content src/app.ts | Select-Object -First 80"', None, "file_read"),
    ("ls -la", None, "directory_listing"),
    ("npm run build && ls dist", None, "command_output"),
    ("sed -i 's/a/b/' a.py && echo done", None, "command_output"),
])
def test_kind_from_command(cmd, files, want):
    assert kind_from_command(cmd, files) == want


# ── from the content (no tool call) ──

def test_content_fallback():
    assert kind_from_content("1\timport os\n2\timport re\n3\tx = 1\n") == "file_read"
    assert kind_from_content('{"chunk_id": "a", "exit_code": 0, "output": "src/a.py:3:def f():\\nsrc/b.py:9:def g():"}') \
        == "search_result"
    assert kind_from_content("The file /w/a.py has been edited. Here's the result of running `cat -n` on a "
                             "snippet:\n 1\tx\n") == "tool_result"
    # a numbered file that mentions "[tool_calls]:" is still a file, not an agent message
    assert kind_from_content('1\tx = "[tool_calls]: [{"\n2\ty = 1\n3\tz = 2\n') == "file_read"
    # a two-line CLI usage error is not a config file
    assert kind_from_content("Usage: cli search [OPTIONS]\n\nError: query required\n") == "command_output"


def test_shell_without_command_falls_back_to_content():
    assert label("1\timport os\n2\timport re\n3\tx = 1\n", "exec_command", {}) == ("file_read", "content")


# ── whole conversations, all three formats ──

def test_anthropic_tracks_scripts_the_agent_wrote():
    conv = [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "Write",
                                           "input": {"file_path": "/w/probe.py", "content": "print(1)"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "File created"}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "b", "name": "Bash",
                                           "input": {"command": "python probe.py"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "b",
                                      "content": [{"type": "text", "text": "1"}]}]},
    ]
    assert [r["kind"] for r in label_conversation(conv)] == ["tool_result", "agent_generated_result"]
    assert kinds_by_position(conv) == {(1, 0): "tool_result", (3, 0): "agent_generated_result"}


def test_openai_chat_openhands_trajectory():
    conv = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {
            "name": "str_replace_editor", "arguments": json.dumps({"command": "create", "path": "/w/repro.py"})}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "File created successfully at: /w/repro.py"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c2", "function": {
            "name": "execute_bash", "arguments": json.dumps({"command": "cd /w && python repro.py"})}}]},
        {"role": "tool", "tool_call_id": "c2", "content": "Traceback ..."},
    ]
    assert kinds_by_position(conv) == {(1, None): "tool_result", (3, None): "agent_generated_result"}


def test_responses_codex_items():
    conv = [
        {"type": "function_call", "call_id": "x", "name": "exec_command",
         "arguments": json.dumps({"cmd": "rg -n useState src"})},
        {"type": "function_call_output", "call_id": "x", "output": "src/a.tsx:3:..."},
    ]
    assert kinds_by_position(conv) == {(1, None): "search_result"}


def test_kinds_by_position_never_raises():
    assert kinds_by_position(None) == {}
    # an output with no call to go by is left to the pipeline's content sniffing
    assert kinds_by_position([None, 3, "x", {"role": "tool"}]) == {}


# ── what reaches the model ──

def test_wire_kind_v1_maps_onto_names_v1_was_trained_on():
    v1_names = {"file_read", "tool_result", "log_output", "directory_listing"}
    assert set(V1_KIND.values()) <= v1_names
    assert wire_kind("search_result", "v1") == "tool_result"
    assert wire_kind("agent_generated_result", "v1") == "log_output"
    assert wire_kind("test_output", "v1") == "log_output"
    assert wire_kind("file_read", "v1") == "file_read"
    assert wire_kind("assistant_thinking", "v1") == "assistant_thinking"   # history kinds pass through


def test_wire_kind_v2_sends_the_kind_as_is():
    assert wire_kind("search_result", "v2") == "search_result"
    assert wire_kind(None, "v1") is None


# ── Codex code mode: `exec` runs JavaScript; the commands are inside it ──
# Scripts below are the ones gpt-6.1-sol wrote in live Codex 0.169.0 sessions.

JS_AWAIT = ('text(await tools.exec_command({cmd:"Get-Content src/calc.py"}));\n'
            'text(await tools.exec_command({cmd:"rg \'def \' src/calc.py"}));\n')
JS_ALLSETTLED = ('const results = await Promise.allSettled([\n'
                 'tools.exec_command({cmd:"Get-Content src/chunking.py; Get-Content src/calc.py",'
                 'max_output_tokens:12000}),\n'
                 'tools.exec_command({cmd:"rg \'def \' src",max_output_tokens:6000})\n'
                 ']);for(let i=0;i<results.length;i++)text({i,...results[i]});\n')
JS_ONE = ('const r = await tools.exec_command({cmd:"Get-Content src/chunking.py; Get-Content src/calc.py; '
          'rg -n \'def \' src","max_output_tokens":12000});text(r.output);\n')


def test_code_mode_calls_read_each_command_out_of_the_script():
    assert code_mode_calls(JS_AWAIT) == [("exec_command", "Get-Content src/calc.py"),
                                        ("exec_command", "rg 'def ' src/calc.py")]
    assert code_mode_kinds(JS_AWAIT) == ["file_read", "search_result"]
    assert code_mode_kinds(JS_ALLSETTLED) == ["file_read", "search_result"]
    # one command that reads two files and greps: the files are most of it
    assert code_mode_kinds(JS_ONE) == ["file_read"]


@pytest.mark.parametrize("script, want", [
    ("for (const f of files) text(await tools.exec_command({cmd: `cat ${f}`}));", ["file_read"]),
    ("text(await tools.exec_command({'cmd': 'python -c \"print(1)\"'}));", ["agent_generated_result"]),
    ('text(await tools.exec_command({cmd:["bash","-lc","pytest -q"]}));', ["test_output"]),
    ('text(await tools.exec_command({cmd: command}));', [None]),             # not a literal
    ('text(await tools.write_stdin({session_id: 3, chars: ""}));', ["command_output"]),
    ('text(await tools.view_image({path: "a.png"})); text(await tools.exec_command({cmd:"git diff"}));',
     ["vcs_output"]),                                                         # only exec results count
    ('text(await tools.exec_command({cmd:"cat > probe.py <<\'EOF\'\\nprint(1)\\nEOF"}));\n'
     'text(await tools.exec_command({cmd:"python probe.py"}));', ["command_output", "agent_generated_result"]),
])
def test_code_mode_kinds(script, want):
    assert code_mode_kinds(script) == want


def test_code_mode_is_not_read_as_a_shell_command():
    # the whole script used to be parsed as one shell command (-> command_output / file_read)
    assert label("x", "exec", {"command": JS_AWAIT}) == ("command_output", "content")
    assert label("x", "exec", {"command": "ls -la"}) == ("directory_listing", "tool")  # a plain `exec` shell


def test_code_mode_output_in_a_conversation():
    conv = [{"type": "custom_tool_call", "call_id": "c", "name": "exec", "input": JS_AWAIT},
            {"type": "custom_tool_call_output", "call_id": "c", "output": [{"type": "input_text", "text": "x"}]}]
    r = label_conversation(conv)[0]
    assert r["how"] == "code_mode" and r["call_kinds"] == ["file_read", "search_result"]
    assert code_mode_kinds_by_position(conv) == {(1, None): ["file_read", "search_result"]}
    assert kinds_by_position(conv) == {}           # the proxy lines code mode up per result instead


def test_content_fallback_finds_results_nested_by_the_script():
    nested = json.dumps({"i": 0, "status": "fulfilled", "value": {
        "chunk_id": "a", "wall_time_seconds": 0.1, "exit_code": 0,
        "output": "".join(f"{n}\tvalue_{n} = compute({n})\n" for n in range(1, 40))}})
    assert kind_from_content(nested) == "file_read"
