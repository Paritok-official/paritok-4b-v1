"""Kind of a tool output, decided from WHAT PRODUCED it: the tool and, for a shell, the command.

The kind picks the compressor's system prompt and rules, so training labels and the proxy must
agree on it. Both use this module: the proxy labels every tool output from the tool call it
already sees in the request (Anthropic tool_use, OpenAI tool_calls, Responses function_call), and
only falls back to the text itself when there is no call to go by.

    label(out, tool="Bash", tool_input={"command": "pytest -x"})   # -> ("test_output", "tool")
    kinds_by_position(messages)    # {(message index, block index | None): kind}, never raises

Tool-output kinds:
    file_read               a file's content: Read / view / open, cat, sed -n, head, tail of a file
    search_result           grep / rg / Grep / Glob / find / git grep / git ls-files output
    directory_listing       ls / tree / LS / list_dir / a directory view
    test_output             a test run, a linter or a type checker
    agent_generated_result  output of code the agent wrote itself: python -c / node -e, a heredoc piped
                            to an interpreter, a script it created earlier in the conversation, an
                            IPython cell -- decidable only from the command
    vcs_output              git diff / log / show / status / blame
    command_output          any other shell command, or shell output whose command is unknown
    tool_result             any other tool (edit / write confirmations, web fetch, MCP tools, ...)
Agent-message kinds (history compression), as before:
    assistant_thinking, bash_command, file_operation, meta_action

The deployed paritok-4b-v1 was trained on the old kind names only; `wire_kind` maps the kinds above
onto them (compression.kind_scheme: v1, the default) or sends them as is (kind_scheme: v2).
"""
from __future__ import annotations

import json
import re

TOOL_OUTPUT_KINDS = ("file_read", "search_result", "directory_listing", "test_output",
                     "agent_generated_result", "vcs_output", "command_output", "tool_result")
AGENT_MESSAGE_KINDS = ("assistant_thinking", "bash_command", "file_operation", "meta_action")


# ── 1. tool names ────────────────────────────────────────────────────────────────────────
# Claude Code, Codex, Gemini CLI, Cursor-style, OpenHands and SWE-agent names, lower-cased.

READ_TOOLS = {"read", "notebookread", "read_file", "read_many_files", "view", "view_file", "open_file",
              "readfile", "open", "goto", "scroll_up", "scroll_down"}
SEARCH_TOOLS = {"grep", "glob", "search", "search_files", "find_files", "codebase_search", "file_search",
                "grep_search", "ripgrep", "grep_files", "search_file_content", "find_file", "search_dir",
                "search_file"}
LIST_TOOLS = {"ls", "list_dir", "list_directory", "list_files"}
SHELL_TOOLS = {"bash", "powershell", "shell", "exec", "exec_command", "execute_bash", "run_command",
               "terminal", "run_terminal_cmd", "run_shell_command", "local_shell", "container.exec",
               "write_stdin", "bashoutput", "shell_command"}
CODE_CELL_TOOLS = {"execute_ipython_cell", "python", "run_python", "code_interpreter", "jupyter"}
WRITE_TOOLS = {"write", "create_file", "write_file", "create"}
# Codex code mode: `exec` takes JavaScript that calls other tools as `tools.<name>(...)`
# (Codex 0.169 + gpt-6.x by default). Not a shell: the commands are inside the script.
CODE_MODE_TOOLS = {"exec"}
CODE_MODE_EXEC_TOOLS = ("exec_command", "write_stdin")   # calls whose results are exec result objects


def _norm_tool(tool: str | None) -> str:
    t = (tool or "").strip().lower()
    if t.startswith("mcp__"):              # mcp__server__tool -> tool
        t = t.rsplit("__", 1)[-1]
    if t != "container.exec":
        t = t.rsplit(".", 1)[-1]           # functions.exec_command -> exec_command
    return t


# ── 2. shell commands ────────────────────────────────────────────────────────────────────

_INTERP = r"(?:python[\d.]*|py|ipython|node|deno|bun|ruby|perl|php|Rscript|ts-node|tsx)(?:\.exe)?"
_SHELL_NAMES = ("bash", "sh", "zsh", "dash", "pwsh", "powershell", "cmd")
_SCRIPT_EXT = (".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".sh", ".ps1", ".rb", ".pl", ".php", ".r",
               ".go", ".java", ".cs", ".c", ".cc", ".cpp", ".rs", ".kt")
# a heredoc body is data, not command words: `cat > t.py <<'EOF' ... EOF` keeps only its first line
_HEREDOC = re.compile(r"<<-?\s*(['\"]?)(\w+)\1([^\n]*)\n.*?\n[ \t]*\2[ \t]*(?=\n|$)", re.S)
_PREFIX = re.compile(r"^(?:\w+=\S*\s+|sudo\s+|time\s+|timeout\s+\S+\s+|nice\s+|env\s+|command\s+|"
                     r"stdbuf\s+\S+\s+|(?:uv|poetry|pipenv|pdm|hatch|rye)\s+run\s+|npx\s+(?:-\S+\s+)*|pnpx\s+|"
                     r"bunx\s+|(?:npm|pnpm)\s+(?:exec|dlx)\s+|bundle\s+exec\s+|&\s+|(?:do|then|else|elif)\s+|\{\s+)+", re.I)
_QUIET = re.compile(r"^(?:for|while|until|if|done|fi|esac|case|\}|cd|pushd|popd|export|set|unset|source|\.|echo|printf|true|false|clear|mkdir|rm|"
                    r"cp|mv|touch|chmod|chown|ln|sleep|wait|Set-Location|Write-Host|Write-Output)\b", re.I)
_INLINE = re.compile(rf"^{_INTERP}(?:\s+-[\w-]+)*\s+(?:-c|-e|-p|--eval|--print|-Command)\b"
                     rf"|^{_INTERP}(?:\s+-[\w-]+)*\s*(?:-\s*)?(?:<<|$)"
                     rf"|^(?:deno\s+eval|php\s+-r)\b", re.I)
_PIPE_TO_INTERP = re.compile(rf"\|\s*{_INTERP}(?:\s+-[\w-]+)*\s*(?:-\s*)?$", re.I)
_COMPILE_RUN = re.compile(r"(?:^|[\s;&|(])(?:g\+\+|gcc|clang\+\+?|javac|rustc)\s.*(?:&&|;)\s*(?:\./|java\s)", re.I)
_TEST_RUNNER = re.compile(
    r"^(?:pytest|py\.test|tox|nox|nosetests|jest|vitest|mocha|ava|karma|phpunit|rspec|ctest|"
    r"playwright\s+test|cypress\s+run|go\s+(?:test|vet)\b|cargo\s+(?:test|clippy|check)\b|"
    r"(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?(?:test|lint|typecheck|type-check|check)\b|"
    r"(?:mvn\w*|\./mvnw)\s+(?:\S+\s+)*?(?:test|verify)\b|(?:gradle\w*|\./gradlew)\s+(?:\S+\s+)*?(?:test|check)\b|"
    r"dotnet\s+test|make\s+(?:test|check|lint)\b|rake\s+test|"
    r"flake8|mypy|ruff|pylint|pyright|pyflakes|pycodestyle|black\s+--check|isort\s+--check|eslint|tsc|"
    r"prettier\s+--check|biome|golangci-lint|shellcheck|clang-tidy|"
    rf"{_INTERP}(?:\s+-\S+)*\s+-m\s+(?:pytest|unittest|nose2?|tox|nox|doctest|flake8|mypy|pyflakes|"
    r"pycodestyle|pylint|black|isort|ruff|pyright)\b)", re.I)
_TEST_SCRIPT = re.compile(rf"^{_INTERP}(?:\s+-\S+)*\s+\S*(?:runtests?\.py|manage\.py\s+test|setup\.py\s+test|"
                          r"(?:^|/)tests?/\S*\.py)\b", re.I)
_VCS = re.compile(r"^git(?:\s+(?:-C\s+\S+|-c\s+\S+|--no-pager|--[\w-]+=\S+))*\s+"
                  r"(?:diff|log|show|status|blame|shortlog|reflog)\b|^gh\s+pr\s+diff\b", re.I)
_SEARCH = re.compile(r"^(?:grep|egrep|fgrep|rg|ag|ack|find|fd|fdfind|locate|findstr|Select-String|sls)\b|"
                     r"^git(?:\s+-C\s+\S+)*\s+(?:grep|ls-files)\b", re.I)
_DUMP = re.compile(r"^(?:cat|sed\s+(?:-\S+\s+)*-n|head|tail|nl|bat|batcat|less|more|type|"
                   r"Get-Content|gc|awk\s+(?:-\S+\s+)*'NR)\b", re.I)
_LIST = re.compile(r"^(?:ls|ll|tree|dir|du|Get-ChildItem|gci)\b", re.I)
_CREATES = re.compile(r"(?:(?<![\d&])>>?|\btee\s+(?:-a\s+)?|Set-Content\s+(?:-Path\s+)?|"
                      r"Out-File\s+(?:-FilePath\s+)?)\s*['\"]?([^\s'\"<>|;&]+)", re.I)
_ADD_FILE = re.compile(r"^\*\*\* Add File: (\S+)", re.M)   # apply_patch (Codex)
_QUOTED = re.compile(r"'[^']*'|\"(?:[^\"\\]|\\.)*\"")
_REDIRECT_OUT = re.compile(r"(?<![\d&])>>?\s*(?!&)\S")


_VIEW_EXT = re.compile(r"\.[A-Za-z][\w-]{0,9}$|^(makefile|dockerfile|license|readme|changelog|authors|notice)$", re.I)


def _base(path: str) -> str:
    return path.strip().strip("'\"").replace("\\", "/").rsplit("/", 1)[-1].lower()


def unwrap_shell(command) -> str:
    """The script inside `bash -lc "..."`, `powershell -Command "..."` or a Codex ["bash", "-lc", script]."""
    if isinstance(command, (list, tuple)):
        parts = [str(x) for x in command]
        if len(parts) >= 3 and _base(parts[0]).removesuffix(".exe") in _SHELL_NAMES:
            return unwrap_shell(parts[-1])
        return " ".join(parts)
    cmd = str(command or "")
    m = re.match(r"^\s*(?:\S*[/\\])?(?:bash|sh|zsh|dash|pwsh|powershell|cmd)(?:\.exe)?"
                 r"(?:\s+-\w+(?:\s+(?!-)[\w.]+)?)*?\s+(?:-l?c|-Command|/c)\s+(.*)$", cmd, re.S | re.I)
    if m:
        inner = m.group(1).strip()
        if len(inner) >= 2 and inner[0] == inner[-1] and inner[0] in "'\"":
            inner = inner[1:-1]
        return inner
    return cmd


def written_scripts(command: str) -> set[str]:
    """Script files a shell command itself creates (`cat > probe.py <<EOF`, tee, Set-Content, apply_patch)."""
    cmd = unwrap_shell(command)
    names = {_base(m.group(1)) for m in _CREATES.finditer(_HEREDOC.sub(lambda m: "<<" + m.group(2) + m.group(3), cmd))}
    names |= {_base(p) for p in _ADD_FILE.findall(cmd)}
    return {n for n in names if n.endswith(_SCRIPT_EXT)}


def _runs_agent_file(producer: str, agent_files: set[str]) -> bool:
    words = re.findall(r"[^\s'\"<>|;&()=]+", producer)
    if not words:
        return False
    hits = [i for i, w in enumerate(words) if _base(w) in agent_files]
    if not hits:
        return False
    first = _base(words[0]).removesuffix(".exe")
    return hits[0] == 0 or bool(re.match(_INTERP + "$", first, re.I)) or first in _SHELL_NAMES


# a compound command prints several things; the most specific source wins, but a trailing `ls` does not
# turn a build or a curl into a directory listing. A file read beats a search in the same command
# (`cat a.py; rg foo`): the file is most of the output, and file content must keep its line numbers.
_PRIORITY = ("agent_generated_result", "test_output", "vcs_output", "file_read", "search_result",
             "command_output", "directory_listing")


def _producer_kind(producer: str, agent_files: set[str]) -> str | None:
    """Kind of what one command (the first stage of a pipeline) prints; None if it prints nothing useful."""
    if _INLINE.search(producer) or _COMPILE_RUN.search(producer):
        return "agent_generated_result"
    if _QUIET.match(producer):
        return None
    unquoted = _QUOTED.sub("''", producer)            # a `>` inside a grep pattern is not a redirect
    if _REDIRECT_OUT.search(re.sub(r"\d?>&\d|2>>?\s*\S+|&>\s*/dev/null|>\s*/dev/null|>\s*\$null", "", unquoted)):
        return None                                   # stdout goes to a file
    if _TEST_RUNNER.match(producer):
        return "test_output"
    if agent_files and _runs_agent_file(producer, agent_files):
        return "agent_generated_result"
    if _TEST_SCRIPT.match(producer):
        return "test_output"
    if _VCS.match(producer):
        return "vcs_output"
    if _SEARCH.match(producer):
        return "search_result"
    if _DUMP.match(producer):
        return "file_read"
    if _LIST.match(producer):
        return "directory_listing"
    return "command_output"


def kind_from_command(command, agent_files: set[str] | None = None) -> str:
    """Kind of a shell command's output. `agent_files`: script names the agent wrote earlier in the
    conversation (label_conversation tracks them); scripts the command itself writes are added."""
    cmd = unwrap_shell(command)
    files = set(agent_files or ()) | written_scripts(cmd)
    flat = _HEREDOC.sub(lambda m: "<<" + m.group(2) + m.group(3), cmd)
    if _COMPILE_RUN.search(flat):                     # g++ t.cpp -o t && ./t
        return "agent_generated_result"
    kinds = set()
    for pipeline in re.split(r"&&|\|\||;|\n", flat):
        pipeline = pipeline.strip()
        if not pipeline:
            continue
        if _PIPE_TO_INTERP.search(pipeline):          # echo ... | python  /  cat <<EOF | node
            kinds.add("agent_generated_result")
            continue
        producer = _PREFIX.sub("", pipeline.split("|", 1)[0].strip())
        k = _producer_kind(producer, files)
        if k:
            kinds.add(k)
    for k in _PRIORITY:
        if k in kinds:
            return k
    return "command_output"


# ── 2b. Codex code mode ──────────────────────────────────────────────────────────────────

_TOOLS_CALL = re.compile(r"\btools\.([A-Za-z_]\w*)\s*\(")
_JS_CMD_KEY = re.compile(r"""(?:^|[\s{,])["']?(?:cmd|command)["']?\s*:\s*""")
_JS_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "v": "\v", "0": "\0"}


def is_code_mode_script(tool: str | None, script) -> bool:
    return (_norm_tool(tool) in CODE_MODE_TOOLS and isinstance(script, str)
            and _TOOLS_CALL.search(script) is not None)


def _js_inner(src: str, start: int) -> str:
    """Text from `start` up to the bracket that closes the one just before it (strings skipped)."""
    depth, i, quote = 1, start, None
    while i < len(src):
        c = src[i]
        if quote:
            if c == "\\":
                i += 2
                continue
            if c == quote:
                quote = None
        elif c in "\"'`":
            quote = c
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                return src[start:i]
        i += 1
    return src[start:]


def _js_string(src: str) -> str | None:
    """The JS string literal ('…', "…" or `…`) at the start of `src`, unescaped."""
    if not src or src[0] not in "\"'`":
        return None
    quote, out, j = src[0], [], 1
    while j < len(src):
        c = src[j]
        if c == "\\" and j + 1 < len(src):
            n = src[j + 1]
            if n == "u" and re.match(r"[0-9a-fA-F]{4}", src[j + 2:j + 6]):
                out.append(chr(int(src[j + 2:j + 6], 16)))
                j += 6
                continue
            out.append(_JS_ESCAPES.get(n, n))
            j += 2
            continue
        if c == quote:
            return "".join(out)
        out.append(c)
        j += 1
    return None


def code_mode_calls(script: str) -> list[tuple[str, str]]:
    """(tool name, command) for every `tools.<name>(…)` call in a code-mode script, in source
    order. The command is the literal `cmd` / `command` argument ('' when it isn't a literal,
    e.g. a variable); a template literal keeps its `${…}` placeholders."""
    calls = []
    for m in _TOOLS_CALL.finditer(script or ""):
        args = _js_inner(script, m.end())
        cmd = ""
        key = _JS_CMD_KEY.search(args)
        if key:
            value = args[key.end():].lstrip()
            if value[:1] == "[":
                parts = [_js_string(p.group(0)) for p in re.finditer(
                    r"\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'|`(?:[^`\\]|\\.)*`", _js_inner(value, 1))]
                cmd = unwrap_shell([p for p in parts if p is not None])
            else:
                cmd = _js_string(value) or ""
        calls.append((m.group(1), cmd))
    return calls


def code_mode_kinds(script: str, agent_files: set[str] | None = None) -> list[str | None]:
    """Kind of each exec_command / write_stdin result a code-mode script asks for, in source
    order (None when the command isn't a literal). Other tool calls are not counted: their
    results are not exec result objects."""
    files = set(agent_files or ())
    kinds: list[str | None] = []
    for name, cmd in code_mode_calls(script):
        if name not in CODE_MODE_EXEC_TOOLS:
            continue
        if cmd:
            kinds.append(kind_from_command(cmd, files))
            files |= written_scripts(cmd)
        else:
            kinds.append("command_output" if name == "write_stdin" else None)
    return kinds


def _command_of(tool_input) -> str:
    inp = tool_input if isinstance(tool_input, dict) else {}
    c = inp.get("command") or inp.get("cmd") or inp.get("script") or ""
    return unwrap_shell(c) if c else ""


def kind_from_tool(tool: str | None, tool_input: dict | None = None,
                   agent_files: set[str] | None = None, content: str | None = None) -> str | None:
    """Kind from the producing tool; None when there is no tool or it is a shell without its command.
    `content` is only used to tell a viewed directory from a viewed file (the editor's `view` takes both)."""
    if not tool:
        return None
    t = _norm_tool(tool)
    inp = tool_input if isinstance(tool_input, dict) else {}
    if t == "str_replace_editor":                     # OpenHands / SWE-agent editor
        if str(inp.get("command") or "") == "view":
            head = (content or "").lstrip()[:200]
            if head.startswith("Here's the files and directories"):
                return "directory_listing"
            if head.startswith("Here's the result of running `cat -n`"):
                return "file_read"
            # no OpenHands header: a directory has no letter extension ("repo__4.3" is a directory)
            path = str(inp.get("path") or "")
            return "directory_listing" if path.endswith("/") or not _VIEW_EXT.search(_base(path)) else "file_read"
        return "tool_result"                          # create / str_replace / insert / undo_edit
    if t in READ_TOOLS:
        return "file_read"
    if t in SEARCH_TOOLS:
        return "search_result"
    if t in LIST_TOOLS:
        return "directory_listing"
    if t in CODE_CELL_TOOLS:
        return "agent_generated_result"
    if is_code_mode_script(t, inp.get("command")):
        return None                                   # several commands inside: see code_mode_kinds
    if t in SHELL_TOOLS:
        cmd = _command_of(inp)
        return kind_from_command(cmd, agent_files) if cmd else None
    return "tool_result"


# ── 3. fallback: the content alone ───────────────────────────────────────────────────────

NUM = re.compile(r"^(\s*)(\d+)(\t|→)")
OH_VIEW = re.compile(r"^Here's the result of running `cat -n` on (\S+?):\s*\n")
EDIT_CONFIRM = re.compile(r"^(The file \S+ has been (created|edited|updated|saved)|File created successfully|"
                          r"Successfully (wrote|edited|replaced|created)|Applied \d+ edits?)", re.I)
OH_FOOT = re.compile(r"^\[(The command completed with exit code|Current working directory:|Python interpreter:|"
                     r"Command finished|The command has no new output|Below is the output of the previous)")
CODEX_HEAD = ("Script completed", "Exit code:", "Wall time", "Total output lines:", "Output:", "Chunk ID:", "Process exited",
              "Warning: truncated output")
TEST_RUN = re.compile(
    r"=+ test session starts =+|^collected \d+ items?|^=+ .*\b\d+ (passed|failed|error)|"
    r"^(FAILED|PASSED|ERROR) \S+::|^Ran \d+ tests? in [\d.]+s|^(OK|FAILED) \((failures|errors|skipped)|"
    r"^Tests:\s+\d+|^Test Suites:|^test result: (ok|FAILED)|^--- (FAIL|PASS):|^ok\s+\S+\s+[\d.]+s$|"
    r"^\S+\.py::\S+ (PASSED|FAILED|ERROR|SKIPPED)|^Creating test database|^System check identified|"
    r"^\s*(✓|✗|×) .+\(\d+ ?ms\)|^ (PASS|FAIL) \S+", re.M)
LINT = re.compile(r"^\S+\.\w+:\d+:\d+: [A-Z]+\d{2,4}\b|\berror TS\d{3,5}\b|^Found \d+ errors? in \d+ files?|"
                  r"^\S+\.py:\d+: (error|note|warning):|^All checks passed!|would reformat|^\d+ problems? \(|"
                  r"^Success: no issues found", re.M)
DIFF = re.compile(r"^diff --git |^@@ -\d+(,\d+)? \+\d+(,\d+)? @@|^\+\+\+ [ab]/|^--- a/|^index [0-9a-f]{7,}\.\.", re.M)
VCS = re.compile(r"^commit [0-9a-f]{7,40}$|^On branch \S|^Changes (not staged|to be committed)|^Your branch is|"
                 r"^Author: .+<", re.M)
GREP_PL = re.compile(r"^[\w.~\-/\\@+]+\.\w{1,8}[:-]\d+[:-]")
GREP_P = re.compile(r"^[\w.~\-/\\@+]+\.\w{1,8}:\S")
GREP_L = re.compile(r"^\d+[:-]")
PATHLINE = re.compile(r"^\.?/?[\w.~\-@+]+(/[\w.~\-@+]+)*/?$")
BARE_NAME = re.compile(r"^[\w.\-@+]+/?$")                    # plain `ls`: one name per line
REPR = re.compile(r"^\s*(\{['\"]\w|\[\{|\(\d|<\w[\w.]* object at 0x)")   # printed Python / JSON records
LS = re.compile(r"^[dl\-][rwxsStT\-]{9}[.+@]?\s|^total \d+$|[├└]──|^d----|^-a---", re.M)
PY = re.compile(r"^\s*(def |class |import |from [\w.]+ import|elif |except\b|@\w)|\bself\.|\bNone\b|\bTrue\b", re.M)
JS = re.compile(r"\b(const|let|var) \w+\s*=|=>|\bfunction\s*\w*\s*\(|^\s*import .* from |^\s*export |"
                r"require\(|^\s*interface \w+|: (string|number|boolean)\b|console\.log|className=", re.M)
OTHER_CODE = re.compile(r"\b(public|private|protected) (static )?(class|void|int|String|async)\b|#include\b|"
                        r"using System|^namespace \w+|std::|^func |^package \w+|:= |\bfn \w+|let mut |^\s*impl\b|"
                        r"^\s*(SELECT|INSERT|CREATE TABLE)\b|\bend$|^#!/", re.M)
MD = re.compile(r"^#{1,6} \S|^\s*[-*+] \S|\[[^\]]+\]\([^)]+\)|^```|^\.\. \w+::|^={3,}$|^-{3,}$|^\|.*\|$", re.M)
MD_FEATURES = [re.compile(p, re.M) for p in (r"^#{1,6} \S", r"^\s*[-*+] \S", r"\[[^\]]+\]\([^)]+\)", r"^```",
                                              r"^\|.*\|$", r"^\.\. \w+::|^={3,}$")]
CONFIG = re.compile(r"^\s*[\w.\-\"']+\s*[:=]\s*\S|^\[[\w.\- \"]+\]\s*$|^\s*<\??[\w!/]", re.M)
HISTORY = re.compile(r"^\[(user|assistant)\]: ", re.M)
_TOOLCALL_KIND = {"think": "assistant_thinking", "task_tracker": "meta_action", "finish": "meta_action",
                  "str_replace_editor": "file_operation", "execute_bash": "bash_command", "bash": "bash_command"}


_EXEC_RESULT_KEYS = frozenset({"chunk_id", "wall_time_seconds", "exit_code", "session_id", "original_token_count"})


def _codex_exec_outputs(v, depth: int = 0) -> list[str]:
    """The `output` of every Codex exec_command result object inside parsed JSON, in order."""
    if depth > 8:
        return []
    if isinstance(v, dict):
        keys = set(v) - {"output"}
        if isinstance(v.get("output"), str) and keys and keys <= _EXEC_RESULT_KEYS:
            return [v["output"]]
        return [o for x in v.values() for o in _codex_exec_outputs(x, depth + 1)]
    if isinstance(v, list):
        return [o for x in v for o in _codex_exec_outputs(x, depth + 1)]
    return []


def unwrap_envelope(text: str) -> tuple[str, str] | None:
    """A tool result delivered as JSON: (inner text, envelope), else None.
    Codex exec_command: {"chunk_id", "exit_code", "output", "wall_time_seconds", ...};
    MCP: {"content": [{"type": "text", "text": ...}], "isError": ...}."""
    s = (text or "").strip()
    if not s.startswith(("{", "[")):
        return None
    try:
        j = json.loads(s)
    except ValueError:
        return None
    nested = _codex_exec_outputs(j)       # code mode: results alone, in arrays, or nested by the script
    if nested:
        return "\n".join(nested), "codex_exec"
    if not isinstance(j, dict):
        return None
    for key in ("output", "content", "result", "value", "text", "stdout"):
        v = j.get(key)
        if isinstance(v, str) and v.strip():
            return v, ("codex_exec" if key == "output" and ("chunk_id" in j or "exit_code" in j) else f"json_{key}")
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            texts = [x.get("text") for x in v if isinstance(x.get("text"), str)]
            if texts:
                return "\n".join(texts), f"json_{key}_list"
    return None


def _density(rx, text, n):
    return len(rx.findall(text)) / max(n, 1)


def strip_frame(text: str) -> str:
    """Drop OpenHands shell footers and the Codex `Exit code / Wall time / Output:` header."""
    lines = [ln for ln in (text or "").lstrip().split("\n") if not OH_FOOT.match(ln)]
    i = 0
    while i < len(lines) and (not lines[i].strip() or lines[i].startswith(CODEX_HEAD)):
        i += 1
    if i and any(x.startswith(CODEX_HEAD) for x in lines[:i]):
        lines = lines[i:]
    return "\n".join(lines)


def kind_from_content(text: str) -> str:
    """Fallback kind from the text alone. Cannot tell agent_generated_result from command_output
    (that needs the command): a program's output with no other marks is command_output."""
    text = text or ""
    env = unwrap_envelope(text)
    if env is not None:
        return kind_from_content(env[0])
    t = text.lstrip()
    if EDIT_CONFIRM.match(t):
        return "tool_result"
    if OH_VIEW.match(t):
        return "file_read"
    if t.startswith("Here's the files and directories"):
        return "directory_listing"
    body = strip_frame(t)
    lines = [ln for ln in body.split("\n") if ln.strip()]
    n = len(lines)
    head = lines[:60]
    numbered = sum(1 for ln in head if NUM.match(ln))
    if (n >= 3 and numbered >= 0.6 * len(head)) or (n and numbered == len(head)):   # a Read can be 1-2 long lines
        return "file_read"            # before the history checks: a file may itself contain "[tool_calls]:"
    if re.search(r'(^|\n)\[tool_calls\]: \[', text):
        name = re.search(r'\[tool_calls\]: \[\{"name": "(\w+)"', text)
        return _TOOLCALL_KIND.get(name.group(1) if name else "", "assistant_thinking")
    if len(HISTORY.findall(text)) >= 2:
        return "assistant_thinking"
    if len(LS.findall(body)) >= max(3, 0.6 * n):   # a doc that draws a tree is not a listing
        return "directory_listing"
    if TEST_RUN.search(body) or LINT.search(body):
        return "test_output"
    if len(DIFF.findall(body)) >= 2 or VCS.search(body):
        return "vcs_output"
    if n and (sum(1 for ln in lines if GREP_PL.match(ln)) >= 0.4 * n
              or sum(1 for ln in lines if GREP_L.match(ln)) >= 0.6 * n
              or sum(1 for ln in lines if GREP_P.match(ln)) >= 0.5 * n):
        return "search_result"
    if re.match(r"\s*(Found \d+ files?|No (matches|files) found)", body):
        return "search_result"
    if n >= 3 and sum(1 for ln in lines if PATHLINE.match(ln.strip()) and "/" in ln) >= 0.7 * n:
        return "search_result"
    if n >= 3 and sum(1 for ln in lines if BARE_NAME.match(ln.strip())) >= 0.8 * n:
        return "directory_listing"
    # Unnumbered file content (cat / sed / a config dump). Thresholds measured on training file views with
    # numbers stripped vs program outputs: 88.7% of reads caught, 12.2% of program outputs mistaken --
    # content alone cannot do better; the command can.
    cn = max(n, 1)
    code_hits = max(len(PY.findall(body)), len(JS.findall(body)), len(OTHER_CODE.findall(body)))
    if sum(1 for ln in lines if REPR.match(ln)) < 0.3 * cn and (
            (code_hits >= 2 and code_hits / cn >= 0.08)
            or _density(MD, body, cn) >= 0.15
            or (len(CONFIG.findall(body)) >= 3 and _density(CONFIG, body, cn) >= 0.4)   # not "Usage: …\nError: …"
            or sum(bool(rx.search(body)) for rx in MD_FEATURES) >= 3):     # a long doc diluted by a tree / code
        return "file_read"
    return "command_output"


# ── 4. entry points ──────────────────────────────────────────────────────────────────────

def label(content: str, tool: str | None = None, tool_input: dict | None = None,
          agent_files: set[str] | None = None) -> tuple[str, str]:
    """(kind, how) of one tool output; how = "tool" when decided from the tool / command."""
    k = kind_from_tool(tool, tool_input, agent_files, content)
    if k is not None:
        return k, "tool"
    return kind_from_content(content), "content"


def _as_args(a) -> dict:
    if isinstance(a, dict):
        return a
    if isinstance(a, str):
        try:
            j = json.loads(a)
        except ValueError:
            return {"command": a}      # free-form tools (Codex apply_patch) pass raw text
        return j if isinstance(j, dict) else {"command": a}
    return {}


def _text_of(c) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, dict):
        return c.get("text") if isinstance(c.get("text"), str) else _text_of(c.get("content") or c.get("output"))
    if isinstance(c, list):
        return "\n".join(_text_of(x) for x in c)
    return ""


def _remember_written(tool: str | None, inp: dict, files: set[str]) -> None:
    t = _norm_tool(tool)
    path = str(inp.get("file_path") or inp.get("path") or inp.get("filename") or "")
    if t in WRITE_TOOLS or (t == "str_replace_editor" and inp.get("command") == "create"):
        if _base(path).endswith(_SCRIPT_EXT):
            files.add(_base(path))
    if t in SHELL_TOOLS or t == "apply_patch":
        files |= written_scripts(_command_of(inp) or " ".join(v for v in inp.values() if isinstance(v, str)))


def label_conversation(messages: list[dict]) -> list[dict]:
    """Kind of every tool output in a conversation, from the call that produced it. Understands
      OpenAI chat (OpenHands / SWE-rebench trajectories, chat-completions agents):
          assistant {"tool_calls": [{"id", "function": {"name", "arguments"}}]} -> {"role": "tool", "tool_call_id", "content"}
      Anthropic Messages (Claude Code):
          assistant [{"type": "tool_use", "id", "name", "input"}] -> user [{"type": "tool_result", "tool_use_id", "content"}]
      OpenAI Responses (Codex):
          {"type": "function_call" | "custom_tool_call", "call_id", "name", "arguments" | "input"} -> {"type": "..._output", "call_id", "output"}
    Scripts the agent writes along the way are tracked, so running one later is agent_generated_result.
    Returns one dict per tool output: {"index", "block", "tool", "kind", "how"} (block: position inside
    an Anthropic user message, else None). A Codex code-mode output (`exec` running JavaScript)
    has how="code_mode" and "call_kinds": the kind of each exec_command the script ran, in order;
    its "kind" is that kind when they all agree, else sniffed from the content."""
    calls: dict = {}
    unanswered: list = []           # chat-format calls in order, for tool messages without an id
    files: set[str] = set()
    out = []

    def emit(i, block, call, content):
        name, args = call if call else (None, {})
        script = args.get("command") if isinstance(args, dict) else None
        if is_code_mode_script(name, script):
            call_kinds = code_mode_kinds(script, files)
            agreed = set(call_kinds)
            k = call_kinds[0] if len(agreed) == 1 and call_kinds[0] else kind_from_content(_text_of(content))
            out.append({"index": i, "block": block, "tool": name, "kind": k, "how": "code_mode",
                        "call_kinds": call_kinds})
            for _, cmd in code_mode_calls(script):
                files.update(written_scripts(cmd))
            return
        k, how = label(_text_of(content), name, args, files)
        out.append({"index": i, "block": block, "tool": name, "kind": k, "how": how})
        if name:
            _remember_written(name, args, files)

    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        typ, role = m.get("type"), m.get("role")
        if typ in ("function_call", "custom_tool_call", "local_shell_call"):
            args = m.get("arguments", m.get("input", m.get("action")))
            calls[m.get("call_id") or m.get("id")] = (m.get("name") or "local_shell", _as_args(args))
        elif typ in ("function_call_output", "custom_tool_call_output", "local_shell_call_output"):
            emit(i, None, calls.get(m.get("call_id")), m.get("output"))
        elif role == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                calls[tc.get("id")] = (fn.get("name"), _as_args(fn.get("arguments")))
                unanswered.append(tc.get("id"))
            if isinstance(m.get("content"), list):
                for b in m["content"]:
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        calls[b.get("id")] = (b.get("name"), _as_args(b.get("input")))
        elif role == "tool":
            cid = m.get("tool_call_id")
            if cid is None and unanswered:
                cid = unanswered[0]
            if cid in unanswered:
                unanswered.remove(cid)
            emit(i, None, calls.get(cid), m.get("content"))
        elif role == "user" and isinstance(m.get("content"), list):
            for j, b in enumerate(m["content"]):
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    emit(i, j, calls.get(b.get("tool_use_id")), b.get("content"))
    return out


# ── 5. what to send to the model ─────────────────────────────────────────────────────────

# paritok-4b-v1 was trained on the old, text-sniffed kind names (file_read / tool_result / log_output /
# directory_listing / ...). Each kind maps to the old name v1 saw most for that content, measured on the
# SWE-rebench trajectories its training set was cut from (call-based kind vs the stored training label).
V1_KIND = {
    "file_read": "file_read",                  # 95%
    "search_result": "tool_result",            # 55% (file_read 28%)
    "directory_listing": "directory_listing",  # 94%
    "test_output": "log_output",               # 38% (tool_result 35%)
    "agent_generated_result": "log_output",    # 64%; also the v1 kind that keeps the most result lines
    "vcs_output": "tool_result",               # 55%
    "command_output": "tool_result",           # 38% (file_read 35%)
    "tool_result": "file_read",                # 86%: mostly edit confirmations carrying a cat -n snippet
}
KIND_SCHEMES = ("v1", "v2")


def wire_kind(kind: str | None, scheme: str = "v1") -> str | None:
    """The kind to send to the compression model: as is for "v2", mapped onto the names
    paritok-4b-v1 was trained on for "v1" (names v1 already knows pass through)."""
    if not kind or scheme == "v2":
        return kind
    return V1_KIND.get(kind, kind)


def code_mode_kinds_by_position(messages) -> dict:
    """{(message index, block index or None): [kind of each exec_command the script ran]} for
    every Codex code-mode output. The proxy lines these up with the results printed in the
    output. Never raises."""
    try:
        return {(r["index"], r["block"]): r["call_kinds"]
                for r in label_conversation(messages or []) if r["how"] == "code_mode"}
    except Exception:
        return {}


def kinds_by_position(messages) -> dict:
    """{(message index, block index or None): kind} for every tool output in `messages` whose
    kind was decided from its tool call. Outputs with no call to go by are left out, so the
    compression pipeline sniffs them from the text the model will actually see. Never raises:
    on an unexpected shape it returns {} and every output falls back to sniffing."""
    try:
        return {(r["index"], r["block"]): r["kind"]
                for r in label_conversation(messages or []) if r["how"] == "tool"}
    except Exception:
        return {}
