"""What a shell tool result is: the agent's own experiment, a test run, or the user's files.

Compression earns its keep on the user's files (a Read, a `cat`/`sed -n` of the repo):
large, mostly irrelevant to the step at hand. The output of a script the agent wrote
itself is the opposite: it printed exactly what it wanted to see, every line is the
result. Compressing it saves little and the agent expands it back, often repeatedly
(loopbench, all ON runs: 24 such outputs compressed saved ≈ $0.24; the 16 read_original
calls they triggered cost ≈ $2.22).

`classify_command(command, agent_files)` sorts a Bash / PowerShell command into:

  agent_script  an inline program (`python - <<EOF`, `python -c`, `node -e`, a here-string
                piped to an interpreter) or a run of a script the agent wrote itself
  test_run      the project's test runner (runtests.py, pytest, manage.py test, ...) or a
                linter / type checker
  project_run   runs the project's own code or CLI (`python -m django dumpfields ...`)
  file_dump    prints files (cat / sed -n / head / tail / Get-Content ...)
  search        grep / rg / find / ls / git grep / Select-String ...
  vcs           git diff / log / show / status / blame
  other

`agent_written(command)` returns the script paths a shell command itself creates
(`cat > probe.py <<EOF`, `tee`, Set-Content), so a later run of them is recognised.
"""

from __future__ import annotations

import re

_INTERP = r"(?:python[\d.]*|py|node|deno|ruby|perl|php|Rscript)(?:\.exe)?"
# An interpreter fed a program inline: `-c '...'`, `-e '...'`, `-` / no file + heredoc.
_INLINE = re.compile(
    rf"(?:^|[\s;&|(`'\"]){_INTERP}(?:\s+[^\s;&|]+)*?\s+(?:-c|-e|-Command)\s+[\"'$]"
    rf"|(?:^|[\s;&|(`'\"]){_INTERP}(?:\s+-[\w-]+)*\s+-?\s*<<-?\s*['\"]?\w+"
    # a lone `-` = program on stdin, even with a redirect before the heredoc
    rf"|(?:^|[\s;&|(`'\"]){_INTERP}(?:\s+-[\w-]+)*\s+-(?:\s|$)"
    rf"|['\"]@\s*\|\s*{_INTERP}"
    rf"|<<-?\s*['\"]?\w+['\"]?\s*\|\s*{_INTERP}", re.I | re.M)
_TEST = re.compile(r"runtests\.py|\bpytest\b|py\.test|manage\.py\s+test|-m\s+unittest|\btox\b|"
                   r"\bnpm\s+(?:run\s+)?test|\byarn\s+test|\bjest\b|\bvitest\b|\bgo\s+test|"
                   r"\bcargo\s+test|\bmvn\s+test|\bgradle\w*\s+test|"
                   # linters / type checkers: a check run, like the tests
                   r"-m\s+(?:flake8|mypy|pyflakes|pycodestyle|pylint|black|isort|ruff)\b|"
                   r"(?:^|[\s;&|(])(?:flake8|mypy|ruff|pylint|eslint|tsc|prettier)\b", re.I)
_DUMP = re.compile(r"(?:^|[\s;&|(])(?:cat|sed\s+-n|head|tail|nl|awk|less|more|bat|type|"
                   r"Get-Content|gc)\b", re.I)
_SEARCH = re.compile(r"(?:^|[\s;&|(])(?:grep|egrep|rg|ag|ack|find|ls|tree|dir|wc|"
                     r"Select-String|Get-ChildItem|gci)\b|git\s+(?:grep|ls-files)", re.I)
# Running the project's own code: an interpreter on a module / repo script (not a version
# query), or the project's CLI — e.g. trying out a management command the agent added.
_RUN = re.compile(rf"(?:^|&&|\|\||[;|\n(])\s*{_INTERP}\s+(?!-V\b|--version\b)\S"
                  r"|django-admin|manage\.py\s+\w", re.I)
_VCS = re.compile(r"\bgit(?:\s+(?:-C\s+\S+|-c\s+\S+|--[\w-]+(?:=\S+)?))*\s+"
                  r"(?:diff|log|show|status|blame)\b", re.I)
_SCRIPT_EXT = (".py", ".js", ".mjs", ".cjs", ".ts", ".sh", ".ps1", ".rb", ".pl", ".php", ".r")
# A shell command writing a script file: `cat > x.py <<EOF`, `tee x.py`, Set-Content x.ps1.
_CREATES = re.compile(r"(?:>|\btee\s+(?:-a\s+)?|Set-Content\s+(?:-Path\s+)?|Out-File\s+(?:-FilePath\s+)?)"
                      r"\s*['\"]?([^\s'\"<>|;&]+)", re.I)
_PATHISH = re.compile(r"[^\s'\"<>|;&()=]+")


def _base(path: str) -> str:
    return path.strip().strip("'\"").replace("\\", "/").rsplit("/", 1)[-1].lower()


def agent_written(command: str) -> set[str]:
    """Basenames of script files this shell command creates."""
    out = set()
    for m in _CREATES.finditer(command or ""):
        name = _base(m.group(1))
        if name.endswith(_SCRIPT_EXT):
            out.add(name)
    return out


def runs_agent_file(command: str, agent_files: set[str]) -> bool:
    """Does the command run (not just print) a script the agent wrote?"""
    if not agent_files:
        return False
    for seg in re.split(r"&&|\|\||[;|\n]", command or ""):
        words = [_base(w) for w in _PATHISH.findall(seg)]
        hits = [i for i, w in enumerate(words) if w in agent_files]
        if not hits:
            continue
        first = words[0] if words else ""
        if _DUMP.match(" " + first) or first in ("cat", "type", "head", "tail", "less", "more"):
            continue  # printing the script, not running it
        if hits[0] == 0 or re.match(_INTERP + r"$", first, re.I) or first in (
                "bash", "sh", "zsh", "pwsh", "powershell", "source", ".", "node", "deno"):
            return True
        if any(re.match(_INTERP + r"$", w, re.I) for w in words[:hits[0]]):
            return True
    return False


def classify_command(command: str, agent_files: set[str] | None = None) -> str:
    cmd = command or ""
    # include scripts this same command writes (`cat > t.py <<EOF ... EOF; python t.py`)
    files = set(agent_files or ()) | agent_written(cmd)
    if _INLINE.search(cmd) or runs_agent_file(cmd, files):
        return "agent_script"
    if _TEST.search(cmd):
        return "test_run"
    if _RUN.search(cmd):
        return "project_run"
    if _VCS.search(cmd):
        return "vcs"
    if _DUMP.search(cmd):
        return "file_dump"
    if _SEARCH.search(cmd):
        return "search"
    return "other"
