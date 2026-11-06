"""classify_command sorts shell tool calls into the agent's own experiments vs the user's
files. Cases are real commands from the loopbench runs (paths shortened), including the
three the first version missed."""
import pytest

from paritok.pipelines.agent_output import agent_written, classify_command

AGENT_SCRIPTS = [
    "python -c \" from django.utils.dateparse import parse_datetime as p\nfor s in ['2012-04-23T10:20:30Z']: print(p(s))\"",
    "cd /repo && python - <<'EOF'\nimport datetime\nprint(1)\nEOF",
    "cd /repo/django/db/models && python - <<'E'\nimport re\nE",
    # missed at first: an option with an argument before -c
    "python -W error -c \" import typing, inspect\nimport django.db.models.deletion as d\"",
    # missed at first: a redirect between `-` and the heredoc
    "cd /repo && mkdir -p \"$TEMP/p\" && python - > \"$TEMP/p/out.txt\" 2>&1 <<'EOF'\nprint(2)\nEOF",
    # missed at first: the script is written and run by the same command
    "sed -n 14,17p django/db/models/query.py; cat > /tmp/t.py <<'EOF'\nimport django\nEOF\npython /tmp/t.py",
    "grep -n datetime_re -A8 django/utils/dateparse.py | head -30; python -c \" from django.utils import x\"",
    "node -e \"console.log(require('./pkg.json').version)\"",
    "@'\nprint(3)\n'@ | python -",
]


@pytest.mark.parametrize("cmd", AGENT_SCRIPTS)
def test_inline_programs_are_agent_scripts(cmd):
    assert classify_command(cmd) == "agent_script"


def test_running_a_script_the_agent_wrote():
    files = {"probe.py", "check_delete_sql.py"}
    assert classify_command("python -I /c/Users/u/AppData/Local/Temp/autodetect/probe.py", files) == "agent_script"
    assert classify_command('cd "/runs/off/django" && python "/runs/off/check_delete_sql.py"', files) == "agent_script"
    # printing it is not running it
    assert classify_command("cat /tmp/probe.py", files) == "file_dump"
    # a repo script of the same kind, not written by the agent
    assert classify_command("python -I /tmp/probe.py") == "project_run"


@pytest.mark.parametrize("cmd,cat", [
    ("cd /repo/tests && python runtests.py delete delete_regress 2>&1 | tail -5", "test_run"),
    ("python -m flake8 ../django/db/models/deletion.py delete/models.py", "test_run"),
    ("python -m mypy --version 2>&1; python --version", "test_run"),
    ("python -m django dumpfields nonexistent --settings=test_sqlite --pythonpath=tests", "project_run"),
    ("python ../django/bin/django-admin.py dumpfields auth --settings=test_sqlite", "project_run"),
    ("git -C /repo diff django/db/models/deletion.py", "vcs"),
    ('cd "/repo" && git --no-pager diff', "vcs"),
    ("git show f110de5c04 -- tests/delete/models.py", "vcs"),
    ("sed -n 452,612p sql/compiler.py", "file_dump"),
    ("cat TASK.md; cat django/db/models/deletion.py", "file_dump"),
    ("grep -n \"def explain\" -A12 django/db/models/query.py; grep -rn explain django/db/backends", "search"),
    ('grep -n "^from\\|^import\\|^class \\| def test_" tests/delete/tests.py', "search"),
    ("sed -n '214,228p' tests/user_commands/tests.py && python --version", "file_dump"),
    ("rm _18", "other"),
])
def test_other_categories(cmd, cat):
    assert classify_command(cmd) == cat


def test_agent_written_finds_scripts_a_command_creates():
    assert agent_written("cat > /tmp/t.py <<'EOF'\nx\nEOF") == {"t.py"}
    assert agent_written("echo hi | tee probe.sh") == {"probe.sh"}
    assert agent_written("Set-Content -Path C:\\tmp\\p.ps1 -Value x") == {"p.ps1"}
    assert agent_written("python runtests.py 2>&1 > out.txt") == set()  # not a script
