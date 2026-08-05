"""Codex shell-authored edit recovery against lossily-compressed file content.

Codex edits by running a shell command that does a literal string replace. When the file
read it saw was compressed, the 4B summary reflowed multi-line code onto one line, so the
`old` codex copied out no longer matches the real file and the replace is a no-op. These
tests pin that:

  - recover_shell_replace maps codex's (old -> new) onto the true file so that
    `original.replace(real_old, real_new)` yields the correctly-edited file,
  - it returns None (leave codex alone) when the edit already matches, is ambiguous, or
    the anchor is genuinely absent,
  - parse_shell_replaces extracts the literal pairs from the common shell shapes,
  - rewrite_shell_command splices recovered literals back in, and abstains safely when a
    recovered literal carries quotes/backslashes or no file resolves.
"""
from paritok.pipelines.codex_edit_recovery import (
    parse_shell_replaces,
    recover_shell_replace,
    rewrite_function_call_arguments,
    rewrite_shell_command,
)


# ---------------------------------------------------------------------------
# Fixtures: real files + the reflowed (old -> new) a model authored against a summary.
# ---------------------------------------------------------------------------

AWL = (
    "def average_word_length(text):\n"
    "    words = tokenize(text)\n"
    "    if not words:\n"
    "        return 0.0\n"
    "    total = sum(len(w) for w in words)\n"
    "    return total / (len(words) + 1)\n"
)

DOC = 'def scale(x):\n    """Scale x."""\n    return x * 2\n'

SC = (
    "def score(hits, tries):\n"
    "    base = hits * 100\n"
    "    return base / (tries + 1)\n"
)

SIG = (
    "def connect(\n"
    "    host,\n"
    "    port=8080,\n"
    "):\n"
    "    return (host, port)\n"
)

CMT = (
    "def h(a):\n"
    "    # step one\n"
    "    a = a + 1\n"
    "\n"
    "    return a\n"
)

AMB = "def f():\n    a = x + 1\n    b = x + 1\n    return a, b\n"

OP = "def area(w, h):\n    scaled = w * h\n    return scaled\n"


def _assert_recovers(original, old, new, expected):
    """recover_shell_replace maps (old, new) onto `original` so applying the recovered
    literal replace reproduces `expected` (the correctly-edited file)."""
    rec = recover_shell_replace(old, new, original)
    assert rec is not None, "expected a recovery"
    real_old, real_new = rec
    assert real_old in original, "recovered OLD must be a real substring of the file"
    assert real_old != real_new, "the change must actually land"
    assert original.replace(real_old, real_new, 1) == expected


# ---------------------------------------------------------------------------
# recover_shell_replace — structural reflow (the core codex failure mode)
# ---------------------------------------------------------------------------

def test_structural_reflow_offbyone_denominator():
    # 4B inlined `total = sum(...)` into a ternary; codex's whole OLD no longer matches,
    # but the real change `(len(words) + 1)` -> `len(words)` is recovered.
    old = "sum(len(w) for w in words) / (len(words) + 1) if words else 0.0"
    new = "sum(len(w) for w in words) / len(words) if words else 0.0"
    _assert_recovers(AWL, old, new, AWL.replace("(len(words) + 1)", "len(words)"))


def test_structural_reflow_const_denominator():
    old = "base = hits * 100; return base / (tries + 1)"
    new = "base = hits * 100; return base / tries"
    _assert_recovers(SC, old, new, SC.replace("(tries + 1)", "tries"))


def test_structural_reflow_unique_operator():
    old = "scaled = w * h; return scaled"
    new = "scaled = w + h; return scaled"
    _assert_recovers(OP, old, new, OP.replace("w * h", "w + h"))


# ---------------------------------------------------------------------------
# recover_shell_replace — whitespace/docstring/comment reflow (whole-string maps)
# ---------------------------------------------------------------------------

def test_docstring_dropped_reflow():
    # Summary dropped the docstring; the whole OLD still maps canonically and the docstring
    # is restored into the rewritten replacement.
    old = "def scale(x):\n    return x * 2"
    new = "def scale(x):\n    return x * 3"
    _assert_recovers(DOC, old, new, DOC.replace("x * 2", "x * 3"))
    # (sanity) the recovered new keeps the docstring the summary had dropped
    _real_old, real_new = recover_shell_replace(old, new, DOC)
    assert '"""Scale x."""' in real_new


def test_multiline_signature_collapsed_reflow():
    # Summary collapsed a multi-line signature onto one line; a default-value change maps
    # back onto the real (indented, trailing-comma) signature.
    old = "def connect(host, port=8080):"
    new = "def connect(host, port=9090):"
    _assert_recovers(SIG, old, new, SIG.replace("8080", "9090"))


def test_comment_and_blank_line_dropped_reflow():
    old = "def h(a):\n    a = a + 1\n    return a"
    new = "def h(a):\n    a = a + 2\n    return a"
    _assert_recovers(CMT, old, new, CMT.replace("a + 1", "a + 2"))
    _real_old, real_new = recover_shell_replace(old, new, CMT)
    assert "# step one" in real_new  # comment restored


# ---------------------------------------------------------------------------
# recover_shell_replace — must return None (leave codex's command untouched)
# ---------------------------------------------------------------------------

def test_old_already_matches_file_returns_none():
    # `return x * 2` is verbatim in the file: no reflow, nothing to fix.
    assert recover_shell_replace("return x * 2", "return x * 3", DOC) is None


def test_old_equals_new_returns_none():
    assert recover_shell_replace("total / (len(words) + 1)",
                                 "total / (len(words) + 1)", AWL) is None


def test_empty_old_returns_none():
    assert recover_shell_replace("", "anything", AWL) is None


def test_whitespace_only_old_returns_none():
    assert recover_shell_replace("   ", "x", AWL) is None


def test_ambiguous_change_returns_none():
    # The minimal changed span (`1`) occurs more than once -> not uniquely locatable.
    old = "a = x + 1; b = x + 1"
    new = "a = x + 2; b = x + 1"
    assert recover_shell_replace(old, new, AMB) is None


def test_hallucinated_anchor_returns_none():
    # codex's target names a symbol that isn't in the file at all.
    hal = "def g(z):\n    return baz(z)\n"
    assert recover_shell_replace("return foo(z)", "return bar(z)", hal) is None


# ---------------------------------------------------------------------------
# parse_shell_replaces — the shell shapes codex actually emits
# ---------------------------------------------------------------------------

def test_parse_python_single_quoted_replace():
    reps = parse_shell_replaces("s = s.replace('old text', 'new text')")
    assert len(reps) == 1
    assert (reps[0].old, reps[0].new) == ("old text", "new text")


def test_parse_dotnet_double_quoted_replace():
    reps = parse_shell_replaces('$c = $c.Replace("foo", "bar")')
    assert len(reps) == 1
    assert (reps[0].old, reps[0].new) == ("foo", "bar")


def test_parse_powershell_replace_operator():
    reps = parse_shell_replaces("$c = $c -replace 'aaa', 'bbb'")
    assert len(reps) == 1
    assert (reps[0].old, reps[0].new) == ("aaa", "bbb")


def test_parse_powershell_variable_form():
    cmd = "$old = 'A B'; $new = 'A C'; $c = $c.Replace($old, $new)"
    reps = parse_shell_replaces(cmd)
    assert len(reps) == 1
    assert (reps[0].old, reps[0].new) == ("A B", "A C")
    # the spans point at the $old/$new assignment literals (so a rewrite fixes those)
    assert cmd[reps[0].old_span[0]:reps[0].old_span[1]] == "'A B'"


def test_parse_replace_spanning_newlines():
    cmd = "s = s.replace('sum(x) / (n + 1)',\n            'sum(x) / n')"
    reps = parse_shell_replaces(cmd)
    assert len(reps) == 1
    assert (reps[0].old, reps[0].new) == ("sum(x) / (n + 1)", "sum(x) / n")


def test_parse_no_replace_returns_empty():
    assert parse_shell_replaces("python -m pytest test_x.py -q") == []


def test_parse_doubled_quote_escape():
    # PowerShell doubles a quote to escape it inside a single-quoted string.
    reps = parse_shell_replaces("$c.Replace('it''s', 'it is')")
    assert len(reps) == 1
    assert reps[0].old == "it's"


# ---------------------------------------------------------------------------
# rewrite_shell_command — end-to-end splice + safety abstains
# ---------------------------------------------------------------------------

def test_rewrite_python_heredoc_reflow_fix():
    old = "sum(len(w) for w in words) / (len(words) + 1) if words else 0.0"
    new = "sum(len(w) for w in words) / len(words) if words else 0.0"
    cmd = f"p.write_text(s.replace('{old}', '{new}'))"
    out, n = rewrite_shell_command(cmd, lambda _c: [AWL])
    assert n == 1
    # the rewritten command's OLD literal is now a real substring of the file
    rep = parse_shell_replaces(out)[0]
    assert rep.old in AWL
    assert AWL.replace(rep.old, rep.new, 1) == AWL.replace("(len(words) + 1)", "len(words)")


def test_rewrite_variable_form_reflow_fix():
    old = "base = hits * 100; return base / (tries + 1)"
    new = "base = hits * 100; return base / tries"
    cmd = (f"$old = '{old}'; $new = '{new}'; "
           "$c = Get-Content -Raw score.py; $c = $c.Replace($old, $new); "
           "Set-Content score.py $c")
    out, n = rewrite_shell_command(cmd, lambda _c: [SC])
    assert n == 1
    rep = parse_shell_replaces(out)[0]
    assert rep.old in SC
    assert SC.replace(rep.old, rep.new, 1) == SC.replace("(tries + 1)", "tries")


def test_rewrite_abstains_when_recovered_literal_has_quotes():
    # The docstring recovery carries `"""..."""`; re-quoting could corrupt the command, so
    # rewrite must leave it untouched (codex falls back to its own re-read). Real newlines
    # here so the whole-string (docstring-restoring) recovery fires.
    old = "def scale(x):\n    return x * 2"
    new = "def scale(x):\n    return x * 3"
    cmd = f"s.replace('{old}', '{new}')"
    out, n = rewrite_shell_command(cmd, lambda _c: [DOC])
    assert n == 0
    assert out == cmd


def test_rewrite_noop_when_no_file_resolves():
    cmd = "s.replace('sum(x) / (n + 1)', 'sum(x) / n')"
    out, n = rewrite_shell_command(cmd, lambda _c: [])
    assert n == 0
    assert out == cmd


def test_rewrite_noop_when_old_already_matches():
    cmd = "s.replace('return x * 2', 'return x * 3')"
    out, n = rewrite_shell_command(cmd, lambda _c: [DOC])
    assert n == 0
    assert out == cmd


# ---------------------------------------------------------------------------
# rewrite_function_call_arguments — the Responses `function_call.arguments` wiring
# ---------------------------------------------------------------------------

def test_function_call_arguments_reflow_fix():
    # The exact shape codex emitted in the 19-item thrash run: a python edit whose OLD is
    # the reflowed one-liner. The wiring rewrites it so OLD matches the real file.
    import json
    old = "sum(len(w) for w in words) / (len(words) + 1) if words else 0.0"
    new = "sum(len(w) for w in words) / len(words) if words else 0.0"
    script = f"s = open('textstats.py').read()\ns = s.replace('{old}', '{new}')"
    args = json.dumps({"command": ["python", "-c", script]})
    out_json, n = rewrite_function_call_arguments(args, [AWL])
    assert n == 1
    new_script = json.loads(out_json)["command"][2]
    rep = parse_shell_replaces(new_script)[0]
    assert rep.old in AWL  # OLD now matches the real file, so codex's replace lands
    assert AWL.replace(rep.old, rep.new, 1) == AWL.replace("(len(words) + 1)", "len(words)")


def test_function_call_arguments_string_command():
    import json
    args = json.dumps({"command": "s.replace('base = hits * 100; return base / (tries + 1)',"
                                  " 'base = hits * 100; return base / tries')"})
    out_json, n = rewrite_function_call_arguments(args, [SC])
    assert n == 1
    rep = parse_shell_replaces(json.loads(out_json)["command"])[0]
    assert rep.old in SC


def test_function_call_arguments_noop_when_matches_or_unparseable():
    import json
    # already-correct edit -> unchanged
    ok = json.dumps({"command": "s.replace('return x * 2', 'return x * 3')"})
    assert rewrite_function_call_arguments(ok, [DOC]) == (ok, 0)
    # non-JSON arguments -> returned untouched
    assert rewrite_function_call_arguments("not json {", [DOC]) == ("not json {", 0)
    # no candidate files -> untouched
    assert rewrite_function_call_arguments(ok, []) == (ok, 0)
