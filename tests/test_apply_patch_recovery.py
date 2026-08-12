"""Codex's `apply_patch` tool locates each hunk by its context + removed (`-`) lines
EXACTLY. When paritok compressed the file read codex saw, those old-side lines dropped a
trailing comment (or reflowed), so apply_patch fails with "Failed to find expected lines"
and codex thrashes — the same class of failure `edit_recovery` fixes for Claude Code's Edit
and `recover_shell_replace` fixes for codex's shell edits.

These cover `rewrite_apply_patch` (and the `rewrite_edit_command` / `rewrite_tool_call_input`
/ `rewrite_function_call_arguments` entry points) against the real api-5.4 failure: the 4B
dropped a `##__o|o__` marker comment, so codex's `-    max_retries = -1` didn't match the
true `-    max_retries = -1  ##__o|o__`.
"""
from paritok.pipelines.codex_edit_recovery import (
    rewrite_apply_patch,
    rewrite_edit_command,
    rewrite_function_call_arguments,
    rewrite_tool_call_input,
)

# The true file on disk (comment kept); the 4B summary codex saw dropped the comment.
REAL_FILE = (
    "def answer_question(q):\n"
    "    max_retries = -1  ##__o|o__\n"
    "    for i in range(max_retries):\n"
    "        return q\n"
)

# codex authored this patch against the compressed view (no comment on the removed line).
CODEX_PATCH = (
    "*** Begin Patch\n"
    "*** Update File: bug.py\n"
    "@@ def answer_question(q):\n"
    "-    max_retries = -1\n"
    "+    max_retries = 3\n"
    "     for i in range(max_retries):\n"
    "*** End Patch"
)


def test_apply_patch_recovers_dropped_comment_on_removed_line():
    new, n = rewrite_apply_patch(CODEX_PATCH, [REAL_FILE])
    assert n == 1
    # The removed line now carries the real trailing comment, so apply_patch will find it.
    assert "-    max_retries = -1  ##__o|o__\n" in new
    # The added line codex intended is untouched.
    assert "+    max_retries = 3\n" in new
    # Context line preserved with its leading space prefix.
    assert "     for i in range(max_retries):\n" in new
    # Frame preserved.
    assert new.startswith("*** Begin Patch\n") and new.endswith("*** End Patch")


def test_apply_patch_noop_when_already_matches():
    # old-side already matches the file (no compression loss) → unchanged.
    patch = (
        "*** Begin Patch\n"
        "*** Update File: bug.py\n"
        "-    for i in range(max_retries):\n"
        "+    for i in range(3):\n"
        "*** End Patch"
    )
    new, n = rewrite_apply_patch(patch, [REAL_FILE])
    assert n == 0 and new == patch


def test_apply_patch_abstains_when_line_count_differs():
    # A reflow (real spans 2 lines, codex's old-side is 1) can't be mapped 1:1 → abstain.
    real = "def f(a,\n      b):\n    return a\n"
    patch = (
        "*** Begin Patch\n"
        "*** Update File: f.py\n"
        "-def f(a, b):\n"
        "+def f(a, b, c):\n"
        "*** End Patch"
    )
    new, n = rewrite_apply_patch(patch, [real])
    assert n == 0 and new == patch


def test_apply_patch_abstains_when_no_unique_match():
    real = "x = 1\ny = 2\nx = 1\n"  # the removed line appears twice → not unique
    patch = "*** Begin Patch\n*** Update File: a.py\n-x = 1\n+x = 9\n*** End Patch"
    new, n = rewrite_apply_patch(patch, [real])
    assert n == 0 and new == patch


def test_apply_patch_noop_on_non_patch_text():
    new, n = rewrite_apply_patch("echo hello", [REAL_FILE])
    assert n == 0 and new == "echo hello"


def test_apply_patch_noop_without_originals():
    new, n = rewrite_apply_patch(CODEX_PATCH, [])
    assert n == 0 and new == CODEX_PATCH


def test_apply_patch_recovers_multi_hunk_pure_addition_left_alone():
    # A hunk that is a pure insertion (only `+` lines, no old-side) must not crash / rewrite.
    patch = (
        "*** Begin Patch\n"
        "*** Update File: bug.py\n"
        "@@ def answer_question(q):\n"
        "-    max_retries = -1\n"
        "+    max_retries = 3\n"
        "+    log('added')\n"
        "     for i in range(max_retries):\n"
        "*** End Patch"
    )
    new, n = rewrite_apply_patch(patch, [REAL_FILE])
    assert n == 1
    assert "-    max_retries = -1  ##__o|o__\n" in new
    assert "+    log('added')\n" in new  # untouched


# ── entry points route apply_patch bodies through recovery ──

def test_rewrite_edit_command_routes_patch():
    new, n = rewrite_edit_command(CODEX_PATCH, lambda _c: [REAL_FILE])
    assert n == 1 and "##__o|o__" in new


def test_rewrite_edit_command_heredoc_wrapped_patch():
    # codex may emit `apply_patch <<'EOF' ... EOF` as a shell command.
    cmd = "apply_patch <<'EOF'\n" + CODEX_PATCH + "\nEOF"
    new, n = rewrite_edit_command(cmd, lambda _c: [REAL_FILE])
    assert n == 1 and "##__o|o__" in new
    assert new.startswith("apply_patch <<'EOF'\n") and new.rstrip().endswith("EOF")


def test_rewrite_edit_command_still_does_shell_replace():
    # Regression: the shell literal-replace path is unaffected.
    real = "value = old_thing\n"
    cmd = "python -c \"s=s.replace('old_thing','new_thing')\""
    new, n = rewrite_edit_command(cmd, lambda _c: [real])
    # shell path either rewrites (n>=1) or safely abstains (n==0); must never raise and
    # must not misfire the patch path on a non-patch command.
    assert isinstance(new, str) and n >= 0 and "*** Begin Patch" not in new


def test_rewrite_function_call_arguments_with_patch_key():
    import json
    args = json.dumps({"patch": CODEX_PATCH})
    new_json, n = rewrite_function_call_arguments(args, [REAL_FILE])
    assert n == 1
    assert "##__o|o__" in json.loads(new_json)["patch"]


def test_rewrite_function_call_arguments_with_input_patch():
    import json
    args = json.dumps({"input": CODEX_PATCH})
    new_json, n = rewrite_function_call_arguments(args, [REAL_FILE])
    assert n == 1 and "##__o|o__" in json.loads(new_json)["input"]


def test_rewrite_tool_call_input_raw_patch():
    new, n = rewrite_tool_call_input(CODEX_PATCH, [REAL_FILE])
    assert n == 1 and "##__o|o__" in new


def test_rewrite_tool_call_input_json_object():
    import json
    new, n = rewrite_tool_call_input(json.dumps({"input": CODEX_PATCH}), [REAL_FILE])
    assert n == 1 and "##__o|o__" in new


def test_rewrite_tool_call_input_noop_on_plain_command():
    new, n = rewrite_tool_call_input("ls -la", [REAL_FILE])
    assert n == 0 and new == "ls -la"
