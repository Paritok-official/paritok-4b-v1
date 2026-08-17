"""Regression tests for eval_model.reanchor — no Docker, no network (a fake fetch
supplies the "true" source). Run: python -m pytest eval_model/tests/
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval_model import reanchor  # noqa: E402
from eval_model.dataset import line_number_context  # noqa: E402

# A real file whose signature is MULTI-LINE (the compressed context the model saw
# had it reflowed onto one line — the classic reflow-breaks-apply case).
REAL = (
    "class C:\n"
    "    def method(self,\n"
    "               a,\n"
    "               b):\n"
    "        x = a + b\n"
    "        return x\n"
)
FULL_CONTEXT = "# File: pkg/util/mod.py\n" + REAL + "\n"


def _fake_fetch(real_at):
    def fetch(repo, commit, path):
        return REAL if path == real_at else None
    return fetch


def test_reflow_and_truncated_path_reanchor():
    # Model wrote the signature on one line (reflow) AND truncated the path to the
    # basename (the `# File:`-header compression bug).
    patch = (
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ -1,6 +1,6 @@\n"
        " class C:\n"
        "     def method(self, a, b):\n"
        "         x = a + b\n"
        "-        return x\n"
        "+        return x * 2\n"
    )
    out = reanchor.reanchor(patch, "o/r", "abc", _fake_fetch("pkg/util/mod.py"),
                            full_context=FULL_CONTEXT)
    assert out is not None
    # path restored to the full repo-relative path
    assert "a/pkg/util/mod.py" in out
    # anchored onto the REAL (multi-line) file: it removes the real `return x`
    # and adds the model's change
    assert "-        return x\n" in out
    assert "+        return x * 2" in out


def test_docstring_targeted_edit_reanchors():
    # Editing content that lives inside a docstring — the docstring-dropping
    # canonical erases it, so this must recover via the whitespace-only fallback.
    real = (
        'def f():\n'
        '    """Return the first of the available choices.\n'
        '    Extended help text here.\n'
        '    """\n'
        '    return 1\n'
    )
    fc = "# File: pkg/d.py\n" + real + "\n"
    patch = (
        "--- a/pkg/d.py\n"
        "+++ b/pkg/d.py\n"
        "@@ -1,5 +1,5 @@\n"
        " def f():\n"
        '-    """Return the first of the available choices.\n'
        '+    """Return the first available choice.\n'
        '     Extended help text here.\n'
        '     """\n'
        "     return 1\n"
    )
    out = reanchor.reanchor(patch, "o/r", "abc", lambda *a: real, full_context=fc)
    assert out is not None
    assert "first available choice" in out


def test_unmatchable_returns_none():
    # Content the model invented that isn't in the file -> can't re-anchor -> None
    # (the caller then keeps the original patch / recount fallback).
    patch = (
        "--- a/pkg/util/mod.py\n"
        "+++ b/pkg/util/mod.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-        totally_made_up_line_not_in_file()\n"
        "+        something_else()\n"
    )
    out = reanchor.reanchor(patch, "o/r", "abc", _fake_fetch("pkg/util/mod.py"),
                            full_context=FULL_CONTEXT)
    assert out is None


def test_complete_docstring_replacement_reanchors():
    # Replacing a WHOLE multi-line docstring: the removed block is the entire
    # `"""..."""`, which the docstring-dropping canon empties -> it must be located AND
    # grafted with the whitespace-only matcher. Regression for the graft canon mismatch
    # that used to raise IndexError (empty offset map) and drop the patch to recount.
    real = (
        "def f():\n"
        '    """Old complete docstring spanning.\n'
        "    a second distinctive line.\n"
        '    """\n'
        "    return 1\n"
    )
    fc = "# File: pkg/dd.py\n" + real + "\n"
    patch = (
        "--- a/pkg/dd.py\n"
        "+++ b/pkg/dd.py\n"
        "@@ -1,5 +1,3 @@\n"
        " def f():\n"
        '-    """Old complete docstring spanning.\n'
        "-    a second distinctive line.\n"
        '-    """\n'
        '+    """New one-line docstring."""\n'
        "     return 1\n"
    )
    out = reanchor.reanchor(patch, "o/r", "abc", lambda *a: real, full_context=fc)
    assert out is not None                       # did not IndexError / drop the patch
    assert "New one-line docstring" in out


def test_strip_patch_line_numbers():
    p = ("--- a/m.py\n+++ b/m.py\n@@ -1,2 +1,2 @@\n"
         "      1\tdef f():\n-      2\t    return 1\n+      2\t    return 2\n")
    s = reanchor.strip_patch_line_numbers(p)
    assert "\n def f():" in s and "\n-    return 1" in s and "\n+    return 2" in s
    # a diff carrying no line numbers is returned unchanged
    clean = "--- a/m.py\n+++ b/m.py\n@@ -1 +1 @@\n-    return 1\n+    return 2\n"
    assert reanchor.strip_patch_line_numbers(clean) == clean


def test_line_number_context_roundtrips():
    from paritok.pipelines.edit_recovery import strip_read_line_numbers
    ln = line_number_context("# File: pkg/m.py\ndef f():\n    return 1\n")
    assert "# File: pkg/m.py" in ln and "\tdef f():" in ln   # header bare, body numbered
    assert "def f():\n    return 1" in strip_read_line_numbers(ln)


def test_reanchor_strips_line_numbers_before_matching():
    real = "def f():\n    x = 1\n    return x\n"
    fc = "# File: pkg/m.py\n" + real + "\n"
    # the model, shown line-numbered source, copied the <n>\t prefixes into the diff
    patch = (
        "--- a/pkg/m.py\n+++ b/pkg/m.py\n@@ -1,3 +1,3 @@\n"
        "     1\tdef f():\n-     2\t    x = 1\n+     2\t    x = 2\n     3\t    return x\n"
    )
    out = reanchor.reanchor(patch, "o/r", "abc", lambda *a: real,
                            full_context=fc, strip_line_numbers=True)
    assert out is not None and "-    x = 1" in out and "+    x = 2" in out
    # without stripping, the numbered hunk cannot match the real (unnumbered) source
    assert reanchor.reanchor(patch, "o/r", "abc", lambda *a: real,
                             full_context=fc, strip_line_numbers=False) is None


def test_empty_patch_returns_none():
    assert reanchor.reanchor("", "o/r", "abc", _fake_fetch("x")) is None
