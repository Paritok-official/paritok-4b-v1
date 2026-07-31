"""Edit-recovery against lossily-compressed file content.

Each frozen case is (original, old_string, new_string) where old_string/new_string
were authored by a model against the lint-compressed view of `original` (docstrings
dropped, multi-line signatures collapsed, blank lines removed, ...). We assert the
recovery maps the edit back onto the true original:

  - match_region locates the exact original region (never a wrong/ambiguous one),
  - recover_edit's client_old matches the real file byte-for-byte,
  - the model's change is applied,
  - dropped docstrings / comments are restored,
  - the result still compiles.

Fixtures live in tests/data/edit_recovery_cases.json and were generated once from
the real training lint; the test itself has no lint dependency.
"""
import json
import re
from pathlib import Path

import pytest

from paritok.pipelines.edit_recovery import match_region, recover_edit

_CASES = json.loads(
    (Path(__file__).parent / "data" / "edit_recovery_cases.json").read_text(encoding="utf-8")
)


def _ids():
    return [c["name"] for c in _CASES]


@pytest.mark.parametrize("case", _CASES, ids=_ids())
def test_edit_recovery(case):
    original = case["original"]
    old_s = case["old_string"]
    new_s = case["new_string"]

    # 1. The region is uniquely located and is a real substring of the file.
    recovered = match_region(old_s, original)
    assert recovered is not None, "region must be uniquely located"
    assert recovered in original

    out = recover_edit(old_s, new_s, original)
    assert out is not None
    client_old, client_new = out

    # 2. client_old matches the real file exactly (so the client Edit will succeed).
    assert client_old == recovered
    assert client_old in original

    # 3. Docstrings and comments inside the edited region survive.
    for doc in re.findall(r'"""(?:.|\n)*?"""|\'\'\'(?:.|\n)*?\'\'\'', recovered):
        assert doc.strip() in client_new, f"docstring dropped: {doc!r}"
    for cmt in re.findall(r"#[^\n]*", recovered):
        assert cmt.rstrip() in client_new, f"comment dropped: {cmt!r}"

    # 4. The model's change actually landed (result differs from the untouched region).
    assert client_new != client_old

    # 5. Applying the edit yields source that still compiles.
    full = original.replace(client_old, client_new, 1)
    compile(full, "<edit_recovery_test>", "exec")


def test_ambiguous_match_returns_none():
    """Two identical regions -> no unique match -> None (caller falls back to expand)."""
    original = "def f():\n    return 1\n\n\ndef f():\n    return 1\n"
    assert match_region("def f(): return 1", original) is None


def test_no_match_returns_none():
    """A genuinely dropped-content old_string (not just reformatted) does not match."""
    original = "def f(a):\n    x = compute(a)\n    y = refine(x)\n    return y\n"
    # model skipped a real content line (y = refine(x)) -> unsafe -> must not match
    assert match_region("def f(a):\n    x = compute(a)\n    return y", original) is None
