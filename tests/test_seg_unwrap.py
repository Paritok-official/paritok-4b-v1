"""Regression: a half-wrapped model reply (opening [SEG ...] but the closing
[/SEG] truncated by num_predict on a long body) used to fall through _unwrap_seg's
fallback and leak the raw '[SEG id=... kind=... level=...]' marker into the
compressed output. It must be scrubbed so the output is clean regardless.
"""
from paritok.strategies.local_model import _unwrap_seg


def test_well_formed_pair_returns_body():
    assert _unwrap_seg("[SEG id=s1 kind=file_read level=L1]\nhello\n[/SEG]") == "hello"


def test_opening_tag_without_close_is_scrubbed():
    # the exact leak seen on the gpu server for a long file
    raw = "[SEG id=s1 kind=file_read level=L1][imports: a, b]\ndef f(): pass\n"
    out = _unwrap_seg(raw)
    assert "[SEG" not in out
    assert out.startswith("[imports: a, b]")
    assert "def f(): pass" in out


def test_dangling_close_tag_is_scrubbed():
    assert _unwrap_seg("def f(): pass\n[/SEG]") == "def f(): pass"


def test_no_tags_returns_text_unchanged():
    assert _unwrap_seg("just some content") == "just some content"


def test_dropped_segment_is_empty():
    assert _unwrap_seg("[SEG id=s1 kind=file_read level=L0]\n[/SEG]") == ""


def test_reunwrap_is_idempotent_on_clean_body():
    # gpu_server re-unwraps an already-clean body; must be a no-op
    body = "def answer_question():\n    return 1"
    assert _unwrap_seg(body) == body
