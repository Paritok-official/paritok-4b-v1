"""`compress()` returns str for four different events; these pin them apart.

Before `compress_result()` a caller could not tell "the model says this segment is
irrelevant to your query" from "the GPU is offline and echoed your bytes back",
because both arrive as a `str` and three of the four arrive as the caller's own
content. That is the root of #20: an empty body was accepted as a summary and a
tool result silently became an empty `[REF:...]` tag.

The contract these tests lock:

* `compress()` keeps its signature and never returns an empty string (the #20 fix),
* `compress_result()` says which of the four things happened, and
* `IRRELEVANT` is only reachable when the response was complete, so a torn stream
  cannot be mistaken for a verdict.
"""

import httpx
import pytest

from paritok.config import GpuServerConfig
from paritok.strategies.gpu_server import CompressionResult, GpuServerStrategy, Outcome

CONTENT = "def charge(amount):\n    return round(amount, 2)\n" * 20


def _strategy(monkeypatch, *, json_body=None, status=200, raises=None):
    def fake_post(url, **kwargs):
        if raises is not None:
            raise raises
        return httpx.Response(
            status,
            json=json_body if json_body is not None else {},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    return GpuServerStrategy(GpuServerConfig(api_key="k"))


def test_compressed_body_is_reported_as_compressed(monkeypatch):
    s = _strategy(monkeypatch, json_body={"compressed": "charge(amount)", "gpu_available": True})
    r = s.compress_result(CONTENT, query="how are charges rounded")
    assert r.outcome is Outcome.COMPRESSED
    assert r.content == "charge(amount)"
    assert r.is_compressed


def test_empty_body_on_a_complete_response_is_a_verdict_not_a_failure(monkeypatch):
    """The behaviour this whole PR exists to make visible."""
    s = _strategy(monkeypatch, json_body={"compressed": "", "gpu_available": True})
    r = s.compress_result(CONTENT, query="something unrelated to charges")
    assert r.outcome is Outcome.IRRELEVANT
    # The caller's own content comes back, so a tool result is never destroyed.
    assert r.content == CONTENT


def test_gpu_offline_is_distinguishable_from_irrelevant(monkeypatch):
    s = _strategy(monkeypatch, json_body={"compressed": "", "gpu_available": False})
    r = s.compress_result(CONTENT, query="anything")
    assert r.outcome is Outcome.UNAVAILABLE
    assert r.content == CONTENT


def test_a_torn_stream_cannot_present_as_a_verdict(monkeypatch):
    """The safety property: transport failure must not look like IRRELEVANT."""
    s = _strategy(monkeypatch, raises=httpx.ReadError("connection reset"))
    r = s.compress_result(CONTENT, query="anything")
    assert r.outcome is Outcome.UNAVAILABLE
    assert r.outcome is not Outcome.IRRELEVANT
    assert r.content == CONTENT


def test_rejected_key_is_its_own_outcome(monkeypatch):
    s = _strategy(monkeypatch, status=401, json_body={})
    r = s.compress_result(CONTENT, query="anything")
    assert r.outcome is Outcome.UNAVAILABLE
    assert "401" in r.detail


def test_non_string_body_is_malformed_not_compressed(monkeypatch):
    s = _strategy(monkeypatch, json_body={"compressed": 42, "gpu_available": True})
    r = s.compress_result(CONTENT, query="anything")
    assert r.outcome is Outcome.MALFORMED
    assert r.content == CONTENT


@pytest.mark.parametrize(
    "body,gpu",
    [("", True), ("", False), ("summary", True)],
)
def test_compress_never_returns_an_empty_string(monkeypatch, body, gpu):
    """#20: an empty string reaching the pipeline is what destroys a tool result."""
    s = _strategy(monkeypatch, json_body={"compressed": body, "gpu_available": gpu})
    out = s.compress(CONTENT, query="anything")
    assert isinstance(out, str)
    assert out.strip()


def test_result_stringifies_to_its_content_for_existing_callers():
    r = CompressionResult(Outcome.COMPRESSED, "body")
    assert f"{r}" == "body"
