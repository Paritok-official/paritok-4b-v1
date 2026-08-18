"""`paritok doctor` — a preflight that fails when compression is a silent no-op.

The gap this closes: GET /test reporting `gpu_available: true` does not mean the
worker is compressing. When it is not, compress() correctly returns the original
(gpu_server.py), the proxy answers HTTP 200, and /stats shows `tokens_saved: 0` —
which looks the same as "nothing in this content was compressible". Issues #20,
#30 and #31 all start from someone unable to tell those apart.

So the doctor does not just probe reachability; it round-trips a block and fails
if the result is not materially smaller.
"""
import click.testing

from paritok.cli import main


class _FakeStrategy:
    """Stands in for GpuServerStrategy / LocalModelStrategy."""

    def __init__(self, available=True, message="GPU server online.", ratio=0.05, raises=None):
        self._available = available
        self._message = message
        self._ratio = ratio
        self._raises = raises

    def check(self):
        return self._available, self._message

    def is_available(self):
        return self._available

    def compress(self, content, **kwargs):
        if self._raises is not None:
            raise self._raises
        # ratio 1.0 == the backend echoed the input back == a silent no-op.
        return content[: max(1, int(len(content) * self._ratio))]


def _run(monkeypatch, strategy, use_gpu_server=True, api_key="pk_live_test"):
    from paritok.config import GpuServerConfig, ParitokConfig

    cfg = ParitokConfig()
    cfg.use_gpu_server = use_gpu_server
    if use_gpu_server:
        cfg.gpu_server = GpuServerConfig(
            base_url="https://www.paritok.com/api", model="paritok-4b-v1",
            api_key=api_key, timeout=10.0,
        )
    monkeypatch.setattr(ParitokConfig, "load", staticmethod(lambda *a, **k: cfg))
    monkeypatch.setattr(
        "paritok.strategies.gpu_server.GpuServerStrategy", lambda *a, **k: strategy
    )
    monkeypatch.setattr(
        "paritok.strategies.local_model.LocalModelStrategy", lambda *a, **k: strategy
    )
    # No proxy running: the doctor should warn, not fail.
    import httpx

    def _no_proxy(*a, **k):
        raise httpx.ConnectError("nothing listening")

    monkeypatch.setattr(httpx, "get", _no_proxy)
    return click.testing.CliRunner().invoke(main, ["doctor"])


def test_passes_when_backend_reachable_and_compressing(monkeypatch):
    result = _run(monkeypatch, _FakeStrategy(ratio=0.05))
    assert result.exit_code == 0, result.output
    assert "All critical checks passed" in result.output or "All checks passed" in result.output


def test_fails_when_backend_answers_but_does_not_compress(monkeypatch):
    """The regression that motivates this command: healthy endpoint, no compression."""
    result = _run(monkeypatch, _FakeStrategy(ratio=1.0))
    assert result.exit_code == 1
    assert "did not compress" in result.output


def test_fails_when_backend_unreachable(monkeypatch):
    result = _run(monkeypatch, _FakeStrategy(available=False, message="unreachable"))
    assert result.exit_code == 1
    assert "unreachable" in result.output
    # The smoke test must not run against a backend we already know is down.
    assert "skipped" in result.output


def test_fails_when_api_key_missing(monkeypatch):
    result = _run(monkeypatch, _FakeStrategy(), api_key="")
    assert result.exit_code == 1
    assert "api_key" in result.output


def test_fails_when_compress_raises(monkeypatch):
    result = _run(monkeypatch, _FakeStrategy(raises=RuntimeError("boom")))
    assert result.exit_code == 1
    assert "RuntimeError" in result.output


def test_missing_proxy_is_a_warning_not_a_failure(monkeypatch):
    result = _run(monkeypatch, _FakeStrategy(ratio=0.05))
    assert result.exit_code == 0
    assert "nothing listening" in result.output
