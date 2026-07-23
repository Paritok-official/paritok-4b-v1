"""GPU-server strategy: compress via the Paritok hosted endpoint.

Same `compress(...)` surface as LocalModelStrategy, so CompressionPipeline can
use either backend. Instead of talking to a local Ollama model, this POSTs each
segment to the hosted compression endpoint:

    POST  {base_url}/compress   -> { "compressed": "<body>", "gpu_available": bool }
    GET   {base_url}/test       -> { "gpu_available": bool, "message": "<why>" }

The hosted endpoint fronts the Paritok GPU inference server. While that server
is offline the endpoint echoes the original text back (gpu_available: false);
this strategy then returns the ORIGINAL content unchanged, so the agent keeps
working uncompressed rather than erroring. Any network/HTTP failure degrades the
same way — never break the caller's turn over a compression hiccup.

Config (paritok.yaml):
    use_gpu_server: true
    gpu_server:
      base_url: https://www.paritok.com/api
      model: paritok-4b-v1
      api_key: ""        # or env PARITOK_API_KEY
"""

from __future__ import annotations

import logging
import threading

from paritok.config import GpuServerConfig

logger = logging.getLogger("paritok.gpu_server")

# If a single compression waits longer than this, the hosted GPU is most likely
# cold-starting / rebooting — tell the user in the proxy console so a slow first
# request doesn't look like a hang.
_REBOOT_NOTICE_AFTER_S = 30.0


class GpuServerStrategy:
    name = "gpu_server"

    def __init__(self, config: GpuServerConfig):
        self.config = config
        self._key_warned = False  # warn about a rejected API key only once

    def compress(
        self,
        content: str,
        *,
        query: str | None = None,
        level: str | None = None,
        kind: str | None = None,
        target_ratio: str | None = None,
        system_prompt: str | None = None,
        **kwargs,
    ) -> str:
        """Compress via the hosted endpoint; return the original on any failure.

        The remote server owns chunking / SEG formatting, so we hand it the whole
        segment plus the intent and read back the compressed body.
        """
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for the gpu_server strategy. "
                "Install with: pip install paritok[llm]"
            )

        payload = {
            "model": self.config.model,
            "content": content,
            "query": (query or "").strip(),
            "level": level,
            "kind": kind,
        }
        # The upstream LLM the user is actually calling (e.g. claude-sonnet-4,
        # gpt-5) — passed through so the hosted gateway can attribute savings
        # per model. Omitted when unknown.
        upstream_model = kwargs.get("upstream_model")
        if upstream_model:
            payload["upstream_model"] = upstream_model
        headers = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        # Warn once, in the proxy console, if we're still waiting after 30s (the
        # hosted GPU is probably rebooting). Cancelled the moment the call returns.
        reboot_timer = threading.Timer(
            _REBOOT_NOTICE_AFTER_S,
            lambda: print("gpu server is rebooting please wait", flush=True),
        )
        reboot_timer.daemon = True
        reboot_timer.start()
        try:
            resp = httpx.post(
                f"{self._base()}/compress",
                json=payload,
                headers=headers,
                timeout=self.config.timeout,
            )
            # Key rejected: the endpoint won't compress without a valid key. Warn
            # once and pass the content through uncompressed (don't spam per call).
            if resp.status_code in (401, 403):
                self._warn_invalid_key_once(resp.status_code)
                return content
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:  # noqa: BLE001 — any failure degrades to passthrough
            logger.warning(
                "GPU server compression unavailable (%s); passing content through "
                "uncompressed.", e,
            )
            return content
        finally:
            reboot_timer.cancel()

        if not data.get("gpu_available", False):
            # Endpoint is up but the GPU backend is offline — it echoed the
            # original back. Keep the original; no compression happened.
            return content
        compressed = data.get("compressed")
        if not isinstance(compressed, str):
            return content
        # Defensive: the hosted server already unwraps SEG tags, but a truncated
        # closing tag can leak a stray opening [SEG ...] marker. Re-unwrap here so
        # the proxy's output is clean and identical regardless of that hiccup.
        from paritok.strategies.local_model import _unwrap_seg
        return _unwrap_seg(compressed)

    def check(self) -> tuple[bool, str]:
        """Probe {base_url}/test WITH the API key. Returns (available, message).

        Sends `Authorization: Bearer <api_key>` exactly like compress(), so an
        auth-gated /test also validates the key at startup (that's how `paritok
        up` tells the user whether their Paritok API key is good). A 401/403 is
        reported as a rejected key — distinct from an unreachable endpoint — so
        the message can point at the key rather than the network. Only used on
        the hosted backend (use_gpu_server: true); self-hosted never calls this.
        """
        try:
            import httpx
        except ImportError:
            return False, (
                "httpx is required for the gpu_server strategy "
                "(pip install paritok[llm])."
            )

        headers = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        try:
            resp = httpx.get(f"{self._base()}/test", headers=headers, timeout=10.0)
        except Exception as e:  # noqa: BLE001 — network failure = unreachable
            return False, (
                f"Could not reach the Paritok hosted endpoint at {self._base()} "
                f"({e}). Compression will pass through uncompressed."
            )

        # Auth failures are the key's fault, not the server's — say so explicitly.
        if resp.status_code in (401, 403):
            got = "no API key was sent" if not self.config.api_key else "the API key was rejected"
            return False, (
                f"Paritok API key check failed (HTTP {resp.status_code}: {got}) at "
                f"{self._base()}/test. Set a valid gpu_server.api_key in paritok.yaml "
                "(or the PARITOK_API_KEY env var)."
            )
        if resp.status_code >= 400:
            return False, (
                f"Paritok hosted endpoint returned HTTP {resp.status_code} from /test. "
                "Compression will pass through uncompressed."
            )

        # Separate JSON-parse failure from transport failure (don't mask one as the other).
        try:
            data = resp.json()
        except ValueError:
            return False, (
                f"Paritok /test returned a non-JSON response (HTTP {resp.status_code}). "
                "Compression will pass through uncompressed."
            )
        return bool(data.get("gpu_available", False)), str(data.get("message", ""))

    def _warn_invalid_key_once(self, status: int) -> None:
        """Print a one-time console warning when the hosted endpoint rejects our
        API key (HTTP 401/403). Requests keep working (passed through
        uncompressed) until the key is fixed; we warn once, not per request, so
        the proxy console isn't flooded during a session with a bad key."""
        if self._key_warned:
            return
        self._key_warned = True
        reason = "no API key configured" if not self.config.api_key else "the API key was rejected"
        print(
            f"[paritok] Paritok API key not valid (HTTP {status}: {reason}). "
            "Compression is passing through UNCOMPRESSED — set a valid "
            "gpu_server.api_key in paritok.yaml (or the PARITOK_API_KEY env var).",
            flush=True,
        )

    def is_available(self) -> bool:
        available, _ = self.check()
        return available

    def _base(self) -> str:
        return self.config.base_url.rstrip("/")
