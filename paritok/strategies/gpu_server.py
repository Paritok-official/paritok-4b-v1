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

from paritok.config import GpuServerConfig

logger = logging.getLogger("paritok.gpu_server")


class GpuServerStrategy:
    name = "gpu_server"

    def __init__(self, config: GpuServerConfig):
        self.config = config

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
        headers = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        try:
            resp = httpx.post(
                f"{self._base()}/compress",
                json=payload,
                headers=headers,
                timeout=self.config.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:  # noqa: BLE001 — any failure degrades to passthrough
            logger.warning(
                "GPU server compression unavailable (%s); passing content through "
                "uncompressed.", e,
            )
            return content

        if not data.get("gpu_available", False):
            # Endpoint is up but the GPU backend is offline — it echoed the
            # original back. Keep the original; no compression happened.
            return content
        compressed = data.get("compressed")
        return compressed if isinstance(compressed, str) else content

    def check(self) -> tuple[bool, str]:
        """Probe {base_url}/test. Returns (available, message).

        message carries the server's explanation (e.g. why the GPU backend is
        offline) so the proxy can surface it to the user at startup.
        """
        try:
            import httpx

            resp = httpx.get(f"{self._base()}/test", timeout=10.0)
            data = resp.json()
            return bool(data.get("gpu_available", False)), str(data.get("message", ""))
        except Exception as e:  # noqa: BLE001
            return False, (
                f"Could not reach the Paritok hosted endpoint at {self._base()} "
                f"({e}). Compression will pass through uncompressed."
            )

    def is_available(self) -> bool:
        available, _ = self.check()
        return available

    def _base(self) -> str:
        return self.config.base_url.rstrip("/")
