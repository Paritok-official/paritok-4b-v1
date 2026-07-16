"""Paritok CLI: compress agent context from the command line."""

import os

import click

from paritok.config import ParitokConfig

# Starter config written by `paritok init`. Every value here is also the built-in
# default, so this file is optional — it exists only to make customization easy
# without cloning the repo.
_STARTER_YAML = """\
# Backend switch — the one setting most people touch.
#   false → self-host the open Paritok model locally (via Ollama).
#   true  → use the Paritok GPU server (needs an API key from https://paritok.com).
use_gpu_server: false

# Paritok GPU server — only used when use_gpu_server: true.
gpu_server:
  base_url: https://www.paritok.com/api   # POST /compress, GET /test
  model: paritok-4b-v1
  api_key: ""            # paste your key, or set env PARITOK_API_KEY
  timeout: 60.0

# Self-hosted local model — only used when use_gpu_server: false.
local_model:
  base_url: http://localhost:11434/v1
  # ollama pull paritok/paritok-4b-v1 && ollama cp paritok/paritok-4b-v1 paritok-4b-v1
  model: paritok-4b-v1
  temperature: 0
  timeout: 300.0

compression:
  min_tokens: 512          # skip compression below this
  max_tokens: 50000        # skip compression above this
  refusal_threshold: 0.05  # must save at least 5% or keep the original

history:
  enabled: true
  keep_recent_turns: 4     # keep last N turns intact
  context_threshold: 0.8   # compress old turns when >80% of the window is used
  context_window: 200000

tool_discovery:
  strategy: relevance      # "relevance" | "passthrough"
  top_k: 5                 # keep top-K tool schemas, stub the rest

# Per-compression debug trace. Turn on to log every original→compressed pair.
trace:
  enabled: false
  path: compress_trace.jsonl

shadow_storage: memory     # "memory" | "redis"
"""

_DEFAULT_CONFIG_NAME = "paritok.yaml"


@click.group()
@click.version_option()
def main():
    """Paritok: Open-source agent context compression.

    Compress agent context via local Ollama model, transparently.
    """
    pass


@main.command()
@click.option("--force", is_flag=True, help="Overwrite an existing paritok.yaml.")
@click.option("--path", default=_DEFAULT_CONFIG_NAME, show_default=True,
              help="Where to write the config.")
def init(force, path):
    """Write a starter paritok.yaml into the current folder.

    Optional — `paritok proxy` already runs on built-in defaults with no config
    file. Use this only to customize (backend, ports, thresholds) without
    cloning the repo.
    """
    if os.path.exists(path) and not force:
        raise click.ClickException(
            f"{path} already exists. Re-run with --force to overwrite it."
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write(_STARTER_YAML)
    click.echo(f"Wrote {path}. Next:")
    click.echo("  # self-host the model (skip if you set use_gpu_server: true)")
    click.echo("  ollama pull paritok/paritok-4b-v1 && ollama cp paritok/paritok-4b-v1 paritok-4b-v1")
    click.echo("  paritok proxy")


@main.command()
def config():
    """Show current configuration."""
    cfg = ParitokConfig.load()
    backend = "GPU server (paritok.com)" if cfg.use_gpu_server else "self-hosted (local)"
    click.echo(f"Backend: {backend}  (use_gpu_server: {cfg.use_gpu_server})")
    click.echo("\nCompression:")
    click.echo(f"  min_tokens: {cfg.compression.min_tokens}")
    click.echo(f"  max_tokens: {cfg.compression.max_tokens}")
    click.echo(f"  refusal_threshold: {cfg.compression.refusal_threshold}")
    if cfg.use_gpu_server:
        click.echo("\nGPU Server:")
        click.echo(f"  base_url: {cfg.gpu_server.base_url}")
        click.echo(f"  model: {cfg.gpu_server.model}")
        click.echo(f"  api_key: {'set' if cfg.gpu_server.api_key else 'MISSING — set PARITOK_API_KEY'}")
    else:
        click.echo("\nLocal Model (Ollama):")
        click.echo(f"  base_url: {cfg.local_model.base_url}")
        click.echo(f"  model: {cfg.local_model.model}")
        click.echo(f"  temperature: {cfg.local_model.temperature}")
    click.echo("\nTool Discovery:")
    click.echo(f"  strategy: {cfg.tool_discovery.strategy}")
    click.echo(f"  top_k: {cfg.tool_discovery.top_k}")
    click.echo(f"\nShadow Storage: {cfg.shadow_storage}")


@main.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8080, type=int, help="Port to listen on")
@click.option("--anthropic-url", default="https://api.anthropic.com",
              help="Upstream Anthropic API URL")
@click.option("--openai-url", default="https://api.openai.com",
              help="Upstream OpenAI API URL")
@click.option("--config-file", default=None, type=click.Path(exists=True),
              help="Path to YAML config file")
@click.option("--log-level", default="info",
              type=click.Choice(["debug", "info", "warning", "error"]))
def proxy(host, port, anthropic_url, openai_url, config_file, log_level):
    """Start the HTTP proxy server.

    Sits between your AI agent and the LLM API, compressing context
    via the local Ollama model.

    Runs on built-in defaults with no config file. If a paritok.yaml is present
    in the current folder it's picked up automatically (create one with
    `paritok init`).

    Examples:
        paritok proxy
        paritok proxy --port 9000
        paritok proxy --config-file paritok.yaml

    Then configure your agent:
        export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
    """
    from paritok.proxy.server import run_proxy

    # Fall back to ./paritok.yaml when no explicit config was passed.
    if config_file is None and os.path.exists(_DEFAULT_CONFIG_NAME):
        config_file = _DEFAULT_CONFIG_NAME
        click.echo(f"Using {_DEFAULT_CONFIG_NAME} from the current folder.")

    run_proxy(
        host=host,
        port=port,
        anthropic_base_url=anthropic_url,
        openai_base_url=openai_url,
        config_path=config_file,
        log_level=log_level,
    )


@main.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8080, type=int, help="Port to listen on")
@click.option("--anthropic-url", default="https://api.anthropic.com",
              help="Upstream Anthropic API URL")
@click.option("--openai-url", default="https://api.openai.com",
              help="Upstream OpenAI API URL")
@click.option("--config-file", default=None, type=click.Path(exists=True),
              help="Path to YAML config file")
@click.option("--registry-model", default="paritok/paritok-4b-v1", show_default=True,
              help="Ollama registry name to pull the model from.")
@click.option("--log-level", default="info",
              type=click.Choice(["debug", "info", "warning", "error"]))
def up(host, port, anthropic_url, openai_url, config_file, registry_model, log_level):
    """Pull the model if missing, then start the proxy - one command.

    The pip-only, no-clone equivalent of deploy.sh. On the self-host backend it
    makes sure Ollama has the model (pulls + tags it if not), then serves. With
    use_gpu_server: true it just starts the proxy.

    Leave this running - the proxy must stay up for the whole agent session.
    If you already pulled a variant yourself (q4 or :f16), it is auto-detected.

    \b
    Examples:
      pip install "paritok[proxy]"
      paritok up                                              # q4, ~2.5GB
      paritok up --registry-model paritok/paritok-4b-v1:f16   # full precision
      # then, in a separate shell that launches your agent:
      export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
    """
    if config_file is None and os.path.exists(_DEFAULT_CONFIG_NAME):
        config_file = _DEFAULT_CONFIG_NAME
        click.echo(f"Using {_DEFAULT_CONFIG_NAME} from the current folder.")

    cfg = ParitokConfig.load(config_file) if config_file else ParitokConfig.load()
    if not cfg.use_gpu_server:
        ok = _ensure_ollama_model(cfg.local_model.model, registry_model,
                                  cfg.local_model.base_url)
        if not ok:
            raise click.ClickException(
                "Could not prepare the local model. Fix the issue above, or "
                "switch to the hosted backend (use_gpu_server: true)."
            )

    from paritok.proxy.server import run_proxy
    run_proxy(
        host=host,
        port=port,
        anthropic_base_url=anthropic_url,
        openai_base_url=openai_url,
        config_path=config_file,
        log_level=log_level,
    )


def _ensure_ollama_model(local_model: str, registry_model: str, base_url: str) -> bool:
    """Make sure Ollama is up and `local_model` is available; pull + tag if not.

    Returns True on success. Prints actionable guidance and returns False on any
    unrecoverable problem (Ollama not installed, server unreachable, pull fails).
    """
    import shutil
    import subprocess
    import time

    if shutil.which("ollama") is None:
        click.echo("Ollama is not installed. Install it from "
                   "https://ollama.com/download and re-run `paritok up`.", err=True)
        return False

    # Ollama's native API lives at the base without the OpenAI-compat /v1 suffix.
    api_base = base_url.rstrip("/")
    if api_base.endswith("/v1"):
        api_base = api_base[:-3]

    def _tags():
        try:
            import httpx
            r = httpx.get(f"{api_base}/api/tags", timeout=5.0)
            if r.status_code == 200:
                return [m.get("name", "") for m in r.json().get("models", [])]
        except Exception:
            return None
        return None

    tags = _tags()
    if tags is None:
        click.echo("Starting 'ollama serve' in the background...")
        try:
            subprocess.Popen(["ollama", "serve"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:  # noqa: BLE001
            click.echo(f"Could not launch 'ollama serve' ({e}). "
                       f"Start it manually and re-run.", err=True)
            return False
        for _ in range(30):
            time.sleep(1)
            tags = _tags()
            if tags is not None:
                break
        if tags is None:
            click.echo("Ollama did not become reachable. Run 'ollama serve' "
                       "manually and re-run `paritok up`.", err=True)
            return False

    def _present(name: str) -> bool:
        # A bare name (no ':') matches its ':latest'; a tagged name is exact.
        if ":" in name:
            return name in tags
        return any(t == name or t == f"{name}:latest" for t in tags)

    # 1. Bare runtime tag already there — run whatever variant it points at.
    if _present(local_model):
        click.echo(f"Model '{local_model}' already present.")
        return True

    # 2. Honor a variant the user pulled themselves (q4 ':latest' or ':f16') by
    #    tagging it as the bare runtime name the config expects. If they asked
    #    for a specific tag via --registry-model, that one is tried first.
    repo = registry_model.split(":")[0]
    for cand in (registry_model, f"{repo}:latest", f"{repo}:f16"):
        if _present(cand):
            click.echo(f"Found '{cand}' — tagging it as '{local_model}'.")
            if subprocess.run(["ollama", "cp", cand, local_model]).returncode != 0:
                click.echo(f"'ollama cp {cand} {local_model}' failed.", err=True)
                return False
            return True

    # 3. Nothing local — pull the requested registry model, then tag it.
    click.echo(f"Pulling '{registry_model}' (first run downloads ~2.5GB)...")
    if subprocess.run(["ollama", "pull", registry_model]).returncode != 0:
        click.echo(f"'ollama pull {registry_model}' failed.", err=True)
        return False
    if registry_model != local_model:
        if subprocess.run(["ollama", "cp", registry_model, local_model]).returncode != 0:
            click.echo(f"'ollama cp {registry_model} {local_model}' failed.", err=True)
            return False
        click.echo(f"Tagged '{registry_model}' as '{local_model}'.")
    return True


if __name__ == "__main__":
    main()
