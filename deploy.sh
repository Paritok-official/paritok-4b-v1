#!/usr/bin/env bash
# One-command runtime setup for the Paritok compression proxy (self-hosted).
#
# What it does:
#   1. verify Ollama is installed (and running; starts `ollama serve` if not)
#   2. pull the compression model if it isn't already present
#   3. install paritok[proxy] if it isn't importable
#   4. start the proxy, printing the one env var your agent needs
#
# Usage:
#   ./deploy.sh                 # port 8080, model paritok-4b-v1, ./paritok.yaml
#   PARITOK_PORT=9000 ./deploy.sh
#   PARITOK_MODEL=paritok-4b-v1 ./deploy.sh
#

set -euo pipefail

MODEL="${PARITOK_MODEL:-paritok-4b-v1}"                          # local runtime tag (paritok.yaml)
REGISTRY_MODEL="${PARITOK_REGISTRY_MODEL:-paritok/paritok-4b-v1}"  # public Ollama pull name
PORT="${PARITOK_PORT:-8080}"
CONFIG="${PARITOK_CONFIG:-paritok.yaml}"
OLLAMA_URL="${PARITOK_OLLAMA_URL:-http://localhost:11434}"

echo "=== [1/4] Checking Ollama ==="
if ! command -v ollama >/dev/null 2>&1; then
  echo "ERROR: Ollama is not installed." >&2
  echo "  Install it from https://ollama.com/download and re-run this script." >&2
  echo "  (Or use the hosted GPU server instead: set use_gpu_server: true in $CONFIG.)" >&2
  exit 1
fi

if ! curl -sf "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
  echo "Ollama server not responding — starting 'ollama serve' in the background..."
  ollama serve >/tmp/paritok-ollama.log 2>&1 &
  for _ in $(seq 1 30); do
    curl -sf "${OLLAMA_URL}/api/tags" >/dev/null 2>&1 && break
    sleep 1
  done
  curl -sf "${OLLAMA_URL}/api/tags" >/dev/null 2>&1 \
    || { echo "ERROR: could not start Ollama (see /tmp/paritok-ollama.log)." >&2; exit 1; }
fi
echo "Ollama OK at ${OLLAMA_URL}."

echo
echo "=== [2/4] Checking model '${MODEL}' ==="
if ollama list | awk '{print $1}' | grep -qx "${MODEL}\(:latest\)\?"; then
  echo "Model '${MODEL}' already present."
else
  echo "Pulling '${REGISTRY_MODEL}' (first run downloads ~2.5GB)..."
  ollama pull "${REGISTRY_MODEL}"
  # Alias the pulled model to the bare local tag the runtime config expects,
  # so paritok.yaml can keep the clean 'paritok-4b-v1' name.
  if [ "${REGISTRY_MODEL}" != "${MODEL}" ]; then
    ollama cp "${REGISTRY_MODEL}" "${MODEL}"
    echo "Tagged '${REGISTRY_MODEL}' as '${MODEL}'."
  fi
fi

echo
echo "=== [3/4] Checking paritok install ==="
if python -c "import paritok" >/dev/null 2>&1; then
  echo "paritok already importable."
else
  echo "Installing paritok[proxy]..."
  pip install "paritok[proxy]"
fi

echo
echo "=== [4/4] Starting proxy on port ${PORT} ==="
echo "---------------------------------------------------------------"
echo "In the shell where you launch your agent, set:"
echo "    export ANTHROPIC_BASE_URL=http://127.0.0.1:${PORT}   # Claude Code"
echo "    export OPENAI_BASE_URL=http://127.0.0.1:${PORT}      # Codex / OpenAI agents"
echo "Live stats:  http://127.0.0.1:${PORT}/stats"
echo "---------------------------------------------------------------"

CONFIG_ARG=()
[ -f "${CONFIG}" ] && CONFIG_ARG=(--config-file "${CONFIG}")
exec paritok proxy --port "${PORT}" "${CONFIG_ARG[@]}"
