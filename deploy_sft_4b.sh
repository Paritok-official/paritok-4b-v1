#!/usr/bin/env bash
# One-shot RunPod deployment for Paritok SFT (Qwen3-4B-Instruct-2507).
#
# Usage:
#   1. Edit RUNPOD_HOST / RUNPOD_PORT below (from your pod's SSH connect info),
#      OR set them as env vars:  RUNPOD_HOST=... RUNPOD_PORT=... ./deploy_sft_3b.sh
#   2. Run from the project root:   ./deploy_sft_3b.sh
#
# What it does:
#   [1/4] verify SSH + GPU
#   [2/4] prepare /workspace/paritok/ layout on pod
#   [3/4] rsync the 7 required files (~370 MB)
#   [4/4] install unsloth + training deps (idempotent), launch training in tmux
#
# After it finishes, training is running detached in a tmux session named "sft4b".
# Monitor it via the printed instructions at the end.

set -euo pipefail

# ─────────────────────── EDIT THESE ───────────────────────
RUNPOD_HOST="${RUNPOD_HOST:-PUT_RUNPOD_HOST_HERE}"   # e.g. 213.181.99.4 or ssh.runpod.io
RUNPOD_PORT="${RUNPOD_PORT:-PUT_RUNPOD_PORT_HERE}"   # ssh port shown in pod console
RUNPOD_USER="${RUNPOD_USER:-root}"
REMOTE_ROOT="${REMOTE_ROOT:-/workspace/paritok}"
# ──────────────────────────────────────────────────────────

if [[ "${RUNPOD_HOST}" == PUT_* || "${RUNPOD_PORT}" == PUT_* ]]; then
  echo "ERROR: set RUNPOD_HOST and RUNPOD_PORT (env vars or edit the script)." >&2
  exit 2
fi

# Qwen3-4B-Instruct-2507 may or may not require auth depending on your HF account.
# HF_TOKEN is OPTIONAL — if set, we'll auth on the pod;
# if not set, we'll try without (works if the model isn't gated for your account).
# Get a token at https://huggingface.co/settings/tokens (Read type).
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[pre-flight] No HF_TOKEN given — trying without auth (will fail at [smoke] if gated)." >&2
fi

SSH_BASE=(ssh -p "${RUNPOD_PORT}" "${RUNPOD_USER}@${RUNPOD_HOST}")
RSYNC_SSH="ssh -p ${RUNPOD_PORT}"

LOCAL_FILES=(
  "training/train_sft_instruct.py"
  "training/requirements_training.txt"
  "training/configs/sft_config_qwen3_4b.yaml"
  "update/file_read_compressed_all10k_merged_20260625.jsonl"
  "update/other_compressed_all_per_kind.jsonl"
  "update/system_prompt.txt"
  "update/system_prompt_other.txt"
  "update/system_prompt_qwen3.txt"
  "update/system_prompt_other_qwen3.txt"
)

echo "=== Pre-flight: checking local files ==="
for f in "${LOCAL_FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing local file: $f" >&2
    echo "Run this script from the paritok_reborn project root." >&2
    exit 1
  fi
done
echo "All 7 files present."

echo
echo "=== [1/4] Verifying SSH + GPU ==="
"${SSH_BASE[@]}" "echo ok && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"

echo
echo "=== [2/4] Preparing remote layout at ${REMOTE_ROOT} (and installing rsync if missing) ==="
"${SSH_BASE[@]}" "mkdir -p ${REMOTE_ROOT}/training/configs ${REMOTE_ROOT}/update ${REMOTE_ROOT}/training/models && \
  MISSING=''; \
  for pkg in rsync tmux; do \
    if ! command -v \$pkg >/dev/null 2>&1; then MISSING=\"\$MISSING \$pkg\"; fi; \
  done; \
  if [ -n \"\$MISSING\" ]; then \
    echo \"[2/4] installing missing tools:\$MISSING\"; \
    apt-get update -qq && apt-get install -y -qq \$MISSING >/dev/null; \
  else \
    echo '[2/4] rsync + tmux already present.'; \
  fi"

echo
echo "=== [3/4] Uploading 7 files (~370 MB; uses rsync, resumable) ==="
rsync -avzhP --no-owner --no-group -e "${RSYNC_SSH}" \
  training/train_sft_instruct.py \
  training/requirements_training.txt \
  "${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_ROOT}/training/"

rsync -avzhP --no-owner --no-group -e "${RSYNC_SSH}" \
  training/configs/sft_config_qwen3_4b.yaml \
  "${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_ROOT}/training/configs/"

rsync -avzhP --no-owner --no-group -e "${RSYNC_SSH}" \
  update/file_read_compressed_all10k_merged_20260625.jsonl \
  update/other_compressed_all_per_kind.jsonl \
  update/system_prompt.txt \
  update/system_prompt_other.txt \
  update/system_prompt_qwen3.txt \
  update/system_prompt_other_qwen3.txt \
  "${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_ROOT}/update/"

echo
echo "=== [4/4] Installing deps + HF login + launching training in tmux ==="
"${SSH_BASE[@]}" "HF_TOKEN='${HF_TOKEN:-}' bash -s" <<'REMOTE_EOF'
set -euo pipefail
cd /workspace/paritok

# RunPod PyTorch 2.8 template enforces PEP 668 (externally-managed-environment).
export PIP_BREAK_SYSTEM_PACKAGES=1

# ─── Pin torch to 2.8 so we can use prebuilt flash-attn wheels ───
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "missing")
if [[ "$TORCH_VER" != 2.8.* ]]; then
  echo "[install] Pinning torch to 2.8.0+cu126 (was: $TORCH_VER)..."
  pip install --no-cache-dir "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu126
else
  echo "[install] torch 2.8 already present."
fi

# ─── Install prebuilt flash-attn matching torch 2.8 ───
if ! python -c "import flash_attn; from flash_attn import flash_attn_func" 2>/dev/null; then
  ABI=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
  PY_TAG=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
  WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1%2Bcu12torch2.8cxx11abi${ABI}-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"
  echo "[install] Installing flash-attn 2.8.3 wheel (cxx11abi=$ABI, py=$PY_TAG)..."
  pip install --no-cache-dir --no-deps "$WHEEL"
  pip install --no-cache-dir einops
  python -c "from flash_attn import flash_attn_func; from flash_attn.bert_padding import pad_input; print('[install] flash-attn + bert_padding import OK')"
fi

# ─── Install training deps (no unsloth) ───
if ! python -c "import trl, peft, bitsandbytes, transformers" 2>/dev/null; then
  echo "[install] installing training deps (2-3 min on first run)..."
  echo "torch==2.8.0" > /tmp/pip-constraints.txt
  pip install --no-cache-dir -c /tmp/pip-constraints.txt -r training/requirements_training.txt
else
  echo "[install] core deps already present; skipping pip."
fi

# ─── Install Liger Kernel (fused Triton kernels for RoPE/RMSNorm/SwiGLU) ───
if ! python -c "import liger_kernel" 2>/dev/null; then
  echo "[install] installing liger-kernel (~30 sec)..."
  pip install --no-cache-dir liger-kernel
fi

# ─── hf_transfer for fast HF downloads (avoids the ValueError when template sets HF_HUB_ENABLE_HF_TRANSFER=1) ───
if ! python -c "import hf_transfer" 2>/dev/null; then
  pip install --no-cache-dir hf_transfer
fi

# Remove RunPod PyTorch template's pre-installed torchaudio if its .so won't load.
if python -c "import torchaudio" 2>&1 | grep -qi "could not load"; then
  echo "[install] pod's torchaudio is broken (ABI mismatch); uninstalling..."
  pip uninstall -y torchaudio >/dev/null
fi

# HF login — only if HF_TOKEN was provided. Otherwise we'll see if the model
# is accessible without auth (some HF accounts get auto-grant on Qwen3-4B).
if [ -n "${HF_TOKEN:-}" ]; then
  echo "[install] writing HF token for gated model access..."
  mkdir -p ~/.cache/huggingface
  echo -n "$HF_TOKEN" > ~/.cache/huggingface/token
  export HF_TOKEN="$HF_TOKEN"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
else
  echo "[install] no HF_TOKEN provided — will try anonymous download."
fi

# Sanity: tokenizer + chat template work on the chosen (gated) base model
python - <<'PYCHECK'
import sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", trust_remote_code=True)
msgs = [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}, {"role": "assistant", "content": "z"}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
assert "<|im_start|>assistant" in text, "chat template missing assistant header"
print("[smoke] tokenizer + gated-model auth + chat template OK")
PYCHECK

# Replace any prior tmux session of the same name (idempotent re-launch)
tmux kill-session -t sft4b 2>/dev/null || true
mkdir -p training/models

tmux new-session -d -s sft4b "cd /workspace/paritok && \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -u training/train_sft_instruct.py \
    --config training/configs/sft_config_qwen3_4b.yaml \
    2>&1 | tee training/models/sft_4b_run1.log"

sleep 2
echo
echo "[launch] tmux session 'sft4b' started. First minutes will download Qwen3-4B weights (~8GB)."
REMOTE_EOF

cat <<EOF

─────────────────────────────────────────────────────────────
Done. Training is running on ${RUNPOD_HOST}:${RUNPOD_PORT}.

Watch live:
  ssh -p ${RUNPOD_PORT} ${RUNPOD_USER}@${RUNPOD_HOST}
  tmux attach -t sft4b        # Ctrl-B then D to detach without killing

Tail log without attaching:
  ssh -p ${RUNPOD_PORT} ${RUNPOD_USER}@${RUNPOD_HOST} \\
    'tail -f /workspace/paritok/training/models/sft_4b_run1.log'

Check GPU usage:
  ssh -p ${RUNPOD_PORT} ${RUNPOD_USER}@${RUNPOD_HOST} 'nvidia-smi'

Download trained LoRA adapter when done (~200-300 MB):
  scp -P ${RUNPOD_PORT} -r \\
    ${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_ROOT}/training/models/sft-qwen3-4b \\
    ./training/models/
─────────────────────────────────────────────────────────────
EOF
