<h1 align="center">Paritok</h1>

<p align="center"><b>The first open-source compression model trained specifically for coding agents.</b></p>
<p align="center">Trained on <b>45K real coding-agent trajectories</b>, Paritok understands the difference between a function signature and a debug line — so it keeps what matters and drops what doesn't. Fully compatible with <b>Claude Code, Cursor, OpenHands</b>, and any agent framework using standard message format.<br/><br/><b>~74% fewer tokens on typical workloads</b> (up to <b>95%</b> on heavy long-session traffic), cutting your input token bill by <b>up to 95%</b> on Claude / GPT — while <b>matching gpt-4.1-mini</b> on SWE-bench Verified at a fraction of the cost.</p>

<p align="center">
  <a href="https://huggingface.co/paritok/paritok-4b-v1">
    <img src="https://img.shields.io/badge/🤗%20Model-HuggingFace-yellow" alt="HF Model"/>
  </a>
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue" alt="License"/>
  </a>
  <img src="https://img.shields.io/badge/backbone-Qwen3--4B-purple" alt="Qwen3-4B"/>
  <img src="https://img.shields.io/badge/python-3.11+-blue" alt="Python"/>
</p>

<p align="center">
  <a href="#-highlights">Highlights</a> ·
  <a href="#-benchmark-swe-bench-verified">Benchmark</a> ·
  <a href="#-cost-impact">Cost</a> ·
  <a href="#-how-paritok-compares">Compare</a> ·
  <a href="#-quick-start">Quick Start</a> ·
  <a href="#-model-card">Model Card</a> ·
  <a href="#-training">Training</a> ·
  <a href="#-team">Team</a> ·
  <a href="#-citation">Citation</a>
</p>

---

## 📢 News

- **2026-07-14** &nbsp; **Paritok-4B-v1** released on Hugging Face Hub with full SWE-bench Verified end-to-end evaluation.
- **2026-06-25** &nbsp; Finished training. 45K teacher-distilled samples on the Qwen3-4B backbone.

---

## ✨ Highlights

- 🎨 **Code-native.** Trained end-to-end on real coding-agent trajectories (`file_read`, `bash_command`, `log_output`, ...). Paritok knows what an import statement is worth vs a debug line, so it protects function names, paths, and error strings while compressing.
- 🚀 **Up to 95% fewer tokens** (74% on typical workloads) — compresses each segment to **25.7%** of original; drop-aware long sessions push overall cuts to 95%. **2× harder than gpt-4.1-mini** (50.2% CR) and **2.4× harder than gpt-5** (61.9% CR).
- 🎯 **Retains 86.5% of full-context solve quality** on SWE-bench Verified — matching gpt-4.1-mini as compressor at **less than half the token spend**, and within 7pp of gpt-5 at **40% its context length**.
- 💰 **Up to 95% off your input token bill** (74% on typical workloads) at Claude Sonnet pricing. Long-session teams save **thousands per month** — see [Cost Impact](#-cost-impact).
- 🪶 **Small & self-hostable** — 4B LoRA adapter, bf16, runs on a single 24GB GPU. No SaaS, no lock-in, no per-token compressor fee.
- 🔓 **Fully open** — Apache 2.0 weights, reproducible data pipeline, real end-to-end SWE-bench numbers (no cherry-picking).

---

## 📊 Benchmark: SWE-bench Verified

Real end-to-end evaluation on **SWE-bench Verified**. An agent scaffold receives its context through each compressor, then attempts to resolve the issue. Primary metric is **quality retained** (solve rate normalized to the uncompressed baseline) — the fair way to compare compressors of different aggressiveness.

### Solve quality vs compression rate

| Context source            | **Quality retained** ¹ | Compression rate |
| ------------------------- | :--------------------: | :--------------: |
| Uncompressed baseline     |         100.0%         |      100.0%      |
| gpt-4.1-mini (compressor) |          85.6%         |       50.2%      |
| gpt-5 (compressor)        |          93.6%         |       61.9%      |
| **Paritok-4B-v1** ⭐      |       **86.5%**        |    **25.7%**     |

<sub>¹ Quality retained = compressor solve rate ÷ uncompressed baseline solve rate. Higher is better.</sub>

**What this says:**
- Paritok retains **86.5% of uncompressed solve quality** — matching gpt-4.1-mini's retention — while shipping **less than half the tokens**.
- Paritok comes within **7pp of gpt-5's retention** at **~40% of gpt-5's context length**.
- No open compressor comes close on the CR / quality frontier.

**Paritok cuts context by 74% while retaining 86.5% of uncompressed solve quality** — enough headroom to double or triple your monthly turn budget on the same API spend.

<sub>Benchmark numbers above use Paritok's baseline 25.7% CR (74% context cut) — the raw compressor performance in a controlled setting. Real production savings can go higher with drop-aware deployment; see [Cost Impact](#-cost-impact) for the 95% upper bound.</sub>

Evaluation used the standard [SWE-bench Verified](https://www.swebench.com/) harness with a public agent scaffold. Per-issue results and reproduction instructions will be published with the v2 release.

---

## 💰 Cost Impact

**Cut your input token bill by up to 95%.** Paritok compresses input; output is unchanged and left to your provider.

> **How to read the numbers below:** all tables use Paritok's **typical 74% saving** (median compression rate 25.7%). Long agent sessions with heavy tool-output and file-read repetition push savings up to **95%** — real bills often land higher than shown here.

At Claude Sonnet input pricing (`$3 / M input tokens`), **typical scenario (74% saving)**:

| Turn input size    | Uncompressed input cost | With Paritok |
| ------------------ | :---------------------: | :----------: |
| Short (8K)         |         $0.024          |    $0.006    |
| Typical (15K)      |         $0.045          |    $0.012    |
| Long session (30K) |         $0.090          |    $0.023    |

### Real project scenarios (typical 15K/turn workload, 74% saving)

| Scenario                                         | Uncompressed input | With Paritok |     Saved      |
| ------------------------------------------------ | :----------------: | :----------: | :------------: |
| Solo dev, 1-week prototype (5d × 300 turns)      |      $67.50        |    $17.34    |    **$50**     |
| Startup, 1-month project (20d × 400 turns)       |       $360         |      $92     |   **$268**    |
| 10-person team, 3-month project (60d × 10 × 500) |     $13,500        |    $3,468    |   **~$10K**    |

Deployment overhead pays for itself in **days**, not weeks — and there's no lock-in: it's your own 4B model on your own hardware.

---

## 🆚 How Paritok compares

|                                                | **Paritok-4B-v1** | LLMLingua-2  | gpt-4.1-mini prompt |
| ---------------------------------------------- | :---------------: | :----------: | :-----------------: |
| **Trained on real coding-agent trajectories**  |     ✅            |     ❌       |         ❌          |
| **Preserves function names / imports / paths** | ✅ (by design)    |   partial    |     partial         |
| **Compression rate** (lower = harder)          |  **25.7%** ⭐    |    ~40%      |       50.2%         |
| **SWE-bench Verified — quality retained**      |  **86.5%** ⭐    | not evaluated|       85.6%         |
| **Self-hostable open weights**                 |    Apache 2.0     |     MIT      |    closed API       |
| **Per-token compressor fee**                   |  zero (self-host) |  zero (open) |  pay-per-token      |

Paritok is the only entry trained end-to-end on real coding-agent trajectories — that's why it compresses **~2× harder than gpt-4.1-mini prompting** on the same task while keeping the same solve rate.

---

## 🚀 Quick Start

Paritok runs as a **middle layer between your agent and the LLM API**. It intercepts each request, compresses the context, and forwards it upstream — your agent doesn't change, it just points at Paritok.

```
Your Agent (Claude Code / Cursor / Codex)
  → builds request (tool results + history + tool schemas)
     ★ Paritok middleware compresses here ★
  → forwarded to Anthropic / OpenAI  (billed on the compressed tokens)
  ← response flows back unchanged; compressed refs expand on demand
```

### Fastest path (self-host, no clone needed)

Everything ships in the PyPI package — you do **not** need to `git clone` the repo. In a fresh environment (with [Ollama](https://ollama.com/download) installed):

```bash
pip install "paritok[proxy]"     # the middleware + CLI
paritok up                       # pulls the model if missing, then starts the proxy
```

`paritok up` is the pip-only, no-clone equivalent of a setup script: it checks Ollama, `ollama pull`s `paritok/paritok-4b-v1` and tags it as the local `paritok-4b-v1` if it isn't already there (~2.5GB, first run only), then serves on port 8080.

> **Leave that terminal running.** The proxy is a foreground server — every agent request flows through it, so it must stay up for the whole session. Closing the terminal (or Ctrl-C) stops compression. Open a **separate** terminal for the next step.

In the shell that launches your agent ([step 4](#4-point-your-agent-at-it)):

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080   # then start Claude Code / Cursor / …
```

No config file is needed — `paritok up` and `paritok proxy` run on built-in defaults. Run `paritok init` to drop a starter `paritok.yaml` only if you want to tweak settings. (Cloned the repo? `./deploy.sh` does the same pull + start.)

**q4 vs full precision.** `paritok up` uses the q4 model (`:latest`, ~2.5GB) by default. For full precision, run `paritok up --registry-model paritok/paritok-4b-v1:f16` (~8GB). If you already `ollama pull`ed either variant yourself, `up` **auto-detects** which one you have and uses it — no re-download. Prefer to run each step by hand? They're below.

### 1. Install

```bash
pip install "paritok[proxy]"
# or, from a clone of this repo:
pip install -e ".[proxy]"
```

### 2. Pick a backend — self-host **or** the GPU server

One boolean in [`paritok.yaml`](paritok.yaml) decides where compression runs:

```yaml
use_gpu_server: false   # ← the only switch that matters
```

**Option A — self-host** (`false`, default). Run the open 4B model on your own machine. No key, nothing leaves your box. Simplest is **Ollama**:

```bash
ollama pull paritok/paritok-4b-v1               # one-time, ~2.5GB
ollama cp   paritok/paritok-4b-v1 paritok-4b-v1  # tag it as the runtime name
```

The registry (pull) name is namespaced; the second line tags it as the bare `paritok-4b-v1` the config uses. Default [`paritok.yaml`](paritok.yaml) already points `local_model` at Ollama (`http://localhost:11434/v1`, `model: paritok-4b-v1`) — nothing else to set. (`./deploy.sh` does both steps for you.)

> Want full precision? Pull `paritok/paritok-4b-v1:f16` instead (~8GB, needs more RAM/VRAM). The default `:latest` is q4_K_M (~2.5GB) — the quantization the runtime is validated against.

<details>
<summary><b>Alternative: vLLM</b> (serves the HF LoRA adapter directly, no GGUF)</summary>

The Hugging Face weights ([`paritok/paritok-4b-v1`](https://huggingface.co/paritok/paritok-4b-v1)) are a **LoRA adapter** over `Qwen/Qwen3-4B-Instruct-2507`. vLLM can serve it as an OpenAI-compatible endpoint on a 24GB GPU:

```bash
pip install vllm
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --enable-lora \
  --lora-modules paritok-4b-v1=paritok/paritok-4b-v1 \
  --max-lora-rank 32 \
  --port 8000
```

Then in `paritok.yaml`, set `local_model.base_url: http://localhost:8000/v1` (keep `model: paritok-4b-v1`, the `--lora-modules` name).
</details>

> Whatever you name the served model, make `local_model.model` match it (or override once with `PARITOK_MODEL=<name>`).

**Option B — Paritok GPU server** (`true`). No GPU required. Create an API key at **[paritok.com](https://paritok.com) → dashboard → API keys**, then:

```yaml
use_gpu_server: true
gpu_server:
  api_key: "pk_live_..."   # or: export PARITOK_API_KEY=pk_live_...
```

### 3. Start the proxy

```bash
paritok proxy --port 8080 --config-file paritok.yaml
```

**Keep this terminal open** — the proxy must stay running for the whole session; run your agent from a separate terminal. On startup it checks the backend and warns (never aborts) if it can't reach one — e.g. the Ollama model isn't pulled, or the hosted server isn't reachable.

### 4. Point your agent at it

Set the base URL in the shell that launches your agent, **then start the agent**:

```bash
# macOS / Linux
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080   # Claude Code
export OPENAI_BASE_URL=http://127.0.0.1:8080      # Codex / OpenAI-style agents
```

```powershell
# Windows PowerShell
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:8080"
$env:OPENAI_BASE_URL    = "http://127.0.0.1:8080"
```

Keep your real provider API key set as usual (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`) — the proxy only rewrites the request body and forwards your headers upstream. That's it: Claude Code, Cursor, Codex — any agent that honors `BASE_URL` — now routes through Paritok. Compressed prompts go upstream; original responses come back unchanged.

### 5. Check it's working

The proxy exposes two endpoints:

```bash
curl http://127.0.0.1:8080/health   # {"status":"ok","version":"..."}
curl http://127.0.0.1:8080/stats    # live compression totals
```

`/stats` reports cumulative savings across the session:

```json
{
  "requests_processed": 42,
  "total_original_tokens": 512340,
  "total_compressed_tokens": 138221,
  "total_saved_tokens": 374119,
  "saved_percent": 73.0,
  "avg_saved_tokens_per_request": 8907.6,
  "items_compressed": 128,
  "total_tools_filtered": 61,
  "uptime_seconds": 613.2
}
```

`saved_percent` is the share of compressible tokens removed — expect **~74%** on typical coding-agent traffic.

### SDK mode (alternative)

Prefer to wrap your client directly in Python?

```python
import anthropic
import paritok

client = paritok.ParitokClient(anthropic.Anthropic())
resp = client.messages.create(
    model="claude-sonnet-4-20250514", max_tokens=4096, messages=[...]
)
print(resp._paritok_savings.saved_tokens, resp._paritok_savings.ratio)
```

### Just the raw model?

To load the LoRA adapter and compress a single `[SEG]` block yourself (no middleware), see [`examples/inference/basic.py`](examples/inference/basic.py).

---

## 🧩 Use Cases

**Paritok is most useful when:**
- Your AI coding agent (Claude Code / Cursor / Copilot / OpenHands / custom SDK) sends **> 5 000 tokens per turn**.
- You're paying per token to Anthropic / OpenAI / other API providers.
- You want lower per-turn latency (fewer input tokens = faster prefill).
- You can tolerate **~300 ms** of compression overhead per request.

**Paritok is less useful when:**
- Your context is already short (< 2 000 tokens).
- You need lossless compression (in that case, just don't compress).
- Your workflow is single-turn Q&A (context doesn't accumulate).

### How the middleware works

The [`paritok/`](paritok/) package is the middle layer. On every request the engine ([`paritok/middleware/wrapper.py`](paritok/middleware/wrapper.py)) runs four steps before forwarding upstream:

1. **Tool discovery** — 70+ tool schemas → top-K kept in full, the rest stubbed ([`pipelines/tool_discovery.py`](paritok/pipelines/tool_discovery.py)).
2. **Compress tool outputs** — each `tool_result` is compressed by the 4B model ([`pipelines/compress.py`](paritok/pipelines/compress.py)).
3. **Compress old history** — turns beyond the recent window are summarized once the context fills up.
4. **Inject virtual tools** — `expand_context` and `gateway_search_tools` ([`pipelines/virtual.py`](paritok/pipelines/virtual.py)) let the model pull back anything it needs.

**Never destructive.** Compressed content is tagged `[REF:id]`; if the LLM needs the original it calls the `expand_context` virtual tool and the middleware returns the full text locally. `gateway_search_tools` recovers any tool schema that discovery stubbed out.

Deploy it either way — the same package powers both:

```
Your Agent  ──►  Paritok middleware  ──►  Anthropic / OpenAI
   (raw)          (self-host or GPU server)   (billed on compressed)
```

Configure everything in [`paritok.yaml`](paritok.yaml); flip `use_gpu_server` to move between your own hardware and our hosted GPU endpoint without touching your agent.

---

## 📋 Model Card

| Property               | Value                                                                                    |
| ---------------------- | ---------------------------------------------------------------------------------------- |
| **Base model**         | [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)        |
| **Adapter type**       | LoRA, r=32, α=64, dropout=0.0                                                            |
| **Target modules**     | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`                          |
| **Training steps**     | 2000 (selected from a 5-checkpoint sweep, best on SWE-bench Verified subset)             |
| **Training precision** | bf16                                                                                     |
| **Effective batch**    | 32 (per_device=2 × grad_accum=16)                                                        |
| **Learning rate**      | 1e-5, linear decay, 10% warmup                                                           |
| **Optimizer**          | AdamW (8-bit)                                                                            |
| **Max seq length**     | 16 384                                                                                   |
| **Dataset size**       | 45 000 samples across `file_read`, `bash_command`, `log_output`, etc.                    |
| **Teacher**            | gpt-4.1-mini                                                                             |
| **Weights**            | [🤗 HF Hub](https://huggingface.co/paritok/paritok-4b-v1)                      |
| **License**            | Apache 2.0 (weights); base model under its own Qwen license                              |

Full training config: [`training/configs/sft_config_qwen3_4b.yaml`](training/configs/sft_config_qwen3_4b.yaml).

---

## 🎓 Training

### Reproduce from scratch

High-level pipeline; see [`training/`](training/) and [`data_pipeline/`](data_pipeline/) for the actual scripts.

```bash
# 1. Prepare data (regenerate pools from agent-trajectory dumps)
python data_pipeline/extract/extract_file_read_pool.py --n 10000
python data_pipeline/extract/extract_other_kinds_pool.py

# 2. Distill via teacher (requires OPENAI_API_KEY, ~$300 in API cost)
python data_pipeline/compress/compress_pool_file_read.py
python data_pipeline/compress/compress_pool_other.py

# 3. Train SFT (2× A100 80GB or 1× H100 80GB, ~5 hours)
bash deploy_sft_4b.sh
```

Or on a fresh RunPod / Lambda / Modal pod:

```bash
RUNPOD_HOST=<host> RUNPOD_PORT=<port> ./deploy_sft_4b.sh
```

### Pipeline summary

1. **Data collection** — 100k+ raw agent-trajectory turns from open-source SWE-bench-style trajectory dumps.
2. **Segmentation** — split each turn into `[SEG]` blocks by kind (`file_read`, `bash_command`, `log_output`, ...).
3. **Teacher distillation** — gpt-4.1-mini produces the target compression for each segment.
4. **Filter & rebalance** — drop mal-formatted teacher outputs, up-sample rare kinds.
5. **SFT** — LoRA on Qwen3-4B-Instruct-2507, imitation loss on the teacher's compressed reply.
6. **Checkpoint selection** — evaluate on SWE-bench Verified end-to-end, pick step 2000.

---

## 🗺️ Roadmap

Paritok-4B-v1 is our first release. What's next:

- 🎯 **Paritok-4B-v2.** Next-generation training pipeline pushing compression to **under 20%** while closing the gap to uncompressed solve rate. Target: **up to +15pp identifier retention** at even tighter compression.
- 📈 **Frontier-scale backbones.** Larger models (10B+ parameters) for multi-day Claude Code / Cursor sessions with **100K+ token histories** and heavy multi-file workflows.
- 🌍 **Multi-language expansion.** First-class support for TypeScript, Rust, Go, Java, C++, Kotlin — v1 is Python-heavy but the architecture is language-agnostic.
- 🔌 **Native integrations.** Drop-in `mcp add paritok` plugin for Claude Code and Cursor. (Or route to the hosted GPU endpoint with `use_gpu_server: true`.)
- ⚙️ **Adaptive compression.** Per-segment auto-selection of compression aggressiveness based on age, kind, and downstream intent — no manual tuning, no level knobs.

Follow the [🤗 model discussions](https://huggingface.co/paritok/paritok-4b-v1/discussions) or star the repo for release notifications.

---

## 👥 Team

Paritok is built by two engineers — no big lab, no external funding, just months of GPU budget and eval iteration.

- **Jiayu Shi** — training, modeling, reward design, data pipeline.
- **Luzhuo Chen** — evaluation, deployment, product, data pipeline.

We ship on our own budget and share every result transparently. Paritok-4B-v1 is our first release; v2 is in training.

Reach us: [paritok9@gmail.com](mailto:paritok9@gmail.com) · X [@paritok](https://x.com/paritok)

---

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@misc{paritok2026,
  author       = {Paritok Team},
  title        = {Paritok: Fine-tuned Compression for AI Coding-Agent Context},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/Paritok-official/paritok-4b-v1}},
}
```

Please also cite the base model:

```bibtex
@misc{qwen3-4b-instruct,
  author       = {Qwen Team},
  title        = {Qwen3-4B-Instruct},
  year         = {2025},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507}},
}
```

---

## 📄 License

Apache 2.0 — see [LICENSE](./LICENSE).

The base model, Qwen3-4B-Instruct-2507, is released under its own license. Please review it before commercial deployment.

---

## 💬 Community & Support

- 🐛 **Bug reports & feature requests** → [GitHub Issues](https://github.com/Paritok-official/paritok-4b-v1/issues)
- 💭 **Discussion** → [🤗 HF Model discussions](https://huggingface.co/paritok/paritok-4b-v1/discussions)
- 📧 **Contact** → [paritok9@gmail.com](mailto:paritok9@gmail.com)
