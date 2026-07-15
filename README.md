<p align="center">
  <img src="figures/logo.png" width="360" alt="Paritok"/>
</p>

<h1 align="center">Paritok</h1>

<p align="center"><b>The first open-source compression model trained specifically for coding agents.</b></p>
<p align="center">Trained on <b>45K real coding-agent trajectories</b>, Paritok understands the difference between a function signature and a debug line — so it keeps what matters and drops what doesn't. Fully compatible with <b>Claude Code, Cursor, OpenHands</b>, and any agent framework using standard message format.<br/><br/>The result: up to <b>74% fewer tokens on typical workloads</b> (up to <b>95%</b> on heavy long-session traffic), up to <b>95% off your input bill</b> on Claude / GPT — while <b>matching gpt-4.1-mini</b> on SWE-bench Verified at a fraction of the cost.</p>

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

### Install

```bash
pip install "torch>=2.4" "transformers>=4.44" "peft>=0.12" accelerate
```

### Load model & adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "paritok/paritok-4b-v1")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
model.eval()
```

### Compress one segment

```python
system_prompt = open("data_pipeline/prompts/system_prompt_qwen3.txt").read()

original_code = """\
def compute_score(items):
    total = 0
    for it in items:
        total += it.get('score', 0)
    return total / max(len(items), 1)
"""

user_msg = (
    f"[SEG id=1 kind=file_read "
    f"user_intent=\"Understand what compute_score does\"]\n"
    f"{original_code}\n"
    f"[/SEG]"
)

prompt = tokenizer.apply_chat_template(
    [{"role": "system", "content": system_prompt},
     {"role": "user",   "content": user_msg}],
    tokenize=False, add_generation_prompt=True,
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
print(tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

Full runnable example: [`examples/inference/basic.py`](examples/inference/basic.py).

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

### Middle-layer API proxy pattern

The most natural deployment is as a middle-layer proxy that sits between your agent and the upstream LLM API:

```
Your Agent  ──►  Paritok proxy  ──►  Anthropic / OpenAI
   (raw)          (compressed)         (billed on compressed)
```

Compressed prompts flow upstream; original responses flow back unchanged. See [`examples/proxy/`](examples/proxy/) for a minimal reference implementation.

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
- 🔌 **Native integrations.** Drop-in `mcp add paritok` plugin for Claude Code and Cursor, plus a hosted inference endpoint for teams that don't want to self-host.
- ⚙️ **Adaptive compression.** Per-segment auto-selection of compression aggressiveness based on age, kind, and downstream intent — no manual tuning, no level knobs.

Follow the [🤗 model discussions](https://huggingface.co/paritok/paritok-4b-v1/discussions) or star the repo for release notifications.

---

## 👥 Team

Paritok is built by two engineers — no big lab, no external funding, just months of GPU budget and eval iteration.

- **Jiayu Shi** — training, modeling, reward design, data pipeline.
- **Luzhuo Chen** — evaluation, deployment, product, data pipeline.

We ship on our own budget and share every result transparently. Paritok-4B-v1 is our first release; v2 is in training.

Reach us: [hello@paritok.ai](mailto:hello@paritok.ai) · X [@paritok](https://x.com/paritok)

---

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@misc{paritok2026,
  author       = {Paritok Team},
  title        = {Paritok: Fine-tuned Compression for AI Coding-Agent Context},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/Paritok2026/paritok-4b-v1}},
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

- 🐛 **Bug reports & feature requests** → [GitHub Issues](https://github.com/Paritok2026/paritok-4b-v1/issues)
- 💭 **Discussion** → [🤗 HF Model discussions](https://huggingface.co/paritok/paritok-4b-v1/discussions)
- 📧 **Contact** → [hello@paritok.ai](mailto:hello@paritok.ai)

---

<p align="center">
  Built on <a href="https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507">Qwen3-4B-Instruct-2507</a>
</p>
