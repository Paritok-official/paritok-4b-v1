<h1 align="center">Paritok</h1>

<p align="center"><b>A non-destructive compression gateway for coding agents — cut your input-token bill without changing your agent.</b></p>
<p align="center">Paritok sits between your agent and the LLM as a drop-in proxy. On every request it strips the tool-schema bloat, compresses tool results and file reads, and summarizes stale history — then forwards upstream, billed on the compressed tokens. Nothing is ever permanently discarded: the agent pulls back any exact original on demand. Works with <b>Claude Code, Cursor, Codex, OpenHands</b>, and any agent that honors <code>BASE_URL</code> — <b>you don't change a line of your agent</b>.<br/><br/>Powered by the <b>first open-source 4B compression model trained specifically for coding agents</b> (45K real agent trajectories). Cut input token bills from <b>~25% on turn one</b> to <b>past 85% in long, context-saturated sessions</b>, and fit <b>~3× more turns</b> in the same context window.</p>

<p align="center">
  <img src="./paritok-compression.png" alt="Paritok compression: 3,000 tokens shrunk to 780 with semantics intact" width="820"/>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2608.24188">
    <img src="https://img.shields.io/badge/arXiv-2608.24188-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"/>
  </a>
  <a href="https://huggingface.co/paritok/paritok-4b-v1">
    <img src="https://img.shields.io/badge/🤗%20Model-HuggingFace-yellow" alt="HF Model"/>
  </a>
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue" alt="License"/>
  </a>
  <a href="https://discord.gg/SeBJE5Eucp">
    <img src="https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white" alt="Discord"/>
  </a>
  <img src="https://img.shields.io/badge/backbone-Qwen3--4B-purple" alt="Qwen3-4B"/>
  <img src="https://img.shields.io/badge/python-3.11+-blue" alt="Python"/>
</p>

<p align="center">
  <a href="#-what-paritok-does">What it does</a> ·
  <a href="#-the-three-levers">The three levers</a> ·
  <a href="#-savings-compound-over-a-session">Compounding savings</a> ·
  <a href="#-cost-impact">Cost</a> ·
  <a href="#-pricing">Pricing</a> ·
  <a href="#-quick-start">Quick Start</a> ·
  <a href="#-the-engine-4b-compression-model">The engine (model)</a> ·
  <a href="#-team">Team</a>
</p>

---

## 📢 News

- **2026-08-17** &nbsp; **v1.3.8** — a visual `/stats` dashboard (live token savings + per-request original→compressed before/after, in place of the raw JSON) and the new [Paritok VS Code extension](https://github.com/Paritok-official/paritok-vscode).
- **2026-07-31** &nbsp; **v1.3.0** — stability release: edit-recovery, `read_original` API rename (from `expand_context`).
- **2026-07-19** &nbsp; **v1.2.0** ships the embedding-based tool filter — the biggest single-turn lever (~29K → ~8K on a typical Claude Code turn), unlocking prompt-cache-friendly tool selection with `gateway_search_tools` recall.
- **2026-07-15** &nbsp; **Paritok gateway v1.0.0** open-sourced — the proxy/middleware that turns the 4B model into a drop-in Claude Code / Cursor / Codex compression layer.
- **2026-07-14** &nbsp; **Paritok-4B-v1** released on Hugging Face Hub with full SWE-bench Lite end-to-end evaluation.
- **2026-06-25** &nbsp; Finished training. 45K teacher-distilled samples on the Qwen3-4B backbone.

---

## 🚪 What Paritok does

Paritok runs as a **middle layer between your agent and the LLM API** — your agent points at Paritok instead of Anthropic/OpenAI, and everything else stays the same.

```
Your Agent (Claude Code / Cursor / Codex)
  → builds request (tool schemas + history + tool results / file reads)
     ★ Paritok gateway rewrites the request here ★
  → forwarded to Anthropic / OpenAI  (billed on the compressed tokens)
  ← response flows back unchanged; compressed refs expand on demand
```

The token bill for a coding agent is dominated by **input you re-send every turn**: dozens of tool schemas, an ever-growing message history, and big file-read / tool-output blocks. Paritok attacks all three — and because it's a **non-destructive** gateway, anything it compresses or filters is still recoverable on demand. It's lossy on the wire, fully recoverable when it counts.

---

## 🎚️ The three levers

Paritok saves tokens through three independent mechanisms. They stack, and they hit different parts of the bill:

### 1. Tool-schema filter — the biggest single-turn win

Coding agents expose dozens of tools — often **70+** once you add MCP servers — in full JSON schema on **every** request. Most are irrelevant to the task at hand. Paritok filters them semantically (`tool_discovery.strategy: embedding`), keeping only the handful relevant to the user's intent in full schema and stubbing the rest.

- **This is the largest single-turn lever.** On a typical Claude Code turn the tool block alone is ~29K tokens; filtered it drops to ~8K — a saving no amount of file compression matches on a single turn.
- **Prompt-cache friendly.** The selection is frozen per conversation, so the `tools[]` block stays byte-stable turn-to-turn and never invalidates the LLM's KV cache.
- **Never destructive.** Anything filtered is recoverable — the model calls `gateway_search_tools` and gets the full schema back.
- Runs a small open embedding model ([BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), MIT, ~130MB) **entirely locally on CPU** — no API, no per-token fee.
- An agent's **core execution tool** (shell / exec / apply_patch) is never stubbed, so agents like Codex that expose only a handful of tools always keep the one they can't work without.

### 2. Content compression — file reads, tool output, history

Each `tool_result`, file read, and (once the window fills) stale history turn is compressed by the 4B model down to **~26% of its original size**, tagged `[REF:id]`. This is where the trained model earns its keep: it knows a function signature from a debug line, so it protects identifiers, paths, and error strings while dropping the noise.

- Single-turn this is the *smaller* lever (a few % — most of a turn's cost is the fixed prefix). **Its power shows up across a session — see below.**
- Every compressed segment is recoverable: the agent calls `read_original` / `expand_context` to pull back the exact untouched bytes, locally and instantly.

### 3. History summarization — keep long sessions under the window

Turns beyond the recent window are summarized once the context fills up, so a long session stays inside the model's context window instead of overflowing (or forcing an aggressive client-side compaction that drops detail).

---

## 📈 Savings compound over a session

This is the part a single-turn benchmark hides. In a real multi-turn session the two levers **grow at different rates**, and together they compound.

We ran the same read-only "find the bug" task for **5 consecutive turns in one Claude Code session** (Sonnet, GPU model). **Important:** in this A/B the tool filter was left **on for both sides**, so the delta below isolates **content compression alone** — it does *not* include the tool-schema saving.

**Content compression only** (tool filter on both sides):

| Turns | Paritok (files compressed) | Baseline (files raw)  | Content-only saving |
| :---: | :------------------------: | :-------------------: | :-----------------: |
| 1     | 72,041                     | 75,507                | 4.6%                |
| 5     | 293,389 (cumulative)       | 377,099 (cumulative)  | 22.2%               |

But that's only one of the three levers. The real "do I use Paritok or not?" comparison must add the tool filter back — and **without Paritok the agent sends the entire ~29K tool-schema block every turn, not the filtered ~8K.** Folding that ~21K/turn back onto the no-Paritok side:

**Full stack** (filter + compression vs. no Paritok at all):

| Turns | Paritok (filter + compression) | No Paritok (full tools + raw files) | End-to-end saving |
| :---: | :----------------------------: | :---------------------------------: | :---------------: |
| 1     | 72,041                         | ~96,500                             | **~25%**          |
| 5     | 293,389 (cumulative)           | ~482,000 (cumulative)               | **~39%**          |

So the 4.6% / 22.2% above is the **floor** (content only). Against a real no-Paritok baseline it's **~25% on turn 1, past ~39% by turn 5** — because the tool filter saves a fixed ~21K *every* turn on top of the compounding content compression.

**Why it grows:** every file you read stays in history and is re-sent (cache-read) every subsequent turn — so content compression keeps paying off turn after turn, while the tool filter adds a fixed cut on top.

- **Content compression → quadratic.** Cumulative saving ≈ `3,350 × N²` — each turn's compressed reads keep paying off on every later turn.
- **Tool filter → linear.** Cumulative saving ≈ `21,000 × N` — a fixed block saved every turn.
- **Crossover ≈ turn 6:** early on the tool filter dominates; past ~turn 6 content compression overtakes it and the gap widens.

Plugging these formulas into a range of N (baseline ~96,500 tokens/turn), **capped at a ~200K context budget** (typical Sonnet-tier configuration; larger-context models like Opus 1M push the flatten point out proportionally):

| Turn (N) | Content saved | Tool filter saved | Cumulative saved | Cumulative baseline | **% saved** |
|:---:|---:|---:|---:|---:|:---:|
| 1  | 3,350   | 21,000  | 24,350    | 96,500    | **25%** |
| 5  | 83,750  | 105,000 | 188,750   | 482,500   | **39%** |
| 10 | 308,150 | 210,000 | 518,150   | 965,000   | **54%** |
| 12 | 404,150 | 252,000 | 656,150   | 1,158,000 | **57%** |
| 15 | 548,150 | 315,000 | 863,150   | 1,447,500 | **60%** |
| 20 | 788,150 | 420,000 | 1,208,150 | 1,930,000 | **63%** |

<sub>**Capped at a ~200K context budget.** The quadratic only holds while history is still growing. Once the accumulated context fills the configured budget (around turn ~8–12 here on Sonnet-tier ~200K), client-side compaction holds it flat, per-turn content saving stops growing (freezes at ~48K/turn), and **% saved plateaus toward the ~72% default ceiling instead of diverging**. Turns 1–5 match the measured tables above; later rows are in-window projections.</sub>

**Ceilings shift with deployment shape:**

| Deployment scenario                              | Baseline / turn | Ceiling % saved |
| ------------------------------------------------ | :-------------: | :-------------: |
| Default (~40 tools, moderate reads)              | 96,500          | **~72%**        |
| MCP-heavy (70+ tools)                            | 127,500         | **~78%**        |
| Context-saturated (no-Paritok forced to compact) | ~200,000        | **~85%+**       |

<sub>The projection above uses the default ceiling. MCP-heavy setups earn a bigger per-turn tool-filter cut; context-saturated sessions get an even larger effective saving because they're now compared against a compacted, information-lossy baseline.</sub>

> **Honest cap:** the quadratic doesn't run forever — whatever session context budget you configure bounds it. In practice a ~200K budget (typical for Sonnet-tier deployments) makes the curve flatten around turn ~12–20 as the window fills; larger-context models like Opus 1M push the flatten point out proportionally. But that ceiling is itself a feature: **because each turn's prefix is smaller, the agent fits more turns before hitting the budget** — Paritok effectively buys back context length.

**Same budget, more room to think.** Because each turn's prefix is smaller, the agent fits far more turns in the same window before compaction kicks in.

| Context budget                        | Turns without Paritok | Turns with Paritok |
| :------------------------------------ | :-------------------: | :----------------: |
| 128K (typical Claude Code default)    | ~10                   | ~30                |
| 200K (Claude Sonnet standard)         | ~15                   | ~44                |
| 1M (Claude Opus, GPT-5 beta)          | ~75                   | ~220               |

That's roughly **3× the runway on any budget**, giving the agent more turns to think and finish the task before hitting the wall.

**One line:** *use more, save more*. Compression frees up the window and lets the agent go deeper and longer in the same session. Strongest on **long, multi-turn, read-heavy** work (auditing, Q&A over a big codebase, long debugging sessions).

---

## 💰 Cost Impact

The **74%** figure is Paritok's **content compression rate** — file reads, tool output, and history shrink to ~26% of their size. End-to-end savings compound with session length.

**Per-session cost** at Claude Sonnet input pricing (`$3 / M input tokens`):

| Turns | Context (uncompressed → compressed) | Savings | Bill (no Paritok → with Paritok) |
| :---: | :--------------------------------: | :-----: | :------------------------------: |
| 1     | 40K → 12K                          | 25%     | $0.12 → $0.09                    |
| 5     | 84K → 30K                          | 39%     | $0.25 → $0.15                    |
| 10    | 140K → 52K                         | 54%     | $0.42 → $0.19                    |
| 20    | 250K → 95K                         | 63%     | $0.75 → $0.28                    |

**Annual savings by team size** (120 turns/dev/day, ~54% average end-to-end):

| Team size              | Baseline annual bill | Saved with Paritok |
| :--------------------- | :------------------: | :----------------: |
| Solo dev               |        $2.4K         |       ~$1.3K       |
| Small team (5 devs)    |        $12.2K        |       ~$6.6K       |
| Growing team (20 devs) |        $48.8K        |       ~$26K        |
| Enterprise (100 devs)  |         $244K        |       ~$132K       |

<sub>Default ~40-tool config on Claude Sonnet. **MCP-heavy setups (70+ tools) push savings to ~78%; context-saturated sessions past 85%.** Use the [interactive calculator at paritok.com](https://paritok.com) for your specific team size, LLM, and workload — it plugs in Claude Opus ($5/M), GPT-5 ($10/M), Haiku ($1/M), Gemini, or any provider you use.</sub>

Install and running in minutes on your own hardware. No vendor lock-in.

---

## 💵 Pricing

**Self-host (Apache 2.0): Free forever.** The gateway, the 4B model, and every training script ship open. Run on your own hardware. No telemetry, no key, no fees.

**Hosted GPU ([paritok.com](https://paritok.com)): $0.30 per 1M tokens processed.** No GPU required. **Free through end of August** as we launch, with full pricing kicking in September 1st.

---

## 🚀 Quick Start

### Fastest path (self-host, no clone needed)

Everything ships in the PyPI package — you do **not** need to `git clone` the repo. In a fresh environment (with [Ollama](https://ollama.com/download) installed):

```bash
pip install "paritok[proxy]"     # the gateway + CLI
paritok up                       # pulls the model if missing, then starts the proxy
```

`paritok up` checks Ollama, `ollama pull`s `paritok/paritok-4b-v1` and tags it as the local `paritok-4b-v1` if it isn't already there (~2.5GB, first run only), then serves on port 8080.

> **Leave that terminal running.** The proxy is a foreground server — every agent request flows through it, so it must stay up for the whole session. Open a **separate** terminal for the next step.

In the shell that launches your agent:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080   # then start Claude Code / Cursor / …
```

No config file is needed — `paritok up` and `paritok proxy` run on built-in defaults. Run `paritok init` to drop a starter `paritok.yaml` only if you want to tweak settings.

**q4 vs full precision.** `paritok up` uses the q4 model (`:latest`, ~2.5GB) by default. For full precision, run `paritok up --registry-model paritok/paritok-4b-v1:f16` (~8GB). If you already `ollama pull`ed either variant, `up` **auto-detects** which one you have — no re-download.

### 1. Install

```bash
pip install "paritok[proxy]"
# or, from a clone of this repo:
pip install -e ".[proxy]"
```

Enable the tool-schema filter (recommended — it's the biggest single-turn lever):

```bash
pip install "paritok[toolselect]"   # adds sentence-transformers (CPU-only)
```

> **First request warms up the embedding model (~10–15s — downloads bge-small once, then caches locally).** Every request after is instant (~15 ms).

### 2. Pick a backend — self-host **or** the GPU server

One boolean in [`paritok.yaml`](paritok.yaml) decides where compression runs:

```yaml
use_gpu_server: false   # ← the only switch that matters
```

**Option A — self-host** (`false`, default). Run the open 4B model on your own machine. No key, nothing leaves your box. Simplest is **Ollama**:

```bash
ollama pull paritok/paritok-4b-v1                # one-time, ~2.5GB
ollama cp   paritok/paritok-4b-v1 paritok-4b-v1  # tag it as the runtime name
```

Default [`paritok.yaml`](paritok.yaml) already points `local_model` at Ollama — nothing else to set.

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

Then in `paritok.yaml`, set `local_model.base_url: http://localhost:8000/v1`.
</details>

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

**Keep this terminal open** — the proxy must stay running for the whole session; run your agent from a separate terminal. On startup it checks the backend and warns (never aborts) if it can't reach one.

### 4. Point your agent at it

Set the base URL in the shell that launches your agent, **then start the agent**:

```bash
# macOS / Linux
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080   # Claude Code
export OPENAI_BASE_URL=http://127.0.0.1:8080      # Cursor / OpenAI-SDK agents
```

```powershell
# Windows PowerShell
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:8080"
$env:OPENAI_BASE_URL    = "http://127.0.0.1:8080"
```

Keep your real provider API key set as usual (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`) — the proxy only rewrites the request body and forwards your headers upstream. Compressed prompts go upstream; original responses come back unchanged.

> **Codex is the exception — it does *not* read `OPENAI_BASE_URL`.** The Codex CLI takes its endpoint from `~/.codex/config.toml`. See [Codex setup](#codex-cli).

#### Any OpenAI-compatible upstream (Groq, Gemini, OpenRouter, …)

The gateway is independent of the upstream LLM — point the OpenAI path at **any** OpenAI-Chat-Completions-compatible endpoint with `--openai-url`:

```bash
paritok proxy --openai-url https://api.groq.com/openai        # Groq
paritok proxy --openai-url https://openrouter.ai/api/v1       # OpenRouter
```

Then point your OpenAI-SDK agent at the proxy (`OPENAI_BASE_URL=http://127.0.0.1:8080`), set `OPENAI_API_KEY` to the provider's key, and use that provider's model names.

#### Codex CLI

Codex ignores `OPENAI_BASE_URL` — it only reads its endpoint from `~/.codex/config.toml`, so Paritok writes that file for you:

```yaml
codex:
  enabled: true            # paritok writes ~/.codex/config.toml on `paritok up`
  model: gpt-5             # any model your key can call
  api_key: "sk-..."        # your OpenAI key (or leave "" to use env OPENAI_API_KEY)
```

```bash
paritok up     # writes ~/.codex/config.toml (backs up any existing one)
codex          # in another shell — now routed through Paritok
```

Codex custom providers only speak the **`responses`** wire protocol, which the proxy serves at `/v1/responses`.

### 5. Check it's working

```bash
curl http://127.0.0.1:8080/health   # {"status":"ok","version":"..."}
curl http://127.0.0.1:8080/stats    # live compression totals
```

`/stats` reports cumulative savings across the session:

```json
{
  "total_requests": 42,
  "input_tokens_original": 512340,
  "input_tokens_compressed": 138221,
  "compression_ratio": 0.27,
  "tokens_saved": 374119,
  "estimated_cost_saved_usd": "$1.01"
}
```

These numbers are **scoped to what Paritok actually intervenes in** — the content it compresses plus the tool schemas it stubs. Everything it can't affect (your system prompt, the model's output) is excluded. `estimated_cost_saved_usd` is **cache-aware**: compressed content is priced at the full base rate (new input each turn), while the frozen tool block is priced as a prompt-cache hit after turn one — counting it at full list price would massively overstate savings.

### SDK mode (alternative)

```python
import anthropic
import paritok

client = paritok.ParitokClient(anthropic.Anthropic())
resp = client.messages.create(
    model="claude-sonnet-4-20250514", max_tokens=4096, messages=[...]
)
print(resp._paritok_savings.saved_tokens, resp._paritok_savings.ratio)
```

### Direct compression API (no agent)

Just want compressed text — no agent, no proxy (e.g. a RAG / retrieval pipeline)? Call the hosted endpoint directly with your Paritok API key. Unlike the proxy it returns **clean compressed text with no `[REF:]` tag** — that tag belongs to the agentic expand/recall loop, which direct callers don't use — plus the token counts:

```bash
curl -sX POST https://www.paritok.com/api/compress \
  -H "Authorization: Bearer pk_live_..." \
  -H "Content-Type: application/json" \
  -d '{
        "content": "…text to compress…",
        "query": "what the compression should keep in view",
        "upstream_model": "claude-sonnet-4"
      }'
```

```json
{
  "compressed": "…clean compressed text, no [REF:]…",
  "gpu_available": true,
  "original_tokens": 2903,
  "compressed_tokens": 88
}
```

`query` is optional but recommended — it's the intent the compressor keeps (it drops what the intent doesn't name). If the GPU backend is unavailable it degrades safely: `gpu_available: false` and `compressed` echoes your original unchanged (nothing lost, just uncompressed).

---

## 🧩 When to use it

**Paritok is most useful when:**
- Your sessions are **long and multi-turn**. Savings compound from ~25% at turn 1 to 60%+ by turn 20.
- You're **read-heavy or mixed**: auditing, Q&A, debugging over a large codebase.
- You have **many MCP tools** (40+). The tool filter alone drops ~21K tokens per turn.
- You're paying premium input rates (Claude Sonnet $3/M, Opus $5/M, GPT-5 $10/M).
- You want to **fit deeper sessions in the same context window** and avoid forced client-side compaction.

---

## 🧠 The engine: 4B compression model

The gateway's content compression is powered by **Paritok-4B-v1**, the first open-source compression model trained end-to-end on real coding-agent trajectories. This section is the engine spec — you don't need it to run the gateway, but it's why the compression keeps what matters.

### Highlights

- 🎨 **Code-native.** Trained on real coding-agent trajectories (`file_read`, `bash_command`, `log_output`, …). It knows what an import statement is worth vs a debug line, so it protects function names, paths, and error strings while compressing.
- 🚀 **Compresses each segment to 25.7%** of original — **2× harder than gpt-4.1-mini** (50.2% CR) and **2.4× harder than gpt-5** (61.9% CR).
- 🎯 **Retains 86.5% of full-context solve quality** on SWE-bench Lite (**89.3%** fed the line-numbered input it was trained for, where the paired difference against uncompressed is not significant, p = 0.079) — matching gpt-4.1-mini as compressor at **less than half the token spend**.
- 🪶 **Small & self-hostable** — 4B LoRA adapter, bf16, runs on a single 24GB GPU. No SaaS, no lock-in, no per-token compressor fee.
- 🔓 **Fully open** — Apache 2.0 weights, reproducible data pipeline, real end-to-end SWE-bench numbers.

### Benchmark: SWE-bench Lite

Real end-to-end evaluation. An agent scaffold receives its context through each compressor, then attempts to resolve the issue. Primary metric is **quality retained** (solve rate normalized to the uncompressed baseline).

| Context source                       | **Quality retained** ¹ | Compression rate |
| ------------------------------------ | :--------------------: | :--------------: |
| Uncompressed baseline                |         100.0%         |      100.0%      |
| gpt-4.1-mini (compressor)            |          85.6%         |       50.2%      |
| gpt-5 (compressor)                   |          93.6%         |       61.9%      |
| Paritok-4B-v1, raw source            |          86.5%         |       25.7%      |
| **Paritok-4B-v1, line-numbered** ⭐  |       **89.3%**        |    **27.8%**     |

<sub>¹ Quality retained = compressor solve rate ÷ uncompressed baseline solve rate, where solve rate is resolved ÷ all 300 instances — a patch that fails to apply counts as unresolved. Higher is better.</sub>

<sub>The last row feeds the compressor `cat -n` line-numbered source, which is what real agents actually read (`--line-numbers`); it is the configuration we recommend. On that run the agent resolves 122/300 uncompressed and 109/300 compressed — 30 instances solved only without compression, 17 only with it. An exact McNemar test on those 47 discordant pairs gives **p = 0.079**: at 300 instances, compressing the context to roughly a quarter of its size does *not* significantly reduce the solve rate. The GPT rows are quoted against the raw-source row so all three compressors see the same input.</sub>

> **The benchmark is a floor, not a ceiling.** The 86.5% measures the **raw 4B model** with **no recall enabled** — compressed output fed straight to the agent. What you actually deploy is the gateway: every segment is tagged `[REF:id]` and the agent can call `read_original` to pull back the exact bytes at any time. Nothing is permanently discarded, so real-world deployment recovers quality the raw benchmark leaves on the table. We publish the raw-model number because it's the honest, reproducible floor.

To reproduce these SWE-bench Lite numbers yourself, see the [`eval_model/`](eval_model/) folder — one command runs it end to end (pull from source → compress → score).

### How Paritok compares

|                                                | **Paritok-4B-v1** | LLMLingua-2  | gpt-4.1-mini prompt |
| ---------------------------------------------- | :---------------: | :----------: | :-----------------: |
| **Trained on real coding-agent trajectories**  |     ✅            |     ❌       |         ❌          |
| **Preserves function names / imports / paths** | ✅ (by design)    |   partial    |     partial         |
| **Compression rate** (lower = harder)          |  **25.7%** ⭐    |    ~40%      |       50.2%         |
| **SWE-bench Lite — quality retained**          |  **86.5%** ⭐    | not evaluated|       85.6%         |
| **Self-hostable open weights**                 |    Apache 2.0     |     MIT      |    closed API       |
| **Per-token compressor fee**                   |  zero (self-host) |  zero (open) |  pay-per-token      |

---

## 🗺️ Roadmap

- 🎯 **Paritok-4B-v2.** Next-gen pipeline pushing compression **under 20%** while closing the gap to uncompressed solve rate.
- 📈 **Frontier-scale backbones.** Larger models (10B+) for multi-day sessions with **100K+ token histories**.
- 🌍 **Multi-language expansion.** First-class TypeScript, Rust, Go, Java, C++, Kotlin — v1 is Python-heavy but the architecture is language-agnostic.
- 🔌 **Native integrations.** Drop-in `mcp add paritok` plugin for Claude Code and Cursor.
- ⚙️ **Adaptive compression.** Per-segment auto-selection of aggressiveness based on age, kind, and downstream intent — no manual tuning.

---

## 👥 Team

Paritok is built by two engineers — no big lab, no external funding, just months of GPU budget and eval iteration.

- **Jiayu Shi** — Architecture of the 4B on-device compression model, data pipeline (cleaning + construction), reward design and fine-tuning.
- **Luzhuo Chen** — Inference acceleration, full-stack architecture and deployment, evaluation framework, high-concurrency gateway optimization.

Reach us: [paritok9@gmail.com](mailto:paritok9@gmail.com) · X [@Paritok](https://x.com/Paritok)

---

## 📖 Citation

Cite the **paper** for the method and results:

```bibtex
@misc{paritok4b,
  title         = {Paritok-4B: Intent-Conditioned Context Compression for Coding Agents},
  author        = {Shi, Jiayu and Chen, Luzhuo},
  year          = {2026},
  eprint        = {2608.24188},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  doi           = {10.48550/arXiv.2608.24188},
  url           = {https://arxiv.org/abs/2608.24188},
}
```

Cite the **companion paper** for end-to-end cost — what a compression rate does and does not translate into over a full agent session:

```bibtex
@misc{paritok-cost-attribution,
  title     = {An Empirical Cost Attribution of Context-Compression Gateways in
               Multi-Turn Coding Agents: Why Compression Rate Is Not the End-to-End Saving},
  author    = {Chen, Luzhuo and Shi, Jiayu},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22043550},
  url       = {https://doi.org/10.5281/zenodo.22043550},
}
```

Cite the **software** for this release — the gateway, the weights, and the training and evaluation code:

```bibtex
@software{paritok-4b-v1,
  title   = {Paritok-4B-v1: An Open-Source Compression Model for AI Coding Agents},
  author  = {Shi, Jiayu and Chen, Luzhuo},
  year    = {2026},
  version = {4B-v1},
  license = {Apache-2.0},
  url     = {https://github.com/Paritok-official/paritok-4b-v1},
}
```

---

## 📄 License

Apache 2.0 — see [LICENSE](./LICENSE). The base model, Qwen3-4B-Instruct-2507, is released under its own license.

---

## 💬 Community & Support

- 👥 **Join the community** → [Discord](https://discord.gg/SeBJE5Eucp)
- 🐛 **Bug reports & feature requests** → [GitHub Issues](https://github.com/Paritok-official/paritok-4b-v1/issues)
- 💭 **Discussion** → [🤗 HF Model discussions](https://huggingface.co/paritok/paritok-4b-v1/discussions)
- 📧 **Contact** → [paritok9@gmail.com](mailto:paritok9@gmail.com)
