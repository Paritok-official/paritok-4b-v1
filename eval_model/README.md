# Paritok — SWE-bench Lite quality-retained eval

This benchmarks the compression **model** (Paritok-4B) itself — how much of a
frontier model's single-shot solve quality survives compressing the source context.
It is not a script to reproduce a specific headline number — run it yourself.

```bash
python eval_model/run.py
```

**Requirements:** a local Ollama (compression), `ANTHROPIC_API_KEY` (the agent), and
Docker (scoring). Scoring runs in-process on Linux/macOS and through WSL on Windows —
`run.py` handles the routing.

> **Endpoint:** compression uses the OpenAI-compatible **`/v1/chat/completions`** endpoint
> (the default), *not* Ollama's native `/api/chat`. `/api/chat` reuses the cached
> system-prompt KV across requests, which flips this model's compression to ~0.60 after the
> first request; `/v1` re-prefills cleanly each request, so it's stable. (Exact ratios on a
> few knife-edge inputs can still shift slightly with the Ollama/llama.cpp build, but the
> aggregate is version-stable — no specific version is required.)

Full SWE-bench Lite (300 instances, pulled from Hugging Face) is:
1. given oracle file context,
2. compressed with the Paritok SEG model on a **local Ollama** endpoint
   (`paritok-4b-v1:latest`, chunk 3000, level L1) — **one instance at a time**
   (compression is not parallelised),
3. sent one-shot at **temperature 0** to a frontier agent for a unified-diff fix
   — from both the **baseline** (uncompressed) and the **compressed** context,
4. **re-anchored onto the true source before scoring.** On the **compressed** arm
   this is Paritok's own `edit_recovery` doing the real work — the same recovery the
   gateway runs on every Edit in production — so the eval measures the shipped
   gateway (compress + recover), not a compressor alone. The **baseline** arm runs
   the identical pass only for fairness: it has nothing reflowed to recover, but
   re-emitting its diff the same way strips the raw-LLM-diff apply brittleness
   (imperfect `@@` line numbers a fuzzy `git apply` still rejects) that would
   otherwise fail valid baseline patches and inflate the ratio. Same treatment both
   sides, so the only variable is the compression,
5. scored with the official SWE-bench harness (Docker).

It prints each arm's **`resolved / total`** (the standard SWE-bench resolve rate)
and the headline **quality retained = `compressed_resolved / baseline_resolved`**.

`edit_recovery` is **imported from `paritok`**, not re-implemented here; the only
eval-specific piece is decomposing a unified diff into edits so the same recovery
core can run on it (production intercepts Edit tool calls; a SWE-bench answer is a
diff). See `reanchor.py`.

Configure with `--n`, `--chunk`, `--compress-model`, `--ollama-url`,
`--agent-model` (or `PARITOK_EVAL_AGENT_MODEL`). `ANTHROPIC_API_KEY` is read from
the environment.

### Optional: `--line-numbers` (real-agent regime)

By default the eval compresses **raw** source. Real coding agents don't send raw
files — their Read tool emits **line-numbered** text (`cat -n`), which is exactly
what the compression model was trained on. Pass `--line-numbers` to compress that
form instead. The 4B model strips the line numbers as it compresses, so the context
the agent answers on is number-free either way — `--line-numbers` changes only the
**compressor's input**, not what the agent sees. The baseline arm is raw source, as
before.

Because Paritok-4B is trained on line-numbered reads, this is its in-distribution
regime: the model **compresses more** and **retains more solve quality**, so the
results come out **higher**. Given the SWE-bench comparison, we keep it as an optional flag
rather than the default — the raw run stays the headline (we don't quote a fixed
figure here), and `--line-numbers` shows that number is a floor, not a ceiling.
