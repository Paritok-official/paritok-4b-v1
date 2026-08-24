"""Paritok SWE-bench Lite quality-retained evaluation — one command, end to end.

    python eval_model/run.py

What it does, for the full SWE-bench Lite set (300 instances) pulled straight
from Hugging Face:

  1. build the oracle file context for each instance (source files the gold patch
     touches, fetched at base_commit),
  2. compress each context with the Paritok SEG model on a local Ollama endpoint
     (chunk 3000, level L1) — one instance at a time; compression is deliberately
     NOT parallelised (a shared local/GPU worker can corrupt under concurrency),
  3. ask a frontier agent for a one-shot unified-diff fix (temperature 0) from
     BOTH the baseline (uncompressed) and the compressed context,
  4. re-anchor both arms' patches onto the true source before scoring. On the
     compressed arm this is Paritok's own edit_recovery doing the real work — the
     same recovery the gateway runs on every Edit in production — so the eval
     measures the shipped gateway (compress + recover), not a compressor alone. The
     baseline arm runs the identical pass only for fairness: nothing is reflowed to
     recover, but re-emitting its diff the same way strips the raw-LLM-diff apply
     brittleness that would otherwise fail valid baseline patches and inflate the
     ratio — so the only variable stays the compression,
  5. run the official SWE-bench harness (Docker) on both and print:
       - each arm's resolved / total  (the standard SWE-bench resolve rate), and
       - quality retained = compressed_resolved / baseline_resolved.

Defaults: compression model = local Ollama `paritok-4b-v1:latest` (direct, no
gateway); agent model = configurable via --agent-model / PARITOK_EVAL_AGENT_MODEL;
ANTHROPIC_API_KEY is read from the environment. Intermediate artifacts go under
eval_model/_work/.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Allow `python eval_model/run.py` from the repo root: put the repo root on the path so
# both the `eval_model` package and the local `paritok` package import cleanly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_model import agent, compress, harness, postprocess, reanchor  # noqa: E402
from eval_model.dataset import load_instances, line_number_context, _fetch  # noqa: E402
from paritok.token_counter import count_tokens  # noqa: E402


def _p(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Paritok SWE-bench Lite quality-retained eval")
    ap.add_argument("--n", type=int, default=300, help="instances (default: full Lite = 300)")
    ap.add_argument("--chunk", type=int, default=3000)
    ap.add_argument("--compress-model", default="paritok-4b-v1:latest",
                    help="local Ollama model name")
    ap.add_argument("--ollama-url", default="http://localhost:11434/v1/chat/completions",
                    help="Ollama OpenAI-compat endpoint. MUST be /v1/chat/completions — "
                         "native /api/chat reuses the system-prompt KV cache and flips this "
                         "model's compression to ~0.60 after the first request (see compress.py).")
    ap.add_argument("--agent-model",
                    default=os.environ.get("PARITOK_EVAL_AGENT_MODEL", "claude-sonnet-4-5-20250929"))
    ap.add_argument("--agent-workers", type=int, default=6)
    ap.add_argument("--harness-workers", type=int, default=6)
    ap.add_argument("--line-numbers", action="store_true",
                    help="compress the cat -n line-numbered context (the real-agent, "
                         "in-distribution regime) instead of raw source; both arms get "
                         "line-numbered context and their patches are de-numbered before "
                         "re-anchoring. Compresses more and holds quality; off by default "
                         "so the run reproduces the conservative raw headline.")
    ap.add_argument("--depad", action="store_true",
                    help="de-pad the cat -n line numbers to the bare 'N\\t' shape before "
                         "compression (the gateway's production-parity path). OFF by "
                         "default: on SWE-bench, padded compresses tighter and is less "
                         "fragile (48/100 padded vs 42/100 + 8 errors bare; GPU A/B "
                         "18.5% vs 23.7% global kept). Only meaningful with --line-numbers.")
    ap.add_argument("--out", default=os.path.join("eval_model", "_work"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set (needed for the agent step).")

    # 1+2. dataset + oracle context (from source; nothing vendored) then compress on
    # the local Ollama SEG model. Both are cached to a resumable jsonl so a long run
    # survives interruption — re-running continues where it stopped. The compression
    # loop is serial by design: one instance at a time (chunks within a file are also
    # compressed sequentially), so a shared worker never sees concurrent requests.
    # Key the cache on the compression MODEL as well as the chunk size, so re-running
    # the same --out with a different --compress-model doesn't silently reuse the old
    # model's compressed contexts and attribute the QR to the wrong model.
    _slug = re.sub(r"[^\w.-]", "_", args.compress_model)
    # keep raw / line-num / de-padded caches separate (each yields different compressed output)
    _lnsuf = ("_ln" if args.line_numbers else "") + ("_depad" if args.depad else "")
    cache_path = os.path.join(args.out, f"instances_chunk{args.chunk}_{_slug}{_lnsuf}.jsonl")
    done: dict[str, dict] = {}
    if os.path.exists(cache_path):
        for line in open(cache_path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done[r["instance_id"]] = r
        _p(f"[1/5] resuming: {len(done)} instances already prepared in {cache_path}")

    _p(f"[1-2/5] preparing + compressing up to {args.n} instances "
       f"({args.compress_model}, chunk={args.chunk}, level L1, serial)...")
    records: list[dict] = []
    cstats = {"chunks": 0, "passthrough": 0}  # count chunks that Ollama failed to compress
    with open(cache_path, "a", encoding="utf-8") as cf:
        k = 0
        for rec in load_instances(args.n):
            if not rec.get("full_context"):
                continue
            k += 1
            iid = rec["instance_id"]
            if iid in done:
                records.append(done[iid])
                continue
            # Feed the model line-numbered source in --line-numbers mode (its
            # in-distribution form); the raw full_context still drives everything else.
            comp_input = (line_number_context(rec["full_context"])
                          if args.line_numbers else rec["full_context"])
            rec["compressed_context"] = compress.compress_context(
                comp_input, rec["problem_statement"],
                chunk=args.chunk, endpoint=args.ollama_url, model=args.compress_model,
                depad=args.depad, stats=cstats,
            )
            cf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cf.flush()
            records.append(rec)
            if k % 10 == 0:
                _p(f"      prepared {k} (compressed {k - len(done)} new)")
    _p(f"      {len(records)} instances ready")

    rec_by_id = {r["instance_id"]: r for r in records}

    # cached fetch of the true source at base_commit — re-anchoring reads the same
    # files repeatedly, so memoise (a full run fetches each (repo, commit, path) once).
    _fetch_cache: dict[tuple, str | None] = {}

    def fetch(repo: str, commit: str, path: str):
        # Cache only SUCCESSFUL fetches. A None (404 or a transient failure past _fetch's
        # own retries) is never cached, so a transient blip on one file doesn't poison it
        # for the whole run and quietly send both arms to the recount fallback.
        key = (repo, commit, path)
        v = _fetch_cache.get(key)
        if v is None:
            v = _fetch(repo, commit, path)
            if v is not None:
                _fetch_cache[key] = v
        return v

    # 3. one-shot agent patches (temperature 0) for BOTH arms.
    client = agent._client()

    def gen(context_key: str) -> dict[str, str]:
        preds: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=args.agent_workers) as ex:
            futs = {ex.submit(agent.generate_patch, client, args.agent_model,
                              r.get(context_key) or "", r.get("problem_statement") or ""): r["instance_id"]
                    for r in records}
            for f in as_completed(futs):
                preds[futs[f]] = f.result()
        return preds

    _p(f"[3/5] generating one-shot patches with {args.agent_model} @ temp=0 (baseline + compressed)...")
    # --line-numbers only changes the COMPRESSOR's input (raw vs line-numbered source).
    # The 4B model strips the read line numbers as it compresses, so the compressed
    # context the agent answers on is number-free — same as the raw baseline. The agent
    # never sees line numbers; only the compression differs between the two arms.
    baseline_raw = gen("full_context")
    compressed_raw = gen("compressed_context")

    # 3.5. re-anchor both arms onto the true source (identical treatment for fairness).
    # edit_recovery does the real work on the compressed arm (undo reflow, mirror the
    # gateway); on the baseline arm it is a no-op for recovery and just re-emits a
    # clean git-appliable diff. If a patch can't be uniquely re-anchored, fall back to
    # recount() so its hunk headers are at least valid — never invent fix content.
    def reanchor_arm(preds: dict[str, str], strip_line_numbers: bool = False) -> dict[str, str]:
        out: dict[str, str] = {}
        for iid, patch in preds.items():
            r = rec_by_id[iid]
            fixed = reanchor.reanchor(patch, r["repo"], r["base_commit"], fetch,
                                      full_context=r.get("full_context"),
                                      strip_line_numbers=strip_line_numbers)
            out[iid] = fixed or postprocess.recount(patch)
        return out

    _p("[3.5/5] re-anchoring both arms with paritok.edit_recovery...")
    baseline_preds = reanchor_arm(baseline_raw)
    # Compressed output is already number-free (the model strips them); strip here is a
    # defensive no-op for any stray line number the model might keep on a retained line.
    compressed_preds = reanchor_arm(compressed_raw, strip_line_numbers=args.line_numbers)
    json.dump(baseline_preds, open(os.path.join(args.out, "baseline_preds.json"), "w"))
    json.dump(compressed_preds, open(os.path.join(args.out, "compressed_preds.json"), "w"))

    # 4. Docker harness on both arms.
    ids = [r["instance_id"] for r in records]
    ok, why = harness.available()
    if not ok:
        # Everything except scoring is done. Write the predictions so they can be graded
        # on a Unix box, report the compression rate, and say exactly how to finish. (The
        # SWE-bench harness is Linux/macOS/WSL only — swebench imports `resource`.)
        harness.write_predictions(baseline_preds,
                                  os.path.join(args.out, "predictions_baseline.jsonl"), "paritok-baseline")
        harness.write_predictions(compressed_preds,
                                  os.path.join(args.out, "predictions_compressed.jsonl"), "paritok-compressed")
        orig_tok = sum(count_tokens(r["full_context"], "cl100k_base") for r in records)
        comp_tok = sum(count_tokens(r.get("compressed_context", ""), "cl100k_base") for r in records)
        cratio = comp_tok / orig_tok if orig_tok else 0.0
        _p(f"\n[4/5] harness SKIPPED — {why}")
        _p(f"      compression rate (compressed / full) = {comp_tok}/{orig_tok} = {100 * cratio:.1f}%")
        _p("      Compression + patch generation + re-anchoring finished here; only Docker scoring is left.")
        _p(f"      Predictions written to {args.out} (predictions_baseline.jsonl, predictions_compressed.jsonl).")
        _p("      Finish on Linux/WSL with Docker, e.g.:")
        _p("        python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Lite \\")
        _p("          --predictions_path predictions_compressed.jsonl --run_id compressed --cache_level instance --clean False")
        return
    _p("[4/5] running the SWE-bench harness (Docker) on the baseline arm...")
    b_res, _b_comp, b_tot = harness.run(baseline_preds, ids, run_id="baseline",
                                        workdir=args.out, max_workers=args.harness_workers)
    _p("      running the SWE-bench harness (Docker) on the compressed arm...")
    c_res, _c_comp, c_tot = harness.run(compressed_preds, ids, run_id="compressed",
                                        workdir=args.out, max_workers=args.harness_workers)

    # 5. compression rate + resolved/total per arm; quality retained = compressed / baseline.
    orig_tok = sum(count_tokens(r["full_context"], "cl100k_base") for r in records)
    comp_tok = sum(count_tokens(r.get("compressed_context", ""), "cl100k_base") for r in records)
    cratio = comp_tok / orig_tok if orig_tok else 0.0
    b_rate = b_res / b_tot if b_tot else 0.0
    c_rate = c_res / c_tot if c_tot else 0.0
    _p("\n[5/5] RESULTS" + ("  [--line-numbers: real-agent regime]" if args.line_numbers else ""))
    _p(f"  compression rate (compressed / full) = {comp_tok}/{orig_tok} = {100 * cratio:.1f}%")
    if cstats["passthrough"]:
        _p(f"  WARNING: {cstats['passthrough']}/{cstats['chunks']} chunks passed through "
           f"UNCOMPRESSED (Ollama failures) — do not trust QR until this is ~0")
    _p(f"  baseline    resolved/total = {b_res}/{b_tot} = {100 * b_rate:.1f}%")
    _p(f"  compressed  resolved/total = {c_res}/{c_tot} = {100 * c_rate:.1f}%")
    qr = (c_rate / b_rate) if b_rate else 0.0
    _p(f"  QUALITY RETAINED (compressed / baseline) = {100 * qr:.1f}%")


if __name__ == "__main__":
    main()
