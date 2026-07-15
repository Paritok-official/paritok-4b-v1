"""Run current DISTILL_SYSTEM_PROMPT through 4.1-mini on the 18 unique GT samples.

Synchronous OpenAI calls (no batch), parallelized — finishes in ~1 min.

Usage:
  python scripts/bench_run.py
  python scripts/bench_run.py --model gpt-4.1-mini-2025-04-14 --label v2

Output:
  data/distill/bench_outputs/<timestamp>_<label>.jsonl
  Each line: {sample_id, compressed, finish_reason, usage}
"""
import argparse
import importlib.util
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import orjson
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_spec = importlib.util.spec_from_file_location(
    "distill_mod", Path(__file__).resolve().parent / "06_distill.py"
)
_distill_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_distill_mod)
DISTILL_SYSTEM_PROMPT = _distill_mod.DISTILL_SYSTEM_PROMPT
TRAIN_FULL = _distill_mod.TRAIN_FULL

GT_PATH = Path("gt_samples.jsonl")
OUT_DIR = Path("data/distill/bench_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = "gpt-4.1-mini-2025-04-14"


def call_compressor(client, model: str, sample: dict) -> dict:
    user_msg = sample["messages"][1]
    budget = sample["metadata"]["compression_budget"]
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": DISTILL_SYSTEM_PROMPT},
            user_msg,
        ],
        temperature=0,
        max_tokens=int(budget * 1.5),
    )
    msg = resp.choices[0].message
    usage = resp.usage
    return {
        "sample_id": sample["metadata"]["sample_id"],
        "compressed": msg.content,
        "finish_reason": resp.choices[0].finish_reason,
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--label", default="run")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("pip install openai")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY missing")
    client = OpenAI(api_key=api_key)

    # Unique sample IDs from GT
    needed_ids = set()
    with open(GT_PATH, "rb") as f:
        for line in f:
            needed_ids.add(orjson.loads(line)["sample_id"])
    print(f"Need {len(needed_ids)} unique samples")

    # Load just those samples from the pool
    samples = []
    with open(TRAIN_FULL, "rb") as f:
        for line in tqdm(f, desc="Scanning pool"):
            s = orjson.loads(line)
            if s["metadata"]["sample_id"] in needed_ids:
                samples.append(s)
                if len(samples) == len(needed_ids):
                    break
    print(f"Found {len(samples)} samples in pool")

    # Compress in parallel
    print(f"Compressing with {args.model} (workers={args.max_workers}) ...")
    results = [None] * len(samples)
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {
            ex.submit(call_compressor, client, args.model, s): i
            for i, s in enumerate(samples)
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Compressing"):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                results[i] = {"sample_id": samples[i]["metadata"]["sample_id"], "compressed": "", "error": str(e)}

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"{timestamp}_{args.label}.jsonl"
    with open(out_path, "wb") as f:
        for r in results:
            f.write(orjson.dumps(r))
            f.write(b"\n")

    finish_reasons = {}
    for r in results:
        fr = r.get("finish_reason", "?")
        finish_reasons[fr] = finish_reasons.get(fr, 0) + 1
    total_completion = sum(r.get("usage", {}).get("completion_tokens", 0) for r in results if r.get("usage"))
    total_prompt = sum(r.get("usage", {}).get("prompt_tokens", 0) for r in results if r.get("usage"))

    print(f"\nWrote: {out_path}")
    print(f"Finish reasons: {finish_reasons}")
    print(f"Total prompt tokens: {total_prompt:,}, completion: {total_completion:,}")
    print(f"\nNext: python scripts/bench_score.py --candidates {out_path}")


if __name__ == "__main__":
    main()
