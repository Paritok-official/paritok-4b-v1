"""Extract a random pool of file_read SEGs from train_full_80k for gpt-5 compression.

Reads each sample's SFT user message, parses all [SEG ...] blocks, keeps only
kind=file_read SEGs. For each, also captures the `user_intent` from the
sample's first user_turn_history SEG so the per-SEG teacher prompt can use it.

Output format mirrors update/file_read_seg.jsonl so bench_per_seg.py-style
runners (and lint_compressed.py) can be reused without changes.

Usage:
  python update/extract_file_read_pool.py --n 10000 --seed 42
  python update/extract_file_read_pool.py --n 10000 --out update/file_read_pool_10k.jsonl
"""
import argparse
import re
import random
import sys
from pathlib import Path

import orjson
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
POOL = PARENT / "data" / "final" / "train_full_80k.jsonl"
DEFAULT_OUT = ROOT / "file_read_pool_10k.jsonl"

SEG_RE = re.compile(
    r"\[SEG\s+id=([^\s\]]+)\s+kind=([^\s\]]+)\s+level=([^\s\]]+)\](.*?)\[/SEG\]",
    re.DOTALL,
)


def extract_input_text(sample: dict) -> str:
    user_msg = sample["messages"][1]["content"]
    start = user_msg.find("```input_data\n")
    if start == -1:
        return user_msg
    start += len("```input_data\n")
    end = user_msg.find("\n```\n\n### COMPRESSED OUTPUT", start)
    if end == -1:
        return user_msg[start:]
    return user_msg[start:end]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-chars", type=int, default=100,
                        help="Skip SEGs smaller than this — not worth gpt-5 budget")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    if not POOL.exists():
        sys.exit(f"Pool missing: {POOL}")

    records = []
    print(f"Reading {POOL} ...")
    with open(POOL, "rb") as f:
        for line in tqdm(f, desc="scanning samples"):
            sample = orjson.loads(line)
            meta = sample["metadata"]
            sample_id = meta["sample_id"]
            repo = meta.get("repo", "")
            budget = meta.get("compression_budget", 0)
            original_tokens = meta.get("original_tokens", 0)

            input_text = extract_input_text(sample)

            user_intent = ""
            for m in SEG_RE.finditer(input_text):
                if m.group(2) == "user_turn_history":
                    user_intent = m.group(4).strip()
                    break

            for m in SEG_RE.finditer(input_text):
                seg_id, kind, level, content = m.group(1), m.group(2), m.group(3), m.group(4)
                if kind != "file_read":
                    continue
                if len(content) < args.min_chars:
                    continue
                records.append({
                    "entry_id": f"{sample_id}__{seg_id}",
                    "sample_id": sample_id,
                    "seg_id": seg_id,
                    "level": level,
                    "repo": repo,
                    "user_intent": user_intent,
                    "original": content,
                    "seg_original_chars": len(content),
                    "sample_budget_tokens": budget,
                    "sample_original_tokens": original_tokens,
                })

    print(f"\nTotal file_read SEGs (>= {args.min_chars} chars): {len(records)}")
    if len(records) < args.n:
        print(f"WARNING: pool has only {len(records)} candidates, requested {args.n}")
    rng = random.Random(args.seed)
    rng.shuffle(records)
    selected = records[:args.n]

    # Level distribution sanity
    by_level = {}
    for r in selected:
        by_level[r["level"]] = by_level.get(r["level"], 0) + 1
    print(f"Sampled: {len(selected)}")
    print(f"Level distribution: {dict(sorted(by_level.items()))}")
    char_stats = sorted(r["seg_original_chars"] for r in selected)
    n = len(char_stats)
    print(f"Char stats: p25={char_stats[n//4]}, p50={char_stats[n//2]}, p75={char_stats[3*n//4]}, p95={char_stats[int(n*0.95)]}, max={char_stats[-1]}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        for r in selected:
            f.write(orjson.dumps(r))
            f.write(b"\n")
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
