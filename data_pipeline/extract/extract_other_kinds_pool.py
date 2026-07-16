"""Extract a random pool of NON-file_read SEGs from train_full_80k for gpt-4.1-mini compression.

Mirrors extract_file_read_pool.py but excludes kind in {file_read, system_prompt, user_turn_history}.

The output records carry the actual `kind` field so the per-SEG prompt can branch
its rules by kind.

Usage:
  python update/extract_other_kinds_pool.py --n 10000 --seed 42
  python update/extract_other_kinds_pool.py --n 10000 --out update/other_kinds_pool_10k.jsonl
"""
import argparse
import re
import random
import sys
from collections import Counter
from pathlib import Path

import orjson
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
POOL = PARENT / "data" / "final" / "train_full_80k.jsonl"
DEFAULT_OUT = ROOT / "other_kinds_pool.jsonl"

SEG_RE = re.compile(
    r"\[SEG\s+id=([^\s\]]+)\s+kind=([^\s\]]+)\s+level=([^\s\]]+)\](.*?)\[/SEG\]",
    re.DOTALL,
)

EXCLUDE_KINDS = {"file_read", "system_prompt", "user_turn_history"}


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
    parser.add_argument("--n", type=int, default=10000,
                        help="Global random sample size (used only when --per-kind is not set)")
    parser.add_argument("--per-kind", type=int, default=None,
                        help="Stratified sampling: N per kind. Errors out if any kind has < N available. "
                             "Overrides --n.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-chars", type=int, default=100)
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
                if kind in EXCLUDE_KINDS:
                    continue
                if len(content) < args.min_chars:
                    continue
                records.append({
                    "entry_id": f"{sample_id}__{seg_id}",
                    "sample_id": sample_id,
                    "seg_id": seg_id,
                    "kind": kind,
                    "level": level,
                    "repo": repo,
                    "user_intent": user_intent,
                    "original": content,
                    "seg_original_chars": len(content),
                    "sample_budget_tokens": budget,
                    "sample_original_tokens": original_tokens,
                })

    print(f"\nTotal non-file_read SEGs (>= {args.min_chars} chars, excl. system/user_turn): {len(records)}")

    # Per-kind availability
    by_kind_all = Counter(r["kind"] for r in records)
    print("Available per kind:")
    for k, c in by_kind_all.most_common():
        print(f"  {k:<22} {c:>6}")

    rng = random.Random(args.seed)

    if args.per_kind is not None:
        # Stratified: shuffle within kind, take first N per kind. Hard error if any short.
        by_kind_recs: dict[str, list] = {}
        for r in records:
            by_kind_recs.setdefault(r["kind"], []).append(r)
        short = [(k, len(v)) for k, v in by_kind_recs.items() if len(v) < args.per_kind]
        if short:
            print(f"\nERROR: these kinds have < {args.per_kind} SEGs available:")
            for k, n in short:
                print(f"  {k}: {n}")
            print("Either lower --per-kind, or filter out the short kind, or accept partial.")
            sys.exit(1)
        selected = []
        for kind in sorted(by_kind_recs):
            recs = by_kind_recs[kind]
            rng.shuffle(recs)
            selected.extend(recs[:args.per_kind])
        rng.shuffle(selected)
    else:
        if len(records) < args.n:
            print(f"WARNING: pool has only {len(records)} candidates, requested {args.n}")
        rng.shuffle(records)
        selected = records[:args.n]

    by_kind = Counter(r["kind"] for r in selected)
    by_level = Counter(r["level"] for r in selected)
    print(f"Sampled: {len(selected)}")
    print("Kind distribution:")
    for k, c in by_kind.most_common():
        print(f"  {k:<22} {c:>5} ({100*c/len(selected):.1f}%)")
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
