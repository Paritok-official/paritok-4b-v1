"""Merge a freshly re-compressed subset (e.g. fixed file_operation) into an
existing all_per_kind output, replacing entries by entry_id.

Usage:
  python update/merge_replacement.py \
      --base update/other_compressed_all_per_kind_<ts>.jsonl \
      --replacement update/other_compressed_fileop_v2_<ts>.jsonl \
      --out update/other_compressed_all_per_kind_v2.jsonl
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="The big all_per_kind output")
    parser.add_argument("--replacement", required=True, help="Fresh re-compressed subset")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    repl_by_id: dict = {}
    with open(args.replacement, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            repl_by_id[rec["entry_id"]] = rec
    print(f"Loaded {len(repl_by_id)} replacement records from {args.replacement}")

    replaced = 0
    kept = 0
    by_kind_replaced: Counter = Counter()
    with open(args.base, "r", encoding="utf-8") as fin, \
         open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            eid = rec["entry_id"]
            if eid in repl_by_id:
                fout.write(json.dumps(repl_by_id[eid], ensure_ascii=False) + "\n")
                replaced += 1
                by_kind_replaced[rec.get("kind", "?")] += 1
            else:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1

    print(f"Wrote: {args.out}")
    print(f"Replaced: {replaced} ({dict(by_kind_replaced)})")
    print(f"Kept from base: {kept}")
    print(f"Total: {replaced + kept}")
    if replaced != len(repl_by_id):
        missing = len(repl_by_id) - replaced
        print(f"WARNING: {missing} replacement records did not match any base entry_id")


if __name__ == "__main__":
    main()
