"""Dump full input/output of all Stage 0 reject samples for eyeballing.

Usage:
  python scripts/inspect_rejects.py --stage 0
Output:
  data/distill/stage{N}_rejects_inspect.md
"""
import argparse
import sys
from pathlib import Path
from collections import defaultdict

import orjson
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import __file__ as _  # noqa
# Reuse helpers from 06_distill
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "distill_mod", Path(__file__).resolve().parent / "06_distill.py"
)
distill_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(distill_mod)

TRAIN_FULL = distill_mod.TRAIN_FULL
DISTILL_DIR = distill_mod.DISTILL_DIR
extract_input_text = distill_mod.extract_input_text
validate_one = distill_mod.validate_one


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=0)
    args = parser.parse_args()
    stage = args.stage

    output_path = DISTILL_DIR / f"stage{stage}_output.jsonl"
    if not output_path.exists():
        sys.exit(f"Not found: {output_path}")

    print("[inspect] Loading pool ...")
    pool_by_id = {}
    with open(TRAIN_FULL, "rb") as f:
        for line in tqdm(f, desc="Reading pool"):
            s = orjson.loads(line)
            pool_by_id[s["metadata"]["sample_id"]] = s

    print("[inspect] Scanning batch output ...")
    rejects = defaultdict(list)  # reason -> list of (sample_id, sample, compressed, original_input)

    with open(output_path, "rb") as f:
        for line in f:
            result = orjson.loads(line)
            sample_id = result["custom_id"]
            response = result.get("response", {})
            if not response or response.get("status_code") != 200:
                continue
            try:
                compressed = response["body"]["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                continue
            sample = pool_by_id.get(sample_id)
            if sample is None:
                continue
            ok, reason = validate_one(sample, compressed)
            if ok:
                continue
            original = extract_input_text(sample)
            rejects[reason].append((sample_id, sample, compressed, original))

    # Write markdown
    out_path = DISTILL_DIR / f"stage{stage}_rejects_inspect.md"
    with open(out_path, "w") as f:
        f.write(f"# Stage {stage} Rejects — full content\n\n")
        n_total = sum(len(v) for v in rejects.values())
        f.write(f"Total rejects: **{n_total}**\n\n")
        for reason, items in sorted(rejects.items(), key=lambda kv: -len(kv[1])):
            f.write(f"- `{reason}`: {len(items)}\n")
        f.write("\n---\n\n")

        for reason, items in sorted(rejects.items(), key=lambda kv: -len(kv[1])):
            f.write(f"# Reason: `{reason}` ({len(items)} samples)\n\n")
            for idx, (sample_id, sample, compressed, original) in enumerate(items, 1):
                meta = sample["metadata"]
                orig_chars = len(original)
                comp_chars = len(compressed)
                budget = meta["compression_budget"]
                orig_tok = orig_chars // 4
                comp_tok = comp_chars // 4
                len_ratio = comp_chars / max(1, orig_chars)
                budget_ratio = comp_tok / max(1, budget)

                f.write(f"## [{reason} #{idx}] `{sample_id}`\n\n")
                f.write(f"- length_bucket: `{meta.get('length_bucket')}`\n")
                f.write(f"- action: `{meta.get('target_action_name')}`\n")
                f.write(f"- resolved: `{meta.get('resolved')}`\n")
                f.write(f"- original: {orig_chars} chars (~{orig_tok} tok)\n")
                f.write(f"- compressed: {comp_chars} chars (~{comp_tok} tok)\n")
                f.write(f"- budget: {budget} tok\n")
                f.write(f"- **len_ratio (comp/orig)**: {len_ratio:.3f}\n")
                f.write(f"- **budget_ratio (comp/budget)**: {budget_ratio:.3f}\n\n")
                f.write("### ORIGINAL INPUT\n\n")
                f.write("```\n")
                f.write(original)
                f.write("\n```\n\n")
                f.write("### COMPRESSED OUTPUT\n\n")
                f.write("```\n")
                f.write(compressed)
                f.write("\n```\n\n")
                f.write("---\n\n")

    size_mb = out_path.stat().st_size / 1e6
    print(f"\nWrote: {out_path} ({size_mb:.2f} MB)")
    print("Open it in your editor to scroll.")


if __name__ == "__main__":
    main()
