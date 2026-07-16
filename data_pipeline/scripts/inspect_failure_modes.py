"""Classify Stage 0 outputs by compression failure modes.

For each of the 100 batch outputs (not just passed), detect:
  - NO_COMPRESSION: len_ratio > 0.85 (already caught as near_identity)
  - WEAK_COMPRESSION: 0.65 < len_ratio <= 0.85 (passes validate but mostly verbatim)
  - TRUNCATED: output does not end cleanly with [/SEG] or has dangling segment
  - PURE_EXTRACTIVE: no [summary:/abridged/etc.] markers + len_ratio > 0.5
  - EXCEEDS_BUDGET: budget_ratio > 1.3

Output: data/distill/stage0_failure_modes.md
"""
import re
import sys
import importlib.util
from collections import Counter
from pathlib import Path

import orjson
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_spec = importlib.util.spec_from_file_location(
    "distill_mod", Path(__file__).resolve().parent / "06_distill.py"
)
_distill_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_distill_mod)
extract_input_text = _distill_mod.extract_input_text
TRAIN_FULL = _distill_mod.TRAIN_FULL

OUTPUT_PATH = Path("data/distill/stage0_output.jsonl")
OUT = Path("data/distill/stage0_failure_modes.md")

ABSTRACTIVE_MARKER_RE = re.compile(
    r"\[summary|\[abridged|\[condensed|\[truncated|\[omitted|\[elided|\[shortened|\[brief|"
    r"\d+ matches|\d+ lines|\d+ more|\d+ files|\.\.\. \(",
    re.IGNORECASE,
)
SEG_OPEN_RE = re.compile(r"\[SEG\s+id=([^\s\]]+)\s+kind=([^\s\]]+)\s+level=([^\s\]]+)\]")
SEG_CLOSE_RE = re.compile(r"\[/SEG\]")


def detect_truncation(compressed: str) -> tuple[bool, str]:
    """Heuristic truncation detection.

    True = looks truncated.
    """
    stripped = compressed.rstrip()
    if not stripped:
        return True, "empty_output"

    n_open = len(SEG_OPEN_RE.findall(compressed))
    n_close = len(SEG_CLOSE_RE.findall(compressed))

    if n_open > n_close:
        return True, f"unclosed_segments ({n_open} open, {n_close} close)"

    # Output should end with [/SEG] (allowing trailing whitespace) OR a closing code fence
    if not stripped.endswith("[/SEG]") and not stripped.endswith("```"):
        # Allow short outputs without segment markers (unusual but not necessarily truncated)
        if n_open == 0 and len(compressed) < 200:
            return False, ""
        return True, "no_closing_marker"

    # Check: does the last segment look abruptly cut? (ends with partial line, no period etc.)
    # Soft signal: last segment content > 100 chars and ends with neither punctuation nor closing brace
    last_seg_match = list(SEG_OPEN_RE.finditer(compressed))
    if last_seg_match:
        last_open = last_seg_match[-1].end()
        last_close = compressed.rfind("[/SEG]")
        if last_close > last_open:
            last_content = compressed[last_open:last_close].rstrip()
            if len(last_content) > 100:
                tail = last_content[-2:]
                # Common clean endings: ., !, ?, ), }, ], ", ', ```, \n
                if tail and not any(tail.endswith(c) for c in [".", "!", "?", ")", "}", "]", '"', "'", "`", "\n"]):
                    # don't flag — too noisy for code, just informational
                    pass

    return False, ""


def main():
    print("[1/2] Loading SFT pool ...")
    pool_by_id = {}
    with open(TRAIN_FULL, "rb") as f:
        for line in tqdm(f, desc="Pool"):
            s = orjson.loads(line)
            pool_by_id[s["metadata"]["sample_id"]] = s
    print(f"  {len(pool_by_id)} samples")

    print("[2/2] Classifying outputs ...")
    rows = []
    with open(OUTPUT_PATH, "rb") as f:
        for line in f:
            result = orjson.loads(line)
            sid = result["custom_id"]
            response = result.get("response", {})
            if not response or response.get("status_code") != 200:
                continue
            try:
                compressed = response["body"]["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                continue
            sample = pool_by_id.get(sid)
            if not sample:
                continue

            original = extract_input_text(sample)
            budget = sample["metadata"]["compression_budget"]
            orig_chars = len(original)
            comp_chars = len(compressed)
            comp_tokens = comp_chars // 4

            len_ratio = comp_chars / max(1, orig_chars)
            budget_ratio = comp_tokens / max(1, budget)
            marker_count = len(ABSTRACTIVE_MARKER_RE.findall(compressed))
            truncated, trunc_reason = detect_truncation(compressed)

            # Usage from API
            usage = response["body"].get("usage", {})
            real_completion_tokens = usage.get("completion_tokens", 0)
            real_finish_reason = response["body"]["choices"][0].get("finish_reason", "")

            # Classify
            flags = []
            if len_ratio > 0.85:
                flags.append("NO_COMPRESSION")
            elif 0.65 < len_ratio <= 0.85:
                flags.append("WEAK_COMPRESSION")
            if truncated:
                flags.append(f"TRUNCATED({trunc_reason})")
            if real_finish_reason == "length":
                flags.append("HIT_MAX_TOKENS")
            if marker_count == 0 and len_ratio > 0.5:
                flags.append("PURE_EXTRACTIVE_NO_REWRITE")
            if budget_ratio > 1.3:
                flags.append("EXCEEDS_BUDGET")

            rows.append({
                "sample_id": sid,
                "orig_chars": orig_chars,
                "comp_chars": comp_chars,
                "len_ratio": len_ratio,
                "budget_ratio": budget_ratio,
                "marker_count": marker_count,
                "truncated": truncated,
                "trunc_reason": trunc_reason,
                "finish_reason": real_finish_reason,
                "flags": flags,
                "compressed": compressed,
                "original": original,
            })

    # Counts
    flag_counts = Counter()
    for r in rows:
        if r["flags"]:
            for f in r["flags"]:
                # strip TRUNCATED arg
                key = f.split("(")[0]
                flag_counts[key] += 1
        else:
            flag_counts["HEALTHY"] += 1

    # Per-row label = primary flag
    def primary(r):
        if not r["flags"]:
            return "HEALTHY"
        # priority order
        for k in ["EXCEEDS_BUDGET", "NO_COMPRESSION", "TRUNCATED", "HIT_MAX_TOKENS", "WEAK_COMPRESSION", "PURE_EXTRACTIVE_NO_REWRITE"]:
            if any(f.startswith(k) for f in r["flags"]):
                return k
        return r["flags"][0].split("(")[0]

    primary_counts = Counter(primary(r) for r in rows)

    # Write report
    with open(OUT, "w") as f:
        n = len(rows)
        f.write(f"# Stage 0 — failure mode classification ({n} outputs)\n\n")

        f.write("## Failure flag tally (a sample can have multiple flags)\n\n")
        f.write("| flag | count | % |\n|---|---|---|\n")
        for k, c in flag_counts.most_common():
            f.write(f"| `{k}` | {c} | {100*c/n:.0f}% |\n")
        f.write("\n")

        f.write("## Primary class (worst flag wins, priority order)\n\n")
        f.write("Priority: EXCEEDS_BUDGET > NO_COMPRESSION > TRUNCATED > HIT_MAX_TOKENS > WEAK_COMPRESSION > PURE_EXTRACTIVE_NO_REWRITE > HEALTHY\n\n")
        f.write("| primary | count | % |\n|---|---|---|\n")
        for k, c in primary_counts.most_common():
            f.write(f"| `{k}` | {c} | {100*c/n:.0f}% |\n")
        f.write("\n")

        f.write("## Distribution stats\n\n")
        lrs = sorted([r["len_ratio"] for r in rows])
        brs = sorted([r["budget_ratio"] for r in rows])
        mkrs = sorted([r["marker_count"] for r in rows])
        f.write("len_ratio (output_chars / input_chars):\n")
        f.write(f"  - p25 {lrs[n//4]:.2f}, p50 {lrs[n//2]:.2f}, p75 {lrs[3*n//4]:.2f}, p95 {lrs[int(n*0.95)]:.2f}\n\n")
        f.write("budget_ratio:\n")
        f.write(f"  - p25 {brs[n//4]:.2f}, p50 {brs[n//2]:.2f}, p75 {brs[3*n//4]:.2f}, p95 {brs[int(n*0.95)]:.2f}\n\n")
        f.write("abstractive marker count per output:\n")
        f.write(f"  - p25 {mkrs[n//4]}, p50 {mkrs[n//2]}, p75 {mkrs[3*n//4]}, p95 {mkrs[int(n*0.95)]}\n")
        f.write(f"  - n_zero_markers: {sum(1 for m in mkrs if m == 0)}\n\n")

        # Show worst examples by category
        for cat in ["TRUNCATED", "WEAK_COMPRESSION", "NO_COMPRESSION", "PURE_EXTRACTIVE_NO_REWRITE", "HIT_MAX_TOKENS"]:
            cat_rows = [r for r in rows if primary(r) == cat]
            if not cat_rows:
                continue
            f.write(f"## Examples — `{cat}` (showing up to 3)\n\n")
            for r in cat_rows[:3]:
                f.write(f"### `{r['sample_id']}`\n\n")
                f.write(f"- len_ratio: {r['len_ratio']:.3f}, budget_ratio: {r['budget_ratio']:.3f}, markers: {r['marker_count']}\n")
                f.write(f"- finish_reason: {r['finish_reason']}, truncated: {r['truncated']} ({r['trunc_reason']})\n")
                f.write(f"- flags: {r['flags']}\n\n")
                f.write("<details><summary>compressed output tail (last 800 chars)</summary>\n\n```\n")
                f.write(r["compressed"][-800:])
                f.write("\n```\n\n</details>\n\n---\n\n")

    print(f"\nWrote: {OUT}")

    print("\n=== Failure flag tally ===")
    for k, c in flag_counts.most_common():
        print(f"  {k:<35} {c:>3} ({100*c/n:.0f}%)")
    print("\n=== Primary class (worst flag wins) ===")
    for k, c in primary_counts.most_common():
        print(f"  {k:<35} {c:>3} ({100*c/n:.0f}%)")


if __name__ == "__main__":
    main()
