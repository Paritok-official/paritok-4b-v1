"""Check whether teacher's PASSED samples contain abstractive rewrites
or are purely extractive (segment selection).

For each passed sample:
  - count abstractive markers in output ([summary, truncated, omitted, etc.)
  - count [SEG ...] segments in input vs output (drop ratio)
  - per-segment: did the segment content shrink (rewritten) or get dropped wholesale?

Output:
  data/distill/stage0_abstractive_inspect.md
"""
import re
import sys
from pathlib import Path
from collections import Counter

import orjson

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib.util
spec = importlib.util.spec_from_file_location(
    "distill_mod", Path(__file__).resolve().parent / "06_distill.py"
)
distill_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(distill_mod)
extract_input_text = distill_mod.extract_input_text
DISTILL_DIR = distill_mod.DISTILL_DIR

VALIDATED = DISTILL_DIR / "stage0_validated.jsonl"

# Abstractive markers — case-insensitive substring scan
ABSTRACTIVE_MARKERS = [
    r"\[summary",
    r"\[abridged",
    r"\[condensed",
    r"\[truncated",
    r"\[omitted",
    r"\[elided",
    r"\[shortened",
    r"\[brief",
    r"\.\.\. \(",          # "... (N more)"
    r"\(omitted",
    r"\(elided",
    r"\(truncated",
    r"\(summarized",
    r"\d+ matches",        # "47 matches"
    r"\d+ files",          # "12 files in src/"
    r"\d+ lines",          # "150 lines elided"
    r"\d+ more",           # "and 30 more"
]

SEG_RE = re.compile(r"\[SEG\s+id=([^\s\]]+)\s+kind=([^\s\]]+)\s+level=([^\s\]]+)\](.*?)\[/SEG\]", re.DOTALL)


def count_markers(text: str) -> Counter:
    found = Counter()
    for pat in ABSTRACTIVE_MARKERS:
        n = len(re.findall(pat, text, re.IGNORECASE))
        if n:
            found[pat] = n
    return found


def parse_segments(text: str) -> dict:
    """Return dict {seg_id: (kind, level, content_len)}."""
    out = {}
    for m in SEG_RE.finditer(text):
        seg_id, kind, level, content = m.group(1), m.group(2), m.group(3), m.group(4)
        out[seg_id] = (kind, level, len(content))
    return out


def main():
    samples = []
    with open(VALIDATED, "rb") as f:
        for line in f:
            samples.append(orjson.loads(line))
    print(f"Loaded {len(samples)} passed samples")

    # Per-sample analysis
    per_sample = []
    total_markers = Counter()
    n_with_any_marker = 0

    seg_in_total = 0
    seg_out_total = 0
    seg_shrunk_total = 0   # segments that appear in both but output is shorter
    seg_intact_total = 0   # segments that appear in both with same length
    seg_dropped_total = 0  # segments in input but not output
    seg_added_total = 0    # segments in output but not input

    for s in samples:
        original = extract_input_text(s)
        compressed = s["messages"][2]["content"]
        markers = count_markers(compressed)
        total_markers.update(markers)
        if markers:
            n_with_any_marker += 1

        in_segs = parse_segments(original)
        out_segs = parse_segments(compressed)

        seg_in_total += len(in_segs)
        seg_out_total += len(out_segs)

        dropped = []
        shrunk = []
        intact = []
        added = []
        for sid, (k, l, in_len) in in_segs.items():
            if sid not in out_segs:
                dropped.append((sid, k, l, in_len))
                seg_dropped_total += 1
            else:
                _, _, out_len = out_segs[sid]
                if out_len < in_len * 0.7:
                    shrunk.append((sid, k, l, in_len, out_len))
                    seg_shrunk_total += 1
                else:
                    intact.append((sid, k, l, in_len, out_len))
                    seg_intact_total += 1
        for sid in out_segs:
            if sid not in in_segs:
                added.append(sid)
                seg_added_total += 1

        per_sample.append({
            "sample_id": s["metadata"]["sample_id"],
            "length_bucket": s["metadata"]["length_bucket"],
            "budget": s["metadata"]["compression_budget"],
            "orig_chars": len(original),
            "comp_chars": len(compressed),
            "len_ratio": len(compressed) / max(1, len(original)),
            "in_segs": len(in_segs),
            "out_segs": len(out_segs),
            "n_dropped": len(dropped),
            "n_shrunk": len(shrunk),
            "n_intact": len(intact),
            "n_added": len(added),
            "markers": dict(markers),
            "shrunk_details": shrunk[:5],
            "compressed_text": compressed,
            "original_text": original,
        })

    # Aggregate report
    out_path = DISTILL_DIR / "stage0_abstractive_inspect.md"
    n = len(samples)
    with open(out_path, "w") as f:
        f.write("# Stage 0 — Abstractive vs Extractive Analysis\n\n")
        f.write(f"Passed samples analyzed: **{n}**\n\n")

        f.write("## Summary stats\n\n")
        f.write(f"- Samples with ANY abstractive marker: **{n_with_any_marker}/{n} ({100*n_with_any_marker/n:.0f}%)**\n")
        f.write(f"- Total segments in inputs: {seg_in_total}\n")
        f.write(f"- Total segments in outputs: {seg_out_total}\n")
        f.write(f"- Segments DROPPED (in input, not output): **{seg_dropped_total} ({100*seg_dropped_total/max(1,seg_in_total):.0f}%)**\n")
        f.write(f"- Segments SHRUNK (in both, out_len < 0.7×in_len): **{seg_shrunk_total} ({100*seg_shrunk_total/max(1,seg_in_total):.0f}%)** ← abstractive signal\n")
        f.write(f"- Segments INTACT (in both, out_len >= 0.7×in_len): **{seg_intact_total} ({100*seg_intact_total/max(1,seg_in_total):.0f}%)** ← pure extractive signal\n")
        f.write(f"- Segments ADDED (synthesized in output): {seg_added_total}\n\n")

        f.write("## Marker frequencies (across all 68 outputs)\n\n")
        if total_markers:
            for pat, count in total_markers.most_common():
                f.write(f"- `{pat}` × {count}\n")
        else:
            f.write("- **NONE FOUND** — teacher is not using abstractive markers at all\n")
        f.write("\n")

        # Sort samples by abstractive likelihood: shrunk segment ratio + marker count
        per_sample.sort(
            key=lambda x: (x["n_shrunk"] / max(1, x["in_segs"]) + 0.1 * sum(x["markers"].values())),
            reverse=True,
        )

        f.write("## Top 5 most abstractive samples (highest shrunk-segment ratio)\n\n")
        for x in per_sample[:5]:
            shrunk_pct = 100 * x["n_shrunk"] / max(1, x["in_segs"])
            dropped_pct = 100 * x["n_dropped"] / max(1, x["in_segs"])
            f.write(f"### `{x['sample_id']}`\n\n")
            f.write(f"- segments in: {x['in_segs']}, out: {x['out_segs']}\n")
            f.write(f"- dropped: {x['n_dropped']} ({dropped_pct:.0f}%), shrunk: {x['n_shrunk']} ({shrunk_pct:.0f}%), intact: {x['n_intact']}\n")
            f.write(f"- markers: {x['markers']}\n")
            f.write(f"- len_ratio: {x['len_ratio']:.2f}\n")
            if x["shrunk_details"]:
                f.write("- shrunk segments examples:\n")
                for sid, k, l, in_len, out_len in x["shrunk_details"][:3]:
                    f.write(f"  - seg {sid} kind={k} level={l}: {in_len} → {out_len} chars\n")
            f.write(f"\n<details><summary>compressed output</summary>\n\n```\n{x['compressed_text']}\n```\n\n</details>\n\n---\n\n")

        f.write("## Bottom 5 most extractive samples (lowest shrunk-segment ratio, pure delete)\n\n")
        for x in per_sample[-5:]:
            shrunk_pct = 100 * x["n_shrunk"] / max(1, x["in_segs"])
            dropped_pct = 100 * x["n_dropped"] / max(1, x["in_segs"])
            f.write(f"### `{x['sample_id']}`\n\n")
            f.write(f"- segments in: {x['in_segs']}, out: {x['out_segs']}\n")
            f.write(f"- dropped: {x['n_dropped']} ({dropped_pct:.0f}%), shrunk: {x['n_shrunk']} ({shrunk_pct:.0f}%), intact: {x['n_intact']}\n")
            f.write(f"- markers: {x['markers']}\n")
            f.write(f"- len_ratio: {x['len_ratio']:.2f}\n")
            f.write(f"\n<details><summary>compressed output</summary>\n\n```\n{x['compressed_text']}\n```\n\n</details>\n\n---\n\n")

    print(f"\nWrote: {out_path}")

    # Console summary
    print("\n=== Console summary ===")
    print(f"Samples with abstractive markers: {n_with_any_marker}/{n} ({100*n_with_any_marker/n:.0f}%)")
    print("Segment-level breakdown:")
    print(f"  DROPPED  : {seg_dropped_total}/{seg_in_total} ({100*seg_dropped_total/max(1,seg_in_total):.1f}%)")
    print(f"  SHRUNK   : {seg_shrunk_total}/{seg_in_total} ({100*seg_shrunk_total/max(1,seg_in_total):.1f}%)  ← abstractive")
    print(f"  INTACT   : {seg_intact_total}/{seg_in_total} ({100*seg_intact_total/max(1,seg_in_total):.1f}%)  ← extractive")
    if total_markers:
        print("Top markers:")
        for pat, count in total_markers.most_common(8):
            print(f"  {pat:<25} {count}")
    else:
        print("NO abstractive markers found at all — pure delete-based compression.")


if __name__ == "__main__":
    main()
