"""Per-segment compression analysis on Stage 0 passed samples.

For each preserved segment in each passed output, compute:
  - in_len / out_len ratio
  - longest verbatim substring of output that's in input segment
  - "lazy extractive" flag: long segment kept with high verbatim overlap

Focus on segment kinds where compression opportunity is highest:
  file_read / observation / tool_output / etc.
"""
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import orjson
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "distill_mod", Path(__file__).resolve().parent / "06_distill.py"
)
_distill_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_distill_mod)
extract_input_text = _distill_mod.extract_input_text

VALIDATED = Path("data/distill/stage0_validated.jsonl")
OUT = Path("data/distill/stage0_per_segment.md")

SEG_RE = re.compile(
    r"\[SEG\s+id=([^\s\]]+)\s+kind=([^\s\]]+)\s+level=([^\s\]]+)\](.*?)\[/SEG\]",
    re.DOTALL,
)


def parse_segments(text: str) -> dict:
    """Return {seg_id: {'kind': ..., 'level': ..., 'content': ...}}."""
    out = {}
    for m in SEG_RE.finditer(text):
        sid, kind, level, content = m.group(1), m.group(2), m.group(3), m.group(4)
        out[sid] = {"kind": kind, "level": level, "content": content}
    return out


def longest_verbatim_run(out_content: str, in_content: str, min_len: int = 60) -> int:
    """Length of the longest substring of out_content that appears verbatim in in_content.

    Approximate via sliding chunk match: walk windows of in_content, find the largest
    window that's a substring of out_content. Good for catching "copied long stretches".
    """
    if not out_content or not in_content:
        return 0
    # Try decreasing window sizes
    longest = 0
    # Use lines as units — file_read content is line-structured
    out_lines = out_content.splitlines()
    in_text = in_content
    run = 0
    for line in out_lines:
        if len(line) > 5 and line in in_text:
            run += len(line) + 1
            if run > longest:
                longest = run
        else:
            run = 0
    return longest


def main():
    print("[1/2] Loading passed samples ...")
    samples = []
    with open(VALIDATED, "rb") as f:
        for line in f:
            samples.append(orjson.loads(line))
    print(f"  {len(samples)} samples")

    # Per-segment records across all samples
    segs_by_kind = defaultdict(list)  # kind -> [seg_record]
    sample_has_lazy = []  # list of (sample_id, lazy_segs)

    for s in tqdm(samples, desc="Analyzing"):
        sid = s["metadata"]["sample_id"]
        original = extract_input_text(s)
        compressed = s["messages"][2]["content"]
        in_segs = parse_segments(original)
        out_segs = parse_segments(compressed)

        lazy_in_sample = []
        for seg_id, out_seg in out_segs.items():
            in_seg = in_segs.get(seg_id)
            if not in_seg:
                continue  # added segment, skip
            in_len = len(in_seg["content"])
            out_len = len(out_seg["content"])
            kind = in_seg["kind"]
            level = in_seg["level"]

            # Compute compression
            ratio = out_len / max(1, in_len)
            verb_run = longest_verbatim_run(out_seg["content"], in_seg["content"])
            verb_frac = verb_run / max(1, out_len)  # fraction of output that's a verbatim run

            # "Lazy extractive" criteria:
            #   - in_len > 500 (worth compressing)
            #   - out_len > 200 (not basically gone)
            #   - ratio > 0.5 (kept more than half)
            #   - verb_frac > 0.5 (more than half output is verbatim run)
            lazy = (in_len > 500 and out_len > 200 and ratio > 0.5 and verb_frac > 0.5)

            record = {
                "sample_id": sid,
                "seg_id": seg_id,
                "kind": kind,
                "level": level,
                "in_len": in_len,
                "out_len": out_len,
                "ratio": ratio,
                "verb_run": verb_run,
                "verb_frac": verb_frac,
                "lazy": lazy,
            }
            segs_by_kind[kind].append(record)
            if lazy:
                lazy_in_sample.append(record)

        if lazy_in_sample:
            sample_has_lazy.append((sid, lazy_in_sample))

    # Aggregate by kind
    with open(OUT, "w") as f:
        f.write("# Stage 0 — per-segment compression by kind (passed samples)\n\n")
        f.write(f"Samples: **{len(samples)}**\n\n")
        f.write(f"Samples with at least one LAZY EXTRACTIVE segment: ")
        f.write(f"**{len(sample_has_lazy)} / {len(samples)} ({100*len(sample_has_lazy)/len(samples):.0f}%)**\n\n")
        f.write("Definition of LAZY: `in_len > 500 AND out_len > 200 AND ratio > 0.5 AND verbatim_run_frac > 0.5`\n\n")

        f.write("## Per-kind segment-level stats (kept segments only)\n\n")
        f.write("| kind | n_segs | avg ratio | avg verb_frac | n_lazy | %_lazy |\n")
        f.write("|---|---|---|---|---|---|\n")
        kinds_sorted = sorted(segs_by_kind.keys(), key=lambda k: -len(segs_by_kind[k]))
        for kind in kinds_sorted:
            recs = segs_by_kind[kind]
            n = len(recs)
            avg_ratio = sum(r["ratio"] for r in recs) / n
            avg_vf = sum(r["verb_frac"] for r in recs) / n
            n_lazy = sum(1 for r in recs if r["lazy"])
            f.write(f"| `{kind}` | {n} | {avg_ratio:.2f} | {avg_vf:.2f} | {n_lazy} | {100*n_lazy/n:.0f}% |\n")
        f.write("\n")

        # Focus on the kinds most likely to have lazy extractive
        f.write("## Worst offenders (top 10 LAZY segments by in_len)\n\n")
        all_lazy = [r for recs in segs_by_kind.values() for r in recs if r["lazy"]]
        all_lazy.sort(key=lambda r: -r["in_len"])
        for r in all_lazy[:10]:
            f.write(f"### `{r['sample_id']}` seg `{r['seg_id']}` kind=`{r['kind']}` level=`{r['level']}`\n\n")
            f.write(f"- in_len: {r['in_len']}, out_len: {r['out_len']}, ratio: {r['ratio']:.2f}\n")
            f.write(f"- longest verbatim run: {r['verb_run']} chars ({r['verb_frac']:.0%} of output)\n\n")

        # Examples by kind
        for kind in kinds_sorted:
            recs = segs_by_kind[kind]
            lazy_recs = [r for r in recs if r["lazy"]]
            if not lazy_recs:
                continue
            f.write(f"## Lazy `{kind}` examples (up to 3)\n\n")
            for r in lazy_recs[:3]:
                f.write(f"- `{r['sample_id']}` seg=`{r['seg_id']}` ")
                f.write(f"in={r['in_len']} out={r['out_len']} ratio={r['ratio']:.2f} verb_frac={r['verb_frac']:.0%}\n")
            f.write("\n")

    print(f"\nWrote: {OUT}")
    print(f"\n=== Per-kind compression behavior ===")
    print(f"{'kind':<25} {'n_segs':>7} {'avg_ratio':>10} {'avg_verb':>9} {'n_lazy':>7} {'%_lazy':>7}")
    for kind in kinds_sorted:
        recs = segs_by_kind[kind]
        n = len(recs)
        avg_ratio = sum(r["ratio"] for r in recs) / n
        avg_vf = sum(r["verb_frac"] for r in recs) / n
        n_lazy = sum(1 for r in recs if r["lazy"])
        print(f"  {kind:<23} {n:>7} {avg_ratio:>10.2f} {avg_vf:>9.2f} {n_lazy:>7} {100*n_lazy/n:>6.0f}%")
    print(f"\nSamples w/ ≥1 lazy seg: {len(sample_has_lazy)}/{len(samples)} ({100*len(sample_has_lazy)/len(samples):.0f}%)")


if __name__ == "__main__":
    main()
