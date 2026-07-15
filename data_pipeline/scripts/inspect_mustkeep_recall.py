"""Measure must-keep recall on Stage 0 passed samples.

For each passed sample:
  - join must_keep_spans from data/labeled/ (not preserved in SFT metadata)
  - count how many span texts appear in the compressed output
  - per-kind + per-sample recall

Output:
  data/distill/stage0_mustkeep_recall.md
"""
import re
import sys
import importlib.util
from collections import Counter, defaultdict
from pathlib import Path

import orjson
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse extract_input_text from 06_distill
_spec = importlib.util.spec_from_file_location(
    "distill_mod", Path(__file__).resolve().parent / "06_distill.py"
)
_distill_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_distill_mod)
extract_input_text = _distill_mod.extract_input_text

LABELED = Path("data/labeled/swe_rebench.jsonl")
VALIDATED = Path("data/distill/stage0_validated.jsonl")
OUT = Path("data/distill/stage0_mustkeep_recall.md")

SEG_RE = re.compile(r"\[SEG\s+id=([^\s\]]+)\s+kind=([^\s\]]+)\s+level=([^\s\]]+)\](.*?)\[/SEG\]", re.DOTALL)

SHRINK_THRESHOLD = 0.7  # out_len < 0.7 * in_len → SHRUNK; else INTACT


def parse_segments(text: str) -> dict:
    """Return {seg_id: {'kind': k, 'level': l, 'len': content_len}}."""
    out = {}
    for m in SEG_RE.finditer(text):
        sid, kind, level, content = m.group(1), m.group(2), m.group(3), m.group(4)
        out[sid] = {"kind": kind, "level": level, "len": len(content)}
    return out


def classify_segments(in_segs: dict, out_segs: dict) -> dict:
    """For each seg_id, return state: 'INTACT', 'SHRUNK', 'DROPPED', or 'ADDED'."""
    states = {}
    for sid, info in in_segs.items():
        if sid not in out_segs:
            states[sid] = "DROPPED"
        else:
            in_len = info["len"]
            out_len = out_segs[sid]["len"]
            if out_len < in_len * SHRINK_THRESHOLD:
                states[sid] = "SHRUNK"
            else:
                states[sid] = "INTACT"
    for sid in out_segs:
        if sid not in in_segs:
            states[sid] = "ADDED"
    return states


def main():
    # 1. Load passed samples: {sample_id: (original_text, compressed_text)}
    print("[1/3] Loading passed samples ...")
    passed = {}
    with open(VALIDATED, "rb") as f:
        for line in f:
            s = orjson.loads(line)
            original = extract_input_text(s)
            compressed = s["messages"][2]["content"]
            passed[s["metadata"]["sample_id"]] = (original, compressed)
    print(f"  {len(passed)} passed samples")

    # 2. Scan labeled, pull must_keep_spans for those sample_ids
    print("[2/3] Scanning labeled data for must_keep_spans ...")
    spans_by_id = {}
    needed = set(passed.keys())
    with open(LABELED, "rb") as f:
        for line in tqdm(f, desc="Reading labeled"):
            s = orjson.loads(line)
            sid = s["sample_id"]
            if sid in needed:
                spans_by_id[sid] = s["must_keep_spans"]
                if len(spans_by_id) == len(needed):
                    break
    print(f"  Joined spans for {len(spans_by_id)}/{len(passed)} samples")

    missing = [sid for sid in passed if sid not in spans_by_id]
    if missing:
        print(f"  WARNING: {len(missing)} sample_ids not found in labeled data")

    # 3. Compute recall — split by segment state (INTACT / SHRUNK / DROPPED)
    print("[3/3] Computing recall by segment state ...")
    # by_state[state][kind] -> (kept, total)
    by_state_kept = defaultdict(Counter)
    by_state_total = defaultdict(Counter)
    raw_kept = Counter()
    raw_total = Counter()
    per_sample_recall = []
    by_kind_losses = defaultdict(list)  # losses on INTACT segments only (real issues)

    seg_state_counts = Counter()  # global INTACT/SHRUNK/DROPPED tally

    for sid, spans in spans_by_id.items():
        original, compressed = passed[sid]
        in_segs = parse_segments(original)
        out_segs = parse_segments(compressed)
        states = classify_segments(in_segs, out_segs)

        # tally segment states (only count segments that have at least one must-keep)
        seg_ids_in_spans = {span["seg_id"] for span in spans}
        for seg_id in seg_ids_in_spans:
            seg_state_counts[states.get(seg_id, "MISSING")] += 1

        # per-sample
        sample_kept_intact = 0
        sample_total_intact = 0

        seen_texts = set()
        for span in spans:
            key = (span["kind"], span["text"], span["seg_id"])
            if key in seen_texts:
                continue
            seen_texts.add(key)
            kind = span["kind"]
            text = span["text"]
            state = states.get(span["seg_id"], "MISSING")
            in_output = text in compressed

            raw_total[kind] += 1
            if in_output:
                raw_kept[kind] += 1

            by_state_total[state][kind] += 1
            if in_output:
                by_state_kept[state][kind] += 1

            if state == "INTACT":
                sample_total_intact += 1
                if in_output:
                    sample_kept_intact += 1
                else:
                    if len(by_kind_losses[kind]) < 30:
                        by_kind_losses[kind].append({
                            "sample_id": sid, "text": text,
                            "seg_id": span["seg_id"], "state": state,
                        })

        # per-sample uses INTACT view (strict)
        per_sample_recall.append({
            "sample_id": sid,
            "recall": sample_kept_intact / sample_total_intact if sample_total_intact else 1.0,
            "n_spans": sample_total_intact,
            "n_kept": sample_kept_intact,
        })

    # Write report
    def kind_table(kept_counter, total_counter):
        lines = ["| kind | kept / total | recall |", "|---|---|---|"]
        gk, gt = 0, 0
        for kind in sorted(total_counter.keys(), key=lambda k: -total_counter[k]):
            kept, total = kept_counter[kind], total_counter[kind]
            gk += kept; gt += total
            lines.append(f"| `{kind}` | {kept} / {total} | **{100*kept/total:.1f}%** |")
        lines.append(f"| **OVERALL** | {gk} / {gt} | **{100*gk/max(1,gt):.1f}%** |")
        return "\n".join(lines), gk, gt

    with open(OUT, "w") as f:
        f.write("# Stage 0 — must-keep recall by segment state\n\n")
        f.write(f"Samples analyzed: **{len(per_sample_recall)}**\n\n")
        f.write("## Segment state distribution (only segments containing must-keep)\n\n")
        total_segs_with_mk = sum(seg_state_counts.values())
        for state in ("INTACT", "SHRUNK", "DROPPED"):
            n = seg_state_counts[state]
            f.write(f"- {state}: {n} ({100*n/max(1,total_segs_with_mk):.0f}%)\n")
        f.write("\n")

        f.write("## A. INTACT segments — the strict expectation\n\n")
        f.write("Teacher kept the segment essentially as-is (out_len ≥ 0.7×in_len). Must-keep span SHOULD be in the output.\n")
        f.write("**This is the smoking-gun metric.** Low recall here means the teacher dropped span content despite committing to keep the segment.\n\n")
        tbl, intact_gk, intact_gt = kind_table(by_state_kept["INTACT"], by_state_total["INTACT"])
        f.write(tbl + "\n\n")

        f.write("## B. SHRUNK segments — abstractive rewrite\n\n")
        f.write("Teacher rewrote/shrunk the segment (out_len < 0.7×in_len). Span loss is **expected**.\n")
        f.write("If recall is non-trivial here, regex's verbatim substring check is too strict (partial preservation not credited).\n\n")
        tbl, shrunk_gk, shrunk_gt = kind_table(by_state_kept["SHRUNK"], by_state_total["SHRUNK"])
        f.write(tbl + "\n\n")

        f.write("## C. DROPPED segments — span loss is correct behavior\n\n")
        f.write("Whole segment removed. Span loss is by design (L3 stale, etc.). Showing recall here just as sanity (should be ~0).\n\n")
        tbl, dropped_gk, dropped_gt = kind_table(by_state_kept["DROPPED"], by_state_total["DROPPED"])
        f.write(tbl + "\n\n")

        f.write("## D. RAW recall (everything, no split)\n\n")
        f.write("For backward comparison.\n\n")
        tbl, _, _ = kind_table(raw_kept, raw_total)
        f.write(tbl + "\n\n")

        # Per-sample recall distribution (INTACT view)
        recalls = sorted([x["recall"] for x in per_sample_recall if x["n_spans"] > 0])
        n = len(recalls)
        f.write("## Per-sample INTACT-only recall distribution\n\n")
        f.write(f"- min: {recalls[0]:.3f}\n")
        f.write(f"- p25: {recalls[n//4]:.3f}\n")
        f.write(f"- p50: {recalls[n//2]:.3f}\n")
        f.write(f"- p75: {recalls[3*n//4]:.3f}\n")
        f.write(f"- p95: {recalls[int(n*0.95)]:.3f}\n")
        f.write(f"- max: {recalls[-1]:.3f}\n")
        f.write(f"- avg: {sum(recalls)/n:.3f}\n\n")
        n_below_80 = sum(1 for r in recalls if r < 0.80)
        n_below_90 = sum(1 for r in recalls if r < 0.90)
        f.write(f"- samples with recall < 90%: **{n_below_90}/{n} ({100*n_below_90/n:.0f}%)**\n")
        f.write(f"- samples with recall < 80%: **{n_below_80}/{n} ({100*n_below_80/n:.0f}%)**\n\n")

        # Examples of LOST spans on INTACT segments (the real concern)
        f.write("## Examples of LOST spans on INTACT segments per kind (up to 15 each)\n\n")
        f.write("These are spans where the teacher kept the segment essentially as-is but still dropped the span.\n\n")
        for kind in sorted(by_kind_losses.keys()):
            n_lost = by_state_total["INTACT"][kind] - by_state_kept["INTACT"][kind]
            f.write(f"### `{kind}` ({n_lost} lost on INTACT)\n\n")
            for ex in by_kind_losses[kind][:15]:
                txt = ex["text"].replace("`", "\\`")
                f.write(f"- `{txt}`  (from {ex['sample_id']})\n")
            f.write("\n")

    print(f"\nWrote: {OUT}")

    # Console summary
    def print_table(name, kept_counter, total_counter):
        print(f"\n=== {name} ===")
        gk, gt = 0, 0
        for kind in sorted(total_counter.keys(), key=lambda k: -total_counter[k]):
            kept, total = kept_counter[kind], total_counter[kind]
            gk += kept; gt += total
            print(f"  {kind:<22} {kept:>5}/{total:<5}  {100*kept/total:.1f}%")
        print(f"  {'OVERALL':<22} {gk:>5}/{gt:<5}  {100*gk/max(1,gt):.1f}%")
        return gk, gt

    print(f"\nSegments containing must-keep: INTACT={seg_state_counts['INTACT']}  "
          f"SHRUNK={seg_state_counts['SHRUNK']}  DROPPED={seg_state_counts['DROPPED']}")
    print_table("A. INTACT segments — STRICT (smoking gun)", by_state_kept["INTACT"], by_state_total["INTACT"])
    print_table("B. SHRUNK segments — abstractive (loss expected)", by_state_kept["SHRUNK"], by_state_total["SHRUNK"])
    print_table("C. DROPPED segments — sanity (should be ~0)", by_state_kept["DROPPED"], by_state_total["DROPPED"])

    print(f"\nPer-sample INTACT recall <90%: {n_below_90}/{n} ({100*n_below_90/max(1,n):.0f}%)")
    print(f"Per-sample INTACT recall <80%: {n_below_80}/{n} ({100*n_below_80/max(1,n):.0f}%)")


if __name__ == "__main__":
    main()
