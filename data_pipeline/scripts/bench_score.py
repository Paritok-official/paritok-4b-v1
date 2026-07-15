"""LLM-as-judge scoring against the 20-entry GT bench.

Usage:
  # Score the existing 'compressed' field in gt_samples.jsonl (4.1-mini baseline)
  python scripts/bench_score.py

  # Score a fresh run from bench_run.py
  python scripts/bench_score.py --candidates data/distill/bench_outputs/<file>.jsonl

  # Override judge model
  python scripts/bench_score.py --judge gpt-4.1-2025-04-14

Output:
  data/distill/bench_scores.md         (human-readable)
  data/distill/bench_scores.jsonl       (per-entry detail)
"""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import orjson
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

GT_PATH = Path("gt_samples.jsonl")
OUT_DIR = Path("data/distill")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_JUDGE = "gpt-4.1-2025-04-14"

SEG_RE = re.compile(
    r"\[SEG\s+id=([^\s\]]+)\s+kind=([^\s\]]+)\s+level=([^\s\]]+)\](.*?)\[/SEG\]",
    re.DOTALL,
)


def categorize(gt_label: str) -> str:
    """Group 20 unique labels into 4 buckets with actionable n."""
    s = gt_label.lower()
    if "dropped" in s:
        return "dropped"
    if "docstring" in s:
        return "docstring"
    if "bash" in s or "grep" in s:
        return "bash_grep"
    return "code_kept"


JUDGE_SYSTEM = """You are a careful, fair evaluator of text compression quality for code-agent context.
Score candidate compressions against a human gold standard.
Output only valid JSON matching the requested schema. No prose outside the JSON."""


def build_judge_user_prompt(entry: dict, candidate_compressed: str, candidate_dropped: bool) -> str:
    gt_action = entry["gt_action"]
    return f"""You are evaluating ONE compression of a single segment.

=== ORIGINAL SEGMENT (the input) ===
{entry["original"]}

=== GROUND TRUTH ===
- GT action: {gt_action}   ("compress" = produce shorter version; "drop" = remove entirely)
- GT rationale: {entry.get("gt_rationale", "")}
- GT label: {entry.get("gt_label", "")}
- GT compressed (if action=compress; empty if drop):
{entry.get("gt_compressed", "")}

=== CANDIDATE ===
- Candidate dropped this segment: {candidate_dropped}
- Candidate compressed text (if not dropped):
{candidate_compressed if not candidate_dropped else "(dropped)"}

Score the CANDIDATE against GT on three integer dimensions:

1. content_fidelity (0-4): Does candidate preserve the same key info (function names, identifiers, signatures, error classes, paths) as GT?
   - GT=drop: 4 if candidate also dropped; 0 if kept verbatim; 1-3 if shrunk but kept some content.
   - GT=compress: 4 if same critical info preserved; 0 if all key info missing or wrong.

2. compression_rate (0-3): Is candidate's length close to GT?
   - GT=drop: 3 if candidate length = 0; 0 if same as original; 1-2 partial shrink.
   - GT=compress: 3 if candidate length within ~50% of GT; 1 if 2-3x off; 0 if more.

3. style_match (0-3): Same compression approach as GT?
   - Same tactic family: placeholder markers ([summary: ...]), verbatim subset selection, abstractive rewrite.
   - If candidate uses a DIFFERENT but reasonable tactic that achieves similar compression AND faithfulness, score 2 (not 0).
   - Only score 0-1 if candidate's tactic is clearly worse or sloppier than GT (e.g., one-sentence summary that loses substance when GT preserved signatures).

total = sum of the three (0-10).

Then write a 1-2 sentence verdict explaining the main differences.

Output strict JSON ONLY:
{{"content_fidelity": int, "compression_rate": int, "style_match": int, "total": int, "verdict": "..."}}"""


def parse_segments(text: str) -> dict:
    out = {}
    for m in SEG_RE.finditer(text):
        out[m.group(1)] = {"kind": m.group(2), "level": m.group(3), "content": m.group(4)}
    return out


def load_candidates_from_run(path: Path) -> dict[str, str]:
    """Load fresh run output: {sample_id: full_compressed_text}."""
    out = {}
    with open(path, "rb") as f:
        for line in f:
            r = orjson.loads(line)
            out[r["sample_id"]] = r["compressed"]
    return out


def get_candidate(entry: dict, candidates_by_sample: dict[str, str] | None) -> tuple[str, bool]:
    """Return (candidate_compressed_text, candidate_dropped)."""
    if candidates_by_sample is None:
        # Baseline: use existing fields. `compressed` may be None for dropped segments.
        cand = entry.get("compressed") or ""
        return cand, bool(entry.get("dropped"))
    sample_text = candidates_by_sample.get(entry["sample_id"])
    if sample_text is None:
        return "", True
    segs = parse_segments(sample_text)
    target_seg_id = entry["seg_id"]
    if target_seg_id not in segs:
        return "", True
    return segs[target_seg_id]["content"], False


def call_judge(client, model: str, entry: dict, candidate: str, dropped: bool) -> dict:
    """Send one judge request."""
    user_prompt = build_judge_user_prompt(entry, candidate, dropped)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    return json.loads(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=None,
                        help="Path to fresh run jsonl (from bench_run.py). If omitted, use existing 'compressed' field.")
    parser.add_argument("--judge", default=DEFAULT_JUDGE)
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

    # Load GT
    entries = []
    with open(GT_PATH, "rb") as f:
        for line in f:
            entries.append(orjson.loads(line))
    print(f"Loaded {len(entries)} GT entries")

    candidates_by_sample = None
    if args.candidates is not None:
        candidates_by_sample = load_candidates_from_run(args.candidates)
        print(f"Loaded {len(candidates_by_sample)} fresh candidate samples from {args.candidates}")
    else:
        print("Using existing `compressed` field from gt_samples.jsonl as baseline")

    # Build (entry, candidate, dropped) triples
    tasks = []
    for entry in entries:
        cand, dropped = get_candidate(entry, candidates_by_sample)
        tasks.append((entry, cand, dropped))

    # Judge in parallel
    print(f"Judging with {args.judge} (workers={args.max_workers}) ...")
    results = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {
            ex.submit(call_judge, client, args.judge, entry, cand, dropped): i
            for i, (entry, cand, dropped) in enumerate(tasks)
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Judging"):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                results[i] = {"error": str(e), "content_fidelity": 0, "compression_rate": 0, "style_match": 0, "total": 0, "verdict": f"ERROR: {e}"}

    # Aggregate
    scored = []
    for (entry, cand, dropped), score in zip(tasks, results):
        rec = {
            "entry_id": entry["entry_id"],
            "sample_id": entry["sample_id"],
            "seg_id": entry["seg_id"],
            "level": entry["level"],
            "gt_action": entry["gt_action"],
            "gt_label": entry["gt_label"],
            "candidate_dropped": dropped,
            "candidate_chars": len(cand),
            "gt_chars": entry.get("gt_chars", 0),
            "scores": score,
        }
        scored.append(rec)

    # Write detail jsonl
    out_jsonl = OUT_DIR / "bench_scores.jsonl"
    with open(out_jsonl, "wb") as f:
        for r in scored:
            f.write(orjson.dumps(r))
            f.write(b"\n")

    # Write markdown report
    totals = [r["scores"].get("total", 0) for r in scored]
    avg_total = sum(totals) / max(1, len(totals))
    by_action = {"compress": [], "drop": []}
    by_level = {"L0": [], "L1": [], "L2": [], "L3": []}
    by_bucket = {"dropped": [], "docstring": [], "code_kept": [], "bash_grep": []}
    for r in scored:
        by_action.setdefault(r["gt_action"], []).append(r["scores"].get("total", 0))
        by_level.setdefault(r["level"], []).append(r["scores"].get("total", 0))
        bucket = categorize(r["gt_label"])
        by_bucket.setdefault(bucket, []).append(r["scores"].get("total", 0))

    out_md = OUT_DIR / "bench_scores.md"
    candidate_source = str(args.candidates) if args.candidates else "baseline (gt_samples.compressed field)"
    with open(out_md, "w") as f:
        f.write("# Bench scores\n\n")
        f.write(f"- Candidate source: `{candidate_source}`\n")
        f.write(f"- Judge: `{args.judge}`\n")
        f.write(f"- Entries: {len(scored)}\n\n")
        f.write(f"## Aggregate\n\n")
        f.write(f"- **avg total**: **{avg_total:.2f} / 10**\n\n")
        f.write(f"### By action\n")
        for act, arr in by_action.items():
            if arr:
                f.write(f"- `{act}`: {sum(arr)/len(arr):.2f} (n={len(arr)})\n")
        f.write(f"\n### By level\n")
        for lvl, arr in by_level.items():
            if arr:
                f.write(f"- `{lvl}`: {sum(arr)/len(arr):.2f} (n={len(arr)})\n")
        f.write(f"\n### By content bucket\n")
        for buck, arr in by_bucket.items():
            if arr:
                f.write(f"- `{buck}`: {sum(arr)/len(arr):.2f} (n={len(arr)})\n")
        f.write("\n## Per-entry\n\n")
        f.write("| entry_id | level | action | label | fid | comp | style | **total** | candidate_chars | gt_chars | verdict |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        scored_sorted = sorted(scored, key=lambda r: r["scores"].get("total", 0))
        for r in scored_sorted:
            s = r["scores"]
            verdict = s.get("verdict", "").replace("\n", " ").replace("|", "\\|")[:140]
            f.write(
                f"| `{r['entry_id'][-22:]}` | {r['level']} | {r['gt_action']} | {r['gt_label']} | "
                f"{s.get('content_fidelity', '?')} | {s.get('compression_rate', '?')} | {s.get('style_match', '?')} | "
                f"**{s.get('total', '?')}** | {r['candidate_chars']} | {r['gt_chars']} | {verdict} |\n"
            )

    print(f"\nWrote: {out_md}")
    print(f"Wrote: {out_jsonl}")
    print(f"\n=== Aggregate ===")
    print(f"Avg total: {avg_total:.2f} / 10")
    for act, arr in by_action.items():
        if arr:
            print(f"  action `{act}`: {sum(arr)/len(arr):.2f}  (n={len(arr)})")
    for lvl, arr in by_level.items():
        if arr:
            print(f"  level  `{lvl}`: {sum(arr)/len(arr):.2f}  (n={len(arr)})")
    for buck, arr in by_bucket.items():
        if arr:
            print(f"  bucket `{buck}`: {sum(arr)/len(arr):.2f}  (n={len(arr)})")


if __name__ == "__main__":
    main()
