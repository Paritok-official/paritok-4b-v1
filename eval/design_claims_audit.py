"""Audit the two design claims the paper makes about the training signal.

The paper claims the compressor is (a) level-aware -- four importance levels
L0-L3 with distinct compression budgets -- and (b) intent-conditioned -- what
is kept depends on the agent's current task. Both are properties of the
distilled targets before they are properties of the student, so both can be
checked directly on the corpus. This script does that.

Claim 1 -- level separation
  For each level, the realized compression ratio distribution, plus the
  probability that a random segment from level i is compressed harder than a
  random segment from level i+1. A value near 0.5 means the two levels are
  indistinguishable; the budget table says it should be well below 0.5.

Claim 2 -- intent conditioning
  Segments are bucketed by how much of their identifier set is named in the
  USER INTENT. If the teacher is intent-conditioned, high-overlap segments
  should be dropped less. Reported twice: raw, and controlled for (kind,
  level) -- because file_read both overlaps intent more and is kept more, so
  the raw effect is heavily confounded and the controlled one is the honest
  number.

Usage:
  python eval/design_claims_audit.py [--stride 2]
"""

import argparse
import json
import random
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SOURCES = [
    (ROOT / "update/file_read_compressed_all10k_merged_20260625.jsonl", "file_read"),
    (ROOT / "update/other_compressed_all_per_kind.jsonl", None),
]

LEVELS = ["L0", "L1", "L2", "L3"]
BUDGET = {"L0": 0.50, "L1": 0.35, "L2": 0.25, "L3": 0.20}

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")
# Boilerplate shared by every SWE-bench intent, plus generic English. Left in,
# these dominate the overlap and wash the signal out.
STOP = set("""self this that from import return None True False def class the and for with
uploaded_files workspace issue_description repository description consider following python
code directory have been should would could there their which when what will your""".split())


def identifiers(text):
    return {w for w in IDENT.findall(text or "") if w.lower() not in STOP}


def load(stride):
    recs = []
    for path, default_kind in SOURCES:
        if not path.exists():
            raise SystemExit(f"missing corpus: {path}")
        for i, raw in enumerate(open(path)):
            if i % stride:
                continue
            d = json.loads(raw)
            dropped = bool(d.get("dropped") or not d.get("compressed"))
            chars = d.get("seg_original_chars") or len(d["original"])
            recs.append({
                "kind": d.get("kind") or default_kind,
                "level": d.get("level"),
                "dropped": dropped,
                "cr": None if dropped or not chars else d["compressed_chars"] / chars,
                "intent": d.get("user_intent"),
                "original": d.get("original"),
            })
    return recs


def claim_level(recs):
    print("=" * 72)
    print("CLAIM 1 -- level separation")
    print("=" * 72)
    cr = defaultdict(list)
    drop = defaultdict(lambda: [0, 0])
    for r in recs:
        drop[r["level"]][1] += 1
        if r["dropped"]:
            drop[r["level"]][0] += 1
        else:
            cr[r["level"]].append(r["cr"])

    print(f"{'level':6s} {'budget':>7s} {'n':>7s} {'p25':>7s} {'median':>7s} "
          f"{'p75':>7s} {'drop rate':>10s}")
    for lv in LEVELS:
        v = sorted(cr[lv])
        if not v:
            continue
        n = len(v)
        print(f"{lv:6s} {BUDGET[lv]:7.2f} {n:>7,} {v[n//4]:7.3f} {v[n//2]:7.3f} "
              f"{v[3*n//4]:7.3f} {drop[lv][0]/drop[lv][1]:>9.1%}")

    print("\nP(level i compressed harder than level i+1) -- 0.5 means no separation:")
    rng = random.Random(0)
    for a, b in zip(LEVELS, LEVELS[1:]):
        if not cr[a] or not cr[b]:
            continue
        p = sum(1 for _ in range(20000)
                if rng.choice(cr[a]) < rng.choice(cr[b])) / 20000
        verdict = "SEPARATED" if p < 0.40 else "NOT SEPARATED"
        print(f"  P({a} < {b}) = {p:.3f}   {verdict}")


def claim_intent(recs):
    print()
    print("=" * 72)
    print("CLAIM 2 -- intent conditioning")
    print("=" * 72)
    rows = []
    for r in recs:
        ii, si = identifiers(r["intent"]), identifiers(r["original"])
        if not ii or not si:
            continue
        rows.append((len(ii & si) / len(si), r["dropped"], r["cr"], r["kind"], r["level"]))

    rows.sort(key=lambda x: x[0])
    print(f"\nRAW (confounded), {len(rows):,} segments bucketed by intent overlap:")
    print(f"{'overlap':16s} {'n':>7s} {'drop rate':>10s} {'CR if kept':>11s}")
    buckets = 6
    per = len(rows) // buckets
    for b in range(buckets):
        chunk = rows[b*per:(b+1)*per] if b < buckets-1 else rows[b*per:]
        dr = sum(1 for r in chunk if r[1]) / len(chunk)
        kept = [r[2] for r in chunk if r[2] is not None]
        print(f"{chunk[0][0]:.3f}-{chunk[-1][0]:.3f}    {len(chunk):>7,} "
              f"{dr:>9.1%} {st.mean(kept):>10.3f}")

    # Controlled: split within each (kind, level) cell so neither can drive it.
    cells = defaultdict(list)
    for ov, dropped, _, kind, lv in rows:
        cells[(kind, lv)].append((ov, dropped))

    print(f"\nCONTROLLED for (kind, level) -- cells with n>=150:")
    print(f"{'kind':20s} {'lv':4s} {'n':>6s} {'low-ov drop':>12s} "
          f"{'high-ov drop':>13s} {'delta':>8s}")
    lo_w = hi_w = n_w = 0
    right = total = 0
    for (kind, lv), v in sorted(cells.items(), key=lambda x: -len(x[1])):
        if len(v) < 150:
            continue
        v.sort(key=lambda x: x[0])
        h = len(v) // 2
        lo = sum(1 for _, d in v[:h] if d) / h
        hi = sum(1 for _, d in v[h:] if d) / (len(v) - h)
        total += 1
        right += hi < lo
        lo_w += lo * len(v); hi_w += hi * len(v); n_w += len(v)
        print(f"{kind:20s} {lv:4s} {len(v):>6,} {lo:>11.1%} {hi:>12.1%} {hi-lo:>+8.1%}")

    print(f"\nweighted: low-overlap {lo_w/n_w:.1%} -> high-overlap {hi_w/n_w:.1%} "
          f"({hi_w/n_w - lo_w/n_w:+.1%})")
    print(f"cells in the expected direction: {right}/{total}")
    print("\nNote: cells governed by rule rather than intent (meta_action stale-plan,"
          "\nassistant_thinking decision-or-drop, bash_command keep-if-short) are"
          "\nexpected to show no effect, and do not.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()
    recs = load(args.stride)
    claim_level(recs)
    claim_intent(recs)


if __name__ == "__main__":
    main()
