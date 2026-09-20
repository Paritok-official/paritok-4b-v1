"""Audit a finished SWE-bench Lite compression run from its cache.

`run.py` prints one compression rate and then moves on to scoring. That single
number hides two things a reader should be able to check: how the rate is
distributed across instances (a corpus-level ratio is dominated by the largest
files), and whether the output is actually extractive on data the model never
trained on. Both are recoverable from the resumable cache `run.py` already
writes, with no GPU and no API calls -- so this script re-derives them offline.

  python eval_model/audit_swebench.py \
      --cache eval_model/_work/instances_chunk3000_paritok-4b-v1_latest_ln.jsonl

Compression rate is reported exactly as run.py computes it -- total compressed
tokens divided by total original tokens over every instance, cl100k_base -- so
the headline figure here is the same quantity, not a re-definition. The
per-instance mean is printed beside it because the two differ substantially and
quoting either one alone invites the wrong reading.

Extractiveness mirrors eval/extractiveness.py, which measures the distilled
training corpus. Running the identical measure here answers the question that
one cannot: does the copy behavior hold up on data the model was not trained on?
SWE-bench Lite is held out end to end -- no instance of it appears in training --
though the underlying repositories are popular enough that some also appear in
the SWE-rebench and SWE-Gym trajectories the corpus was distilled from, so this
is unseen-instance generalization, not unseen-repository generalization.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paritok.token_counter import count_tokens  # noqa: E402

# Kept byte-identical to eval/extractiveness.py so the two numbers are comparable.
MARKER = re.compile(
    r"^\s*\[(file|body|lines|imports|thinking|plan)\b.*\]\s*$"
    r"|^\s*\[\d+\s+(more|lines|tests)\b.*\]\s*$"
    r"|^\s*\[.*[x×]\s*\d+\]\s*$",
    re.I,
)
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_\.]{2,}|\d+")

# A chunk that failed compression is emitted byte-for-byte, so a long verbatim
# window of the source surviving in the output is the signature of passthrough.
# 2000 chars sits below run.py's 3000-char chunk size and well above any span a
# genuine keep decision produces.
PASSTHROUGH_WINDOW = 2000


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load(cache: Path) -> list[dict]:
    if not cache.exists():
        raise SystemExit(f"no such cache: {cache}")
    recs = [json.loads(line) for line in cache.open(encoding="utf-8")]
    if not recs:
        raise SystemExit(f"cache is empty: {cache}")
    return recs


def compression(recs: list[dict], seed: int) -> None:
    print("=" * 72)
    print("COMPRESSION RATE")
    print("=" * 72)

    per, missing = [], 0
    for r in recs:
        compressed = r.get("compressed_context") or ""
        if not compressed:
            missing += 1
        o = count_tokens(r["full_context"], "cl100k_base")
        c = count_tokens(compressed, "cl100k_base")
        per.append((r["instance_id"], o, c, c / o if o else 0.0))

    orig = sum(x[1] for x in per)
    comp = sum(x[2] for x in per)
    n = len(per)

    print(f"instances: {n}   with no compressed output: {missing}")
    print(f"corpus-level (run.py's figure): {comp:,}/{orig:,} = {100 * comp / orig:.1f}%")

    # Resample instances, not tokens: the instance is the independent unit.
    rng = random.Random(seed)
    boots = []
    for _ in range(5000):
        s = [per[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(x[2] for x in s) / sum(x[1] for x in s))
    boots.sort()
    print(f"  95% bootstrap CI over instances: "
          f"[{100 * boots[125]:.1f}%, {100 * boots[4875]:.1f}%]")

    ratios = sorted(x[3] for x in per)
    print(f"per-instance mean: {100 * st.mean(ratios):.1f}%   "
          f"median: {100 * ratios[n // 2]:.1f}%")
    print(f"  p10 {100 * ratios[n // 10]:.1f}%   p25 {100 * ratios[n // 4]:.1f}%   "
          f"p75 {100 * ratios[3 * n // 4]:.1f}%   p90 {100 * ratios[9 * n // 10]:.1f}%")
    print(f"  min {100 * ratios[0]:.1f}%   max {100 * ratios[-1]:.1f}%")
    print(f"mean tokens per instance: {orig / n:,.0f} -> {comp / n:,.0f}")

    hits = 0
    for r in recs:
        o, c = r["full_context"], r.get("compressed_context") or ""
        step = PASSTHROUGH_WINDOW
        if any(o[i:i + step] in c for i in range(0, max(len(o) - step, 0), step)):
            hits += 1
    print(f"instances containing a >={PASSTHROUGH_WINDOW}-char verbatim window "
          f"(passthrough upper bound): {hits}/{n}")


def extractiveness(recs: list[dict]) -> None:
    print()
    print("=" * 72)
    print("EXTRACTIVENESS (SWE-bench Lite is held out: never used for training)")
    print("=" * 72)

    verbatim = marker = novel = 0
    copied = emitted = 0
    per_inst = []
    for r in recs:
        original = r["full_context"]
        compressed = r.get("compressed_context") or ""
        flat = normalize(original)

        v = m = nv = 0
        for line in compressed.split("\n"):
            norm = normalize(line)
            if not norm:
                continue
            if MARKER.match(line):
                m += 1
            elif norm in flat:
                v += 1
            else:
                nv += 1
        verbatim += v
        marker += m
        novel += nv

        src = set(TOKEN.findall(original))
        out = TOKEN.findall(compressed)
        c = sum(1 for t in out if t in src)
        copied += c
        emitted += len(out)
        if out:
            per_inst.append((r["instance_id"], c / len(out), len(out)))

    total = verbatim + marker + novel
    print(f"lines emitted: {total:,}")
    print(f"  verbatim {verbatim / total:.1%}   marker {marker / total:.1%}   "
          f"novel {novel / total:.1%}")
    print(f"identifier/path/number tokens emitted: {emitted:,}")
    print(f"  present in the input: {copied / emitted:.1%}   "
          f"not present: {1 - copied / emitted:.1%}")

    per_inst.sort(key=lambda x: x[1])
    n = len(per_inst)
    print(f"per-instance token copy: min {per_inst[0][1]:.1%}   "
          f"p10 {per_inst[n // 10][1]:.1%}   median {per_inst[n // 2][1]:.1%}   "
          f"max {per_inst[-1][1]:.1%}")
    print(f"  instances at >=95%: {sum(1 for x in per_inst if x[1] >= 0.95)}/{n}")
    # The low-copy tail is the near-total-drop tail: when almost everything is
    # discarded the surviving output is mostly summary framing, so the ratio is
    # taken over a handful of tokens. Print the denominators so that is visible.
    print("  lowest 5 (with the token count the ratio is over):")
    for iid, ratio, tok in per_inst[:5]:
        print(f"    {iid:32s} {ratio:6.1%}  over {tok:,} tokens")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", type=Path, required=True,
                    help="the instances_chunk*.jsonl run.py wrote")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    recs = load(args.cache)
    compression(recs, args.seed)
    extractiveness(recs)


if __name__ == "__main__":
    main()
