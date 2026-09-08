"""Measure how extractive the compressor actually is.

"Extractive" is a design commitment, not a proof: the system prompt forbids
emitting content that is not in the input, but a closed set of structural
markers is allowed, three kinds are rewritten by rule (assistant_thinking,
meta_action, long string literals), and tool-call JSON is unwrapped into a
compact template. This script turns the adjective into two numbers.

  line level   fraction of emitted lines that are byte-identical (modulo
               whitespace) to a span of the input, vs. closed-vocabulary
               markers, vs. newly generated text. Strict, but it penalizes
               the tool-call unwrapping, where the payload is copied and
               only the framing is regenerated.

  token level  fraction of emitted identifier-like tokens (identifiers,
               dotted paths, numbers) that already appear in the input.
               Robust to reformatting; this is the number that answers
               "can the model inject an identifier the input never had?"

Usage:
  python eval/extractiveness.py                    # full corpus
  python eval/extractiveness.py --stride 3         # 1/3 sample (fast)
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SOURCES = [
    (ROOT / "update/file_read_compressed_all10k_merged_20260625.jsonl", "file_read"),
    (ROOT / "update/other_compressed_all_per_kind.jsonl", None),
]

# The closed marker vocabulary from the two system prompts.
MARKER = re.compile(
    r"^\s*\[(file|body|lines|imports|thinking|plan)\b.*\]\s*$"
    r"|^\s*\[\d+\s+(more|lines|tests)\b.*\]\s*$"
    r"|^\s*\[.*[x×]\s*\d+\]\s*$",
    re.I,
)

# Identifiers, dotted paths, and numbers -- the tokens whose invention would
# actually hurt an agent (a wrong function name breaks an exact-match edit).
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_\.]{2,}|\d+")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def measure(stride: int):
    lines = defaultdict(lambda: {"verbatim": 0, "marker": 0, "novel": 0})
    tokens = defaultdict(lambda: [0, 0])  # [copied, total]

    for path, default_kind in SOURCES:
        if not path.exists():
            raise SystemExit(f"missing corpus: {path}")
        for i, raw in enumerate(open(path)):
            if i % stride:
                continue
            rec = json.loads(raw)
            if rec.get("dropped") or not rec.get("compressed"):
                continue
            kind = rec.get("kind") or default_kind
            original, compressed = rec["original"], rec["compressed"]

            flat = normalize(original)
            for line in compressed.split("\n"):
                norm = normalize(line)
                if not norm:
                    continue
                if MARKER.match(line):
                    lines[kind]["marker"] += 1
                elif norm in flat:
                    lines[kind]["verbatim"] += 1
                else:
                    lines[kind]["novel"] += 1

            src = set(TOKEN.findall(original))
            out = TOKEN.findall(compressed)
            tokens[kind][0] += sum(1 for t in out if t in src)
            tokens[kind][1] += len(out)

    return lines, tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=1,
                    help="evaluate every Nth record (1 = full corpus)")
    args = ap.parse_args()

    lines, tokens = measure(args.stride)

    print(f"{'kind':22s} {'verbatim':>9s} {'marker':>8s} {'novel':>8s} "
          f"{'tok copy':>9s} {'lines':>9s}")
    print("-" * 70)
    agg = {"verbatim": 0, "marker": 0, "novel": 0}
    tc = tt = 0
    for kind in sorted(lines, key=lambda k: -sum(lines[k].values())):
        v = lines[kind]
        n = sum(v.values())
        for key in agg:
            agg[key] += v[key]
        c, t = tokens[kind]
        tc, tt = tc + c, tt + t
        print(f"{kind:22s} {v['verbatim']/n:8.1%} {v['marker']/n:7.1%} "
              f"{v['novel']/n:7.1%} {c/t:8.1%} {n:>9,}")

    total = sum(agg.values())
    print("-" * 70)
    print(f"{'ALL':22s} {agg['verbatim']/total:8.1%} {agg['marker']/total:7.1%} "
          f"{agg['novel']/total:7.1%} {tc/tt:8.1%} {total:>9,}")

    # assistant_thinking is abstractive by rule; report the rest separately so
    # the headline number is not dragged by a carve-out we already disclose.
    at_c, at_t = tokens.get("assistant_thinking", (0, 0))
    if at_t:
        print(f"{'ALL minus thinking':22s} {'':8s} {'':7s} {'':7s} "
              f"{(tc - at_c)/(tt - at_t):8.1%}")


if __name__ == "__main__":
    main()
