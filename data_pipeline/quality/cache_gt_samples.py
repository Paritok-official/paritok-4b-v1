"""One-shot: extract the 18 GT-needed samples from 8GB train_full_80k.jsonl
to gt_pool_cache.jsonl so message.txt can load them in <1s."""
import json
from pathlib import Path

ROOT = Path(__file__).parent
GT_PATH = ROOT / "gt_samples.jsonl"
TRAIN_FULL = ROOT / "train_full_80k.jsonl"
OUT = ROOT / "gt_pool_cache.jsonl"

try:
    import orjson
    loads = orjson.loads
    dumps = lambda o: orjson.dumps(o).decode("utf-8")
    print("using orjson")
except ImportError:
    loads = json.loads
    dumps = json.dumps
    print("using stdlib json (slower)")

needed = set()
with open(GT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            needed.add(json.loads(line)["sample_id"])
print(f"need {len(needed)} samples")

found = []
seen = 0
with open(TRAIN_FULL, "rb") as f:
    for raw in f:
        seen += 1
        if seen % 5000 == 0:
            print(f"  scanned {seen:,} lines, found {len(found)}/{len(needed)}")
        if not raw.strip():
            continue
        s = loads(raw)
        if s["metadata"]["sample_id"] in needed:
            found.append(s)
            if len(found) == len(needed):
                print(f"  scanned {seen:,} lines — all samples found")
                break

with open(OUT, "w", encoding="utf-8") as f:
    for s in found:
        f.write(dumps(s) + "\n")
print(f"wrote {len(found)} samples -> {OUT.name}")
