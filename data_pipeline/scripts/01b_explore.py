"""scripts/01b_explore.py — Memory-safe exploration."""
from pathlib import Path
import pyarrow.parquet as pq
from collections import Counter

DATA_DIR = Path("data/raw/swe_rebench")
parquet_files = sorted(DATA_DIR.rglob("*.parquet"))
print(f"Found {len(parquet_files)} parquet files")
for f in parquet_files:
    print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")

# 1. Total rows from metadata (instant)
total_rows = 0
for f in parquet_files:
    pf = pq.ParquetFile(f)
    total_rows += pf.metadata.num_rows
print(f"\nTotal rows: {total_rows}")

# 2. Resolved distribution — read only one column (~3 sec, low memory)
print("\nReading 'resolved' column only...")
all_resolved = []
for f in parquet_files:
    t = pq.read_table(f, columns=["resolved"])
    all_resolved.extend(t["resolved"].to_pylist())
print(f"Resolved distribution: {Counter(all_resolved)}")
print(f"Success rate: {sum(all_resolved) / len(all_resolved):.2%}")

# 3. Turn length on 200 sampled rows (~5 sec, low memory)
print("\nSampling 200 rows for turn length distribution...")
table_sample = pq.read_table(
    parquet_files[0],
    columns=["trajectory"],
).slice(0, 200)
turn_lengths = [len(t) for t in table_sample["trajectory"].to_pylist()]
turn_lengths.sort()
print(f"Turn count (n=200): "
      f"min={turn_lengths[0]}, "
      f"p25={turn_lengths[50]}, "
      f"p50={turn_lengths[100]}, "
      f"p75={turn_lengths[150]}, "
      f"max={turn_lengths[-1]}, "
      f"avg={sum(turn_lengths)/200:.1f}")

# 4. Inspect one sample structure
print("\n=== Sample 0 structure ===")
sample_traj = table_sample["trajectory"][0].as_py()
print(f"Trajectory length: {len(sample_traj)}")
roles = Counter(msg.get("role") for msg in sample_traj)
print(f"Role distribution in sample 0: {dict(roles)}")
print(f"First message role: {sample_traj[0].get('role')}")
content = str(sample_traj[0].get('content', ''))[:200]
print(f"First message content (preview):\n{content}")

print("\n Done")