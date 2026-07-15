"""Diagnose path occurrence in segmented data."""
import json
import re
from collections import Counter

path_pattern = re.compile(r'"path":\s*"([^"]+)"')
path_count = Counter()
sample_stats = []

with open('data/segmented/swe_rebench.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 100:
            break
        d = json.loads(line)
        sample_paths = []
        for seg in d['input_segments']:
            if seg['kind'] not in ('file_operation', 'file_read'):
                continue
            for m in path_pattern.finditer(seg['content']):
                sample_paths.append(m.group(1))
        c = Counter(sample_paths)
        sample_stats.append({
            'sample_id': d['sample_id'],
            'n_paths': len(sample_paths),
            'unique_paths': len(c),
            'max_repeats': max(c.values()) if c else 0,
        })
        for p in sample_paths:
            path_count[p] += 1

n = len(sample_stats)
print('=== 100 samples path statistics ===')
print(f'Avg paths per sample:     {sum(s["n_paths"] for s in sample_stats)/n:.1f}')
print(f'Avg unique paths per sample: {sum(s["unique_paths"] for s in sample_stats)/n:.1f}')
print(f'Max single-path repeats:  {max(s["max_repeats"] for s in sample_stats)}')

print()
print('=== Top 20 most frequent paths (across all samples) ===')
for p, c in path_count.most_common(20):
    print(f'  {c:>4}x  {p[:80]}')

print()
print('=== First 5 samples ===')
for s in sample_stats[:5]:
    print(f'{s["sample_id"]}: paths={s["n_paths"]} unique={s["unique_paths"]} max_repeats={s["max_repeats"]}')