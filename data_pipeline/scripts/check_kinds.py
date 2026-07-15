import json
from collections import Counter
kc = Counter()
with open('data/labeled/swe_rebench.jsonl') as f:
    for line in f:
        d = json.loads(line)
        for s in d['input_segments']:
            kc[s['kind']] += 1
for k, c in kc.most_common():
    print(f"  {k:<25} {c}")