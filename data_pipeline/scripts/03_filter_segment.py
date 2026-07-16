"""segment splitting + meaningful-sample filtering + budget truncation."""
import sys
from pathlib import Path
import orjson
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

IN = Path("data/parsed/swe_rebench.jsonl")
OUT = Path("data/segmented/swe_rebench.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

MIN_LONG_SEG_TOKENS = 1000          # at least one segment must be this long
MAX_INPUT_TOKENS = 32000            # truncate if exceeded (don't drop)
MAX_SEGMENT_TOKENS = 4000           # split if exceeded

PROTECTED_KINDS = {
    "system_prompt",
    "user_turn_current",
    "user_turn_history",
}

# Tool names that count as "real actions" worth training on.
# (think / task_tracker etc don't depend on input -> low training signal)
REAL_ACTION_NAMES = {
    # OpenHands
    "str_replace_editor",
    "execute_bash",
    # Claude Code style (won't appear in this dataset but kept for future-proof)
    "Read",
    "Edit",
    "Write",
    "Bash",
    "Grep",
    "Glob",
    "MultiEdit",
}

def split_long_segment(seg: dict) -> list[dict]:
    """Split segment > MAX_SEGMENT_TOKENS by line boundaries."""
    if seg["tokens"] <= MAX_SEGMENT_TOKENS:
        return [seg]

    content = seg["content"]
    lines = content.split("\n")
    chunk_chars_limit = MAX_SEGMENT_TOKENS * 4

    chunks = []
    current = []
    current_chars = 0
    for line in lines:
        line_size = len(line) + 1
        if current_chars + line_size > chunk_chars_limit and current:
            chunks.append("\n".join(current))
            current = [line]
            current_chars = line_size
        else:
            current.append(line)
            current_chars += line_size
    if current:
        chunks.append("\n".join(current))

    sub_segs = []
    for i, chunk_text in enumerate(chunks):
        sub = dict(seg)
        sub["seg_id"] = f"{seg['seg_id']}_c{i}"
        sub["content"] = chunk_text
        sub["tokens"] = len(chunk_text) // 4
        sub_segs.append(sub)
    return sub_segs

def has_compressible_content(segments: list[dict]) -> bool:
    """At least one non-protected segment must be sufficiently long."""
    for seg in segments:
        if seg["kind"] in PROTECTED_KINDS:
            continue
        if seg["tokens"] >= MIN_LONG_SEG_TOKENS:
            return True
    return False


def is_meaningful_target(target_action: dict) -> bool:
    """Target must be a real action that depends on input context."""
    if target_action.get("type") != "tool_call":
        return False
    name = target_action.get("name", "")
    return name in REAL_ACTION_NAMES

def truncate_to_budget(segments: list[dict], budget: int) -> list[dict]:
    """If total tokens > budget, drop middle segments while keeping head+tail."""
    total = sum(s["tokens"] for s in segments)
    if total <= budget:
        return segments

    # 策略:保留前 30% segments (含 system / user) + 后 50% segments (近期 context)
    # 中间 20% 删除
    n = len(segments)
    n_head = max(2, n * 30 // 100)
    n_tail = max(2, n * 50 // 100)

    # Drop largest middle segments first
    while total > budget:
        n = len(segments)
        if n_head + n_tail >= n:
            break
        mid_start = n_head
        mid_end = n - n_tail
        if mid_end <= mid_start:
            break
        mid_segs = segments[mid_start:mid_end]
        largest_offset = max(range(len(mid_segs)), key=lambda i: mid_segs[i]["tokens"])
        largest_idx = mid_start + largest_offset
        removed = segments.pop(largest_idx)
        total -= removed["tokens"]

    # Still over budget? drop largest from anywhere (except first 2 = system+user)
    while total > budget and len(segments) > 4:
        candidates = list(range(2, len(segments)))
        largest_idx = max(candidates, key=lambda i: segments[i]["tokens"])
        removed = segments.pop(largest_idx)
        total -= removed["tokens"]

    return segments


def process_sample(sample: dict) -> dict | None:
    # 1. Filter: target must be a real action
    if not is_meaningful_target(sample["target_action"]):
        return None

    # 2. Split long segments
    new_segs = []
    for seg in sample["input_segments"]:
        new_segs.extend(split_long_segment(seg))

    # 3. Filter: must have compressible content
    if not has_compressible_content(new_segs):
        return None

    # 4. Truncate if over budget — track whether truncation happened
    original_total = sum(s["tokens"] for s in new_segs)            
    new_segs = truncate_to_budget(new_segs, MAX_INPUT_TOKENS)
    final_total = sum(s["tokens"] for s in new_segs)              
    sample["was_truncated"] = (final_total < original_total * 0.95) 

    # After truncation, re-check that we still have compressible content
    if not has_compressible_content(new_segs):
        return None

    sample["input_segments"] = new_segs
    sample["total_input_tokens"] = final_total
    return sample


def main():
    n_total = 0
    n_no_real_action = 0
    n_no_compressible = 0
    n_truncated = 0
    n_kept = 0

    # Optional: track target action distribution for sanity
    from collections import Counter
    action_dist_in = Counter()
    action_dist_out = Counter()

    with open(IN, "rb") as fin, open(OUT, "wb") as fout:
        pbar = tqdm(total = 469518, desc="Filtering")
        for line in fin:
            n_total += 1
            pbar.update(1)
            sample = orjson.loads(line)

            tgt_name = sample["target_action"].get("name", "<text>")
            action_dist_in[tgt_name] += 1

            # Check action first (cheapest check)
            if not is_meaningful_target(sample["target_action"]):
                n_no_real_action += 1
                continue

            original_tokens = sample["total_input_tokens"]

            processed = process_sample(sample)
            if processed is None:
                n_no_compressible += 1
                continue

            if processed["total_input_tokens"] < original_tokens * 0.95:
                n_truncated += 1

            action_dist_out[tgt_name] += 1
            fout.write(orjson.dumps(processed))
            fout.write(b"\n")
            n_kept += 1  
        pbar.close()

    print("\n=== Filter Report ===")
    print(f"Total input:        {n_total:>8}")
    print(f"No real action:     {n_no_real_action:>8} ({100*n_no_real_action/n_total:.1f}%)")
    print(f"No compressible:    {n_no_compressible:>8} ({100*n_no_compressible/n_total:.1f}%)")
    print(f"Truncated:          {n_truncated:>8} ({100*n_truncated/n_total:.1f}%)")
    print(f"Kept:               {n_kept:>8} ({100*n_kept/n_total:.1f}%)")
    out_size = OUT.stat().st_size / 1e9
    print(f"Output size:        {out_size:.2f} GB")

    print("\n=== Top action targets (input) ===")
    for name, c in action_dist_in.most_common(10):
        kept = action_dist_out.get(name, 0)
        print(f"  {name:<25} in: {c:>7}  kept: {kept:>7}")


if __name__ == "__main__":
    main()