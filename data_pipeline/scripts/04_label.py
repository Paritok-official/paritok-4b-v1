"""Label each sample with token_labels (L0-L3) and must_keep_spans."""
import sys
import re
from pathlib import Path
import orjson
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mustkeep import find_must_keep_spans

IN = Path("data/segmented/swe_rebench.jsonl")
OUT = Path("data/labeled/swe_rebench.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

PROTECTED_KINDS = {"system_prompt", "user_turn_current", "user_turn_history"}
TOOL_RESULT_KINDS = {"tool_result", "log_output", "file_read"}


def detect_stale_files(segments: list[dict]) -> set[int]:
    """Find stale segments: file accesses that were superseded by later access."""
    path_pattern = re.compile(r'"path":\s*"([^"]+)"')

    fop_positions = {}
    for i, seg in enumerate(segments):
        if seg["kind"] != "file_operation":
            continue
        for m in path_pattern.finditer(seg["content"]):
            path = m.group(1)
            if not path.startswith('/'):
                continue
            if '.' not in path.rsplit('/', 1)[-1]:
                continue
            fop_positions.setdefault(path, []).append(i)

    stale = set()
    for path, positions in fop_positions.items():
        if len(positions) <= 1:
            continue
        for pos in positions[:-1]:
            stale.add(pos)
            if pos + 1 < len(segments) and segments[pos + 1]["kind"] == "tool_result":
                stale.add(pos + 1)

    return stale


def assign_level(seg: dict, seg_idx: int, total_segs: int,
                 stale_indices: set[int]) -> tuple[str, str]:
    """Assign L0/L1/L2/L3 using both relative position and absolute distance."""
    kind = seg["kind"]
    is_current = seg.get("is_current_turn", False)
    relative_pos = seg_idx / max(1, total_segs - 1)
    turns_from_end = total_segs - 1 - seg_idx

    # 综合判断:相对位置或绝对距离任一满足即可
    is_very_recent = (relative_pos > 0.85) or (turns_from_end <= 1)
    is_recent = (relative_pos > 0.7) or (turns_from_end <= 4)
    is_ancient = (relative_pos < 0.25) and (turns_from_end > 8)

    # L0
    if kind in PROTECTED_KINDS:
        return "L0", f"protected={kind}"
    if is_current:
        return "L0", "is_current_turn"
    if is_very_recent and kind in TOOL_RESULT_KINDS:
        return "L0", "very_recent_tool_result"

    # L3
    if seg_idx in stale_indices:
        return "L3", "stale_file_read"
    if is_ancient and kind not in ("file_operation", "bash_command"):
        return "L3", f"ancient(pos={relative_pos:.2f})"

    # L1: 近期 file/result
    if is_recent:
        if kind == "file_read":
            return "L1", "recent_file_read"
        if kind in TOOL_RESULT_KINDS:
            return "L1", f"recent_{kind}"

    # L1: action
    if kind in ("file_operation", "bash_command", "tool_call"):
        return "L1", f"action_{kind}"

    # L2: 中段
    if kind == "file_read":
        return "L2", "mid_file_read"
    if kind == "assistant_thinking":
        return "L2", "thinking"

    if kind in TOOL_RESULT_KINDS:
        if relative_pos < 0.4:
            return "L3", f"old_{kind}"
        return "L2", f"mid_{kind}"

    return "L2", f"default_{kind}"

def reclassify_tool_result(seg: dict) -> str:
    """Re-classify tool_result based on actual content patterns."""
    if seg["kind"] != "tool_result":
        return seg["kind"]

    content = seg["content"]
    head = content[:300]

    # OpenHands 的 cat -n 输出
    if "Here's the result of running `cat -n`" in head:
        return "file_read"

    # str_replace_editor view 输出 (通常是文件内容)
    if head.startswith("Here's the result of running") and "cat" in head:
        return "file_read"

    # 已编辑/创建的文件确认
    if re.match(r"^The file .+ has been (created|edited|saved)", head):
        return "file_edit_confirm"

    # 目录列表
    if "Here's the files and directories" in head:
        return "directory_listing"

    # 包含大量代码的(import / def / class 出现且换行多)
    code_indicators = sum(1 for kw in ["import ", "def ", "class ", "from ", "function ", "package "] if kw in head)
    if code_indicators >= 2 and content.count("\n") > 10:
        return "file_read"

    return "tool_result"

def label_sample(sample: dict) -> dict:
    segments = sample["input_segments"]

    for seg in segments:
        new_kind = reclassify_tool_result(seg)
        if new_kind != seg["kind"]:
            seg["kind"] = new_kind
            seg["_reclassified"] = True

    stale_indices = detect_stale_files(segments)

    token_labels = []
    for i, seg in enumerate(segments):
        level, reason = assign_level(seg, i, len(segments), stale_indices)
        token_labels.append({
            "seg_id": seg["seg_id"],
            "level": level,
            "reason": reason,
        })

    must_keep = []
    for seg in segments:
        spans = find_must_keep_spans(seg["content"], seg["seg_id"], seg["kind"])
        must_keep.extend(spans)

    sample["token_labels"] = token_labels
    sample["must_keep_spans"] = must_keep
    sample["stale_segment_ids"] = [segments[i]["seg_id"] for i in sorted(stale_indices)]

    return sample


def main():
    n_total = n_with_stale = 0
    total_must_keep = 0
    level_counts = {"L0": 0, "L1": 0, "L2": 0, "L3": 0}
    must_keep_kind_counts = {}

    pbar = tqdm(total=423358, desc="Labeling")  # 1000 for testing
    with open(IN, "rb") as fin, open(OUT, "wb") as fout:
        for line in fin:
            n_total += 1
            pbar.update(1)
            sample = orjson.loads(line)
            labeled = label_sample(sample)

            if labeled["stale_segment_ids"]:
                n_with_stale += 1
            total_must_keep += len(labeled["must_keep_spans"])
            for tl in labeled["token_labels"]:
                level_counts[tl["level"]] += 1
            for mk in labeled["must_keep_spans"]:
                k = mk["kind"]
                must_keep_kind_counts[k] = must_keep_kind_counts.get(k, 0) + 1

            fout.write(orjson.dumps(labeled))
            fout.write(b"\n")
            
    pbar.close()

    print("\n=== Label Report ===")
    print(f"Total samples:           {n_total:>8}")
    print(f"Samples with stale files:{n_with_stale:>8} ({100*n_with_stale/n_total:.1f}%)")
    total_segs = sum(level_counts.values())
    print("\nLevel distribution (across all segments):")
    for lvl in ("L0", "L1", "L2", "L3"):
        c = level_counts[lvl]
        print(f"  {lvl}: {c:>10} ({100*c/total_segs:.1f}%)")
    print(f"\nTotal must_keep_spans: {total_must_keep}")
    print(f"Avg per sample:        {total_must_keep/n_total:.1f}")
    print("\nMust-keep span breakdown:")
    for k, c in sorted(must_keep_kind_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:<20} {c:>10} ({100*c/total_must_keep:.1f}%)")
    out_size = OUT.stat().st_size / 1e9
    print(f"\nOutput size: {out_size:.2f} GB")


if __name__ == "__main__":
    main()