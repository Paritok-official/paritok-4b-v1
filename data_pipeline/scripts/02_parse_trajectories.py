"""Parse SWE-rebench trajectories → list of TurnSample (jsonl), streaming."""
import sys
import json
from pathlib import Path
import orjson
import pyarrow.parquet as pq
from tqdm import tqdm

# 让 import 能找到 src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schema import Segment, TurnSample
# from src.tokenizer_utils import count_tokens

IN_DIR = Path("data/raw/swe_rebench")
OUT = Path("data/parsed/swe_rebench.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

# 控制单条 trajectory 转出多少个训练样本
MAX_SAMPLES_PER_TRAJ = 8  # 太长的 trajectory 只取前 30 个 decision 点
BATCH_SIZE = 200  # parquet row group 批大小


def classify_segment_kind(msg: dict, position_idx: int) -> str:
    role = msg.get("role")
    if role == "system":
        return "system_prompt"
    if role == "user":
        return "user_turn_current" if position_idx == 0 else "user_turn_history"
    if role == "assistant":
        if msg.get("tool_calls"):
            # 看第一个 tool_call 的 name 来细分
            tc = msg["tool_calls"][0]
            name = tc.get("function", {}).get("name", "")
            if name in ("think",):
                return "assistant_thinking"
            if name in ("task_tracker", "finish"):
                return "meta_action"
            if name in ("str_replace_editor",):
                return "file_operation"
            if name in ("execute_bash", "bash"):
                return "bash_command"
            return "tool_call"  # 兜底
        return "assistant_thinking"
    if role == "tool":
        content = msg.get("content", "") or ""
        head = content[:200]
        if "Traceback" in content or "FAILED" in content or "Error:" in head:
            return "log_output"
        if head.strip().startswith(("/", ".", "#!")) or "@@" in head:
            return "file_read"
        if "\n" in head and head.count("\n") > 5:
            return "log_output"
        return "tool_result"
    return "tool_result"


def build_segments(history: list[dict]) -> list[Segment]:
    segments = []
    for j, msg in enumerate(history):
        kind = classify_segment_kind(msg, j)
        content = msg.get("content", "") or ""

        # 截断 OpenHands 的超长 system prompt
        if kind == "system_prompt" and len(content) > 500:
            content = content[:500] + "\n[...truncated...]"
            
        # tool_calls 规范化处理
        if msg.get("tool_calls"):
            normalized_calls = []
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                args = fn.get("arguments", "")
                # 反序列化 arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass  # 解析失败就保持 string
                normalized_calls.append({
                    "name": fn.get("name", ""),
                    "args": args,
                })
            # 用规范化的格式拼到 content
            tool_str = json.dumps(normalized_calls, ensure_ascii=False)
            if content:
                content = content + "\n[tool_calls]: " + tool_str
            else:
                content = "[tool_calls]: " + tool_str

        seg = Segment(
            seg_id=f"s{j}",
            role=msg.get("role", "unknown"),
            kind=kind,
            content=content,
            tokens=len(content) // 4,
            turn_idx=j,
            is_current_turn=(j == len(history) - 1),
        )
        segments.append(seg)
    return segments


def build_target_action(target_msg: dict) -> dict:
    if target_msg.get("tool_calls"):
        tc = target_msg["tool_calls"][0]
        args = tc.get("function", {}).get("arguments", "")
        # arguments 在原始 parquet 里是 JSON string,反序列化一下
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                pass
        return {
            "type": "tool_call",
            "name": tc.get("function", {}).get("name", ""),
            "args": args,
        }
    return {
        "type": "text_response",
        "content": target_msg.get("content", "") or "",
    }


def trajectory_to_samples(row: dict) -> list[TurnSample]:
    """One trajectory → multiple turn-level samples."""
    samples = []
    trajectory = row["trajectory"]

    decision_indices = [
        i for i, msg in enumerate(trajectory)
        if msg.get("role") == "assistant"
    ]

    # 限制每条 trajectory 最多产出 N 个样本
    if len(decision_indices) > MAX_SAMPLES_PER_TRAJ:
        # 均匀采样
        step = len(decision_indices) / MAX_SAMPLES_PER_TRAJ
        decision_indices = [
            decision_indices[int(i * step)] for i in range(MAX_SAMPLES_PER_TRAJ)
        ]

    for idx_i, dec_i in enumerate(decision_indices):
        if dec_i < 3:
            continue  # 没有 history,跳过

        history = trajectory[:dec_i]
        target_msg = trajectory[dec_i]

        segments = build_segments(history)
        target_action = build_target_action(target_msg)

        sample = TurnSample(
            sample_id=f"{row['trajectory_id']}_t{idx_i}",
            trajectory_id=row["trajectory_id"],
            turn_idx=idx_i,
            repo=row.get("repo", ""),
            resolved=bool(row.get("resolved", 0)),
            input_segments=segments,
            target_action=target_action,
            total_input_tokens=sum(s.tokens for s in segments),
        )
        samples.append(sample)

    return samples


def main():
    parquet_files = sorted(IN_DIR.rglob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")

    n_in = n_out = n_err = 0
    pbar = tqdm(total=67074, desc="Parsing")

    with open(OUT, "wb") as fout:
        for pf_path in parquet_files:
            pf = pq.ParquetFile(pf_path)
            # 按 batch 流式读取,避免 OOM
            for batch in pf.iter_batches(batch_size=BATCH_SIZE):
                rows = batch.to_pylist()
                for row in rows:
                    n_in += 1
                    pbar.update(1)
                    try:
                        samples = trajectory_to_samples(row)
                        for s in samples:
                            fout.write(orjson.dumps(s.to_dict()))
                            fout.write(b"\n")
                            n_out += 1
                    except Exception as e:
                        n_err += 1
                        if n_err <= 5:
                            tqdm.write(f"[error] {row.get('trajectory_id')}: {e}")
                        continue         
    pbar.close()

    print(f"\nDone. trajectories: {n_in}, samples: {n_out}, errors: {n_err}")
    out_size = OUT.stat().st_size / 1e9
    print(f"Output size: {out_size:.2f} GB")


if __name__ == "__main__":
    main()