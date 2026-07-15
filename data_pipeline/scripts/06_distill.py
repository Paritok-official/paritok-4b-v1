# scripts/06_distill.py
"""distillation: gpt-4.1-mini via OpenAI Batch API.

Three-stage rollout:
  Stage 0: 100 samples ($1.5)   — sanity check prompt
  Stage 1: 1000 samples ($15)   — measure reject rate
  Stage 2: 30000 samples ($340) — main training data

Usage:
  python scripts/06_distill.py prepare --stage 0   # 准备 batch input
  python scripts/06_distill.py submit  --stage 0   # 提交 batch
  python scripts/06_distill.py status              # 查 batch 进度
  python scripts/06_distill.py download --batch-id batch_xxx  # 下载结果
  python scripts/06_distill.py validate --stage 0  # 验证 + 统计
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from collections import Counter, defaultdict
import random
import orjson
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# === Config ===
TRAIN_FULL = Path("data/final/train_full_80k.jsonl")  # 80k 池子
DISTILL_DIR = Path("data/distill")
DISTILL_DIR.mkdir(parents=True, exist_ok=True)

STAGE_SIZES = {
    0: 100,
    1: 1000,
    2: 30000,
    3: 50000,  # incremental 备用
}

MODEL = "gpt-4.1-mini-2025-04-14"
SEED = 42

# === Distillation prompt ===
# Loaded from a file so the embedded few-shot examples can contain triple-quotes
# (Python docstrings, code blocks) without escape hell.

DISTILL_SYSTEM_PROMPT = (Path(__file__).resolve().parent.parent / "data" / "distill" / "system_prompt.txt").read_text()


# === Stage 0/1/2/3: Stratified subsampling from train pool ===

def stratified_subsample(pool_path: Path, target_n: int, seed: int = SEED) -> list:
    """Stratified subsample from train pool."""
    print(f"[subsample] Reading {pool_path} ...")
    
    # Group by stratum (length_bucket × action × resolved)
    by_stratum = defaultdict(list)
    with open(pool_path, "rb") as f:
        for i, line in enumerate(tqdm(f, desc="Reading pool")):
            sample = orjson.loads(line)
            meta = sample["metadata"]
            stratum = (
                meta["length_bucket"],
                meta["target_action_name"],
                meta["resolved"],
            )
            by_stratum[stratum].append((i, sample))
    
    total_pool = sum(len(v) for v in by_stratum.values())
    print(f"  Pool size: {total_pool}, strata: {len(by_stratum)}")
    
    # Allocate target per stratum
    rng = random.Random(seed)
    selected = []
    for stratum, items in by_stratum.items():
        share = max(1, int(round(target_n * len(items) / total_pool)))
        share = min(share, len(items))
        sampled = rng.sample(items, share)
        selected.extend(sampled)
    
    # Trim/extend to exact target_n
    rng.shuffle(selected)
    if len(selected) > target_n:
        selected = selected[:target_n]
    
    print(f"  Selected: {len(selected)}")
    return [s for _, s in selected]


# === Prepare: build batch JSONL for OpenAI Batch API ===

def prepare_batch(stage: int):
    """Stage N → data/distill/stage{N}_input.jsonl (OpenAI batch format)."""
    target_n = STAGE_SIZES[stage]
    print(f"[prepare] Stage {stage}: {target_n} samples")
    
    selected = stratified_subsample(TRAIN_FULL, target_n, seed=SEED + stage)
    
    # Save selected sample IDs for later validation
    selected_ids_path = DISTILL_DIR / f"stage{stage}_selected.jsonl"
    with open(selected_ids_path, "wb") as f:
        for s in selected:
            f.write(orjson.dumps({
                "sample_id": s["metadata"]["sample_id"],
                "trajectory_id": s["metadata"]["trajectory_id"],
                "compression_budget": s["metadata"]["compression_budget"],
                "length_bucket": s["metadata"]["length_bucket"],
            }))
            f.write(b"\n")
    print(f"  Wrote selected metadata: {selected_ids_path}")
    
    # Build OpenAI Batch API input format
    batch_input_path = DISTILL_DIR / f"stage{stage}_input.jsonl"
    with open(batch_input_path, "wb") as f:
        for s in tqdm(selected, desc="Building batch input"):
            request = {
                "custom_id": s["metadata"]["sample_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": DISTILL_SYSTEM_PROMPT},
                        # 用 SFT user message(已经是正确格式)
                        # 注意:SFT data 中 messages[1] 是已经构建好的 user message
                        s["messages"][1],
                    ],
                    "temperature": 0.0,  # deterministic
                    # max_tokens 设成 budget × 1.5 给 buffer
                    "max_tokens": int(s["metadata"]["compression_budget"] * 1.5),
                },
            }
            f.write(orjson.dumps(request))
            f.write(b"\n")
    
    size_mb = batch_input_path.stat().st_size / 1e6
    print(f"  Wrote batch input: {batch_input_path} ({size_mb:.1f} MB)")
    
    # Sanity check: print one request
    print(f"\n=== Sample request (first item) ===")
    with open(batch_input_path, "rb") as f:
        first = orjson.loads(f.readline())
        print(f"custom_id: {first['custom_id']}")
        print(f"model: {first['body']['model']}")
        print(f"max_tokens: {first['body']['max_tokens']}")
        print(f"system prompt length: {len(first['body']['messages'][0]['content'])} chars")
        print(f"user prompt length: {len(first['body']['messages'][1]['content'])} chars")
    
    print(f"\nNext step: python scripts/06_distill.py submit --stage {stage}")


# === Submit batch to OpenAI ===

def submit_batch(stage: int):
    """Upload batch input file and create batch."""
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("Install: pip install openai")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Set OPENAI_API_KEY environment variable")
    
    client = OpenAI(api_key=api_key)
    
    batch_input_path = DISTILL_DIR / f"stage{stage}_input.jsonl"
    if not batch_input_path.exists():
        sys.exit(f"Run prepare first: {batch_input_path} not found")
    
    print(f"[submit] Uploading {batch_input_path} ...")
    with open(batch_input_path, "rb") as f:
        upload = client.files.create(file=f, purpose="batch")
    print(f"  File ID: {upload.id}")
    
    print(f"[submit] Creating batch ...")
    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"stage": str(stage), "project": "paritok"},
    )
    print(f"  Batch ID: {batch.id}")
    print(f"  Status: {batch.status}")
    
    # Save batch ID for later
    batch_meta_path = DISTILL_DIR / f"stage{stage}_batch.json"
    with open(batch_meta_path, "w") as f:
        json.dump({
            "stage": stage,
            "batch_id": batch.id,
            "input_file_id": upload.id,
            "submitted_at": time.time(),
        }, f, indent=2)
    print(f"  Saved batch meta: {batch_meta_path}")
    print(f"\nCheck progress: python scripts/06_distill.py status")


# === Status check ===

def check_status():
    """List all submitted batches and their status."""
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("pip install openai")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    
    print(f"{'Stage':<8} {'Batch ID':<35} {'Status':<15} {'Progress':<20}")
    print("-" * 80)
    
    for stage in [0, 1, 2, 3]:
        meta_path = DISTILL_DIR / f"stage{stage}_batch.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        
        batch = client.batches.retrieve(meta["batch_id"])
        completed = batch.request_counts.completed
        total = batch.request_counts.total
        progress = f"{completed}/{total}" if total else "—"
        print(f"{stage:<8} {batch.id[:33]:<35} {batch.status:<15} {progress:<20}")
        
        if batch.status == "completed":
            print(f"  → Output file: {batch.output_file_id}")
            print(f"  → Download: python scripts/06_distill.py download --stage {stage}")


# === Download results ===

def download_batch(stage: int):
    """Download completed batch output."""
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("pip install openai")
    
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    meta_path = DISTILL_DIR / f"stage{stage}_batch.json"
    if not meta_path.exists():
        sys.exit(f"No batch for stage {stage}")
    with open(meta_path) as f:
        meta = json.load(f)
    
    batch = client.batches.retrieve(meta["batch_id"])
    if batch.status != "completed":
        sys.exit(f"Batch not completed yet. Status: {batch.status}")
    
    print(f"[download] Downloading output file {batch.output_file_id} ...")
    content = client.files.content(batch.output_file_id)
    
    output_path = DISTILL_DIR / f"stage{stage}_output.jsonl"
    with open(output_path, "wb") as f:
        f.write(content.read())
    
    size_mb = output_path.stat().st_size / 1e6
    print(f"  Wrote: {output_path} ({size_mb:.1f} MB)")
    print(f"\nNext: python scripts/06_distill.py validate --stage {stage}")


# === Validation ===

SEG_RE = re.compile(
    r"\[SEG\s+id=([^\s\]]+)\s+kind=([^\s\]]+)\s+level=([^\s\]]+)\](.*?)\[/SEG\]",
    re.DOTALL,
)
SHRINK_RATIO = 0.7
LAZY_MIN_IN_LEN = 500
LAZY_MIN_OUT_LEN = 200
LAZY_RATIO = 0.5
LAZY_VERB_FRAC = 0.5
LAZY_FILEREADS_REJECT = 2          # ≥2 lazy file_read segs → reject sample
SHRUNK_MUSTKEEP_RECALL_MIN = 0.7   # L0/L1 SHRUNK segs must keep ≥70% must-keep
LABELED_PATH = Path("data/labeled/swe_rebench.jsonl")


def parse_segments(text: str) -> dict:
    """Return {seg_id: {'kind', 'level', 'content'}}."""
    out = {}
    for m in SEG_RE.finditer(text):
        sid, kind, level, content = m.group(1), m.group(2), m.group(3), m.group(4)
        out[sid] = {"kind": kind, "level": level, "content": content}
    return out


def longest_verbatim_run_lines(out_content: str, in_content: str) -> int:
    """Length of longest run of consecutive output lines that all appear verbatim in input."""
    if not out_content or not in_content:
        return 0
    longest, run = 0, 0
    for line in out_content.splitlines():
        if len(line) > 5 and line in in_content:
            run += len(line) + 1
            if run > longest:
                longest = run
        else:
            run = 0
    return longest


def extract_input_text(sample: dict) -> str:
    """Get the original input text from the SFT sample for comparison."""
    user_msg = sample["messages"][1]["content"]
    # Extract content between ```input_data ... ```
    start = user_msg.find("```input_data\n")
    if start == -1:
        return user_msg
    start += len("```input_data\n")
    end = user_msg.find("\n```\n\n### COMPRESSED OUTPUT", start)
    if end == -1:
        return user_msg[start:]
    return user_msg[start:end]


def validate_one(sample: dict, compressed: str, must_keep_spans: list | None = None) -> tuple[bool, str]:
    """Run all validation rules on one (sample, compressed) pair.

    must_keep_spans: optional list of {seg_id, kind, text}. When provided, runs
    segment-level recall check on L0/L1 SHRUNK segments.
    """
    budget = sample["metadata"]["compression_budget"]
    original = extract_input_text(sample)

    # 1. Length sanity (chars/4 estimate)
    compressed_tokens = len(compressed) // 4
    if compressed_tokens > budget * 1.3:
        return False, "exceeds_budget"
    if compressed_tokens < budget * 0.2:
        return False, "too_short"

    # 2. No identity — output must be meaningfully shorter than input.
    # Tightened from 0.85 to 0.65 to catch WEAK_COMPRESSION leak.
    len_ratio = len(compressed) / max(1, len(original))
    if len_ratio > 0.65:
        return False, "near_identity"

    # 3. Truncation — segment open/close must balance.
    n_open = compressed.count("[SEG id=")
    n_close = compressed.count("[/SEG]")
    if n_open != n_close:
        return False, "truncated_segments"

    # 4. Hallucinated preamble check
    bad_starts = [
        "Here is the compressed", "Here's the compressed",
        "I have compressed", "Below is the compressed",
        "Compressed version:", "The compressed",
        "I've compressed", "Sure, here",
    ]
    stripped = compressed.lstrip()
    for bad in bad_starts:
        if stripped.lower().startswith(bad.lower()):
            return False, "hallucinated_preamble"

    # 5. Code fence wrapping
    if stripped.startswith("```") and compressed.rstrip().endswith("```"):
        return False, "wrapped_in_codefence"

    # 6. QA-style response
    qa_starts = ["Yes,", "No,", "The answer is", "To answer", "Based on"]
    for qa in qa_starts:
        if stripped.startswith(qa):
            return False, "looks_like_qa_response"

    # 7-8. Segment-level checks (need parsed segments)
    in_segs = parse_segments(original)
    out_segs = parse_segments(compressed)

    # 7. Lazy file_read detection — count file_read segs that are mostly verbatim.
    n_lazy_filereads = 0
    for seg_id, out_seg in out_segs.items():
        in_seg = in_segs.get(seg_id)
        if in_seg is None or in_seg["kind"] != "file_read":
            continue
        in_len = len(in_seg["content"])
        out_len = len(out_seg["content"])
        if in_len < LAZY_MIN_IN_LEN or out_len < LAZY_MIN_OUT_LEN:
            continue
        if out_len / max(1, in_len) < LAZY_RATIO:
            continue
        verb_run = longest_verbatim_run_lines(out_seg["content"], in_seg["content"])
        if verb_run / max(1, out_len) > LAZY_VERB_FRAC:
            n_lazy_filereads += 1
    if n_lazy_filereads >= LAZY_FILEREADS_REJECT:
        return False, "lazy_file_reads"

    # 8. SHRUNK must-keep recall on L0/L1 segments (anti-destructive).
    if must_keep_spans:
        spans_by_seg = defaultdict(set)
        for span in must_keep_spans:
            spans_by_seg[span["seg_id"]].add(span["text"])
        for seg_id, out_seg in out_segs.items():
            in_seg = in_segs.get(seg_id)
            if in_seg is None:
                continue
            if in_seg["level"] not in ("L0", "L1"):
                continue
            in_len = len(in_seg["content"])
            out_len = len(out_seg["content"])
            if out_len >= in_len * SHRINK_RATIO:
                continue  # INTACT — separate path
            texts = spans_by_seg.get(seg_id, set())
            if len(texts) < 3:
                continue  # too few spans to draw a recall conclusion
            kept = sum(1 for t in texts if t in out_seg["content"])
            recall = kept / len(texts)
            if recall < SHRUNK_MUSTKEEP_RECALL_MIN:
                return False, "shrunk_lost_mustkeep"

    return True, "ok"


def validate_stage(stage: int):
    """Validate batch output, compute reject statistics."""
    output_path = DISTILL_DIR / f"stage{stage}_output.jsonl"
    selected_path = DISTILL_DIR / f"stage{stage}_selected.jsonl"
    
    if not output_path.exists():
        sys.exit(f"Run download first: {output_path} not found")
    
    # Load original SFT samples by sample_id (need user message + budget)
    print(f"[validate] Loading SFT pool to find inputs ...")
    pool_by_id = {}
    with open(TRAIN_FULL, "rb") as f:
        for line in tqdm(f, desc="Reading pool"):
            sample = orjson.loads(line)
            pool_by_id[sample["metadata"]["sample_id"]] = sample
    print(f"  Loaded {len(pool_by_id)} samples")

    # Get the sample_ids we need (from batch output) and join must_keep_spans
    # from labeled data — required for SHRUNK-level recall check.
    needed_ids = set()
    with open(output_path, "rb") as f:
        for line in f:
            r = orjson.loads(line)
            needed_ids.add(r["custom_id"])

    must_keep_by_id: dict[str, list] = {}
    if LABELED_PATH.exists():
        print(f"[validate] Joining must_keep_spans from {LABELED_PATH} ...")
        with open(LABELED_PATH, "rb") as f:
            for line in tqdm(f, desc="Reading labeled"):
                s = orjson.loads(line)
                sid = s["sample_id"]
                if sid in needed_ids:
                    must_keep_by_id[sid] = s["must_keep_spans"]
                    if len(must_keep_by_id) == len(needed_ids):
                        break
        print(f"  Got spans for {len(must_keep_by_id)}/{len(needed_ids)}")
    else:
        print(f"[validate] WARNING: {LABELED_PATH} missing — SHRUNK recall check skipped")

    # Iterate batch outputs
    print(f"[validate] Validating outputs ...")
    reject_reasons = Counter()
    passed_samples = []
    n_total = 0

    with open(output_path, "rb") as f:
        for line in tqdm(f, desc="Validating"):
            n_total += 1
            result = orjson.loads(line)

            # Batch API result format
            sample_id = result["custom_id"]
            response = result.get("response", {})
            if not response or response.get("status_code") != 200:
                reject_reasons["api_error"] += 1
                continue

            try:
                compressed = response["body"]["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                reject_reasons["malformed_response"] += 1
                continue

            sample = pool_by_id.get(sample_id)
            if sample is None:
                reject_reasons["sample_not_in_pool"] += 1
                continue

            spans = must_keep_by_id.get(sample_id)
            ok, reason = validate_one(sample, compressed, must_keep_spans=spans)
            if not ok:
                reject_reasons[reason] += 1
                continue
            
            # Build SFT sample with assistant filled
            sft_sample = {
                "messages": [
                    sample["messages"][0],  # system
                    sample["messages"][1],  # user
                    {"role": "assistant", "content": compressed},
                ],
                "metadata": sample["metadata"],
            }
            passed_samples.append(sft_sample)
    
    # Stats
    n_passed = len(passed_samples)
    n_rejected = sum(reject_reasons.values())
    print(f"\n=== Stage {stage} Validation Report ===")
    print(f"Total: {n_total}, passed: {n_passed} ({100*n_passed/n_total:.1f}%)")
    print(f"Rejected: {n_rejected} ({100*n_rejected/n_total:.1f}%)")
    print(f"\nReject breakdown:")
    for reason, count in reject_reasons.most_common():
        print(f"  {reason:<25} {count:>5} ({100*count/n_total:.1f}%)")
    
    # Length distribution of passed
    if passed_samples:
        compressed_lens = [
            len(s["messages"][2]["content"]) // 4
            for s in passed_samples
        ]
        budget_lens = [s["metadata"]["compression_budget"] for s in passed_samples]
        
        print(f"\nLength adherence (compressed_tokens / budget):")
        ratios = [c / b for c, b in zip(compressed_lens, budget_lens)]
        ratios.sort()
        n = len(ratios)
        print(f"  p25: {ratios[n//4]:.2f}")
        print(f"  p50: {ratios[n//2]:.2f}")
        print(f"  p75: {ratios[3*n//4]:.2f}")
        print(f"  p95: {ratios[int(n*0.95)]:.2f}")
    
    # Save validated samples for SFT
    final_path = DISTILL_DIR / f"stage{stage}_validated.jsonl"
    with open(final_path, "wb") as f:
        for s in passed_samples:
            f.write(orjson.dumps(s))
            f.write(b"\n")
    print(f"\nSaved validated SFT data: {final_path}")
    
    # Save reject report
    report_path = DISTILL_DIR / f"stage{stage}_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "stage": stage,
            "n_total": n_total,
            "n_passed": n_passed,
            "n_rejected": n_rejected,
            "pass_rate": n_passed / n_total if n_total else 0,
            "reject_reasons": dict(reject_reasons),
        }, f, indent=2)
    print(f"Saved report: {report_path}")
    
    # Decision gate
    pass_rate = n_passed / n_total if n_total else 0
    print(f"\n=== Decision Gate ===")
    if pass_rate < 0.70:
        print(f"⚠️  Pass rate {pass_rate*100:.1f}% < 70%. STOP. Investigate top reject reasons.")
    elif pass_rate < 0.85:
        print(f"⚠️  Pass rate {pass_rate*100:.1f}% in [70, 85). Consider tuning prompt before next stage.")
    else:
        print(f"✅ Pass rate {pass_rate*100:.1f}% ≥ 85%. SAFE to proceed to next stage.")


# === Main ===

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "submit", "status", "download", "validate"])
    parser.add_argument("--stage", type=int, choices=[0, 1, 2, 3], default=None)
    args = parser.parse_args()
    
    if args.action == "prepare":
        prepare_batch(args.stage)
    elif args.action == "submit":
        submit_batch(args.stage)
    elif args.action == "status":
        check_status()
    elif args.action == "download":
        download_batch(args.stage)
    elif args.action == "validate":
        validate_stage(args.stage)


if __name__ == "__main__":
    main()