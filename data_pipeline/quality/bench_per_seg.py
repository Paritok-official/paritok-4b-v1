"""Per-SEG compression bench.

Instead of feeding the full ~29k-token sample context to the model and
parsing one SEG out of the multi-SEG response, this script feeds ONE SEG
at a time directly from gt_samples.jsonl. Each call is small (a few
hundred to a few thousand tokens), so 20 entries finish in <1 min.

Output is written directly in the format review_app reads:
  gt_<label>_samples.jsonl
with v_* / v5_* fields populated, so no build_v5_compare.py step needed.

Usage:
  python bench_per_seg.py --label v8           # gpt-4.1-mini
  python bench_per_seg.py --label v8 --model gpt-4.1-mini-2025-04-14
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROMPT_PATH = ROOT / "system_prompt.txt"
GT_PATH = ROOT / "gt_samples.jsonl"
ENV_PATH = ROOT / ".env"

DEFAULT_MODEL = "gpt-4.1-mini-2025-04-14"


def load_env():
    if not ENV_PATH.exists():
        return
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def build_user_message(entry: dict) -> str:
    """Single-SEG compression task. Mirrors the per-SEG slice of a real
    sample message so the prompt sees the same shape it would in production."""
    intent = (entry.get("user_intent") or "").strip()
    seg_id = entry["seg_id"]
    # kind/level are not stored on entry; the original body includes "[SEG ...]" wrapper
    # but gt_samples' `original` field is the SEG inner body only. Wrap it.
    # Look up kind from entry.original — we don't have it explicitly, so reconstruct
    # the wrapper with placeholders that won't break the prompt's level rules.
    level = entry["level"]
    return (
        "USER INTENT:\n"
        f"{intent}\n\n"
        "Compress the following segment under the rules in your system prompt. "
        "Output only the compressed [SEG]...[/SEG] block (or an empty one to drop):\n\n"
        f"[SEG id={seg_id} kind=file_read level={level}]\n"
        f"{entry['original']}\n"
        f"[/SEG]\n"
    )


def call(client, model: str, system_prompt: str, entry: dict) -> dict:
    user_msg = build_user_message(entry)
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
    }
    # gpt-5 family rejects temperature param; older models accept temperature=0
    if not model.startswith(("gpt-5", "o1", "o3")):
        kwargs["temperature"] = 0
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:
        # If gpt-5 needs different params, surface error verbatim
        raise
    msg = resp.choices[0].message
    usage = resp.usage
    return {
        "entry_id": entry["entry_id"],
        "compressed": msg.content,
        "finish_reason": resp.choices[0].finish_reason,
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
        },
    }


try:
    from lint_compressed import lint as _lint_compressed
except ImportError:
    _lint_compressed = lambda x: x  # noqa: E731


def parse_seg_body(text: str, seg_id: str) -> str | None:
    """Extract the body of the matching [SEG id=seg_id ...]...[/SEG] block.
    Returns None if the SEG is absent or has empty body (=DROP)."""
    if not text:
        return None
    import re
    pattern = re.compile(
        r"\[SEG\s+id=" + re.escape(seg_id) + r"[^\]]*\](.*?)\[/SEG\]",
        re.DOTALL,
    )
    m = pattern.search(text)
    if not m:
        return None
    body = m.group(1).strip("\n")
    return body if body.strip() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--label", required=True, help="output label, e.g. v8")
    parser.add_argument("--max-workers", type=int, default=20)
    args = parser.parse_args()

    load_env()
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("pip install openai")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY missing")
    client = OpenAI(api_key=api_key)

    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    print(f"Loaded system_prompt.txt: {len(system_prompt)} chars")

    entries = [json.loads(l) for l in open(GT_PATH, "r", encoding="utf-8") if l.strip()]
    print(f"GT entries: {len(entries)}")

    print(f"Compressing per-SEG with {args.model} (workers={args.max_workers}) ...")
    t0 = time.time()
    raw_results = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(call, client, args.model, system_prompt, e): e["entry_id"]
                for e in entries}
        done = 0
        for fut in as_completed(futs):
            eid = futs[fut]
            try:
                raw_results[eid] = fut.result()
            except Exception as e:
                raw_results[eid] = {"entry_id": eid, "compressed": "", "error": str(e)}
            done += 1
            print(f"  [{done}/{len(entries)}] done")

    # Save raw API responses for inspection
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    raw_path = ROOT / f"{timestamp}_{args.label}_raw.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(raw_results[e["entry_id"]], ensure_ascii=False) + "\n")

    # Build review-app-ready file
    out_path = ROOT / f"gt_{args.label}_samples.jsonl"
    finish_reasons = {}
    total_prompt = 0
    total_completion = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for e in entries:
            r = raw_results[e["entry_id"]]
            body = parse_seg_body(r.get("compressed") or "", e["seg_id"])
            body = _lint_compressed(body)
            if body is not None and not body.strip():
                body = None  # treat post-lint empty as drop
            chars = len(body) if body else 0
            ratio = round(chars / e["seg_original_chars"], 3) if e["seg_original_chars"] else 0.0

            fr = r.get("finish_reason", r.get("error", "?"))
            finish_reasons[fr] = finish_reasons.get(fr, 0) + 1
            if r.get("usage"):
                total_prompt += r["usage"].get("prompt_tokens", 0)
                total_completion += r["usage"].get("completion_tokens", 0)

            out = dict(e)
            for prefix in ("v5", "v"):
                out[f"{prefix}_compressed"] = body
                out[f"{prefix}_dropped"] = body is None
                out[f"{prefix}_chars"] = chars
                out[f"{prefix}_ratio"] = ratio
                out[f"{prefix}_finish_reason"] = r.get("finish_reason")
                out[f"{prefix}_completion_tokens"] = r.get("usage", {}).get("completion_tokens") if r.get("usage") else None
            out["v_label"] = args.label
            out["gt_approved"] = None
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"\nWrote: {out_path.name}  (and raw: {raw_path.name})")
    print(f"Took: {time.time() - t0:.1f}s")
    print(f"Finish reasons: {finish_reasons}")
    print(f"Total prompt tokens: {total_prompt:,}, completion: {total_completion:,}")
    print(f"\nReview: open review_app  (default points to gt_v7_samples.jsonl; use Open file...")
    print(f"        to switch to {out_path.name})")


if __name__ == "__main__":
    main()
