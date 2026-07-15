"""Compress a slice of the file_read SEG pool with gpt-5 (per-SEG, sync).

Reuses update/system_prompt.txt (v15) + lint_compressed.lint(). Modeled after
bench_per_seg.py but reads the random pool created by extract_file_read_pool.py.

Usage:
  # Smoke test: first 1000 SEGs
  python update/compress_pool_file_read.py --limit 1000 --label test1k

  # Full 10000
  python update/compress_pool_file_read.py --limit 10000 --label full10k

  # Try a different model
  python update/compress_pool_file_read.py --limit 1000 --model gpt-4.1-mini-2025-04-14 --label mini_baseline
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
PROMPT_PATH = ROOT / "system_prompt.txt"
DEFAULT_POOL = ROOT / "file_read_pool_10k.jsonl"
ENV_PATH = ROOT / ".env"

sys.path.insert(0, str(ROOT))
try:
    from lint_compressed import lint as _lint
except ImportError:
    _lint = lambda x: x  # noqa: E731

DEFAULT_MODEL = "gpt-5"


def load_env():
    if not ENV_PATH.exists():
        return
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def build_user_message(entry: dict) -> str:
    intent = (entry.get("user_intent") or "").strip()
    seg_id = entry["seg_id"]
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
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_message(entry)},
        ],
    }
    # gpt-5 / o1 / o3 family rejects temperature
    if not model.startswith(("gpt-5", "o1", "o3")):
        kwargs["temperature"] = 0
    resp = client.chat.completions.create(**kwargs)
    return {
        "entry_id": entry["entry_id"],
        "raw_compressed": resp.choices[0].message.content,
        "finish_reason": resp.choices[0].finish_reason,
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
        },
    }


def parse_seg(text: str, seg_id: str) -> tuple[bool, str | None]:
    """Return (seg_found, body).

    seg_found = True iff a [SEG id=seg_id ...]...[/SEG] block exists.
    body     = None if SEG body is empty (= drop) OR seg_found is False.
               non-empty str if model emitted a real kept body.
    """
    if not text:
        return False, None
    pat = re.compile(
        r"\[SEG\s+id=" + re.escape(seg_id) + r"[^\]]*\](.*?)\[/SEG\]",
        re.DOTALL,
    )
    m = pat.search(text)
    if not m:
        return False, None
    body = m.group(1).strip("\n")
    if not body.strip():
        return True, None
    return True, body


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--label", required=True, help="label for output filename")
    parser.add_argument("--pool", default=str(DEFAULT_POOL))
    parser.add_argument("--max-workers", type=int, default=20)
    parser.add_argument("--existing", default=None,
                        help="Previous run jsonl. Entries already present (with no error) are skipped; "
                             "the new output merges existing + newly-compressed entries.")
    parser.add_argument("--retry-errors", action="store_true",
                        help="With --existing, also retry entries that errored last run.")
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

    if not PROMPT_PATH.exists():
        sys.exit(f"Missing prompt: {PROMPT_PATH}")
    if not Path(args.pool).exists():
        sys.exit(f"Pool missing: {args.pool}. Run extract_file_read_pool.py first.")

    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    print(f"Loaded system_prompt.txt: {len(system_prompt)} chars")

    entries = []
    with open(args.pool, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
            if len(entries) == args.limit:
                break
    print(f"Loaded {len(entries)} entries (limit={args.limit}) from {args.pool}")

    # --existing: load and decide which entries to skip / retry
    existing_by_id: dict = {}
    if args.existing:
        ex_path = Path(args.existing)
        if not ex_path.exists():
            sys.exit(f"--existing path missing: {ex_path}")
        with open(ex_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                existing_by_id[rec["entry_id"]] = rec
        print(f"Loaded {len(existing_by_id)} existing records from {ex_path.name}")

    def needs_call(e: dict) -> bool:
        ex = existing_by_id.get(e["entry_id"])
        if ex is None:
            return True
        if args.retry_errors and ex.get("error"):
            return True
        return False

    to_compress = [e for e in entries if needs_call(e)]
    n_skipped = len(entries) - len(to_compress)
    print(f"To compress: {len(to_compress)}; skipped (from existing): {n_skipped}")

    print(f"Compressing {len(to_compress)} new entries with {args.model} (workers={args.max_workers}) ...")
    t0 = time.time()
    results: dict = {}
    completed = 0

    if to_compress:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futs = {ex.submit(call, client, args.model, system_prompt, e): e["entry_id"]
                    for e in to_compress}
            pbar = tqdm(as_completed(futs), total=len(to_compress), desc="compressing", smoothing=0.05)
            for fut in pbar:
                eid = futs[fut]
                try:
                    results[eid] = fut.result()
                except Exception as e:
                    results[eid] = {"entry_id": eid, "raw_compressed": "", "error": str(e)}
                completed += 1
                if completed % 50 == 0 or completed == len(to_compress):
                    elapsed = time.time() - t0
                    rate = completed / max(1, elapsed)
                    eta = (len(to_compress) - completed) / rate if rate > 0 else 0
                    pbar.write(f"  [{completed}/{len(to_compress)}]  rate {rate:.2f}/s  ETA {eta:.0f}s  elapsed {elapsed:.0f}s")

    # Parse + lint + write
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = ROOT / f"file_read_compressed_{args.label}_{timestamp}.jsonl"
    n_drop = 0
    n_error = 0
    n_orphan = 0  # body could not be parsed
    total_prompt = 0
    total_completion = 0
    finish_reasons: dict = {}
    by_level_ratio: dict = {}
    by_level_drop: dict = {}

    with open(out_path, "w", encoding="utf-8") as f:
        for e in entries:
            eid = e["entry_id"]

            # Reuse existing record verbatim for entries we skipped
            if eid in existing_by_id and eid not in results:
                ex_rec = existing_by_id[eid]
                if ex_rec.get("dropped"):
                    n_drop += 1
                    by_level_drop[e["level"]] = by_level_drop.get(e["level"], 0) + 1
                else:
                    by_level_ratio.setdefault(e["level"], []).append(ex_rec.get("ratio", 0.0))
                if ex_rec.get("error"):
                    n_error += 1
                fr = ex_rec.get("finish_reason", "error" if ex_rec.get("error") else "?")
                finish_reasons[fr] = finish_reasons.get(fr, 0) + 1
                f.write(json.dumps(ex_rec, ensure_ascii=False) + "\n")
                continue

            # New / retried entry
            r = results.get(eid, {})
            raw = r.get("raw_compressed", "") or ""
            seg_found, body = parse_seg(raw, e["seg_id"])
            if not seg_found and raw and not r.get("error"):
                n_orphan += 1
            body = _lint(body)
            if body is not None and not body.strip():
                body = None
            chars = len(body) if body else 0
            ratio = round(chars / e["seg_original_chars"], 3) if e["seg_original_chars"] else 0.0

            if body is None:
                n_drop += 1
                by_level_drop[e["level"]] = by_level_drop.get(e["level"], 0) + 1
            else:
                by_level_ratio.setdefault(e["level"], []).append(ratio)

            if r.get("error"):
                n_error += 1
            fr = r.get("finish_reason", "error" if r.get("error") else "?")
            finish_reasons[fr] = finish_reasons.get(fr, 0) + 1
            if r.get("usage"):
                total_prompt += r["usage"].get("prompt_tokens", 0)
                total_completion += r["usage"].get("completion_tokens", 0)

            out = dict(e)
            out["compressed"] = body
            out["dropped"] = body is None
            out["compressed_chars"] = chars
            out["ratio"] = ratio
            out["finish_reason"] = r.get("finish_reason")
            out["error"] = r.get("error")
            out["raw_compressed"] = r.get("raw_compressed", "")
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    print()
    print(f"Wrote: {out_path}")
    print(f"Took: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Drops: {n_drop}/{len(entries)} ({100*n_drop/len(entries):.1f}%)")
    print(f"Errors: {n_error}")
    print(f"Orphan output (couldn't parse SEG body): {n_orphan}")
    print(f"Finish reasons: {finish_reasons}")
    print(f"Tokens: prompt {total_prompt:,}  completion {total_completion:,}")
    print()
    print("Per-level breakdown (kept entries):")
    for level in sorted(set(list(by_level_ratio) + list(by_level_drop))):
        arr = by_level_ratio.get(level, [])
        n_kept = len(arr)
        n_drop_l = by_level_drop.get(level, 0)
        n_total = n_kept + n_drop_l
        avg = sum(arr) / n_kept if n_kept else 0.0
        print(f"  {level}: kept={n_kept}/{n_total} ({100*n_kept/max(1,n_total):.0f}%), "
              f"drop={n_drop_l}, avg ratio (kept)={avg:.3f}")


if __name__ == "__main__":
    main()
