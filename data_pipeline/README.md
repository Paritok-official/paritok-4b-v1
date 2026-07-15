# SWE Trajectory Compression — Teacher Pipeline

Context-compression teacher for distilling a student model that compresses
multi-SEG code-agent contexts. The teacher takes one `[SEG]` block + the
user's task intent and returns a compressed `[SEG]` block (or empty body
to drop). A student model is then SFT-trained on (input, teacher_output) pairs.

## Files

```
system_prompt.txt           # Teacher system prompt (v15, code-form, ✓/✗ examples)
lint_compressed.py          # Deterministic post-processor (stdlib import strip,
                            #   chain assignments, multi-line call collapse, etc.)
bench_per_seg.py            # OpenAI client: reads gt_samples.jsonl → calls teacher
                            #   per-SEG → applies lint → writes gt_<label>_samples.jsonl
cache_gt_samples.py         # One-shot: extracts samples needed for the 18 GT
                            #   entries from the full 80k pool

gt_samples.jsonl            # 18 hand-authored GT entries (anchor set)
gt_sample_index.json        # Index: entry_id → stratified label
gt_v15_gpt5_samples.jsonl   # gpt-5 teacher output on those 18 — the QUALITY BAR
                            #   reference. Distilled student should match these.

file_read_seg.jsonl         # 468 file_read SEGs extracted from stage0_validated
                            #   (the pool the GT 18 were sampled from)
gt_pool_cache.jsonl         # 18 full samples (sample-level context) needed if
                            #   you want to run a sample-level bench instead of
                            #   per-SEG (not used by bench_per_seg.py)

review_app/                 # Electron 3-pane reviewer: ORIGINAL | TEACHER | GT
                            #   (editable, with auto-save and approval scoring)
```

## Pipeline

```
[gt_samples.jsonl + system_prompt.txt + .env]
            ↓
    bench_per_seg.py
            ↓ (per-SEG calls to gpt-5 or gpt-4.1-mini)
    raw model output
            ↓
    parse [SEG]...[/SEG] body
            ↓
    lint_compressed.py        ← stdlib import strip, blank-line collapse,
            ↓                    chain assignment, multi-line call collapse,
                                 orphan decorator strip, line-num prefix strip
    gt_<label>_samples.jsonl  ← teacher output, ready to compare vs GT
```

## How to run a teacher bench (verify quality on the 18 anchors)

```bash
# 1. Install
pip install openai

# 2. Set API key
echo 'OPENAI_API_KEY=sk-...' > .env

# 3. Run (per-SEG, ~10s with gpt-4.1-mini, ~50s with gpt-5)
python bench_per_seg.py --label v15_test                       # default = gpt-4.1-mini
python bench_per_seg.py --model gpt-5 --label v15_gpt5_test
python bench_per_seg.py --model gpt-4.1-2025-04-14 --label v15_full_test

# Output: gt_v15_test_samples.jsonl
```

Compare against the included `gt_v15_gpt5_samples.jsonl` reference — drop
direction and ratios should be similar. If teacher avg ratio diverges from
GT by > 20% or drop direction errors > 4/18, something regressed.

## Verify with the review_app

```bash
cd review_app
npm install         # installs electron + deps
npm start           # opens the 3-pane reviewer
# Click "Open file…" → choose your gt_<label>_samples.jsonl
```

The reviewer auto-saves edits to the gt_compressed field on blur, so if a
human disagrees with the teacher they can fix it inline.

## Scaling to Stage 1 / Stage 2 distillation

The included `bench_per_seg.py` is the anchor-set runner. To scale to
the full distillation (Stage 1 = 1k samples, Stage 2 = 30k):

1. **Don't** call per-SEG for the full corpus — that's `N_segs × API call`
   which is expensive at scale. Use the OpenAI **Batch API** with sample-level
   prompts (see `06_distill.py` in the parent project for the batch-input
   builder).
2. The same `system_prompt.txt` works for sample-level (multi-SEG) prompts —
   model emits all SEGs in one response. Parse with the regex used in
   `bench_per_seg.py`'s `parse_seg_body` (extended for multiple SEGs).
3. Apply `lint_compressed.lint(body)` to every parsed SEG body before
   writing the final SFT dataset.

Cost estimates (30k samples ≈ 500k SEGs):
  - gpt-4.1-mini : ~$5–15
  - gpt-4.1 full : ~$40–60
  - gpt-5        : ~$150–250
  - claude-sonnet-4-6 : ~$40–60

Recommendation: **gpt-5 for Stage 2.** On the 18 GT anchors it hit 17/18
drop-direction correct vs gpt-4.1-mini's 14/18, and it emits the
`[lines L1-L2: ...]` and docstring-continuation markers that smaller
models silently skip. The marker behavior is critical because the student
will learn whatever the teacher demonstrates — if mini omits markers,
student learns to drop those tail accessors entirely.

## Prompt iteration history (for context)

The teacher prompt went through v7 → v15. Key learnings baked into v15:
  - **Code-form output, never prose.** No "X.py excerpt: defines a transform that…"
  - **Drop is expected.** Default to compress, but D1–D5 in Step 1 enumerate
    the cases where drop is correct. Mini undershoots drop count; gpt-5 hits it.
  - **`[lines L1-L2: fnA / fnB — note]` marker** for tail accessors / unrelated
    methods inside a SEG that's otherwise relevant.
  - **Docstring continuation marker** for SEGs that are purely mid-docstring.
  - **String literal abbreviation** for long warn()/raise() messages.
  - **All format/whitespace nits are handled by `lint_compressed.py`**, not
    the prompt — that's why the prompt is 8.7k chars instead of 18k.

## What the student is being trained to do

Given a sample-level prompt of the form
```
<system: distill prompt>
<user: USER INTENT + [SEG s0 kind=... level=L0]...[/SEG] [SEG s1 ...]... etc.>
```
the student should emit the same multi-SEG output the teacher does, with
the same drop/keep/compress decisions and the same markers. Quality target:
match the gpt-5 reference on `gt_v15_gpt5_samples.jsonl` within ratio ±15%
and drop direction ≥ 16/18.

## Known gaps (where teacher still differs from human GT)

These are baked into `gt_v15_gpt5_samples.jsonl` — heads up:
  - **#16 nltk choose**: gpt-5 emits a tiny stub instead of dropping;
    human GT drops. Lossy ~0.07 ratio.
  - Ratio on a few L1 entries (#4, #5) is ~0.40 where GT is ~0.25 — teacher
    is slightly more conservative on kept entries.
  - These are within acceptable bounds for distillation but worth a
    re-pass after the student is trained.
