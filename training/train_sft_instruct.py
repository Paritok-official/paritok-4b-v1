"""Stage 1: SFT Training

Supervised fine-tuning on per-segment teacher (gpt-5 / gpt-4.1-mini) compressions.
The student learns the teacher's per-SEG keep/drop decisions and body rewrites.

Input: the per-segment JSONL files produced under update/ — one record per
(sample, seg) decision, e.g. update/file_read_compressed_all10k_*.jsonl and
update/other_compressed_all_per_kind_*.jsonl. The (system, user) prompt format
mirrors update/compress_pool_{file_read,other}.py exactly; the assistant target
is the teacher's [SEG ...]<body>[/SEG] (empty body for drops).

Usage:
    python training/train_sft_instruct.py --config training/configs/sft_config_qwen3_4b.yaml
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import yaml
from datasets import Dataset
from transformers import EarlyStoppingCallback, set_seed
 
 
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"
 
 
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
 
 
class ResponseOnlyCollator:
    """Data collator that only computes loss on assistant responses.
    
    Masks everything before (and including) the response_template tokens
    with -100 so they don't contribute to the loss.
    """
 
    def __init__(self, tokenizer, response_template: str = RESPONSE_TEMPLATE):
        self.tokenizer = tokenizer
        self.response_template_ids = tokenizer.encode(
            response_template, add_special_tokens=False
        )
        self.template_len = len(self.response_template_ids)
 
    def _find_response_start(self, input_ids: List[int]) -> int:
        """Find the index right after the response template in input_ids."""
        for i in range(len(input_ids) - self.template_len + 1):
            if input_ids[i:i + self.template_len] == self.response_template_ids:
                return i + self.template_len
        return -1
 
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract per-sample weight before tokenizer.pad (which only knows about
        # tokenizer fields). Default 1.0 keeps the un-weighted code path working.
        weights = [float(f.get("weight", 1.0)) for f in features]
        pad_features = [{k: v for k, v in f.items() if k != "weight"} for f in features]

        batch = self.tokenizer.pad(
            pad_features,
            padding=True,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()

        for i in range(len(labels)):
            input_ids = batch["input_ids"][i].tolist()
            resp_start = self._find_response_start(input_ids)

            if resp_start == -1:
                # No response template found — mask everything
                labels[i, :] = -100
            else:
                # Mask everything before response (system + user + template)
                labels[i, :resp_start] = -100

            # Mask padding positions via attention_mask, NOT by matching pad_token_id.
            # On Qwen the default fallback makes pad_token_id == eos_token_id, so a
            # token-id match would also mask the legitimate <|im_end|> at the end of
            # the assistant turn — and the model would never learn to stop.
            labels[i, batch["attention_mask"][i] == 0] = -100

        batch["labels"] = labels
        batch["weight"] = torch.tensor(weights, dtype=torch.float32)
        return batch
 
 
def build_user_message(entry: dict, kind: str) -> str:
    """Same user-message format the teacher saw (see update/compress_pool_*.py)."""
    intent = (entry.get("user_intent") or "").strip()
    return (
        "USER INTENT:\n"
        f"{intent}\n\n"
        "Compress the following segment under the rules in your system prompt. "
        "Output only the compressed [SEG]...[/SEG] block (or an empty one to drop):\n\n"
        f"[SEG id={entry['seg_id']} kind={kind} level={entry['level']}]\n"
        f"{entry['original']}\n"
        f"[/SEG]\n"
    )


def build_assistant_target(entry: dict, kind: str) -> str:
    """Rebuild the teacher's reply: [SEG ...]<body>[/SEG], empty body for drop.

    The `compressed` field is the lint-cleaned body (no wrapper); we re-wrap with
    the same header format the teacher emitted. For drops, body is empty.
    """
    header = f"[SEG id={entry['seg_id']} kind={kind} level={entry['level']}]"
    if entry.get("dropped") or entry.get("compressed") is None:
        return f"{header}[/SEG]"
    return f"{header}{entry['compressed']}[/SEG]"


def load_segment_sft_dataset(
    sources: List[Dict[str, str]],
    tokenizer,
    max_tokens: Optional[int] = None,
    dropped_repeat: int = 1,
    drop_loss_weight: float = 1.0,
    cache_dir: Optional[str] = None,
) -> Dataset:
    """Load per-segment teacher compressions and pre-tokenize them.

    Each source is a dict with:
        path:               JSONL produced by update/compress_pool_*.py
        system_prompt_path: matching teacher system prompt
        default_kind:       fallback used ONLY when a record lacks `kind`.
                            Required for the file_read pool (no `kind` in records);
                            unnecessary for the other pool (every record sets it).
                            If a record lacks `kind` and no default is given, we raise.

    Records with non-null `error` are skipped. Records pass through if `dropped`
    is true — they are the model's "drop" signal and must be learned.

    Two mechanisms to upweight drop signal (compensating the per-token gradient
    deficit — drop targets are ~10 tok vs keep ~200 tok, so without upweighting
    the drop signal is ~40× weaker than keep and DropAcc falls below baseline):

      `dropped_repeat` (default 1): duplicate each dropped record N times.
          Simple but biases sample-level exposure (model sees "output empty"
          pattern N× more often).

      `drop_loss_weight` (default 1.0): scale per-sample loss for dropped
          records. Cleaner — keeps natural sample distribution, only amplifies
          the gradient. Math: at w=20, drop gradient share ≈ 30% (matches the
          natural 30% drop sample fraction).

    The weight is emitted in each record's `weight` field; WeightedSFTTrainer
    applies it in compute_loss.

    `cache_dir` (optional): if given, the tokenized dataset is saved to disk
    under a key derived from (data files' mtime+size, system prompt mtime,
    tokenizer name, max_tokens, dropped_repeat, drop_loss_weight). On subsequent
    runs with the same inputs, the cache is loaded in seconds instead of
    re-tokenizing 40k+ samples (which takes 3-5 minutes).
    """
    # ────────── Cache lookup ──────────
    cache_path = None
    if cache_dir:
        import hashlib
        key_parts = [
            f"tok={tokenizer.name_or_path}",
            f"max_tokens={max_tokens}",
            f"dropped_repeat={dropped_repeat}",
            f"drop_loss_weight={drop_loss_weight}",
        ]
        for src in sources:
            dp = Path(src["path"])
            sp = Path(src["system_prompt_path"])
            try:
                key_parts.append(
                    f"data={dp.name}:{dp.stat().st_mtime_ns}:{dp.stat().st_size}"
                )
                key_parts.append(f"sys={sp.name}:{sp.stat().st_mtime_ns}:{sp.stat().st_size}")
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Cache key build: {e}") from e
            key_parts.append(f"default_kind={src.get('default_kind', '')}")

        cache_key = hashlib.sha256("|".join(key_parts).encode()).hexdigest()[:16]
        cache_path = Path(cache_dir) / f"sft_dataset_{cache_key}"

        if cache_path.exists():
            print(f"  [cache HIT] Loading pre-tokenized dataset from {cache_path}")
            ds = Dataset.load_from_disk(str(cache_path))
            print(f"  Total: {len(ds)} samples (from cache)")
            return ds
        else:
            print(f"  [cache MISS] Will tokenize fresh and save to {cache_path}")

    all_records = []
    total_err = 0
    total_long = 0
    for src in sources:
        path = src["path"]
        sys_prompt = Path(src["system_prompt_path"]).read_text(encoding="utf-8")
        default_kind = src.get("default_kind")  # None unless explicitly set

        n_kept = n_dropped = n_err = n_long = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                if e.get("error"):
                    n_err += 1
                    continue
                kind = e.get("kind") or default_kind
                if kind is None:
                    raise ValueError(
                        f"{Path(path).name}: record {e.get('entry_id')} has no `kind` "
                        f"and source has no `default_kind` configured"
                    )
                user_msg = build_user_message(e, kind)
                assistant_msg = build_assistant_target(e, kind)
                text = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": assistant_msg},
                    ],
                    tokenize=False,
                    add_generation_prompt=False,
                )

                # Pre-tokenize so we own the columns (text-based SFTTrainer prep
                # would strip our `weight` column on the way through dataset.map).
                encoded = tokenizer(text, add_special_tokens=False)
                if max_tokens is not None and len(encoded["input_ids"]) > max_tokens:
                    n_long += 1
                    continue

                is_drop = bool(e.get("dropped"))
                weight = float(drop_loss_weight) if is_drop else 1.0
                record = {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "weight": weight,
                }
                all_records.append(record)
                if is_drop:
                    n_dropped += 1
                    for _ in range(dropped_repeat - 1):
                        all_records.append(dict(record))
                else:
                    n_kept += 1

        total_err += n_err
        total_long += n_long
        name = Path(path).name
        msg = f"  {name}: kept={n_kept}, dropped={n_dropped}, errored={n_err}"
        if dropped_repeat > 1:
            msg += f" (dropped×{dropped_repeat} = +{n_dropped*(dropped_repeat-1)} extra)"
        if drop_loss_weight != 1.0:
            msg += f" (drop_loss_weight={drop_loss_weight})"
        if n_long:
            msg += f", skipped_long(>{max_tokens})={n_long}"
        print(msg)

    print(
        f"  Total: {len(all_records)} samples"
        + (f" (errored={total_err}, too_long={total_long})" if (total_err or total_long) else "")
    )
    ds = Dataset.from_list(all_records)

    if cache_path is not None:
        print(f"  [cache SAVE] Saving tokenized dataset to {cache_path} (~3s)...")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(cache_path))
        print("  [cache SAVE] Done. Future restarts will load this in seconds.")

    return ds
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/configs/sft_config_qwen3_4b.yaml")
    args = parser.parse_args()
 
    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
 
    # ── Reproducibility ──
    set_seed(train_cfg.get("seed", 42))
 
    # ── Load model (vanilla HF + peft + FA2, no unsloth) ──
    # We removed unsloth because it (a) strips logits breaking WeightedSFTTrainer
    # and (b) auto-offloaded gradients causing 10× slowdown at vocab 152k.
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    # Liger Kernel: fused Triton kernels for RoPE / RMSNorm / SwiGLU.
    # Patches Qwen2 module BEFORE model load so when AutoModelForCausalLM
    # constructs the model, these fast ops are used. ~1.2-1.5× speedup.
    # We disable fused linear+CE because WeightedSFTTrainer needs raw logits
    # for per-sample weighting.
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2
        apply_liger_kernel_to_qwen2(
            rope=True,
            rms_norm=True,
            swiglu=True,
            cross_entropy=False,
            fused_linear_cross_entropy=False,
        )
        print("Liger Kernel applied: rope + rms_norm + swiglu")
    except ImportError:
        print("Liger Kernel not installed; running without (slower).")

    print(f"Loading {model_cfg['name']} (HF + FA2, no unsloth)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Required for gradient checkpointing + LoRA: HF Trainer turns on grad ckpt
    # via TrainingArguments.gradient_checkpointing=True, but peft adapters need
    # the input embedding gradient flow re-enabled for backward to reach LoRA.
    if train_cfg.get("gradient_checkpointing", True):
        model.enable_input_require_grads()

    # ── Apply LoRA ──
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
 
    # ── Load data (per-segment teacher distillations) ──
    # Default both train/eval filters to the model's seq length so that any
    # sample SFTTrainer would have to right-truncate (chopping off the
    # assistant target → silent label-masking) is dropped here instead.
    max_seq_length = model_cfg["max_seq_length"]
    train_max_tokens = data_cfg.get("train_max_tokens", max_seq_length)
    eval_max_tokens = data_cfg.get("eval_max_tokens", max_seq_length)

    dropped_repeat = data_cfg.get("dropped_repeat", 1)
    drop_loss_weight = float(data_cfg.get("drop_loss_weight", 1.0))
    # Pre-tokenized dataset cache. Sits in /workspace (volume disk) so it
    # survives pod restarts. Cache key includes data mtimes + tokenizer + all
    # relevant params — change any of them, cache is auto-invalidated.
    sft_cache_dir = data_cfg.get("cache_dir", "training/cache")

    print("Loading train sources...")
    train_dataset = load_segment_sft_dataset(
        data_cfg["train"], tokenizer, max_tokens=train_max_tokens,
        dropped_repeat=dropped_repeat,
        drop_loss_weight=drop_loss_weight,
        cache_dir=sft_cache_dir,
    )
    eval_dataset = None
    eval_sources = data_cfg.get("eval") or []
    if eval_sources:
        print("Loading eval sources...")
        # Eval keeps native distribution — don't duplicate dropped, no weighting
        eval_dataset = load_segment_sft_dataset(
            eval_sources, tokenizer, max_tokens=eval_max_tokens,
            cache_dir=sft_cache_dir,
        )
 
    print(f"Train samples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Eval samples: {len(eval_dataset)}")
 
    # ── Verify response template is found ──
    collator = ResponseOnlyCollator(tokenizer, RESPONSE_TEMPLATE)
    sample_ids = list(train_dataset[0]["input_ids"])
    resp_start = collator._find_response_start(sample_ids)
    total_tokens = len(sample_ids)
    if resp_start == -1:
        snippet = tokenizer.decode(sample_ids[:200], skip_special_tokens=False)
        print(f"WARNING: Response template '{RESPONSE_TEMPLATE}' not found in first sample!")
        print(f"  First 200 toks decoded: {snippet[:300]!r}")
        raise ValueError("Response template not found — check chat template format")
    else:
        masked = resp_start
        trained = total_tokens - resp_start
        print(f"  Response template found at token {resp_start}/{total_tokens}")
        print(f"  Masked (no loss): {masked} tokens ({masked/total_tokens*100:.1f}%)")
        print(f"  Trained (loss):   {trained} tokens ({trained/total_tokens*100:.1f}%)")
        sample_weight = train_dataset[0].get("weight", 1.0)
        print(f"  Sample weight:    {sample_weight} (drop_loss_weight={drop_loss_weight})")
 
    # ── SFT Trainer ──
    from trl.trainer.sft_config import SFTConfig
    from trl.trainer.sft_trainer import SFTTrainer

    class WeightedSFTTrainer(SFTTrainer):
        """SFTTrainer with per-sample loss weighting via batch['weight'].

        Token-level weighted CE: each token's loss is multiplied by its
        sample's weight, then aggregated as a weighted mean over all
        non-masked tokens in the batch. With drop_loss_weight=20 and the
        current 30/70 drop/keep split, drop signal becomes ~30% of total
        gradient (vs ~2% unweighted). When all weights are 1.0, falls back
        to the parent's default compute_loss for parity.
        """

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            weights = inputs.pop("weight", None)

            if weights is None or bool((weights == 1.0).all().item()):
                return super().compute_loss(
                    model, inputs, return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )

            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.logits  # (B, T, V)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
            per_token_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_labels.size())  # (B, T-1)

            mask = (shift_labels != -100).to(per_token_loss.dtype)
            weight_per_token = weights.to(per_token_loss.dtype).unsqueeze(-1) * mask

            loss = (per_token_loss * weight_per_token).sum() / weight_per_token.sum().clamp(min=1e-6)

            if return_outputs:
                return loss, outputs
            return loss

    has_eval = eval_dataset is not None
 
    save_strategy = train_cfg["save_strategy"]
    eval_strategy = train_cfg.get("eval_strategy", "steps")
    if has_eval:
        assert save_strategy == eval_strategy, (
            f"save_strategy ({save_strategy}) and eval_strategy ({eval_strategy}) "
            f"must match when load_best_model_at_end=True"
        )
 
    training_args = SFTConfig(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        optim=train_cfg["optim"],
        bf16=train_cfg["bf16"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=save_strategy,
        save_steps=train_cfg.get("save_steps", 100),
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        # max_seq_length removed: TRL 0.20+ dropped this kwarg. We've already
        # pre-tokenized + filtered too-long samples in load_segment_sft_dataset,
        # so SFTConfig doesn't need to know about truncation.
        seed=train_cfg.get("seed", 42),
        report_to=train_cfg.get("report_to", "none"),
        # Dataset already tokenized + has custom `weight` column. Skip TRL's
        # auto-tokenize/format pass (it would strip `weight`), and tell HF Trainer
        # to keep all columns through to the collator (compute_loss pops weight).
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        # Evaluation
        eval_strategy=eval_strategy if has_eval else "no",
        eval_steps=train_cfg.get("eval_steps", 100),
        # Early stopping
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        packing=False,
    )
 
    callbacks = []
    early_stopping_patience = train_cfg.get("early_stopping_patience", 3)
    if has_eval and early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
 
    trainer = WeightedSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
 
    # ── Train ──
    output_dir = train_cfg["output_dir"]
    print("Starting SFT training...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted, saving current checkpoint...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        raise
 
    # ── Save LoRA adapter ──
    print(f"Saving LoRA adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    log_history = trainer.state.log_history
    with open(f"{output_dir}/train_log.json", "w") as f:
        json.dump(log_history, f, indent=2)
    print(f"  Training log saved to {output_dir}/train_log.json")
    print("SFT training complete.")
    print(f"  To load for GRPO: set model.name to '{output_dir}' in grpo_config.yaml")
 
 
if __name__ == "__main__":
    main()