"""Minimal example: load Paritok and compress a code context.

Usage:
    python examples/inference/basic.py
"""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "paritok/paritok-4b-v1"
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, MODEL_ID)
    model.eval()
    return model, tokenizer


def compress(model, tokenizer, original_code: str, user_intent: str = "") -> str:
    """Compress a code snippet."""
    system_prompt = open("data_pipeline/prompts/system_prompt_qwen3.txt").read()
    intent_attr = f' user_intent="{user_intent}"' if user_intent else ""
    user_msg = (
        f"[SEG id=1 kind=file_read{intent_attr}]\n{original_code}\n[/SEG]"
    )
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    return tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )


if __name__ == "__main__":
    model, tokenizer = load_model()
    sample = """
def process_query(query: str, corpus: list) -> str:
    hits = [c for c in corpus if query.lower() in c["text"].lower()]
    if hits:
        return hits[0]["text"]
    return "no match"
"""
    print(compress(model, tokenizer, sample, user_intent="Understand process_query"))
