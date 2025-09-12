import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "meta-llama/Meta-Llama-3-8B"
ADAPTER = "polyglots/SinLlama_v01"
EXTENDED_TOKENIZER = "polyglots/Extended-Sinhala-LLaMA"

OFFLOAD_DIR = "./offload_cache"
os.makedirs(OFFLOAD_DIR, exist_ok=True)

print("1) Load EXTENDED tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(EXTENDED_TOKENIZER, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("2) Load base model with CPU offload...")
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    dtype=torch.float16,          # (use dtype= — torch_dtype is deprecated)
    device_map="auto",
    low_cpu_mem_usage=True,
    offload_folder=OFFLOAD_DIR,
)

print("3) Resize token embeddings to match extended vocab (no mean_resizing)...")
new_vocab_size = len(tokenizer)
model.resize_token_embeddings(new_vocab_size, mean_resizing=False)
model.config.vocab_size = new_vocab_size

print("4) Attach SinLlama adapter...")
model = PeftModel.from_pretrained(model, ADAPTER)

# --- test generate ---
prompt = "සිංහල භාෂාවෙන් මට කෙටි කතා ඛණ්ඩයක් කියන්න."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(out[0], skip_special_tokens=True))
