import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from accelerate import infer_auto_device_map, dispatch_model

torch.backends.cuda.matmul.allow_tf32 = True  # small perf boost on RTX 40xx

BASE = "meta-llama/Meta-Llama-3-8B"
ADAPTER = "polyglots/SinLlama_v01"
EXTENDED_TOKENIZER = "polyglots/Extended-Sinhala-LLaMA"

OFFLOAD_DIR = "./offload_cache"
os.makedirs(OFFLOAD_DIR, exist_ok=True)

print("1) Load EXTENDED tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(EXTENDED_TOKENIZER, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("2) Load base model fully on CPU (no offload yet)...")
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    device_map={"": "cpu"},     # keep everything on CPU for now
    dtype=torch.float16,
    low_cpu_mem_usage=True,
)

print("3) Resize token embeddings to match extended vocab (no mean_resizing)...")
new_vocab_size = len(tokenizer)
model.resize_token_embeddings(new_vocab_size, mean_resizing=False)
model.config.vocab_size = new_vocab_size

print("4) Attach SinLlama adapter on CPU (no device_map/offload here)...")
model = PeftModel.from_pretrained(
    model,
    ADAPTER,
    device_map=None,          # <— important: don't dispatch during PEFT load
    offload_dir=None,         # <— important: don't offload during PEFT load
    is_trainable=False,
)

print("5) Shard to GPU + CPU with Accelerate...")
max_memory = {0: "5GiB", "cpu": "48GiB"}  # fits a 6GB 4050; adjust if needed
device_map = infer_auto_device_map(
    model,
    max_memory=max_memory,
    no_split_module_classes=["LlamaDecoderLayer"],
)
print("   Device map ready. Dispatching...")
model = dispatch_model(model, device_map=device_map, offload_dir=OFFLOAD_DIR)

print("6) Generate a short Sinhala sample...")
prompt = "සිංහල භාෂාවෙන් මට කෙටි කතාවක් කියන්න."
inputs = tokenizer(prompt, return_tensors="pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=60,     # keep modest for offload speed
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

print("\n=== OUTPUT ===")
print(tokenizer.decode(out[0], skip_special_tokens=True))
