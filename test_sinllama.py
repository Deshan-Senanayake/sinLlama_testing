import os
import torch
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

# ---------- Config ----------
MODEL_NAME = "polyglots/SinLlama_v01"
TOKENIZER_NAME = "polyglots/Extended-Sinhala-LLaMA"

# Choose quantization via env var (default 4-bit):
#   setx USE_4BIT 0   # (PowerShell) to force 8-bit/FP16 next time
USE_4BIT = os.environ.get("USE_4BIT", "1") != "0"

MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "160"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("TOP_P", "0.9"))
# ----------------------------

def main():
    # Check device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"[INFO] CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("[WARN] CUDA not available, running on CPU (slow)")

    print(f"[INFO] Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)

    print(f"[INFO] Loading model: {MODEL_NAME} (4-bit={USE_4BIT})")
    # NOTE: Requires access to meta-llama/Meta-Llama-3-8B on Hugging Face.
    # If you lack access, this will raise an error.
    model, _ = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=2048,
        load_in_4bit=USE_4BIT,  # set False to try 8-bit/FP16 depending on your setup
        dtype=None,             # auto-select (bf16/fp16/cpu)
    )

    # Ensure embeddings match the extended Sinhala vocab
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # A couple of quick prompts to sanity-check Sinhala generation.
    prompts = [
        "සිංහල භාෂාව ගැන වටිනා කෙටි පැහැදිලි කිරීමක් ලියන්න.",
        "ශ්‍රී ලංකාවේ තාක්ෂණික නවෝත්පාදන පිළිබඳ සරල සාරාංශයක් ලියන්න.",
    ]

    for i, prompt in enumerate(prompts, start=1):
        print(f"\n[GEN {i}] Prompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"[GEN {i}] Output:\n{text}\n")

    print("[DONE] If you see coherent Sinhala text above, the model loaded correctly.")

if __name__ == "__main__":
    main()
