# Unsloth must be imported before transformers
from unsloth import FastLanguageModel
import torch
from transformers import AutoTokenizer

MODEL_NAME = "polyglots/SinLlama_v01"
TOKENIZER_NAME = "polyglots/Extended-Sinhala-LLaMA"

def main():
    print("[INFO] CUDA:", torch.cuda.is_available())

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)

    model, _ = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=2048,
        # Start with False to avoid 4-bit (which may use bnb). If it works, you can try True later.
        load_in_4bit=False,
        dtype=None,
    )

    model.resize_token_embeddings(len(tokenizer))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    prompt = "ශ්‍රී ලංකාව පිළිබඳ සරල සාරාංශයක් ලියන්න."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=120, temperature=0.7, top_p=0.9,
                             eos_token_id=tokenizer.eos_token_id)
    print("\n[OUTPUT]\n" + tokenizer.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
