import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ---- Names ----
BASE_MODEL = "meta-llama/Meta-Llama-3-8B"           # requires license acceptance on HF
ADAPTER_MODEL = "polyglots/SinLlama_v01"
TOKENIZER_NAME = "polyglots/Extended-Sinhala-LLaMA"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    # 4-bit quantization config (fits 8B on ~6GB VRAM)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",       # good quality
        bnb_4bit_compute_dtype=torch.float16,
    )

    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)

    print("[INFO] Loading base model in 4-bit...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",               # spreads across CUDA/CPU if needed
        torch_dtype=torch.float16,
    )

    print("[INFO] Applying SinLlama LoRA adapter...")
    model = PeftModel.from_pretrained(base, ADAPTER_MODEL)

    # Ensure embedding size matches extended vocab
    model.resize_token_embeddings(len(tokenizer))

    prompt = "සිංහල භාෂාව පිළිබඳ කෙටි පැහැදිලි කිරීමක් ලියන්න."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )

    print("\n[OUTPUT]\n" + tokenizer.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
