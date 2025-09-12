import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "meta-llama/Meta-Llama-3-8B"           # you now have access
ADAPTER_MODEL = "polyglots/SinLlama_v01"
TOKENIZER_NAME = "polyglots/Extended-Sinhala-LLaMA"

def main():
    use_cuda = torch.cuda.is_available()
    print("[INFO] CUDA:", use_cuda)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)

    # Offload: keep GPU under ~5.6GB, spill the rest to CPU RAM
    max_memory = {0: "5600MB", "cpu": "20GB"} if use_cuda else {"cpu": "24GB"}

    print("[INFO] Loading base model with offload...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.float16 if use_cuda else torch.float32,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )

    print("[INFO] Applying SinLlama LoRA...")
    model = PeftModel.from_pretrained(base, ADAPTER_MODEL)
    model.resize_token_embeddings(len(tokenizer))

    prompt = "සිංහල භාෂාව පිළිබඳ කෙටි පැහැදිලි කිරීමක් ලියන්න."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )

    print("\n[OUTPUT]\n" + tokenizer.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
