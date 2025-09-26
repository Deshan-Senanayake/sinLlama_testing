# SinLlama – Local Testing & Baselines

Small, practical runners to load and sanity-check **SinLlama** on CPU / GPU offload, and to compare quick baselines (PEFT/LoRA and Unsloth variants).

> **Why SinLlama?**  
> SinLlama extends Llama-3-8B with Sinhala-specific tokenizer updates and continual pretraining on a 10.7M-sentence cleaned Sinhala corpus, then fine-tunes with LoRA on several downstream tasks. In your experiments it **consistently beat Llama-3 base/instruct** on Sinhala classification tasks (notably ~**86 F1** on news categorization vs ~61 for a fine-tuned Llama-3 base).  
> Method highlights: tokenizer extension for Sinhala, continual pretraining, LoRA fine-tuning.

---

## Repo structure

```
.
├── run_sinllama_cpu.py        # Load the model fully on CPU; minimal smoke test / prompt
├── run_sinllama_offload.py    # Load on limited VRAM with CPU/GPU offload (accelerate / bitsandbytes)
├── test_sinllama.py           # Quick generation & tokenization sanity checks
├── test_sinllama_offload.py   # Same tests with offload config
├── test_sinllama_peft.py      # Attach/verify LoRA adapters; short eval loop
├── test_sinllama_unsloth.py   # Unsloth-optimized loading for faster inference/finetune
```

---

## Requirements

- Python 3.10+
- PyTorch, `transformers`, `accelerate`, `peft`, `bitsandbytes` (for offload/4-bit), `unsloth` (optional), `tiktoken` or `sentencepiece`, `datasets`, `einops`
- A Hugging Face token if the base model requires it

Install (example):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121  # choose your CUDA/CPU build
pip install transformers accelerate peft bitsandbytes unsloth tiktoken sentencepiece datasets einops
```

---

## Models & data

- **Model weights:** `polyglots/SinLlama_v01` (Hugging Face)  
- **Sinhala corpus (cleaned):** `polyglots/MADLAD_CulturaX_cleaned`  

> SinLlama is built by extending Llama-3-8B with Sinhala tokens and continually pretraining on ~**10.7M sentences (~304M tokens)** from MADLAD-400 + CulturaX (cleaned & deduped).

---

## Quickstart

### 1) CPU-only smoke test
```bash
python run_sinllama_cpu.py --model polyglots/SinLlama_v01 \
  --prompt "ශ්‍රී ලංකාවේ දැන් වෙලාව කීයද?"
```

### 2) Low-VRAM offload
```bash
python run_sinllama_offload.py --model polyglots/SinLlama_v01 \
  --load-in-4bit --device-map "auto"
```

### 3) Basic checks
```bash
python test_sinllama.py --model polyglots/SinLlama_v01
python test_sinllama_offload.py --model polyglots/SinLlama_v01 --load-in-4bit
```

### 4) Attach LoRA (PEFT)
```bash
python test_sinllama_peft.py \
  --base-model polyglots/SinLlama_v01 \
  --lora-path ./checkpoints/lora_sinl/  # your adapter dir
```

### 5) Unsloth path (optional)
```bash
python test_sinllama_unsloth.py --model polyglots/SinLlama_v01
```

---

## Benchmarks (from the paper)

| Task                      | Best Llama-3 baseline (finetuned) | **SinLlama (finetuned)** |
|---------------------------|-----------------------------------|---------------------------|
| **News categorization**   | ~61 F1                            | **~86 F1**                |
| **Sentiment analysis**    | ~68 F1 (instruct FT)              | **~72 F1**                |
| **Writing style**         | lower across models               | **best among all**        |

---

## How this repo fits the paper

- **Tokenizer & continual pretraining** are upstream; these runners assume a ready model on HF.  
- **LoRA/PEFT testing** reflects the paper’s efficient adaptation strategy for modest GPUs.  
- **Offload/4-bit** helps reproduce “limited VRAM” scenarios described in the experimental setup.  

---

## Tips for low-VRAM setups

- Prefer `--load-in-4bit` + `--device-map auto` with `accelerate`.  
- Reduce `max_new_tokens` during quick checks.  
- For batch testing, keep sequence lengths short (the original continual-pretraining used a reduced block size of **512** to fit memory).  

---

## Roadmap / ideas

- Small CLI to select **CPU / offload / unsloth** profiles from one entrypoint.  
- Add tiny eval harnesses for **news / sentiment / style** sample sets to sanity-check local runs.  
- Optional notebook: export traces to compare **tokenization** before/after Sinhala tokenizer extension.  

---

## Acknowledgements

- Underlying corpus: MADLAD-400 & CulturaX (Sinhala slices).  
- Methodological references: tokenizer extension + continual pretraining for LRLs.  

