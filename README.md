# TinyMathReason-1B 🧮

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Parameter Count](https://img.shields.io/badge/Parameters-1.12B-green.svg)]()
[![Training Setup](https://img.shields.io/badge/Pretraining-TPU_v4--32-orange.svg)]()

TinyMathReason-1B is a 1.12 Billion parameter Llama-style decoder-only transformer trained from scratch specifically for mathematical reasoning. The model was pretrained on ~300 Billion tokens of math-heavy text on a TPU v4-32 cluster using MaxText, then fine-tuned (SFT) and preference-optimized (DPO/GRPO) using PyTorch and TRL on GPUs. 

This repository contains the complete end-to-end code to replicate the entire process: tokenizer training, data pipeline, TPU pretraining, HuggingFace conversion, supervised fine-tuning, and preference optimization.

## Architecture 🏛️

The model uses a highly efficient Llama-like architecture optimized for mathematical context:
- **Total Parameters:** ~1.12B
- **Layers:** 22
- **Hidden Dimension:** 2048
- **Attention:** 16 Q-heads, 4 KV-heads (GQA 4:1)
- **Activation:** SwiGLU (Intermediate Dim: 5632)
- **Vocabulary Size:** 32,000 (Custom math-optimized BPE)
- **Max Sequence Length:** 4096

## Results (Placeholder) 📊

| Benchmark | Shots | Base | SFT | DPO | TinyLlama | Qwen2.5-0.5B |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| GSM8K | 8 | TBD% | TBD% | TBD% | 2.5% | 40.0% |
| MATH | 4 | TBD% | TBD% | TBD% | 0.0% | 15.0% |
| ARC-C | 25 | TBD% | TBD% | TBD% | 35.0% | 42.0% |

## Quick Start Guide 🚀

To use the final HuggingFace checkpoint:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "your-hf-username/TinyMathReason-1B-DPO"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).cuda()

prompt = "Solve the equation: 3x + 7 = 22"
chat = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(chat, return_tensors="pt", add_generation_prompt=True).cuda()

outputs = model.generate(inputs, max_new_tokens=256, temperature=0.1)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Reproduction Guide 🛠️

To reproduce the training from scratch, follow the pipeline below. See the respective markdown documents in `docs/` for detailed commands.

1. **Data Processing:** Run `make data` (Downloads datasets, cleans, deduplicates, and packs them into shards).
2. **Tokenizer:** Trained using `src/data/train_tokenizer.py`.
3. **Pretraining (TPU):** Follow `docs/pretraining_setup.md` to run MaxText on a TPU v4-32.
4. **Checkpoint Conversion:** Run `src/train/convert_checkpoint.py` to convert Orbax arrays to HF safetensors.
5. **SFT:** Follow `docs/sft_setup.md` to run `src/sft/train_sft.py`.
6. **DPO/GRPO:** Follow `docs/modal_dpo_setup.md` to run preference optimization.
7. **Evaluation:** Use the scripts in `src/eval/` to benchmark the model.

## Dataset Description 📚

The 300B token pretraining mixture consists of:
- 40% FineWeb-Edu (General knowledge & reasoning)
- 35% OpenWebMath (Raw mathematical text & LaTeX)
- 15% Proof-Pile-2 (Mathematical proofs and textbooks)
- 10% Stack-Edu / Cosmopedia (Code to enhance structural logic)

SFT data utilizes MathInstruct, MetaMathQA, and GSM8K.

## Hardware Requirements 💻

- **Data Processing:** 2x Vultr `c2-standard-30` (30+ vCPUs, 120GB RAM, 1TB+ NVMe SSD)
- **Pretraining:** Google Cloud TPU `v4-32` (Main & Evals)
- **SFT & DPO:** 1x AMD MI300X (192GB VRAM) via AMD Cloud
- **DPO Generation:** Serverless scale-out via Modal
- **Tracking/Demos:** Lightning AI & Thunder Compute

## Citation 📄

```bibtex
@misc{tinymathreason2026,
  author = {Your Name},
  title = {TinyMathReason-1B: A 1 Billion Parameter Mathematical Reasoning LLM Built from Scratch on TPU v4-32},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/your-username/TinyMathReason-1B}}
}
```

## Acknowledgments 🙏
- **MaxText** team for the incredible JAX/TPU framework.
- **HuggingFace** & **TRL** for the Post-Training and Tokenizer ecosystem.
- Dataset creators (OpenWebMath, FineWeb, MathInstruct).

## License
Apache 2.0
