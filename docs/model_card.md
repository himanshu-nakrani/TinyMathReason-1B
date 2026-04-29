---
language:
- en
license: apache-2.0
tags:
- math
- reasoning
- jax
- pytorch
- dpo
datasets:
- HuggingFaceFW/fineweb-edu
- open-web-math/open-web-math
- hoskinson-center/proof-pile-v2
- TIGER-Lab/MathInstruct
- meta-math/MetaMathQA
- gsm8k
metrics:
- accuracy
---

# TinyMathReason-1B

TinyMathReason-1B is a 1.12 Billion parameter Llama-style decoder-only transformer trained from scratch specifically for mathematical reasoning. This repository contains the [Base / SFT / DPO] variant.

## Model Description

- **Developed by:** [Your Name]
- **Model type:** Decoder-only Transformer
- **Language(s):** English, Mathematics, Code
- **License:** Apache 2.0
- **Architecture:** 22 layers, 2048 hidden dimension, 16 Attention heads, 4 KV heads (GQA), SwiGLU activation (5632 intermediate dim).
- **Parameters:** 1.12B total
- **Context Length:** 4096 tokens

## Training Details

### Pretraining (Base Model)
The base model was trained from a random initialization on Google Cloud TPU v4-32 using the [MaxText](https://github.com/google/maxtext) framework.
- **Tokens:** ~300 Billion
- **Optimizer:** AdamW (β1=0.9, β2=0.95, weight_decay=0.1)
- **Learning Rate:** 3e-4 peak, cosine decay to 3e-5
- **Data Mix:** 
  - 40% FineWeb-Edu
  - 35% OpenWebMath
  - 15% Proof-Pile-2
  - 10% Stack-Edu (Code)

### Supervised Fine-Tuning (SFT)
The SFT variant was trained on ~600k instruction-following mathematical examples formatted in ChatML.
- **Hardware:** 1x A100 GPU using PyTorch + TRL
- **Learning Rate:** 2e-5 (Cosine schedule)
- **Epochs:** 2

### Preference Optimization (DPO)
The DPO variant was trained using Direct Preference Optimization on a dataset of 10k generated pairs from GSM8K.
- **Hardware:** 1x A100 GPU using PyTorch + TRL
- **Learning Rate:** 5e-7
- **Beta:** 0.1

## Evaluation Results

*(Note: Replace TBD with actual metrics after running `run_benchmarks.py`)*

| Benchmark | Base Model | SFT Model | DPO Model |
| :--- | :---: | :---: | :---: |
| **GSM8K** (8-shot) | TBD% | TBD% | TBD% |
| **MATH** (4-shot) | TBD% | TBD% | TBD% |
| **ARC-Challenge** | TBD% | TBD% | TBD% |
| **MMLU** (5-shot) | TBD% | TBD% | TBD% |
| **HellaSwag** | TBD% | TBD% | TBD% |

## Intended Uses & Limitations

**Intended Uses:**
- Solving step-by-step grade-school to high-school level math problems.
- Educational assistance and logic-based chain-of-thought generation.
- As a foundation for further fine-tuning in scientific domains.

**Limitations:**
- Being a 1B parameter model, it lacks the broad general knowledge of larger models.
- Prone to arithmetic hallucination on very large numbers.
- May fail on complex topology or advanced undergraduate mathematics.

## Environmental Impact

- **Hardware Type:** TPU v4-32 + NVIDIA A100s
- **Hours used:** ~300 hours (TPU) + ~24 hours (GPU)
- **Cloud Provider:** Google Cloud Platform, Vultr, Modal
- **Compute Region:** us-central
- **Estimated CO2 Emissions:** ~TBD kg CO2 eq.

## Citation

```bibtex
@misc{tinymathreason2026,
  author = {Your Name},
  title = {TinyMathReason-1B: A 1 Billion Parameter Mathematical Reasoning LLM Built from Scratch on TPU v4-32},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/your-username/TinyMathReason-1B}}
}
```
