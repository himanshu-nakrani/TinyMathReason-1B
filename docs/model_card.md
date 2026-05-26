---
language:
- en
license: apache-2.0
tags:
- math
- reasoning
- jax
- pytorch
- grpo
- rl
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

TinyMathReason-1B is a 1.12 Billion parameter Llama-style decoder-only transformer trained from scratch specifically for mathematical reasoning. This repository contains the [Base / SFT / GRPO] variant.

## Model Description

- **Developed by:** Himanshu Nakrani
- **Model type:** Decoder-only Transformer
- **Language(s):** English, Mathematics, Code
- **License:** Apache 2.0
- **Architecture:** 22 layers, 2048 hidden dimension, 16 Attention heads, 4 KV heads (GQA), SwiGLU activation (5632 intermediate dim).
- **Parameters:** 1.12B total
- **Context Length:** 4096 tokens

## Training Details

### Pretraining (Base Model)
The base model was trained from a random initialization on Google Cloud TPU v4-32 using the [MaxText](https://github.com/google/maxtext) framework.
- **Tokens:** ~57 Billion (Run 11 Rerun)
- **Optimizer:** AdamW (β1=0.9, β2=0.95, weight_decay=0.1)
- **Learning Rate:** 3e-4 peak, cosine decay
- **Data Mix:** 
  - 40% FineWeb-Edu
  - 35% OpenWebMath
  - 15% Proof-Pile-2
  - 10% Stack-Edu (Code)

### Supervised Fine-Tuning (SFT)
The SFT variant was trained on reasoning traces (MathInstruct, MetaMathQA) formatted in ChatML.
- **Hardware:** 1x AMD MI300X GPU using PyTorch + TRL
- **Learning Rate:** 2e-5 (Cosine schedule)
- **Epochs:** 2

### Group Relative Policy Optimization (GRPO)
The GRPO variant was trained using Group Relative Policy Optimization to improve reasoning traces and rule-based correctness.
- **Hardware:** 1x NVIDIA L4 GPU using PyTorch + TRL
- **Learning Rate:** 5e-6
- **Beta:** 0.01
- **Group Size (G):** 8

## Evaluation Results

| Benchmark | Base Model | SFT Model | **GRPO Model** |
| :--- | :---: | :---: | :---: |
| **GSM8K** (8-shot) | 1.0% | 1.0% | **2.2%** (Flex) |
| **Minerva Math** (4-shot) | 0.0% | 0.0% | **2.0%** |
| **ARC-Challenge** (25-shot) | 21.7% | 24.7% | TBD |
| **MMLU** (5-shot) | 23.5% | 24.6% | TBD |
| **HellaSwag** (10-shot) | 25.8% | 26.7% | TBD |

## Intended Uses & Limitations

**Intended Uses:**
- Solving step-by-step grade-school to high-school level math problems.
- Educational assistance and logic-based chain-of-thought generation.

**Limitations:**
- Being a 1B parameter model, it lacks the broad general knowledge of larger models.
- Prone to arithmetic hallucination on very large numbers.
- GRPO traces often contain repetitive phrases or mode collapse loops.

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
