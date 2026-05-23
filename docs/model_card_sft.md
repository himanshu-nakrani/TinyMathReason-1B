---
language:
- en
license: apache-2.0
tags:
- math
- reasoning
- sft
- instruction-tuning
- llama
- pytorch
- text-generation
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

# TinyMathReason-1B-sft

TinyMathReason-1B-sft is a 1.12 Billion parameter Llama-style decoder-only transformer trained from scratch specifically for mathematical reasoning. This is the **Supervised Fine-Tuned (SFT)** variant.

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
- **Tokens:** ~300 Billion
- **Optimizer:** AdamW (β1=0.9, β2=0.95, weight_decay=0.1)
- **Learning Rate:** 3e-4 peak, cosine decay to 3e-5

### Supervised Fine-Tuning (SFT)
This variant was trained on ~600k instruction-following mathematical examples formatted in ChatML.
- **Hardware:** 1x A100 GPU using PyTorch + TRL
- **Learning Rate:** 2e-5 (Cosine schedule)
- **Epochs:** 2

## Intended Uses & Limitations

**Intended Uses:**
- Solving step-by-step grade-school to high-school level math problems.
- Educational assistance and logic-based chain-of-thought generation.
- As a foundation for further preference optimization (e.g., DPO, GRPO).

**Limitations:**
- Being a 1B parameter model, it lacks the broad general knowledge of larger models.
- Prone to arithmetic hallucination on very large numbers.
- May fail on complex topology or advanced undergraduate mathematics.

## Citation

```bibtex
@misc{tinymathreason2026,
  author = {Himanshu Nakrani},
  title = {TinyMathReason-1B: A 1 Billion Parameter Mathematical Reasoning LLM Built from Scratch on TPU v4-32},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/himanshu-nakrani/TinyMathReason-1B}}
}
```
