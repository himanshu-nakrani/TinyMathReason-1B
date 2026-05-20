---
language: en
license: apache-2.0
tags:
  - math
  - reasoning
  - llama
  - tinymath
  - from-scratch
  - pretrained
library_name: transformers
pipeline_tag: text-generation
model-index:
  - name: TinyMathReason-1B-base
    results: []
---

# TinyMathReason-1B (Base)

A **1.126 Billion parameter** language model trained **from scratch** for mathematical reasoning, using a Llama-2 style architecture on Google Cloud TPU v4-64.

> **Note:** This is the **base (pretrained) model** — not instruction-tuned. For the fine-tuned version with reasoning capabilities, see [TinyMathReason-1B](https://huggingface.co/himanshunakrani9/TinyMathReason-1B) (coming soon).

## Model Details

### Architecture

| Parameter | Value |
|---|---|
| **Total Parameters** | ~1.126B |
| **Architecture** | Llama-2 style decoder-only transformer |
| **Hidden Dimension** | 2048 |
| **MLP Dimension** | 5632 (SwiGLU) |
| **Number of Layers** | 22 |
| **Query Heads** | 16 |
| **KV Heads** | 4 (Grouped-Query Attention, 4:1 ratio) |
| **Head Dimension** | 128 |
| **Vocab Size** | 32,768 (padded from 32k for FSDP alignment) |
| **Max Sequence Length** | 4096 |
| **Precision** | bfloat16 |
| **Normalization** | RMSNorm (eps=1e-5) |
| **Position Encoding** | RoPE (theta=10000) |
| **Activation** | SiLU (SwiGLU gating) |

### Tokenizer

- **Type:** Custom BPE (tiktoken format, converted to HuggingFace `tokenizers`)
- **Vocabulary:** 32,000 active tokens (padded to 32,768 in model config)
- **Special Tokens:** `<|bos|>`, `<|eos|>`, `<|pad|>`, `<|unk|>`, `<think>`, `</think>`
- **Training Data:** Trained on a sample of the pretraining corpus

## Training

### Framework & Infrastructure

- **Framework:** [MaxText](https://github.com/google/maxtext) (JAX/Flax)
- **Hardware:** Google Cloud TPU v4-64 (8 chips, 4 hosts)
- **Config:** `pure_nnx_decoder: True`, `scan_layers: False`
- **Total Training Steps:** 54,362
- **Throughput:** ~8,900 tokens/sec/chip (~66 TFLOP/s/device)

### Optimizer

| Parameter | Value |
|---|---|
| **Optimizer** | AdamW |
| **Learning Rate** | 3e-4 (peak) |
| **Schedule** | Cosine decay to 0.1× peak |
| **Warmup** | Linear warmup |
| **β1, β2** | 0.9, 0.95 |
| **Weight Decay** | 0.1 |

### Batch Configuration

| Parameter | Value |
|---|---|
| **Per-device Batch Size** | 2 |
| **Global Batch Size** | 64 sequences |
| **Sequence Length** | 4096 tokens |
| **Tokens per Step** | ~262,144 |

### Training Data

Total pretraining corpus: **~57 Billion tokens** in `jsonl.zst` format.

| Dataset | Tokens | Description |
|---|---|---|
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | ~10B | High-quality educational web text |
| [MathPile](https://huggingface.co/datasets/GAIR/MathPile) | ~9.5B | Mathematical documents and textbooks |
| [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math) | ~37.7B* | Mathematical web pages |
| [Stack-Edu](https://huggingface.co/datasets/bigcode/the-stack-v2) | (included above)* | Educational code |

*OpenWebMath and Stack-Edu were processed together (~37.7B tokens combined).

### Training Loss

Final training loss: **~2.6** (cross-entropy).

## Evaluation Results

> **Status:** Evaluation in progress. Results will be added here.

| Benchmark | Shots | Score |
|---|---|---|
| GSM8K | 8 | *pending* |
| MATH (Algebra) | 4 | *pending* |
| ARC-Easy | 0 | *pending* |
| ARC-Challenge | 25 | *pending* |
| HellaSwag | 10 | *pending* |
| MMLU | 5 | *pending* |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "himanshunakrani9/TinyMathReason-1B-base"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

prompt = "The derivative of x^2 + 3x is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Intended Use

This is a **portfolio/learning project** demonstrating the full LLM training stack — from data preparation through pretraining, fine-tuning, and evaluation. It is **not targeting state-of-the-art performance**.

### Intended Uses
- Research and educational purposes
- Base model for fine-tuning on math reasoning tasks
- Demonstration of the from-scratch LLM training pipeline

### Out-of-Scope Uses
- Production deployment without further fine-tuning and safety evaluation
- Tasks requiring strong general-purpose capabilities
- Safety-critical applications

## Training Procedure

The full training pipeline:

1. **Tokenizer Training:** Custom 32k BPE tokenizer trained on a sample of the pretraining corpus
2. **Data Processing:** Datasets cleaned, MinHash deduplicated, and packed into `jsonl.zst` shards across two Vultr bare metal servers
3. **Pretraining:** 54,362 steps on TPU v4-64 using MaxText (JAX)
4. **Checkpoint Conversion:** Orbax (JAX/zarr) → HuggingFace safetensors using custom `convert_checkpoint.py`

## Citation

```bibtex
@misc{tinymathreason1b,
  title={TinyMathReason-1B: A 1.1B Parameter Math Reasoning Model Trained From Scratch},
  author={Himanshu Nakrani},
  year={2026},
  url={https://huggingface.co/himanshunakrani9/TinyMathReason-1B-base}
}
```

## Acknowledgments

- [Google MaxText](https://github.com/google/maxtext) for the training framework
- [Google Cloud TPU Research Cloud](https://sites.research.google/trc/) for compute resources
- The open-source dataset creators (FineWeb-Edu, MathPile, OpenWebMath, The Stack)
