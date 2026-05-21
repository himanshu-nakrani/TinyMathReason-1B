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

TinyMathReason-1B (Base) is a 1.12B-parameter decoder-only language model trained from scratch for mathematical reasoning.

This is the pretrained base checkpoint, intended for continued fine-tuning, evaluation, and research. It is not instruction-tuned.

## Model details

### Architecture

- Type: Llama-2 style decoder-only transformer
- Parameters: ~1.12B
- Hidden size: 2048
- Intermediate size: 5632
- Layers: 22
- Query heads: 16
- KV heads: 4
- Head dimension: 128
- Vocabulary size: 32,768
- Context length: 4096
- Precision: bfloat16
- Normalization: RMSNorm
- Positional encoding: RoPE

### Tokenizer

- Type: custom BPE tokenizer
- Active vocabulary: 32,000 tokens
- Padded vocabulary: 32,768
- Includes special tokens for chat and reasoning workflows

## Training

- Framework: MaxText (JAX/Flax)
- Hardware: Google Cloud TPU v4-64
- Total training steps: 54,362
- Throughput: ~8,900 tokens/sec/chip
- Optimizer: AdamW
- Peak learning rate: 3e-4
- Weight decay: 0.1
- Precision: bfloat16

## Training data

The pretraining corpus totals about 57B tokens and combines:

- FineWeb-Edu
- MathPile
- OpenWebMath
- Stack-Edu

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

## Intended use

This model is intended for:

- research and education
- experimentation with math-focused fine-tuning
- demonstrating a from-scratch LLM training pipeline

It is not intended for production deployment or safety-critical applications without further tuning and evaluation.

## License

Apache 2.0
