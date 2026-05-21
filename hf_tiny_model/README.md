---
language: en
license: apache-2.0
tags:
  - math
  - reasoning
  - llama
  - tinymath
library_name: transformers
pipeline_tag: text-generation
---

# TinyMathReason-1B (Base)

A 1.12B-parameter language model trained from scratch for mathematical reasoning.

## Architecture

- Type: Llama-2 style decoder-only transformer
- Parameters: ~1.12B
- Hidden dim: 2048
- Layers: 22
- Attention: GQA (16 query heads, 4 KV heads)
- MLP: SwiGLU (intermediate dim 5632)
- Context length: 4096
- Vocab size: 32,768

## Training

- Framework: MaxText (JAX) on TPU v4-64
- Data: ~57B tokens from FineWeb-Edu, MathPile, OpenWebMath, and Stack-Edu
- Optimizer: AdamW with cosine decay
- Precision: bfloat16

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("path/to/hf_model")
tokenizer = AutoTokenizer.from_pretrained("path/to/hf_model")

inputs = tokenizer("The sum of 2 and 3 is", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```
