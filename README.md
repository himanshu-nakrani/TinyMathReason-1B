# TinyMathReason-1B 🧮

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Parameter Count](https://img.shields.io/badge/Parameters-1.12B-green.svg)]()
[![Training Setup](https://img.shields.io/badge/Pretraining-TPU_v4--64-orange.svg)]()

TinyMathReason-1B is a 1.12B-parameter Llama-style decoder-only transformer trained from scratch for mathematical reasoning.

This repository contains the full pipeline: tokenizer training, data preparation, TPU pretraining, checkpoint conversion, supervised fine-tuning, preference optimization, and evaluation.

## Model Overview

- Architecture: Llama-2 style decoder-only transformer
- Parameters: ~1.12B
- Layers: 22
- Hidden size: 2048
- Attention: 16 query heads, 4 KV heads (GQA 4:1)
- MLP: SwiGLU, intermediate size 5632
- Vocabulary: 32,768
- Context length: 4096
- Precision: bfloat16

## Training Pipeline

1. Train a custom BPE tokenizer for math-heavy text.
2. Prepare and shard the pretraining corpus.
3. Pretrain on TPU using MaxText.
4. Convert checkpoints to Hugging Face format.
5. Run SFT on curated instruction data.
6. Run DPO / GRPO preference optimization.
7. Evaluate on math and reasoning benchmarks.

## Repository Layout

- src/data/ — tokenizer and data pipeline code
- src/model/ — model architecture
- src/sft/ — supervised fine-tuning
- src/dpo/ — preference optimization
- src/eval/ — evaluation scripts
- docs/ — setup and runbooks

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "your-hf-username/TinyMathReason-1B-DPO"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).cuda()

prompt = "Solve the equation: 3x + 7 = 22"
inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    return_tensors="pt",
    add_generation_prompt=True,
).cuda()

outputs = model.generate(inputs, max_new_tokens=256, temperature=0.1)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Reproduction

- `make data` — download, clean, deduplicate, and shard the corpus
- `make pretrain` — see `docs/pretraining_setup.md`
- `make sft` — prepare SFT data and train
- `make dpo` — generate preferences and train DPO/GRPO
- `make eval` — run the evaluation suite

## Citation

```bibtex
@misc{tinymathreason2026,
  author = {Your Name},
  title = {TinyMathReason-1B: A 1.12B Parameter Mathematical Reasoning Model Built from Scratch},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/your-username/TinyMathReason-1B}}
}
```

## License

Apache 2.0
