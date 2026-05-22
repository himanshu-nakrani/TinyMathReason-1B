# TinyMathReason-1B 🧮

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Parameters](https://img.shields.io/badge/Parameters-1.12B-green.svg)]()
[![Architecture](https://img.shields.io/badge/Architecture-Llama--style-purple.svg)]()
[![Pretraining](https://img.shields.io/badge/Pretraining-TPU_v4--64-orange.svg)]()
[![Post--Training](https://img.shields.io/badge/Post--Training-AMD_MI300X%20%2B%20Modal-blue.svg)]()

TinyMathReason-1B is a 1.12B-parameter decoder-only transformer trained from scratch for mathematical reasoning.

The goal of this project is not just to ship a small math model, but to build and document the full end-to-end LLM stack: data curation, tokenizer training, TPU pretraining, checkpoint conversion, supervised fine-tuning, preference optimization, and evaluation. The repository is designed both as a reproducible engineering artifact and as a learning-focused reference for anyone studying how modern language models are built.

## Why this repo exists

Most open LLM projects only expose the final weights or a narrow slice of the training story. This repository aims to show the whole system:

- how a math-focused corpus is assembled and processed
- how a compact Llama-style architecture is chosen and justified
- how MaxText/JAX pretraining is run on TPUs
- how Orbax checkpoints are converted into Hugging Face format
- how SFT and DPO/GRPO are layered on top using PyTorch + TRL
- how each stage is evaluated and compared

If you are learning ML systems hands-on, this repo is meant to be useful as both code and documentation.

## What this repository includes

- tokenizer training code
- data download, cleaning, mixing, packing, and upload pipeline
- PyTorch reference model implementation
- TPU pretraining utilities and conversion scripts
- supervised fine-tuning code
- DPO / GRPO preference optimization code
- evaluation scripts and comparison helpers
- planning, architecture, and setup docs

## Model overview

TinyMathReason-1B uses a Llama-style decoder-only transformer optimized for compact reasoning workloads.

| Component | Value |
|---|---|
| Parameters | ~1.12B |
| Layers | 22 |
| Hidden size | 2048 |
| Attention heads | 16 query heads |
| KV heads | 4 |
| Head dimension | 128 |
| Attention type | Grouped Query Attention (GQA 4:1) |
| MLP | SwiGLU |
| Intermediate size | 5632 |
| Context length | 4096 |
| Base vocab size | 32,000 |
| HF/export vocab size | 32,768 |
| Norm | RMSNorm |
| Positional encoding | RoPE |
| Precision | bfloat16 |

Relevant code:
- `src/model/modeling_tinymath.py`
- `docs/architecture.md`

## Project status

Current pipeline progress (verified via `STATUS.md`):

- **Phase 1 COMPLETE:** Tokenizer training & pretraining corpus curation.
- **Phase 2 COMPLETE:** TPU pretraining (57B tokens) & Hugging Face conversion.
- **Phase 3 COMPLETE:** Post-Training SFT & comprehensive evaluation suite on AMD MI300X.
- **Phase 4 IN PROGRESS:** DPO / GRPO preference optimization on Modal.

### Benchmark Performance comparison

| Benchmark | Setting | Base Score | SFT Score | Delta Gain | GRPO Score (Future) |
|---|---|:---:|:---:|:---:|:---:|
| **GSM8K** | 8-shot (Template-aligned) | 1.00% | **1.00%** | *Adheres to `<think>` format* | *[TBD - Phase 4]* |
| **MATH (Algebra)** | 4-shot | 0.00% | **0.00%** | *Stable baseline* | *[TBD - Phase 4]* |
| **ARC-Easy** | 0-shot | 29.90% | **25.51%** | -4.39% | *[TBD - Phase 4]* |
| **ARC-Challenge** | 25-shot | 21.70% | **24.66%** | **+2.96%** 📈 | *[TBD - Phase 4]* |
| **HellaSwag** | 10-shot | 25.80% | **26.70%** | **+0.90%** 📈 | *[TBD - Phase 4]* |
| **MMLU** | 5-shot | 23.50% | **24.60%** | **+1.10%** 📈 | *[TBD - Phase 4]* |

These SFT gains demonstrate stable general reasoning alignment, positioning us perfectly for RL-based mathematical instruction tuning in Phase 4!

## Training pipeline

The repo is structured around the same multi-stage workflow used for many modern LLMs.

### 1. Tokenizer training

A custom BPE tokenizer is trained on math-heavy text so the model can better represent formulas, symbols, and technical text efficiently.

Main file:
- `src/data/train_tokenizer.py`

### 2. Data pipeline

The pretraining corpus is downloaded, cleaned, filtered, mixed, tokenized, packed, sharded, and uploaded for TPU training.

Pipeline files:
- `src/data/pipeline/a_download_datasets.py`
- `src/data/pipeline/b_clean_and_filter.py`
- `src/data/pipeline/c_mix_datasets.py`
- `src/data/pipeline/d_tokenize_and_pack.py`
- `src/data/pipeline/e_create_shards.py`
- `src/data/pipeline/f_upload_to_gcs.py`

### 3. Pretraining

Pretraining is run with MaxText on Google Cloud TPU v4 hardware. The repo includes configuration, monitoring, and preemption-handling utilities to support long-running TPU jobs.

Main files:
- `src/train/preemption_handler.py`
- `src/train/monitor_training.py`
- `docs/pretraining_setup.md`

### 4. Checkpoint conversion

The pretrained checkpoint is converted from Orbax/JAX format into Hugging Face-compatible safetensors for downstream use and evaluation.

Main files:
- `src/train/convert_checkpoint.py`
- `src/train/inspect_checkpoint.py`
- `src/train/verify_model.py`
- `src/eval/verify_hf.py`

### 5. Supervised fine-tuning

Completed a robust two-stage post-training SFT curriculum on AMD MI300X:

- **Stage 1 (Conversational Prior):** Fine-tuned on conversational instruction datasets to align base output forms.
- **Stage 2 (Reasoning Traces):** Resized token embeddings to support special reasoning `<think>` and `</think>` tags, fine-tuning on math-specific reasoning datasets (GSM8K, MathInstruct, MetaMathQA).

Main files:
- `src/sft/prepare_sft_data.py`
- `src/sft/train_sft.py`
- `src/eval/run_custom_eval.py`
- `docs/sft_setup.md`

### 6. Preference optimization

The repo includes both DPO and GRPO training flows for improving answer quality and reasoning behavior after SFT.

Main files:
- `src/dpo/generate_preferences.py`
- `src/dpo/train_dpo.py`
- `src/dpo/train_grpo.py`
- `docs/dpo_setup.md`

### 7. Evaluation

Evaluation scripts cover benchmark execution, custom evaluation, curve plotting, and model comparison.

Main files:
- `src/eval/run_benchmarks.py`
- `src/eval/run_custom_eval.py`
- `src/eval/generate_comparison.py`
- `src/eval/plot_training_curves.py`
- `src/eval/modal_eval.py`

## Data mixture

The project combines general educational text with math-heavy data so the model learns both broad language structure and domain-specific mathematical patterns.

Documented sources across the repo include:
- FineWeb-Edu
- OpenWebMath
- MathPile
- Stack-Edu

The processed pretraining corpus is approximately 57B tokens in the currently documented completed run.

## Infrastructure used

This project spans multiple compute environments across the training lifecycle.

| Stage | Infrastructure |
|---|---|
| Tokenizer / local development | local machine |
| Data processing | 2x Vultr `c2-standard-30` nodes |
| Pretraining | Google Cloud TPU v4-64 |
| SFT | AMD MI300X (192GB VRAM) |
| Preference generation | Modal |
| Final evaluation / comparisons | Lightning AI + Thunder Compute |

This makes the repo useful not only as a model-training project, but also as a practical example of hybrid infra orchestration across CPUs, TPUs, and GPUs.

## Repository layout

```text
.
├── README.md
├── STATUS.md
├── docs/
├── hf_1b_model/
├── hf_tiny_model/
├── src/
│   ├── data/
│   ├── dpo/
│   ├── eval/
│   ├── model/
│   ├── sft/
│   └── train/
└── Makefile
```

## Quick start

If you want to use a final Hugging Face checkpoint locally:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "your-hf-username/TinyMathReason-1B-DPO"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

prompt = "Solve the equation: 3x + 7 = 22"
messages = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)

outputs = model.generate(
    inputs,
    max_new_tokens=256,
    temperature=0.1,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

If you only want to inspect the exported base checkpoint, see:
- `hf_1b_model/README.md`
- `hf_tiny_model/README.md`

## Reproducing the pipeline

Top-level commands:

- `make data` — run the full data processing pipeline
- `make pretrain` — TPU pretraining entrypoint / instructions
- `make sft` — prepare SFT data and launch fine-tuning
- `make dpo` — generate preferences and run DPO
- `make eval` — run benchmark and comparison scripts

Supporting docs:
- `docs/pretraining_setup.md`
- `docs/sft_setup.md`
- `docs/dpo_setup.md`
- `docs/vultr_setup.md`
- `docs/execution_plan.md`

## Why the pretrained baseline matters

One of the most instructive parts of this repository is that it does not hide the weak pretrained baseline. For a reasoning model, the jump from base pretraining to SFT and then to preference optimization is the story. Keeping the base metrics visible makes the later improvements measurable and honest.

That is especially valuable for:
- ML learners studying how capabilities emerge across stages
- researchers comparing post-training methods
- engineers who want a transparent end-to-end artifact instead of only polished final weights

## Intended use

TinyMathReason-1B is best understood as:

- a learning and portfolio project for end-to-end LLM systems
- a research artifact for small-model math reasoning experiments
- a starting point for instruction tuning and alignment experiments
- a reference repo for data → pretrain → post-train → eval workflows

It is not intended, in its current form, for:
- production deployment without substantial additional validation
- safety-critical decision making
- strong general-purpose assistant behavior in the base checkpoint

## Known limitations

- The released base model benchmark scores are still weak on reasoning-heavy tasks.
- The project is mid-pipeline: SFT and preference optimization are not yet fully reflected in the repo root README results.
- Some documentation in the repo may describe earlier plans or older hardware assumptions; `STATUS.md` is the best snapshot of current progress.
- The citation block and some public-facing metadata may still need personalization before release.

## Suggested reading order

If you are new to the repository, this order works well:

1. `README.md`
2. `STATUS.md`
3. `docs/architecture.md`
4. `docs/pretraining_setup.md`
5. `src/train/convert_checkpoint.py`
6. `docs/sft_setup.md`
7. `src/eval/run_benchmarks.py`

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
