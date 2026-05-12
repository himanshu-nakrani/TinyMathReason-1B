# TinyMathReason-1B Project Status

This document tracks our progress through the accelerated Execution Plan.

**Last Updated:** 2026-05-12

## Phase 1: Setup & Data Prep (Days 1-5) ✅
- [x] Project setup complete (Repo structured, `venv` created, dependencies installed).
- [x] Train custom 32k math tokenizer on sample data (`tokenizer.tiktoken` generated).
- [x] Spin up two Vultr `c2-standard-30` instances to process data in parallel.
- [x] Node A: Download, clean, MinHash, and pack FineWeb-Edu and MathPile.
- [x] Node B: Download, clean, MinHash, and pack OpenWebMath and Stack-Edu.
- [x] Upload all `jsonl.zst` shards to GCS bucket (`gs://tinymath-reason-data-himanshu/pretraining-data/`). Vultr nodes destroyed.

## Phase 2: Pretraining (Days 6-15) ✅
- [x] Provision `v4-32` cluster and run MaxText smoke test.
- [x] Setup TPU VM (MaxText, dependencies, config, mock injector for JAX/Pallas compatibility).
- [x] Launch main pretraining run on `v4-64` (upgraded from v4-32 for faster throughput).
- [x] Handle multiple spot preemptions (moved to `tinymath-1b-prod-run2` for clean Orbax state).
- [x] Resolve JAX 0.6.2 breaking changes (Pallas import path migration, AbstractMesh, reshard API).
- [x] **Pretraining COMPLETE.** 54,363 steps (~57B tokens) finished.
  - Final checkpoint: `gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/`
  - Architecture: 1.07B params (22 layers, 2048 dim, 16 query heads, 4 KV heads, 5632 MLP dim)
  - Optimizer: AdamW (lr=3e-4, cosine decay, β1=0.9, β2=0.95)
  - Batch size: 256 sequences × 4096 tokens = ~1M tokens/step
  - Checkpoint interval: every 1,000 steps (12 checkpoints saved)
- [ ] Convert Orbax checkpoint to HuggingFace Safetensors (`src/train/convert_checkpoint.py`).

## Phase 3: Post-Training SFT (Days 16-17)
- [ ] Provision GPU instance (AMD MI300X or equivalent).
- [ ] Stage 1 SFT: Prepare data and train on conversational prior (No CoT).
- [ ] Stage 2 SFT: Resize tokenizer (+ `<think>`) and train on reasoning traces (MathInstruct, OpenThoughts).

## Phase 4: Post-Training DPO/GRPO (Days 18-20)
- [ ] Generate preference candidate data using Modal serverless endpoints.
- [ ] Run DPOTrainer or GRPOTrainer.

## Phase 5: Evaluation & Release (Days 21-23)
- [ ] Run full benchmark suite at every stage (base → SFT → DPO):
  - GSM8K (8-shot), MATH (4-shot), ARC-Easy (0-shot), ARC-Challenge (25-shot), HellaSwag (10-shot), MMLU (5-shot)
- [ ] Run custom 30-problem math eval and side-by-side comparison (TinyLlama, Qwen2.5).
- [ ] Plot training curves and draft report.
- [ ] Host Gradio UI demo.
- [ ] Release Report, Model Card, and GitHub Repo.

---

## Key Files

| File | Purpose |
|---|---|
| `scripts/setup_tpu.sh` | TPU setup + training launch (MaxText + JAX compat patches) |
| `src/train/maxtext_config.yml` | MaxText training config |
| `src/train/convert_checkpoint.py` | Orbax → HuggingFace conversion |
| `src/model/modeling_tinymath.py` | PyTorch model definition (1.07B LLaMA-style) |
| `src/eval/run_benchmarks.py` | lm-evaluation-harness wrapper |
| `src/sft/train_sft.py` | SFT training with TRL |
