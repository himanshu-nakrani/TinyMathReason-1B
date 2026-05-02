# TinyMathReason-1B Project Status

This document tracks our progress through the accelerated Execution Plan.

## Phase 1: Setup & Data Prep (Days 1-5)
- [x] Project setup complete (Repo structured, `venv` created, dependencies installed).
- [x] Train custom 32k math tokenizer on sample data (`tokenizer.tiktoken` generated).
- [x] Spin up two Vultr `c2-standard-30` instances to process data in parallel.
- [x] Node A: Download, clean, MinHash, and pack FineWeb-Edu and MathPile.
- [x] Node B: Download, clean, MinHash, and pack OpenWebMath and Stack-Edu.
- [x] Upload all `jsonl.zst` shards to GCS bucket (`gs://tinymath-reason-data-himanshu/pretraining-data/`). Vultr nodes destroyed.

## Phase 2: Pretraining (Days 6-15)
- [ ] Provision `v4-32` cluster and run MaxText smoke test.
- [ ] Launch main pretraining run on `v4-32`.
- [ ] Run continuous `lm-eval` benchmarks on intermediate checkpoints.
- [ ] Pretraining finishes. Final 300B token checkpoint saved.
- [ ] Convert Orbax to HF Safetensors.

## Phase 3: Post-Training SFT (Days 16-17)
- [ ] Provision AMD MI300X instance.
- [ ] Stage 1 SFT: Prepare data and train on conversational prior (No CoT).
- [ ] Stage 2 SFT: Resize tokenizer (+ `<think>`) and train on reasoning traces (MathInstruct, OpenThoughts).

## Phase 4: Post-Training DPO/GRPO (Days 18-20)
- [ ] Generate preference candidate data using Modal serverless endpoints.
- [ ] Run DPOTrainer or GRPOTrainer on the AMD MI300X.

## Phase 5: Evaluation & Release (Days 21-23)
- [ ] Run full suite of benchmarks (`make eval`) on the MI300X.
- [ ] Use Lightning AI to plot training curves and draft report.
- [ ] Use Thunder Compute to host Gradio UI demo.
- [ ] Release Report, Model Card, and GitHub Repo.
