# Phase 4: GRPO Preference Optimization Setup

GRPO (Group Relative Policy Optimization) is used instead of DPO because the SFT model's near-zero math baselines (GSM8K 1%, MATH 0%) make it impossible to generate valid chosen/rejected preference pairs. GRPO's online group exploration bootstraps reasoning from scratch.

## Infrastructure

| Component | Resource | Purpose |
|-----------|----------|---------|
| Training | AMD MI300X (192GB VRAM) | GRPO trainer (policy + reference model + G=8 rollouts) |
| Monitoring | WandB | Reward curves, KL divergence, loss tracking |

## Dependencies

```bash
pip install trl>=0.17.0 math-verify vllm sympy wandb
```

> **Note:** MI300X uses ROCm. If `vllm` is unstable on ROCm, the script falls back to native HF generation (192GB handles G=8 comfortably without vLLM).

## Staged Execution

### Stage A: Smoke Test (~10 min)
```bash
python src/dpo/train_grpo.py \
  --model_path ./models/sft-1.1b-math \
  --output_dir ./outputs/grpo-smoke \
  --max_samples 50
```

Verify: no OOM, WandB logging, non-zero rewards, no conversation simulation.

### Stage B: Calibration (~1-2 hours)
```bash
python src/dpo/train_grpo.py \
  --model_path ./models/sft-1.1b-math \
  --output_dir ./outputs/grpo-calibration \
  --max_samples 500
```

Monitor on WandB:
- `reward/correctness` — should slowly increase from ~0
- `reward/format` — should climb to ~0.8+ within 50 steps
- `reward/repetition` — should stay near 0
- `kl_divergence` — should stay < 5.0

### Stage C: Full Training (~4-8 hours)
```bash
python src/dpo/train_grpo.py \
  --model_path ./models/sft-1.1b-math \
  --output_dir ./outputs/grpo-full
```

### Optional: vLLM Acceleration
```bash
python src/dpo/train_grpo.py \
  --model_path ./models/sft-1.1b-math \
  --output_dir ./outputs/grpo-full \
  --use_vllm
```

## Key Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `num_generations` (G) | 8 | Minimum for stable advantage normalization |
| `beta` (KL) | 0.01 | Protects MMLU/ARC gains while allowing math exploration |
| `learning_rate` | 5e-6 | Escapes 0% MATH local minimum |
| `lr_scheduler` | cosine | Smooth convergence |
| `warmup_ratio` | 0.05 | Gentle ramp-up for high-variance gradients |

## Reward Functions

1. **`correctness_reward_func`** — AST-based math verification via `math_verify` + GSM8K numeric fallback
2. **`format_reward_func`** — Strict regex validation of `<think>...</think><answer>...</answer>` structure
3. **`repetition_penalty_func`** — 3-gram uniqueness check to kill mode collapse loops (penalty: -1.5)

## Post-Training Evaluation

```bash
# Quick checkpoint eval
python src/eval/run_custom_eval.py \
  --model_path ./outputs/grpo-full/checkpoint-200 \
  --output_file ./eval_results/grpo_ckpt200.md

# Full benchmark suite
python src/eval/run_benchmarks.py --model_path ./outputs/grpo-full/final

# Upload to HF Hub
huggingface-cli upload himanshunakrani9/TinyMathReason-1B-grpo ./outputs/grpo-full/final
```
