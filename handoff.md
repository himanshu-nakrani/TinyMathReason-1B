# TinyMathReason-1B: Project Handoff & State Summary

**Last Updated:** 2026-05-12
**Current Status:** Phase 2 (Pretraining) COMPLETE → Phase 3 (Checkpoint Conversion + Evaluation + SFT) is next.

***

## 1. Project Objective & Architecture

**Goal:** Train a 1.07 Billion parameter LLM from scratch, specializing in mathematical reasoning. The full pipeline is: Tokenizer → Pretraining → SFT → DPO/GRPO → Evaluation → Release.

**This is a portfolio/learning project — not targeting SOTA. The goal is to demonstrate the full LLM training stack.**

### Model Architecture (Llama-2 Style)
| Parameter | Value |
|---|---|
| Total Parameters | ~1.07B (0.134B per FSDP shard × 8 workers) |
| Hidden Dim | 2048 |
| MLP Dim | 5632 (SwiGLU) |
| Layers | 22 |
| Query Heads | 16 |
| KV Heads | 4 (GQA, 4:1 ratio) |
| Head Dim | 128 |
| Vocab Size | 32,768 (padded from 32k for FSDP alignment) |
| Max Seq Len | 4096 |
| Precision | bfloat16 |
| Normalization | RMSNorm (eps=1e-5) |
| Position Encoding | RoPE (theta=10000) |

### Tokenizer
- **Type:** Custom BPE via tiktoken format
- **Vocab:** 32,000 active tokens (padded to 32,768 in model config)
- **Location:** `gs://tinymath-reason-data-himanshu/tokenizer/tokenizer.tiktoken` and `tokenizer/tokenizer.tiktoken` in repo
- **Special tokens:** Includes `<think>` and `</think>` for future reasoning trace SFT

### SFT Strategy (Planned)
- **Stage 1:** Conversational instruction-following (no CoT)
- **Stage 2:** Reasoning trace fine-tuning with `<think>` tags. Requires `--resize_token_embeddings` to integrate the special tokens.

---

## 2. Phase 1: Pretraining Data Pipeline (COMPLETED ✅)

Processed across two Vultr Bare Metal servers (now destroyed). All data on GCS.

**Total Pretraining Corpus:** ~57 Billion Tokens in `jsonl.zst` format.

### Dataset Breakdown
| Dataset | Tokens | Shards | GCS Path |
|---|---|---|---|
| FineWeb-Edu | ~10B | 363 | `gs://.../pretraining-data/` (root) |
| GAIR/MathPile | ~9.5B | 225 | `gs://.../pretraining-data/mathpile/` |
| OpenWebMath + Stack-Edu | ~37.7B | 1,041 | `gs://.../pretraining-data/math-and-code/` |

**Bucket:** `gs://tinymath-reason-data-himanshu/pretraining-data/`

### Data Pipeline Gotchas (If Re-running)
- HF deprecated `trust_remote_code=True` — use `datasets==2.21.0`
- Zstandard crashes on HTTP range streaming — use `streaming=False` for `.zst`
- Hard-cap `num_cores = 14` in tokenize/shard scripts to avoid OOM kills
- Skip/resume logic in pipeline: `if out_file.exists(): return True`

---

## 3. Phase 2: Pretraining (COMPLETED ✅)

### Training Configuration
- **Framework:** MaxText (JAX) on GCP TPU v4-64 (8 workers × 4 chips = 32 TPU v4 chips)
- **Config file:** `src/train/maxtext_config.yml`
- **Setup script:** `scripts/setup_tpu.sh` (handles MaxText clone, deps, JAX patches, mock injection)
- **Run name:** `tinymath-1b-prod-run2`
- **Total steps:** 54,363 (0-indexed, last step = 54362)
- **Batch size:** 256 sequences × 4096 tokens = ~1M tokens/step
- **Optimizer:** AdamW (lr=3e-4, cosine decay to 0.1×, β1=0.9, β2=0.95, weight_decay=0.1)
- **Warmup:** ~362 steps (0.67% of total)
- **Gradient clipping:** 1.0
- **Checkpoint interval:** every 1,000 steps (12 total checkpoints)

### Final Checkpoint
```
gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/
├── _CHECKPOINT_METADATA
├── commit_success.txt    ← Confirms successful commit
└── items/                ← Orbax parameter shards
```

### All Available Checkpoints
12 checkpoints exist at steps in `gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/` (steps include 44000, 45000, 46000, ..., 52000, 53000, 54362).

### TensorBoard Logs
```
gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/tensorboard/
```

### Critical Engineering Context

#### JAX 0.6.2 Compatibility
MaxText HEAD (as of May 2026) has breaking changes with JAX 0.6.2. The `setup_tpu.sh` script handles all of these via a **mock injector** prepended to `train.py`:

1. **Pallas Import Migration:** `jax.experimental.pallas` was removed in JAX 0.6.x. A `_PallasMockFinder` meta_path hook intercepts ALL pallas imports and returns mock package modules. This is safe because we use `dot_product` attention (not Pallas GPU kernels).

2. **`jax.sharding.reshard` removed:** Patched via `sed` in setup script.

3. **`flax.nnx.Pytree` → `Object`:** Patched via `sed`.

4. **`jax.set_mesh` missing in 0.6.x:** Shimmed with a context manager.

5. **`jax.jit` decorator factory pattern:** Monkey-patched to support `jax.jit()` without arguments.

6. **Internal Google modules (pathwaysutils, qwix, tokamax, drjax):** All mocked via `MagicMock` with proper exception classes.

7. **Grain compatibility:** `BestFitPackIterDataset` and `PyGrainCheckpointHandler` stubs added.

#### Spot Preemption Handling
- TPU v4-64 is a spot instance — preemptions happen regularly
- `setup_tpu.sh` has a `while true` restart loop for software crashes
- For hardware preemptions, the TPU must be recreated via `gcloud compute tpus tpu-vm create`
- Moved from `tinymath-1b-prod-run1` to `tinymath-1b-prod-run2` to avoid Orbax checkpoint scanning overhead with too many old checkpoints

#### Known Issue: Final Step Restart Loop
When restoring from step 54362 (the final step), MaxText tries to re-checkpoint that step but it already exists → "Checkpoint for step 54362 already exists" → exits → restart loop. This is harmless — training is complete. Just kill the processes.

---

## 4. Next Steps (IN ORDER)

### Step 1: Checkpoint Conversion (MaxText Orbax → HuggingFace)
**Script:** `src/train/convert_checkpoint.py`
**What it does:** Maps MaxText JAX parameter names to HuggingFace `LlamaForCausalLM` format, saves as safetensors.

```bash
python src/train/convert_checkpoint.py \
    --orbax_dir gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362 \
    --hf_out_dir ./hf_model \
    --tokenizer_path ./tokenizer
```

**⚠️ IMPORTANT NOTES:**
- The conversion script uses `vocab_size=32000` in `LlamaConfig` but the MaxText training used `vocab_size=32768`. This needs to be reconciled — **update the conversion script to use 32768**.
- The MaxText param tree structure may differ from what the conversion script assumes (it was written before training). You'll likely need to **inspect the actual checkpoint structure** first:
  ```python
  from orbax import checkpoint as ocp
  ckpt = ocp.PyTreeCheckpointer().restore("gs://...")
  print(jax.tree.map(lambda x: x.shape, ckpt))
  ```
- The conversion requires JAX + PyTorch both installed. Use a GPU instance or Colab.
- The script references `PreTrainedTokenizerFast.from_pretrained()` but our tokenizer is tiktoken format — may need adaptation.

### Step 2: Evaluate Base (Pretrained) Model
**Script:** `src/eval/run_benchmarks.py`

```bash
python src/eval/run_benchmarks.py --model_path ./hf_model --output_dir ./eval_results/base
```

**Benchmarks:**
| Benchmark | Shots |
|---|---|
| GSM8K | 8 |
| MATH (algebra) | 4 |
| ARC-Easy | 0 |
| ARC-Challenge | 25 |
| HellaSwag | 10 |
| MMLU | 5 |

Also run custom eval: `src/eval/run_custom_eval.py` (30 hand-curated math problems)

### Step 3: SFT Training
**Scripts:** `src/sft/train_sft.py`, `src/sft/prepare_sft_data.py`, `src/sft/sft_config.yaml`
- Stage 1: Conversational instruction-following
- Stage 2: `<think>` reasoning traces (MathInstruct, OpenThoughts)
- Framework: TRL SFTTrainer (PyTorch)
- Requires GPU (MI300X, A100, or similar)

### Step 4: Evaluate SFT Model
Same benchmarks as Step 2, on the SFT checkpoint.

### Step 5: DPO/GRPO
**Scripts:** `src/dpo/` directory
- Generate preference data using SFT model
- Train with TRL DPOTrainer

### Step 6: Final Evaluation & Release
- Run all benchmarks on base vs SFT vs DPO
- Side-by-side comparison with TinyLlama-1.1B, Qwen2.5-0.5B (`src/eval/generate_comparison.py`)
- Plot training curves (`src/eval/plot_training_curves.py`)
- Write report, model card, blog post
- Release on HuggingFace + GitHub

---

## 5. Key Files Reference

| File | Purpose |
|---|---|
| `src/train/maxtext_config.yml` | MaxText training configuration (final) |
| `scripts/setup_tpu.sh` | TPU VM setup + training launch script (JAX compat patches, mock injector) |
| `src/train/convert_checkpoint.py` | Orbax → HuggingFace safetensors conversion |
| `src/model/modeling_tinymath.py` | PyTorch model definition (reference implementation) |
| `src/eval/run_benchmarks.py` | lm-evaluation-harness wrapper |
| `src/eval/run_custom_eval.py` | 30 hand-curated math problems |
| `src/eval/generate_comparison.py` | Side-by-side model comparison |
| `src/eval/plot_training_curves.py` | Training curve visualization |
| `src/sft/train_sft.py` | SFT training script (TRL) |
| `src/sft/prepare_sft_data.py` | SFT data preparation |
| `src/sft/sft_config.yaml` | SFT hyperparameters |
| `tokenizer/tokenizer.tiktoken` | Custom 32k BPE tokenizer |
| `prompt.md` | Full project specification (all 13 deliverables) |
| `STATUS.md` | Progress tracker |

## 6. GCS Bucket Layout

```
gs://tinymath-reason-data-himanshu/
├── pretraining-data/
│   ├── *.jsonl.zst                    (FineWeb-Edu, 363 shards)
│   ├── mathpile/*.jsonl.zst           (MathPile, 225 shards)
│   └── math-and-code/*.jsonl.zst      (OpenWebMath+Stack-Edu, 1041 shards)
├── tokenizer/
│   └── tokenizer.tiktoken
├── checkpoints/
│   └── tinymath-1b-prod-run2/
│       ├── checkpoints/
│       │   ├── 44000/ ... 54362/      (12 Orbax checkpoints)
│       └── tensorboard/
└── metrics/
```

***

*End of Handoff Document*