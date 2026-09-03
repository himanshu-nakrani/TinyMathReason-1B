# TinyMathReason-1B: Phase 2 Retrospective & Phase 3 Context

This document captures the critical engineering context, bug fixes, and technical details from the Phase 2 Pretraining chat. Use this to maintain continuity in the next session.

***

## 1. Project Status: Phase 2 COMPLETE ✅
*   **Final Run Name:** `tinymath-1b-prod-run11` (RE-RUN)
*   **Total Tokens Processed:** ~57 Billion
*   **Final Step:** 54,362
*   **Location of Checkpoints (Orbax):** 
    `gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run11/checkpoints/54362/`

***

## 2. Critical Engineering Learnings & Bug Fixes

### A. The "Zero Layer" / 0.134B Parameter Bug
*   **Issue:** Early pretraining runs (run1-run9) produced checkpoints with only 13 keys, containing only embeddings and the output head (0.134B parameters).
*   **Root Cause:** A compatibility bug in MaxText's Linen-to-NNX wrapper. When `scan_layers: True` was used, the `nn.scan` operation silently failed to trace or register the 22 transformer layers, even with `pure_nnx_decoder: True`.
*   **Fix:** Permanently switched to **`pure_nnx_decoder: True`** AND **`scan_layers: False`**. This ensures the native NNX-based transformer blocks are properly instantiated and saved as individual layers, guaranteeing all **1.126 Billion** parameters are captured.

### B. Host RAM OOM during XLA Compilation
*   **Issue:** Once the 1.126B parameters were enabled, the TPU VM would crash with a "Killed" message shortly after initialization.
*   **Root Cause:** XLA compilation of a 1.1B parameter model with a large `per_device_batch_size` (8) requires >300GB of Host RAM (not TPU HBM) to build the HLO graph.
*   **Fix:** Reduced **`per_device_batch_size` to 2**. This brought Host RAM usage into a safe range and allowed compilation to succeed.

### C. JAX 0.6.2 Breaking Changes
*   **AttributeError (`jax.Ref`):** JAX 0.6.2 removed `jax.Ref`.
*   **Fix:** Added a permanent patch in `setup_tpu.sh` and `mock_injector.py` that shims `jax.Ref` to `typing.Any`. Also directly patched MaxText's ragged attention kernels via `sed` in the setup script.

### D. TPU Synchronization (Preemption Handling)
*   **Issue:** Spot TPU clusters frequently hit "SSH 255" or "Connection Refused" errors.
*   **Finding:** This is often an "Uptime Desync." If one worker reboots while others stay alive, the JAX distributed mesh fails.
*   **Standard Procedure:** If connectivity hangs, perform a sequential **`sudo reboot`** loop across all 8 workers. Wait 5 minutes for synchronization. All workers must have a nearly identical, fresh uptime to successfully form a mesh.

***

## 3. Technical Configuration (Active)

**Model Architecture (`src/train/tinymath-1b.yml`):**
*   22 Decoder Layers
*   Hidden Dim: 2048
*   MLP Dim: 5632 (SwiGLU)
*   GQA: 16 Query Heads / 4 KV Heads
*   Vocab Size: 32,768 (padded)

**Training Config (`src/train/maxtext_config.yml`):**
*   `pure_nnx_decoder: True`
*   `scan_layers: True`
*   `per_device_batch_size: 2`
*   `max_target_length: 4096`

***

## 4. Phase 3: Immediate Next Steps

### Step 1: Checkpoint Conversion
Run the conversion script to move from JAX/Orbax to PyTorch/Safetensors.
*   **Source:** `gs://.../tinymath-1b-prod-run9/checkpoints/54362/`
*   **Destination:** `./hf_model/`
*   **Verification:** Ensure the resulting `model.safetensors` is ~2.2GB (for bfloat16 weights).

### Step 2: Base Model Evaluation
Run benchmarks on the converted model to establish the reasoning baseline.
*   **Targets:** GSM8K, MATH, ARC, MMLU.

### Step 3: SFT Stage 1
*   Prepare the `MathInstruct` and `GSM8K` training data in ChatML format.
*   Provision a GPU (Thunder Compute) and launch `src/sft/train_sft.py`.

***

**End of Phase 2 Context Document**
