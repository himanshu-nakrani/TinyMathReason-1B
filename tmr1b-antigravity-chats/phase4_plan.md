# Phase 4: Optimal GRPO Training Plan

**Model:** TinyMathReason-1B-SFT → TinyMathReason-1B-GRPO  
**Infra:** Modal (generation) + AMD MI300X 192GB VRAM (training)  
**Timeline:** Days 18–20 (starting now)

---

## Overview

Phase 3 delivered an SFT model with stable format compliance (`<think>` traces) and marginal benchmark gains (ARC-C +2.96%, HellaSwag +0.90%, MMLU +1.10%), but math reasoning remains at floor (GSM8K 1.00%, MATH 0.00%). Phase 4 applies GRPO to bootstrap actual mathematical reasoning capability while protecting the general knowledge gains from SFT.

The grpo_optimization.md analysis identified **6 critical vulnerabilities** in the current `train_grpo.py`. This plan integrates all fixes into a concrete execution sequence.

---

## Step 0: Decision — GRPO-Only vs DPO+GRPO

> [!IMPORTANT]
> **Recommendation: GRPO-only.** Skip DPO entirely.

| Factor | DPO | GRPO |
|--------|-----|------|
| Data requirement | Needs pre-generated chosen/rejected pairs | Online — generates its own rollouts |
| Memory | Dual model (policy + reference) | Single model + frozen ref copy (vLLM sleep mode handles this) |
| Reasoning depth | Optimizes surface preferences | Directly optimizes for correct mathematical derivation |
| Fit for 0% MATH baseline | Poor — the SFT model can barely produce correct answers, so preference pairs will be almost entirely rejected | Designed for exactly this — bootstrap from near-zero via exploration |

The current `generate_preferences.py` would produce nearly zero valid chosen/rejected pairs given the 0% MATH and 1% GSM8K baselines. GRPO's group exploration is the only viable path.

---

## Step 1: Environment & Dependencies Setup

**Where:** AMD MI300X instance  
**Time estimate:** 30 min

```bash
# SSH into AMD MI300X
source ~/sft_env/bin/activate

# Install/upgrade critical dependencies
pip install --upgrade trl>=0.17.0 math-verify vllm>=0.7.0 sympy wandb

# Verify vLLM ROCm compatibility (MI300X uses ROCm, not CUDA)
python -c "import vllm; print(vllm.__version__)"

# Verify math_verify
python -c "from math_verify import parse, verify; print('math_verify OK')"

# Pull SFT model weights (if not already on instance)
# Option A: From HF Hub
# huggingface-cli download himanshunakrani9/TinyMathReason-1B-sft --local-dir ./models/sft-1.1b-math

# Option B: From previous SFT output
ls -la ./sft_output/final/
```

> [!WARNING]
> **ROCm vs CUDA:** The MI300X runs ROCm. The vLLM colocate mode described in grpo_optimization.md assumes CUDA. You must verify that `vllm` is installed with ROCm support (`pip install vllm-rocm` or the ROCm-compatible build). If vLLM ROCm colocate mode is unstable, fall back to native HF `transformers` generation with the memory optimizations below.

### Fallback: If vLLM Colocate Doesn't Work on ROCm

```python
# Remove vLLM flags entirely. Use native generation instead:
# In GRPOConfig, remove:
#   use_vllm, vllm_mode, vllm_gpu_memory_utilization, vllm_enable_sleep_mode
# The 192GB VRAM of MI300X is large enough for G=8 with native generation.
```

---

## Step 2: Harden the Training Script

**Where:** Local machine (edit, commit, push) → pull on MI300X  
**Time estimate:** 1–2 hours  
**Files to modify:** [train_grpo.py](file:///Users/himanshu/Git/TinyMathReason-1B/src/dpo/train_grpo.py)

### 2.1 — Reward Functions (Critical Fixes)

The current script has **3 structural vulnerabilities** identified in grpo_optimization.md:

#### Fix 1: Replace string-match correctness with AST verification

```diff
-from trl import GRPOTrainer, GRPOTrainingArguments
+from trl import GRPOTrainer, GRPOConfig
+from math_verify import LatexExtractionConfig, parse, verify

-def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
+def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
+    """AST-based mathematical equivalence verification."""
     rewards = []
-    for comp, gt in zip(completions, answer):
-        pred = extract_answer(comp)
-        truth = extract_answer(gt)
-        if pred and truth and pred.replace(",","").replace(" ","") == truth.replace(",","").replace(" ",""):
-            rewards.append(1.0)
-        else:
-            rewards.append(0.0)
+    completion_contents = [c["content"] if isinstance(c, dict) else c for c in completions]
+    for content, gt in zip(completion_contents, answer):
+        try:
+            gold = parse(gt, extraction_mode="first_match",
+                         extraction_config=[LatexExtractionConfig()])
+            if not gold:
+                rewards.append(0.0); continue
+            pred = parse(content, extraction_mode="first_match",
+                         extraction_config=[LatexExtractionConfig()])
+            rewards.append(1.0 if verify(pred, gold) else 0.0)
+        except Exception:
+            rewards.append(0.0)
     return rewards
```

> **Why:** String matching fails on equivalent representations (1/2 vs 0.5 vs \\frac{1}{2}). With a 1% GSM8K baseline, every correctly-solved problem must be detected — false negatives poison the gradients.

#### Fix 2: Replace heuristic format check with strict regex

```diff
-def format_reward_func(prompts, completions, **kwargs) -> list[float]:
+def format_reward_func(completions, **kwargs) -> list[float]:
+    """Binary structural validation via regex."""
+    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
+    contents = [c["content"] if isinstance(c, dict) else c for c in completions]
+    matches = [bool(re.match(pattern, c, re.DOTALL)) for c in contents]
+    return [1.0 if m else 0.0 for m in matches]
```

> **Why:** The old 0.2 partial reward for just having `<think>` tags lets the model game the format without producing real content. Binary 1.0/0.0 forces sharp compliance.

#### Fix 3: Add n-gram repetition penalty (NEW reward function)

```python
def repetition_penalty_func(completions, **kwargs) -> list[float]:
    """Penalizes mode collapse loops via 3-gram uniqueness ratio."""
    rewards = []
    for c in completions:
        text = c["content"] if isinstance(c, dict) else c
        words = text.split()
        if len(words) < 3:
            rewards.append(0.0); continue
        ngrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        ratio = len(set(ngrams)) / len(ngrams)
        if ratio < 0.2:
            rewards.append(-1.5 * (1.0 - ratio))
        else:
            rewards.append(0.0)
    return rewards
```

> **Why:** The SFT model is known to enter infinite `e^x + e^x...` loops. Without this, GRPO rollouts waste compute on degenerate sequences and the policy never receives useful gradients.

### 2.2 — Stopping Criteria (Conversation Simulation Fix)

```python
# After loading tokenizer:
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Build explicit stop token list
stop_ids = [tokenizer.eos_token_id]
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
if isinstance(im_end_id, int) and im_end_id > 0:
    stop_ids.append(im_end_id)

# Inject into generation kwargs
generation_kwargs = {
    "max_new_tokens": 512,
    "do_sample": True,
    "temperature": 0.9,
    "stop_token_ids": stop_ids,
}
```

> **Why:** Without explicit `<|im_end|>` in stop tokens, the model continues generating fake user/assistant turns after its real answer, wasting the entire 512-token budget on hallucinated conversation.

### 2.3 — Hyperparameter Architecture

| Parameter | Old Value | New Value | Rationale |
|-----------|-----------|-----------|-----------|
| `num_generations` (G) | 4 | **8** | G=4 causes σ→0 when all generations fail, zeroing gradients. G=8 is the minimum for stable advantage normalization. |
| `beta` (KL penalty) | *not set* | **0.01** | Protects MMLU/ARC/HellaSwag gains from SFT while allowing deep math reasoning exploration. |
| `learning_rate` | 1e-6 | **5e-6** | 1e-6 is too slow to escape the 0% MATH local minimum. Empirical GRPO literature on 1.5B models uses 5e-6. |
| `lr_scheduler_type` | linear | **cosine** | Smooth long-tail annealing prevents late-stage instability. |
| `warmup_ratio` | *not set* | **0.05** | Gentle ramp-up before high-variance GRPO gradients hit. |
| `gradient_checkpointing` | True | True | Keep — saves ~40% activation memory. |
| `max_completion_length` | 512 | 512 | Keep — sufficient for GSM8K-level problems. |

### 2.4 — Memory Configuration (MI300X-Specific)

The MI300X has **192GB VRAM** — significantly more than typical A100 80GB setups. This gives us headroom:

```python
# Memory budget breakdown for MI300X (192GB):
# - Model weights (bf16):        ~2.2 GB
# - Optimizer states (AdamW):    ~8.8 GB (4x weights for 32-bit moments)
# - Gradients:                   ~2.2 GB
# - Activations (grad ckpt):    ~3-5 GB
# - KV cache for G=8 rollouts:  ~4-8 GB
# - Buffer/fragmentation:       ~5 GB
# TOTAL: ~25-30 GB out of 192 GB — we have MASSIVE headroom

# Therefore on MI300X:
# Option A (if vLLM ROCm works):
#   vllm_gpu_memory_utilization=0.3  (still conservative, ~57GB for vLLM)
#   vllm_enable_sleep_mode=True
#
# Option B (recommended if ROCm vLLM is unstable):
#   Skip vLLM entirely — 192GB handles G=8 native generation easily
#   Just use per_device_train_batch_size=2 with gradient_accumulation_steps=4
```

> [!TIP]
> With 192GB, you could even try **G=16** for even better advantage normalization. Start with G=8 to validate the pipeline, then bump to G=16 if memory allows.

---

## Step 3: Staged Training Execution

> [!IMPORTANT]
> **Do NOT run the full GSM8K train split (7,473 problems) in one shot.** Use a staged approach to catch issues early.

### Stage A: Smoke Test (10 min)

```bash
python src/dpo/train_grpo.py \
  --model_path ./models/sft-1.1b-math \
  --output_dir ./outputs/grpo-smoke \
  --max_samples 50 \
  --num_train_epochs 1
```

**Success criteria:**
- [ ] No OOM crash
- [ ] WandB logging active
- [ ] Reward values are non-zero (at least some format rewards firing)
- [ ] No conversation simulation (generations end at `<|im_end|>`)
- [ ] Step time is reasonable (< 60s/step)

### Stage B: Calibration Run (1–2 hours)

```bash
python src/dpo/train_grpo.py \
  --model_path ./models/sft-1.1b-math \
  --output_dir ./outputs/grpo-calibration \
  --max_samples 500 \
  --num_train_epochs 1
```

**Monitor on WandB:**
- [ ] `reward/correctness` — should slowly increase from ~0 
- [ ] `reward/format` — should rapidly climb to ~0.8+ within 50 steps
- [ ] `reward/repetition` — should stay near 0 (no collapse)
- [ ] `kl_divergence` — should stay < 5.0 (if spiking, increase β)
- [ ] `train/loss` — should decrease smoothly

**Decision point after Stage B:**

| Observation | Action |
|-------------|--------|
| Format reward stuck at 0 | The regex pattern is too strict — relax to allow flexible whitespace |
| Correctness reward stays exactly 0 | `math_verify` isn't parsing GSM8K answers — check the answer column format |
| KL divergence > 10 | Increase β to 0.04 |
| Repetition penalty firing > 30% of steps | The model is collapsing — reduce learning rate to 3e-6 |
| Everything looks healthy | Proceed to Stage C |

### Stage C: Full Training Run (4–8 hours)

```bash
python src/dpo/train_grpo.py \
  --model_path ./models/sft-1.1b-math \
  --output_dir ./outputs/grpo-full \
  --num_train_epochs 1 \
  --save_steps 100
```

**Checkpointing strategy:**
- Save every 100 steps
- This gives ~75 checkpoints over 7,473 samples / (batch_size=1 × grad_accum=8) ≈ 934 steps
- If training degrades, you can roll back to the best checkpoint

---

## Step 4: Mid-Training Evaluation Gates

Run lightweight evaluation at **step 200, 500, and final** to detect catastrophic forgetting early:

```bash
# Quick GSM8K eval (100 samples, not full 1319)
python src/eval/run_custom_eval.py \
  --model_path ./outputs/grpo-full/checkpoint-200 \
  --output_file ./eval_results/grpo_checkpoint_200.md
```

### Abort Criteria

| Metric | SFT Baseline | Abort If |
|--------|-------------|----------|
| ARC-Challenge | 24.66% | Drops below 20% |
| HellaSwag | 26.70% | Drops below 23% |
| MMLU | 24.60% | Drops below 22% |
| GSM8K (custom) | ~1% | Drops to 0% AND format compliance < 50% |

> If any abort criteria trigger, **roll back to the previous checkpoint** and either:
> 1. Increase β (tighter KL constraint)
> 2. Reduce learning rate
> 3. Reduce number of epochs / training steps

---

## Step 5: Final Evaluation & Model Release

After training completes:

```bash
# 1. Full benchmark suite (matches Phase 3 methodology)
python src/eval/run_benchmarks.py --model_path ./outputs/grpo-full/final

# 2. Custom 30-problem math eval
python src/eval/run_custom_eval.py \
  --model_path ./outputs/grpo-full/final \
  --output_file ./eval_results/grpo_final_custom.md

# 3. Side-by-side comparison
python src/eval/generate_comparison.py \
  --models base:./hf_1b_model sft:./models/sft-1.1b-math grpo:./outputs/grpo-full/final

# 4. Upload to HF Hub
huggingface-cli upload himanshunakrani9/TinyMathReason-1B-grpo ./outputs/grpo-full/final
```

---

## Step 6: Update Project Documentation

### STATUS.md updates:
```markdown
## Phase 4: Post-Training GRPO ✅
- [x] Hardened reward functions (AST correctness, regex format, repetition penalty)
- [x] Configured GRPO hyperparameters (G=8, β=0.01, lr=5e-6, cosine schedule)
- [x] Executed staged training (smoke → calibration → full run)
- [x] Post-GRPO Evaluation:
  * GSM8K: X.XX% (SFT: 1.00%) -- +X.XX% Gain
  * MATH Algebra: X.XX% (SFT: 0.00%) -- +X.XX% Gain
  * ARC-Challenge: XX.XX% (SFT: 24.66%)
  * HellaSwag: XX.XX% (SFT: 26.70%)
  * MMLU: XX.XX% (SFT: 24.60%)
- [x] Model uploaded to HF Hub: `himanshunakrani9/TinyMathReason-1B-grpo`
```

### README.md updates:
- Fill in the GRPO Score column in the benchmark table
- Update model_id in Quick Start to point to GRPO checkpoint
- Move Phase 4 from "IN PROGRESS" to "COMPLETE"

---

## Execution Checklist (Ordered)

```
□ 1. Verify MI300X access & SFT weights available
□ 2. Install dependencies (trl, math_verify, vllm/rocm, wandb)
□ 3. Test vLLM ROCm compatibility — decide vLLM vs native gen
□ 4. Apply all train_grpo.py fixes (reward, stopping, hyperparams)
□ 5. Commit & push hardened script
□ 6. SSH to MI300X, pull latest code
□ 7. Run Stage A smoke test (50 samples) — verify no crashes
□ 8. Run Stage B calibration (500 samples) — tune hyperparams
□ 9. Run Stage C full training (7,473 samples, ~1 epoch)
□ 10. Run mid-training eval gates at step 200, 500
□ 11. Run final benchmark suite
□ 12. Upload GRPO model to HF Hub
□ 13. Update STATUS.md and README.md with results
□ 14. Transition to Phase 5 (report, Gradio demo, release)
```

---

## Risk Mitigation Summary

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| vLLM ROCm incompatibility | **High** | Native HF generation fallback — 192GB handles G=8 easily |
| Mode collapse / infinite loops | Medium | Repetition penalty reward + explicit stop tokens |
| Catastrophic forgetting of MMLU/ARC | Medium | β=0.01 KL penalty + mid-training eval gates |
| `math_verify` can't parse GSM8K `####` format | Medium | Add GSM8K-specific extraction before AST parse |
| Zero correctness reward (all generations wrong) | **High** at start | G=8 increases exploration; temperature=0.9 ensures diversity; format reward provides learning signal even when correctness=0 |
| TRL API version mismatch (`GRPOTrainingArguments` vs `GRPOConfig`) | Medium | Check `trl` version; ≥0.17 uses `GRPOConfig` |

> [!NOTE]
> **Realistic expectations for a 1.1B model:** Even with perfect GRPO execution, don't expect GSM8K to jump from 1% to 50%. Research on similarly-sized models (Qwen2.5-1.5B) shows GRPO typically delivers 5–15% absolute gains on GSM8K for sub-2B models. A jump from 1% to 5–10% would be a significant, publishable result demonstrating that the RL pipeline works.
