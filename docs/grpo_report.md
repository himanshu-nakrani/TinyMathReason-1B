# Phase 4: GRPO Training Report - TinyMathReason-1B

## Executive Summary
GRPO (Group Relative Policy Optimization) was applied to the SFT model to improve mathematical reasoning and adherence to a step-by-step thinking format. While the model shows a measurable improvement in benchmark scores (GSM8K and Minerva Math), it suffers from significant mode collapse and hallucination in open-ended reasoning.

## Training Configuration
- **Dataset:** MathInstruct + MetaMathQA (7,473 samples per epoch, 3 epochs total)
- **Batch Size:** 1 (Global: 80 via Group Size G=8)
- **Group Size (G):** 8
- **Learning Rate:** 5e-6 (Cosine decay)
- **KL Penalty (β):** 0.01
- **Max Length:** 512 tokens
- **Hardware:** 1x NVIDIA L4 (24GB VRAM)
- **Total Steps:** 22,419

## Training Dynamics
- **Final Reward (Mean):** 0.5139
- **Correctness Reward:** 0.01389 (Approx. 1.4% correctness rate in group samples)
- **Format Reward:** 0.50 (Highly stable, indicating consistent use of requested formatting)
- **KL Divergence:** 2.975 (Controlled drift from the SFT policy)
- **Entropy:** 4.679 (Maintenance of token diversity)

## Benchmark Results

| Benchmark | Metric | Base | SFT | **GRPO (Final)** |
| :--- | :--- | :---: | :---: | :---: |
| **GSM8K** | Exact Match | 1.0% | 1.0% | **2.2% (Flex)** |
| **Minerva Math** | Math Verify | 0.0% | 0.0% | **2.0%** |
| **ARC-Easy** | Acc Norm | 29.9% | 25.5% | **28.8%** |
| **ARC-Challenge** | Acc Norm | 21.7% | 24.7% | **22.8%** |
| **HellaSwag** | Acc Norm | 25.8% | 26.7% | **26.3%** |
| **MMLU** | Accuracy | 23.5% | 24.6% | **23.6%** |

## Qualitative Analysis
### Strengths
- **Format Adherence:** The model successfully learned to structure its output with a reasoning trace, although the quality of the trace is low.
- **Score Gain:** Measurable doubling of GSM8K performance from a very low baseline.

### Weaknesses
- **Incoherence:** The reasoning traces often contain repetitive phrases or nonsensical hallucinations (e.g., discussing "poverty" when asked math questions).
- **Mode Collapse:** The model frequently falls into "loops" of generated text.
- **Low Signal:** With only ~1.4% of samples receiving a correctness reward, the RL signal was likely too weak to drive deep reasoning improvements.

## Conclusion & Recommendations
GRPO successfully demonstrated a "formatting boost" and a small reasoning gain. However, for a 1B model, a much stronger SFT baseline and possibly a larger group size or higher-quality reward model (rather than rule-based regex) would be required to achieve significant reasoning capabilities.

---
**Model Path:** `outputs/grpo-full/final`
**Logs:** `grpo_training.log`
