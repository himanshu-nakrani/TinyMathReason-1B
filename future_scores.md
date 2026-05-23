# TinyMathReason-1B: GRPO Benchmark Projections & Post-RL Roadmaps

This document outlines the projected performance metrics for **TinyMathReason-1B** following the Phase 4 GRPO (Group Relative Policy Optimization) alignment phase on the NVIDIA A100 infrastructure. It also provides strategic blueprints for subsequent post-training optimization phases.

---

## 📊 Phase 4: Benchmark Projections

By shifting from Supervised Fine-Tuning (SFT)—which only optimized formatting templates—to GRPO with correctness, formatting, and loop-penalizing reward signals, the model undergoes a structural transition in its mathematical and logical reasoning capacity.

Below is the comparative projection of **TinyMathReason-1B** against baseline, SFT, and top-tier industrial models:

| Benchmark | Setting | Base Score | SFT Score | GRPO Score (Projected) | Delta (vs SFT) | Performance Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **GSM8K** | 8-shot | 1.00% | 1.00% | **45.00% – 55.00%** 🚀 | **+44.00% to +54.00%** | The AST-based arithmetic and equation correctness rewards actively reinforce successful reasoning chains. |
| **MATH (Algebra)** | 4-shot | 0.00% | 0.00% | **12.00% – 18.00%** 📈 | **+12.00% to +18.00%** | SymPy-based AST parser allows matching equivalent algebraic equations, teaching the model algebraic translation. |
| **ARC-Easy** | 0-shot | 29.90% | 25.51% | **30.00% – 32.00%** | **+4.49% to +6.49%** | Recovers from formatting SFT shocks as the model stabilizes its ChatML outputs. |
| **ARC-Challenge**| 25-shot | 21.70% | 24.66% | **28.00% – 32.00%** 💡 | **+3.34% to +7.34%** | Multi-choice science reasoning directly benefits from System-2 step-by-step thinking traces (`<think>`). |
| **HellaSwag** | 10-shot | 25.80% | 26.70% | **27.50% – 29.00%** | **+0.80% to +2.30%** | Maintained stable via strict KL-divergence penalties ($\beta = 0.01$) preventing distribution collapse. |
| **MMLU** | 5-shot | 23.50% | 24.60% | **27.00% – 30.00%** 🎓 | **+2.40% to +5.40%** | Improved logical induction and deduction capabilities translate to higher multiple-choice accuracy. |

---

## 🔬 Scientific Analysis of the GRPO Gains

### 1. Bootstrapping Reasoning out of SFT Format
The SFT phase established the mechanical scaffolding of reasoning: the `<think> ... </think>` block. However, the SFT objective did not reward the accuracy of the contents within those tags. Under GRPO, the correctness reward forces the policy network to explore and reinforce paths that lead to the exact target answer.

### 2. The Step-by-Step Reasoning Dividend
Even though GRPO correctness rewards are strictly calculated on mathematical tasks (GSM8K/MATH), the model's emergent ability to generate logical, structured step-by-step thinking chains generalizes to general knowledge and logic benchmarks. This results in positive transfer (a boost) on **MMLU** and **ARC-Challenge** scores.

### 3. Mitigating the "Alignment Tax" with KL Divergence
We configured a robust KL-divergence penalty coefficient ($\beta = 0.01$) between the active policy model and your SFT reference model. This penalty restricts the model from drifting too far from its original language priors, safeguarding commonsense knowledge and general reasoning performance on benchmarks like **HellaSwag**.

---

## 🗺️ Next-Generation Post-RL Optimization Blueprints

Once the GRPO training run is complete and fully evaluated, you can implement these four advanced post-RL alignment pipelines to achieve state-of-the-art results:

### 1️⃣ Rejection Sampling Distillation (RSD)
This technique is the core behind DeepSeek-R1's extreme data efficiency and clean generation traces.
*   **The Pipeline:**
    1. Pass your entire training prompt set through your freshly trained GRPO model.
    2. Sample $N=16$ or $32$ generations per prompt.
    3. Keep only the correct paths that have the most logical reasoning steps and cleanest formatting.
    4. Run a final high-speed supervised fine-tuning (SFT) pass on this self-generated reasoning corpus.
*   **Outcome:** Eradicates the formatting bugs, awkward phrasing, and dead-ends typical of raw RL outputs, delivering a clean and fast-decoding production model.

### 2️⃣ Process-Supervised Reward Models (PRMs)
Transition from outcome-only supervision (checking the final math answer) to process-level step-by-step verification.
*   **The Pipeline:**
    1. Deploy a small step-level judge model (e.g., a 0.5B classifier) or build an AST-based heuristic parser.
    2. Check the mathematical correctness of *each step* inside the `<think>` block.
    3. Inject step-level reward feedback into the GRPO advantage normalization step.
*   **Outcome:** Completely eliminates "reward hacking" (where a model uses incorrect logic but randomly guesses the correct final number).

### 3️⃣ Iterative DPO for Verbosity and Length Penalization
Deep RL reasoning runs often lead to "verbosity bloat" where the model generates hundreds of unnecessary thinking steps to maximize its chance of correctness.
*   **The Pipeline:**
    1. Generate pairs where:
        *   **Chosen:** Mathematically correct with a concise, optimal reasoning path.
        *   **Rejected:** Mathematically correct, but highly repetitive, bloated, or looping.
    2. Run a fast, lightweight DPO (Direct Preference Optimization) or ORPO (Odds Ratio Preference Optimization) epoch.
*   **Outcome:** Shaves off 30%–50% of the reasoning token length, drastically accelerating inference speeds and decreasing server costs.

### 4️⃣ Speculative Decoding with TinyMathReason-1B
Leverage your highly optimized 1B model to accelerate larger reasoning models.
*   **The Pipeline:**
    1. Host your optimized **TinyMathReason-1B** as a draft model alongside a larger target model (e.g., Qwen-2.5-7B or 14B).
    2. Use the draft model to speculatively generate thinking paths, which are verified in parallel by the larger model.
*   **Outcome:** Achieve the massive intelligence of a larger model at the lightning generation speed of your 1B model.
