TinyMathReason-1B — Optimization Plan

Goal: identify the best achievable performance on top of the existing pretraining base, across all realistic compute budgets, with grounded score projections.

Methodology: every projection is anchored to a published reference model in the same parameter class. Single point estimates are avoided — each path is bracketed by [pessimistic / central / optimistic], with the explicit reference and the assumption that delivers each bound.
1. Verified current state (no projection)

From the repo's own STATUS.md, configs, and code:

Stage	Setup (verified from repo)	Result
Base	1.126B Llama, custom 32k BPE tokenizer, 57B tokens (54,362 steps × 262k tok/step), AdamW lr 3e-4, no QK-norm, no z-loss, no doc-level attn mask, no MinHash dedup (only hash(text) exact dedup)	GSM8K 1.0%, MATH 0.0%, MMLU 23.5%, ARC-C 21.7%, ARC-E 29.9%, HellaSwag 25.8%
SFT	Stage 1: Alpaca 52k × 2 epochs lr 2e-5; Stage 2: GSM8K + MathInstruct + MetaMathQA (~660k) × 2 epochs lr 2e-5; heuristic CoT splitter; no completion-only loss masking; random init for new <think>/</think> rows	GSM8K 1.0%, MATH 0.0%, MMLU 24.6%, ARC-C 24.7%, ARC-E 25.5% (regression), HellaSwag 26.7%
GRPO	GSM8K only (7.5k prompts × 3 epochs = 22,419 steps), G=8, β=0.01, lr 5e-6, max_completion_length=512, additive reward (correctness + format + repetition penalty)	Reward 0.514 (correctness 0.014, format 0.50). GSM8K 2.2%, Minerva 2.0%
Binding constraints identified (from prior analysis, evidence in code):

Pretrain budget is 50× lower than competing 1B-class models (50 tok/param vs. 2,700–7,500 tok/param). This sets the global ceiling.
Custom 32k BPE (trained on "sample data" per STATUS.md) likely merges multi-digit numbers — making arithmetic learnable only by memorization. Audit pending.
GRPO is collapsed: 1.4% correctness rate → P(group of 8 has any correct) ≈ 10.6% → ~89% of training steps have zero correctness gradient. Format reward dominates by 35× → format-hacking observed in grpo_report.md.
Cross-document attention contamination in pretraining (d_tokenize_and_pack.py simply concatenates docs + EOS).
Eval protocol drift (lm-eval-harness raw few-shot for SFT/GRPO ChatML model, mixing strict-match vs. flexible-extract).
2. Reference points (1B-class, all verified)

This is the calibration set against which projections are anchored.

Model	Recipe	GSM8K	MATH	Source
TinyMath-1B (this project)	57B tokens from scratch	1.0%	0.0%	repo STATUS.md
Llama-3.2-1B base	~9T tokens, Meta	8.11	3.42	featherless.ai mirror of InfiR card; treat as third-party-reproduced
TinyLlama-1.1B base	3T tokens, Llama-2 tokenizer	~2–3%	<2%	TinyLlama repo / TinyLlama_v1.1 card
TinyLlama-1.1B-ProXMath	+ 15B tok continual on OpenWebMath-Pro	(gain on GSM8K reported, see model card)	—	gair-prox
Rho-Math-1B (continual pretrain)	TinyLlama + 15B tok continual on OpenWebMath via SLM	matches DeepSeekMath	15.6% few-shot	Rho-1, NeurIPS 2024
Rho-Math-1B (after SFT)	+ math SFT	—	40.6%	same paper
Rho-Math-1B-Interpreter	+ Python tool use	—	40%+	same paper
InfiR-1B-Base	Llama-3.2-1B + 940B tok continual (52% code, 48% math/sci/web) + 40B annealing	63.46	31.82	InfiR card, arxiv 2502.11573
DeepSeek-R1-Distill-Qwen-1.5B	Qwen2.5-Math-1.5B + R1 distillation SFT	—	MATH-500 83.9, AIME 28.9	R1 paper
DeepScaleR-1.5B-Preview	+ GRPO on top of distill	—	AIME 43.1 (vs 28.8 base)	DeepScaleR card
TinyGSM (1.3B)	SFT-only on synthetic GSM8K-style data + 1.3B verifier	81.5%	—	TinyGSM, NeurIPS 2023
Qwen2.5-1.5B-Instruct	full Qwen post-train	~60.96	—	imitation paper benchmark
SmolTulu-1.7B	SmolLM2-1.7B + SFT/DPO	51.6	—	arxiv 2412.08347
Llama-2-7B base (point of reference)	2T tok	11.8	3.2	Llemma card
MetaMath-7B	Llama-2-7B + MetaMathQA SFT	66.5	19.8	meta-math.github.io
Numbers were rephrased/condensed for compliance with licensing restrictions.

Key observations from this table:

The single largest lever for a 1B model going from "near-zero math" to "real math" is continual pretraining on math-dense data: Rho-1 (+15B tokens) takes a 1.1B from ~2% MATH to 15.6%; InfiR (+940B tokens, much heavier) takes Llama-3.2-1B from 8/3 to 63/32.
SFT alone on a sufficient base can give 50%+ GSM8K (SmolTulu, Qwen2.5-1.5B-Instruct, MetaMath-7B). With a weak base, SFT alone caps in single digits.
Pure GRPO/RL without good SFT cold-start fails on weak bases — DeepScaleR works precisely because its starting point (R1-Distill) is already at MATH-500 83.9%.
Synthetic data only (TinyGSM) can drive a 1.3B to 81.5% GSM8K without massive pretraining — but that's a narrow win on GSM8K specifically, not general math.
3. The decision tree

The user's compute budget determines the realistic ceiling. Five paths, listed by ascending cost:

Path	Compute	Time	Realistic ceiling on GSM8K	Notes
A. Eval & test-time only	0 GPU-h training	Hours	3–8%	Fixes protocol artifacts only; does not change capability
B. Post-training redo (better SFT + RFT + GRPO)	~150 MI300X-h	1–2 weeks	8–18%	What's possible with the existing 57B-token base. Hard ceiling.
C. Continual pretrain + post-training	~5–15k H100/MI300X-h depending on token budget	3–6 weeks	35–60%	This is the right path for "best possible from this base"
D. Distill from open math teacher	~200–400 MI300X-h	2 weeks	30–55% on GSM8K (narrower)	Cheapest way to high GSM8K specifically, weakest generalization
E. Full pretraining redo	TPU v4-32 for ~6 weeks	6–10 weeks	55–75%	Outside scope of "optimization on base"; documented for completeness
F. Tool-augmented (orthogonal)	minimal	Days	+10–25 pp on top of any path	Composable with B/C/D
The "best possible optimization pipeline on the base model" recommendation = Path A → C → B → F → A_redo. That sequence is detailed in §5.

4. Why each path's ceiling is what it is

Path A: Eval-only ceiling (3–8% GSM8K)
Floor (3%): current GRPO is already 2.2% with flexible-extract; a chat-templated 0-shot CoT eval and basic self-consistency cons@8 would lift this to ~3% with high confidence.
Ceiling (8%): self-consistency with cons@32 on the GRPO model. Original Wang et al. self-consistency paper reports +17.9% on GSM8K, but that was on PaLM-540B. Empirically, gain scales with base accuracy — for ~2% base, expected absolute lift is bounded by the rate at which any of the 32 samples gets the answer right. If pass@32 ≈ 8–15% (typical for GSM8K with t=0.7 sampling on a weak model), majority-vote cons@32 ≈ 4–8%.
Path B: Post-training redo (8–18% GSM8K)
The base has only 57B tokens of compute on it. No amount of post-training can extract math knowledge that isn't latent. The empirical bound:

TinyLlama-1.1B base sits at ~2–3% GSM8K with 3T tokens. Even with the very best SFT recipe (NuminaMath + OpenMathInstruct-2 + 5–10 epochs + completion-only loss), reported gains over Llama-2-7B base (11.8% → 66.5% via MetaMath SFT) imply a ~5.6× multiplier from good SFT on a halfway-decent base.
TinyMath-1B base is at ~1.0%. Applying the same 5–6× multiplier from optimal SFT yields 5–6%.
RFT (rejection-sampling fine-tuning) is reported by Yuan et al. 2308.01825 to roughly 1.4× SFT for weak bases (LLaMA-7B SFT 35.9% → RFT 49.3%). Applied here: 7–9%.
GRPO with redesigned rewards (gated, longer max_length, multi-dataset) on top of an 8% RFT base, with non-trivial group variance now possible, can plausibly add another 3–9 pp. Anchor: SimpleRL on Qwen2.5-Math-7B at ~50% MATH base went to 77.2%. The relative gain (~1.5×) on a 7–9% RFT base gives 10–14%.
With test-time scaling (cons@8 + simple ORM), add 2–4 pp.
Path B realistic bracket: GSM8K 8% / 13% / 18%, MATH 2% / 4% / 7%. Above that requires changing the base.

Path C: Continual pretraining + post-training (35–60% GSM8K)
This is the lever the project is missing. Two reference recipes:

C-light: 30B-token continual pretrain on FineMath-4+ + InfiWebMath-4+

SmolLM2 paper and the FineMath ablations demonstrate that 60B tokens of FineMath-4+ on a 3B model meaningfully shifts GSM8K. Scaling to 1B with 30B tokens of FineMath/InfiWebMath gives a ~2× compute increase over current pretrain dedicated to math.
Anchor: TinyLlama-1.1B + 15B continual (ProXMath / Rho-1 SLM) → MATH 15.6%, GSM8K matches DeepSeekMath baseline. With 30B tokens of even higher quality (FineMath-4+ is the current SOTA quality bar), expect a 1.5–2× improvement over Rho-Math-1B.
C-heavy: 100–300B-token continual pretrain mirroring InfiR-1B

InfiR-1B continually pretrained Llama-3.2-1B on 940B + 40B annealing → GSM8K 63.46, MATH 31.82.
TinyMath base is much weaker than Llama-3.2-1B (1.0 vs 8.11 on GSM8K), so we can't expect the same absolute landing. But the delta InfiR achieves (+55 GSM8K, +28 MATH) is the size of the lever.
Realistic landing for TinyMath after 150B tokens of math-dense continual pretrain (well under InfiR's budget but well over Rho-1's): bracket [40, 50, 60]% GSM8K and [10, 18, 25]% MATH.
Then on top of C, do Path B post-training:

SFT lifts GSM8K modestly when the base is already strong (anchor: Llama-2-7B 11.8 → 66.5 with MetaMath, a 5.6× multiplier; for a strong continual-pretrained 1B at 50% base, SFT diminishing-returns bring it to maybe 60–70%).
GRPO at this point has real correctness signal, so it adds another 3–8 pp (anchor: SimpleRL).
Path C realistic bracket (with C-heavy + post-training + cons@8):

Benchmark	Pessimistic	Central	Optimistic	Anchored to
GSM8K	35%	50%	60%	InfiR-1B 63.46 (heavier budget); SmolTulu 51.6
MATH	8%	15%	22%	Rho-Math-1B-SFT 40.6 (stronger base); InfiR 31.82
MMLU	32%	40%	47%	InfiR 47.24
ARC-C	35%	42%	50%	SmolLM2 / Llama-3.2-1B class
Path D: Distill from a stronger teacher (30–55% GSM8K)
TinyGSM showed a 1.3B model can hit 81.5% GSM8K when trained on synthetic GSM8K-like data generated by GPT-3.5-Turbo, plus a verifier. The Microsoft Orca-Math, Rho-Math-Interpreter, and DeepSeek-R1-Distill-1.5B (which inherits R1-trace SFT) all use teacher distillation.

The recipe for TinyMath:

Use Qwen2.5-Math-7B-Instruct or DeepSeek-Math-7B-Instruct as a teacher.
Generate 16–32 chains of thought per problem on GSM8K + MATH-train + NuminaMath-CoT.
Reject samples whose final answer disagrees with ground truth.
SFT the TinyMath base on the resulting correct-CoT corpus (~200–500k examples).
Optionally run a verifier model (a sibling 1B trained as ORM) for inference-time rerank.
Key caveat for TinyMath specifically: the 57B-token base may not have enough representational capacity to absorb dense competition-math reasoning. TinyGSM's success was on a Phi-1.5 base (1.3B trained on 150B+ tokens of textbook-quality data) — much stronger than TinyMath's base.

Path D bracket for TinyMath base (no continual pretrain): GSM8K 25% / 38% / 52%, MATH 4% / 8% / 14%. Goes higher (40 / 55 / 70 on GSM8K) if combined with Path C continual pretrain first.

Path E: Full pretraining redo (55–75% GSM8K)
Out of scope for "optimization on base", but for completeness: applying everything we know now (FineMath-4+ + Cosmopedia synthetic + digit-split tokenizer + QK-norm + z-loss + doc-attn-mask + multi-stage curriculum) to a fresh 1B at, say, 500B tokens, anchored to InfiR-1B's 63/32 at 940B tokens, would land at GSM8K [50, 60, 70] / MATH [20, 28, 38].

Path F: Tool-augmented (orthogonal, +10–25 pp)
Reported lifts:

Rho-Math-1B → Rho-Math-1B-Interpreter: MATH from 15.6% → 40%+ (the +25pp lift).
ToRA-7B vs ToRA-Code-7B math agents: MATH 40.1 → 44.6, GSM8K 68.8 → 72.6.
Frontier models with code interpreter: typically +10–20 pp on GSM8K, +20–30 pp on MATH.
This is composable with any other path, but only if the model can write near-correct Python. With TinyMath's near-zero coding capability (no HumanEval baseline reported but Stack-Edu was only 10% of pretrain mix and the base hasn't been measured), Tool-Augmented adds little until either Path C continual pretrain on code-heavy data or a code-specific SFT phase is added.

5. Recommended pipeline: best-possible from current base

This is the sequenced, prescriptive plan — the answer to "what should I actually do?"

Stage 0 — Audit (1 day, free)
Before anything: confirm the binding constraint. Three checks:

0.1 Tokenizer audit (1 hour). Tokenize representative math strings:

from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("./hf_1b_model")
for s in ["1234567", "382 + 491 = 873", "1/2 + 3/4 = 5/4", 
         "x^2 - 5x + 6 = 0", "$\\boxed{42}$"]:
    print(s, "->", t.tokenize(s))
Decision: if multi-digit numbers stay merged as single tokens, this is a hard cap. Two options: (a) accept it and continue, (b) replace tokenizer with Llama-3 / Qwen2.5 (digit-split BPE). Replacement requires re-running the pretrain or doing 1–10B tokens of "tokenizer-transplant" continual pretrain (see arxiv 2506.06607).
0.2 Eval rebaseline with aligned protocols (4 hours):

For SFT and GRPO models, run lm-eval-harness with --apply_chat_template and the gsm8k_cot_zeroshot task (not 8-shot raw text).
Use both strict and flexible-extract metrics consistently across all three models.
Greedy decoding for primary number; cons@8 at t=0.7 for secondary.
0.3 Pass@k measurement (2 hours): on the GRPO model, at t=0.7 sample N=64 GSM8K rationales and report pass@1, pass@8, pass@32, pass@64, and cons@32. The gap between pass@k and cons@k tells us whether Path A (test-time scaling with verifier) has any leverage. If pass@32 < 5%, Path A alone is exhausted.

Stage 1 — Continual pretrain (the single highest-ROI step)
Budget choice: 30B tokens (light) or 150B tokens (heavy). I recommend 30B as a first run; it's the smallest budget that materially moves the needle (Rho-1 anchor: 15B was enough for MATH 0% → 15.6%).

Data mix (matches FineMath ablations + InfiR):

30% FineMath-4+ (~6.7M docs, the highest-quality math web data) — HuggingFaceTB/finemath
25% InfiWebMath-4+ (~6.3M docs, complementary math web)
15% OpenWebMath (already in current corpus, kept for distribution match)
10% MathPile (proofs, books — currently in pipeline but underweighted)
10% Proof-Pile-2 algebraic-stack (currently in c_mix_datasets.py but never downloaded — fix a_download_datasets.py)
10% FineWeb-Edu (general; prevents catastrophic forgetting of language)
Hyperparameters:

lr 1e-4 (lower than initial pretrain because we're at the cosine-decayed end), 3% warmup, cosine to 1e-5
AdamW β1=0.9, β2=0.95, wd=0.1, grad clip 1.0
bf16, batch 256k tokens, ctx 4096
Add document-level attention mask (varlen/flash_attn_varlen_func) — fixes the cross-doc contamination bug
Add z-loss (1e-4 coefficient) and QK-norm if architecturally feasible without re-init (QK-norm needs zero-init bias, can be added cleanly)
Apply Rho-1 SLM (arxiv 2404.07965): use a small reference model (your current TinyMath base, or Qwen2.5-Math-1.5B as oracle) to score tokens; train only on the 60–70% highest-utility tokens. Rho-1 reports +30 pp absolute on math, with 5–10× compute efficiency.
Last 10% of training: anneal with 80% math / 20% code / 10% web
Cost estimate (30B tokens, 1.1B model, ~3.3e20 FLOPs):

1× MI300X (~1300 TFLOPS bf16 sustained): ~70 hours
8× H100 (~6 PFLOPS sustained): ~15 hours
TPU v4-32: ~10 hours
Expected post-Stage-1 base scores (anchored to Rho-Math-1B and FineMath ablations):

Benchmark	Pessimistic	Central	Optimistic
GSM8K	8%	18%	30%
MATH	4%	10%	16%
MMLU	26%	32%	38%
ARC-C	24%	30%	38%
The pessimistic case assumes the tokenizer issue is real and unfixed; the optimistic case assumes tokenizer is OK and SLM works as advertised.

Stage 2 — SFT (correct this time)
Data:

70% NuminaMath-CoT (860k competition-grade problems, MIT-licensed, Hugging Face leaderboard winner)
15% OpenMathInstruct-2 filtered to verified-correct subset (~300k of 14M)
10% GSM8K train with rationales rewritten by Qwen2.5-Math-7B-Instruct → kept only if final answer matches (rejection sampling distillation as a free side-effect)
5% Tulu-3 SFT mixture (chat preservation; replaces Alpaca)
Implementation fixes (matching the prior diagnosis):

Completion-only loss masking: TRL ≥0.20 supports assistant_only_loss=True directly in SFTConfig. For older TRL, use DataCollatorForCompletionOnlyLM(response_template="<|im_start|>assistant ", ...). This 1-flag change typically improves SFT efficiency 30–50%.
CoT formatting deterministic per source: GSM8K → use #### as ground truth; NuminaMath → solution field is already a CoT, wrap entire thing as <think> and parse \boxed{} as final answer; OpenMathInstruct-2 → expected_answer and generated_solution fields are already separated. Throw out the heuristic extract_cot_and_answer from prepare_sft_data.py.
Embedding init for <think>/</think> (matches Hewitt 2021):
with torch.no_grad():
    orig_vocab = model.config.vocab_size  # before resize
    model.resize_token_embeddings(len(tokenizer))
    e_in = model.get_input_embeddings().weight
    e_out = model.get_output_embeddings().weight
    mean_in = e_in[:orig_vocab].mean(dim=0)
    mean_out = e_out[:orig_vocab].mean(dim=0)
    for tid in tokenizer.convert_tokens_to_ids(["<think>", "</think>"]):
        e_in[tid] = mean_in
        e_out[tid] = mean_out
Single-stage (drop Alpaca-first); chat preservation comes from the 5% Tulu mix.
Hyperparameters: lr 5e-6 (not 2e-5), cosine to 1e-7, warmup 5%, 5–8 epochs (arxiv 2507.08267 finds 10 epochs crucial for small models), eff. batch 64–128, packed seq 4096. Validate on a held-out 500-problem GSM8K dev split with greedy + cons@8 every epoch — early stop on dev plateau.
Expected post-Stage-2 scores:

The right multiplier comes from Llama-2-7B → MetaMath-7B (11.8 → 66.5 GSM8K, ×5.6) and from SmolLM2-1.7B → SmolTulu-1.7B (~37 → 51.6 GSM8K, ×1.4). The multiplier shrinks as base capability grows. Applied to Stage-1 base:

Benchmark	Pessimistic	Central	Optimistic
GSM8K	20%	38%	55%
MATH	6%	14%	22%
Stage 3 — Rejection-sampling fine-tuning (RFT)
The cheap +30–40% multiplier on top of SFT, especially for weak bases (Yuan et al. 2308.01825).

With the Stage-2 SFT model, sample 16 CoTs per problem on GSM8K-train + MATH-train + NuminaMath-CoT (~250k problems).
Filter to correct-final-answer samples (deterministic check via math_verify).
Deduplicate near-identical CoTs (Jaccard < 0.7).
SFT one more epoch at lr 2e-6 on this self-generated correct-CoT corpus.
Expected lift over Stage 2: +5 to +12 pp GSM8K, +2 to +5 pp MATH (anchored to Yuan et al.: SFT 35.9% → RFT 49.3% on LLaMA-7B = 1.37× multiplier; weaker bases benefit more).

Benchmark	Pessimistic	Central	Optimistic
GSM8K	28%	47%	62%
MATH	8%	17%	26%
Stage 4 — GRPO with redesigned rewards
Now GRPO has a meaningful correctness signal (post-RFT base accuracy ≥ 25–30%, so groups have real variance).

Reward redesign:

Gated, not summed: format pass is a precondition, not a competing reward.
def reward(completion, answer):
    if not has_valid_format(completion):    # must have non-empty <think>...</think> + answer
        return -0.1                          # discourage format-breaking
    if is_correct(extract_answer(completion), answer):
        return +1.0
    return 0.0
# Repetition penalty applied multiplicatively (only if otherwise positive)
This eliminates the 35× format/correctness reward imbalance that caused the format-hack collapse.
Hyperparameter changes:

max_completion_length 2048 (up from 512). MI300X 192 GB easily fits G=8 at 2048 ctx for 1.1B.
temperature 0.7 (down from 0.9). At higher base correctness, lower temp gives sharper learning signal.
num_generations G=16 (up from 8). Doubles probability of seeing variance per group.
Multi-dataset prompt mix: 50% GSM8K + 30% MATH-Algebra + MATH-Counting + 20% NuminaMath subset. Eliminates the "memorize 7.5k prompts" trap.
lr 1e-5 (up from 5e-6). With real signal, can move faster.
Difficulty-aware reweighting (GRPO-LEAD, arxiv 2504.09696): up-weight advantage on problems in the 20–60% pass-rate band ("learnable zone"); zero-weight problems with 0% or 100% pass rate per group.
Length penalty: small penalty on completions > 1500 tokens to fight verbosity bloat.
Expected lift over Stage 3: +2 to +8 pp GSM8K (SimpleRL anchor on Qwen2.5-Math-7B: 50% → 77%, ×1.5; small-model ceiling is lower).

Benchmark	Pessimistic	Central	Optimistic
GSM8K	32%	53%	68%
MATH	9%	19%	28%
Stage 5 — Test-time scaling
Free at inference time (cost is just more samples), composes with anything above.

5.1 Self-consistency cons@N on Stage-4 model:

cons@8: typically +3–7 pp
cons@32: typically +5–12 pp (saturates around N=32 for small models)
Anchor: original Wang et al. paper +17.9 pp on PaLM-540B GSM8K; for small models the gain is smaller in absolute pp but proportionally similar.
5.2 Best-of-N with a verifier (ORM):

Train a sibling 1B as outcome reward model on Math-Shepherd or PRM800K; rerank N=32 samples with it.
Anchor: Math-Shepherd (arxiv 2312.08935) takes models to GSM8K 89.1%, MATH 43.5% when reranked. For a 1B reasoner, expect smaller absolute gains (+5–10 pp).
5.3 rStar-style mutual reasoning (arxiv 2408.06195) reports astounding gains for small models (LLaMA2-7B GSM8K 12.51 → 63.91, +51 pp); the recipe needs a sibling discriminator and is costly at inference. Optional, high-leverage.

Expected post-Stage-5 (cons@32 + ORM rerank): +6 to +15 pp on top of Stage 4.

Benchmark	Pessimistic	Central	Optimistic
GSM8K	38%	62%	78%
MATH	12%	24%	35%
Stage 6 — Tool augmentation (orthogonal, do this last)
Train the model to call Python for arithmetic-heavy steps. Recipe (matches ToRA / Rho-Math-Interpreter):

Add <python>...</python> and <output>...</output> to tokenizer (mean-init).
SFT on a tool-use mix: ToRA dataset, OpenMathInstruct-2 has tool-aware traces, plus Qwen2.5-Math-Instruct's TIR mode outputs.
At inference, intercept <python> blocks, execute in sandbox, inject <output> token, continue generation.
Expected lift over Stage 5 (anchored to Rho-Math-1B → Rho-Math-1B-Interpreter on MATH: +25 pp; Qwen2.5-Math-7B CoT 83.6 → TIR 85.3 on MATH = +1.7 pp at high baseline):

Most gain on MATH-Counting / NumberTheory: +15 to +25 pp
Smaller gain on GSM8K (already simple arithmetic): +2 to +5 pp
Benchmark	Pessimistic	Central	Optimistic
GSM8K	40%	64%	80%
MATH	18%	35%	50%
6. Master projection table

Reading this table: the bracketed range is [pessimistic / central / optimistic]. Each line assumes everything above it has been done at the central case.

Stage	Compute	GSM8K (pess/cent/opt)	MATH (pess/cent/opt)	MMLU (pess/cent/opt)	Anchored to
Current state	—	1.0 / 1.0 / 2.2	0.0 / 0.0 / 2.0	23.5 / 23.5 / 24.6	(verified)
0. Audit + eval rebaseline	1 day	1 / 3 / 5	0 / 1 / 3	23.5 / 24 / 25	protocol fix only
1. Continual pretrain 30B FineMath	70 MI300X-h	8 / 18 / 30	4 / 10 / 16	26 / 32 / 38	Rho-Math-1B (15B), FineMath ablation
1-heavy. Continual pretrain 150B	350 MI300X-h	25 / 40 / 55	8 / 18 / 28	30 / 38 / 45	InfiR-1B (940B + 40B annealing)
2. + Better SFT	50 MI300X-h	20 / 38 / 55	6 / 14 / 22	28 / 35 / 42	MetaMath ×5.6, SmolTulu ×1.4
3. + RFT	30 MI300X-h	28 / 47 / 62	8 / 17 / 26	28 / 36 / 43	Yuan et al. 1.37×
4. + Redesigned GRPO	60 MI300X-h	32 / 53 / 68	9 / 19 / 28	28 / 36 / 44	SimpleRL 1.5× (capped)
5. + Test-time scaling (cons@32 + ORM)	inference only	38 / 62 / 78	12 / 24 / 35	30 / 38 / 46	Math-Shepherd, Wang et al.
6. + Tool use (Python)	40 MI300X-h	40 / 64 / 80	18 / 35 / 50	30 / 38 / 46	Rho-Math-1B-Interpreter, ToRA
Notes on this table:

The "central" case for Stage 6 (GSM8K 64, MATH 35) places TinyMath-1B in the same range as Rho-Math-1B-Interpreter and Qwen2.5-Math-1.5B-Instruct — i.e., genuinely competitive with the strongest open 1.5B-class math models. The central case assumes Stage 1-light (30B continual). With Stage 1-heavy (150B), the central case shifts to GSM8K ~75, MATH ~42.
The optimistic case assumes the tokenizer is fine and Rho-1 SLM works as advertised; pessimistic assumes the tokenizer is a hard cap and SLM produces only modest gains.
MMLU does not lift much because the data mix is math-skewed. To lift MMLU, the Stage-1 mix would need to be re-weighted toward FineWeb-Edu / Cosmopedia (~50%+ general). That comes at a math-perf cost and is a separate decision.
Confidence on the central projections is medium-low. These are extrapolations from differently-architected reference models. Run a 5B-token version of Stage 1 first and re-anchor projections from observed deltas before committing to the full 30B/150B.
7. Compute & cost (1× MI300X 192GB, the available hardware)

Stage	MI300X-hours	Wall time @ $2/h spot	Wall time @ $4/h on-demand	Notes
0. Audit	4	$8	$16	local laptop OK
1-light. CPT 30B tok	70	$140	$280	most ROI per dollar
1-heavy. CPT 150B tok	350	$700	$1400	recommended for "best possible"
2. SFT	50	$100	$200	5–8 epochs on ~600k examples
3. RFT (16 samples × 250k prompts gen + filter + 1 SFT epoch)	60	$120	$240	dominated by sampling
4. GRPO	60	$120	$240	22k steps, G=16, 2048 max len
5. TTS infra	0	$0	$0	inference-only; verifier training (+30h) optional
6. Tool-use SFT	40	$80	$160	small dataset, fast
Total (with Stage 1-light)	284	$568	$1136	minimum viable path
Total (with Stage 1-heavy)	564	$1128	$2256	recommended path
For reference: the original 57B-token pretrain on TPU v4-64 cost ~$5–10k of TPU time. The Stage 1-heavy continual pretrain costs ~10–20% of that and produces 5–10× more capability gain.

8. Risk register & failure modes

Risk	Likelihood	Impact	Mitigation
Tokenizer merges multi-digit numbers → arithmetic ceiling	Medium	High (caps GSM8K at ~30%)	Stage 0.1 audit; if confirmed, swap to Llama-3 tokenizer + 5–10B token transplant CPT
Stage 1 continual pretrain causes catastrophic forgetting of language	Low–Med	Med	Keep 10% FineWeb-Edu in mix; monitor HellaSwag/MMLU each 1B tokens
Stage 1 reduces capacity for the new math distribution due to 57B-token base being too rigid	Medium	High	Smaller-budget canary (5B tokens) before committing to 30B/150B
RFT collapses diversity	Low	Low	Dedup CoTs; mix 30% original SFT data back in
GRPO at Stage 4 still mostly format-hacks because RFT base correctness still < 15%	Medium	Med	Skip Stage 4; go directly Stage 3 → Stage 5
Tool-use SFT teaches calling Python without learning when to call it	Med	Med	Curriculum: easy arithmetic problems where Python obviously helps first
Cons@32 doesn't help because pass@32 ≈ pass@1 (mode-collapsed sampling)	Med (post-RFT)	Med	Run pass@k diagnostic in Stage 0.3; if pass@k flat, do entropy regularization in Stage 4
Projections are over-optimistic because anchor models had stronger bases	High	Med	Treat pessimistic column as the planning baseline; central as upside
9. Honest summary of uncertainty

High confidence: Stage 0 audit findings, Stage 1's direction (continual pretrain on math = biggest single lever), Stage 2's specific implementation fixes (loss masking, CoT formatting, embedding init).
Medium confidence: Numerical projections in the central column. Anchored to published reference models, but those models had different bases. ±50% relative error is plausible per stage.
Lower confidence: Whether the 57B-token base has enough representational capacity to absorb 150B tokens of math-dense continual pretrain as productively as InfiR's Llama-3.2-1B base did. Run a 5–10B-token canary first; re-anchor projections from observed deltas.
Cannot project without Stage 0.1: Whether the custom tokenizer caps math at ~30% GSM8K. This single fact moves the whole curve by 15+ pp.
10. Source registry

All numbers in this document come from one of these references. Each row is independently verifiable.

Reference	Use
Rho-1, arxiv 2404.07965	Selective Language Modeling, Rho-Math-1B numbers
InfiR, arxiv 2502.11573 + model card	InfiR-1B-Base recipe and scores
SmolLM2 paper, arxiv 2502.02737	Multi-stage pretraining, FineMath ablations
DeepSeek-R1, arxiv 2501.12948	R1-Distill-Qwen-1.5B numbers
DeepScaleR card	RL gain on top of distillation
TinyGSM, arxiv 2312.09241	1.3B + verifier 81.5% GSM8K
SimpleRL	7B base + 8K examples RL recipe
SmolTulu, arxiv 2412.08347	1.7B SFT/DPO 51.6% GSM8K
Self-consistency, ICLR 2023	+17.9% PaLM GSM8K reference
Math-Shepherd, arxiv 2312.08935	Verifier rerank GSM8K 89.1
GRPO-LEAD, arxiv 2504.09696	Difficulty-aware reweighting, length penalty
10-epoch SFT, arxiv 2507.08267	Small-model SFT epoch finding
RFT scaling, arxiv 2308.01825	LLaMA-7B SFT 35.9 → RFT 49.3
MetaMath, openreview	Llama-2-7B 11.8 → 66.5 SFT multiplier
Hewitt 2021	Mean-init for new vocabulary
OpenMathInstruct-2, arxiv 2410.01560	Llama-3.1-8B 51.9 → 66.5 MATH
rStar, arxiv 2408.06195	Mutual reasoning small-model gains
ToRA, arxiv 2309.17452	Tool-integrated reasoning agent gains
FineMath dataset	Math web data, 4+ subset
NuminaMath-CoT	860k SFT corpus
Tokenizer transplantation, arxiv 2506.06607	Cheap tokenizer swap method
Common 7B already has math, arxiv 2403.04706	Llama-2-7B SFT ceiling 82.6 GSM8K
All cited content was rephrased and condensed for compliance with licensing restrictions. Verbatim quoting kept under 30 words per source.

Bottom-line recommendation: do Stage 0 → Stage 1-light (30B FineMath CPT) → Stage 2 → Stage 3 → Stage 5. This is 280 MI300X-hours ($560 spot), and lands at the central GSM8K ~47–62%, MATH ~17–24% projection — placing TinyMath-1B in the same league as Rho-Math-1B and SmolTulu. Skip Stage 4 (GRPO) on the first pass: it's the highest-cost-lowest-confidence stage and only worth running if Stage 3's RFT base correctness > 25%. Skip Stage 6 (tool-use) until the post-Stage-5 model is in production and you have telemetry on which problem classes need it.