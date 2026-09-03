Part 1: Why scores are low — root causes by stage

The headline numbers (Base ~1% / SFT ~1% / GRPO 2.2% on GSM8K) aren't a single failure — they're a cascade. Each stage's ceiling is set by the previous stage, and the pretrain ceiling is the binding constraint here.

1.1 Base model: severely undertrained on math, with weak data quality

This is the dominant cause of low math scores. Everything downstream inherits it.

(A) Token budget is 1–2 orders of magnitude below the modern small-LM regime
maxtext_config.yml shows steps: 54363 × per_device_batch_size: 2 × 32 chips × 4096 tokens ≈ 57B tokens.

Model	Params	Pretrain tokens	Tokens/param
TinyMathReason	1.12B	57B	~50
TinyLlama-1.1B	1.1B	3 T	~2,700
Llama-3.2-1B	1.2B	~9 T	~7,500
Qwen2.5-1.5B	1.5B	~18 T	~12,000
SmolLM2-1.7B	1.7B	11 T	~6,500
Chinchilla-optimal for compute is ~20 tokens/param, but for inference-deployable small models the literature now agrees on heavy overtraining — 1,000–10,000 tokens/param (Beyond Chinchilla-Optimal, ICML 2024; SmolLM2 paper). At 50 tokens/param, TinyMath simply hasn't seen enough math to memorize basic arithmetic identities, let alone compose them. This alone could explain ~80% of the GSM8K gap.

(B) Tokenizer almost certainly hurts arithmetic
STATUS.md says the 32k BPE was "trained on sample data". Modern math-capable models do one of two things:

Split digits at tokenizer time (Llama-3, DeepSeek-Math): each digit is its own token so addition/subtraction are linearly composable.
Reuse a pretrained 100k+ tokenizer (Qwen, Gemma).
A small custom BPE trained on web text typically merges frequent multi-digit numbers ("2024", "100", "50") into single tokens, while rarer numbers ("382391") become an unrecoverable mash. This makes arithmetic essentially unlearnable at small scale. Worth verifying: tokenize "1234567" and "382 + 491 = 873". If they aren't per-digit, this is a hard cap.

(C) Data pipeline is much weaker than advertised
Looking at b_clean_and_filter.py and c_mix_datasets.py:

Claim (docs)	Reality (code)
"MinHash dedup"	Only hash(text) exact-string dedup. No near-duplicate removal.
"Proof-Pile-2 15%"	Listed in mixer ratios (c_mix_datasets.py:23) but never downloaded (a_download_datasets.py:11–16 only has fineweb-edu, openwebmath, mathpile, stack-edu).
Quality filtering	Only len(words) > 20 and alpha_ratio > 0.3. No perplexity filter, no FineWeb-Edu-style classifier, no language detection, no PII scrubbing.
Sequence packing	d_tokenize_and_pack.py simply concatenates docs + EOS into 4096-token sequences. No document-attention mask → cross-document attention contamination, especially harmful for math where each problem must be self-contained.
Curriculum	None. Single-stage uniform mix. Modern recipes (SmolLM2, MiniCPM, Qwen2.5) use staged pretraining with math-heavy annealing in the final 10–20%.
Synthetic data	None. SmolLM2's main edge over TinyLlama is ~28B tokens of synthetic Cosmopedia textbooks.
(D) Hyperparameter / architecture nits that don't help
Warmup: warmup_steps_fraction: 0.0066 ≈ 360 steps out of 54k (~0.7%). Standard is 1–2%. Short warmup with bf16 + math-heavy data risks early-step instability.
No QK-norm, no z-loss, no MuP — the standard small-model stability stack now (Qwen2.5, SmolLM2). The model can train, but it leaves performance on the table.
scan_layers: False was used to fix the "zero-layers" bug, but it's also slower; that's why the run got cut short at 57B tokens of compute budget.
1.2 SFT: format is learned, reasoning isn't

prepare_sft_data.py, train_sft.py, and the configs reveal four independent problems.

(A) CoT extraction is heuristic and produces noisy <think> boundaries
extract_cot_and_answer splits responses by literal substrings like "The answer is", falling back to splitting on the last ". ". For:

GSM8K (#### separator): works correctly.
MetaMathQA: usually works (it has "The answer is X." reliably).
MathInstruct (260k examples): contains program-aided traces, GPT-4 explanations, multi-format outputs. The heuristic frequently puts the answer inside <think> and an empty / partial string outside it. The model learns the syntax of <think> without any consistent semantic boundary.
This is why GRPO's format reward saturates while correctness doesn't — the SFT model learned to open a <think> block but not to use it.

(B) No prompt-only loss masking
train_sft.py uses SFTTrainer with dataset_text_field="text" but does not pass DataCollatorForCompletionOnlyLM or set a response template. By default, SFTTrainer computes loss over the whole sequence — system prompt, user question, and assistant answer.

Effect: ~50–70% of the loss is the model "fitting" the system prompt and user questions, which is wasted gradient. For ~660k examples × 2 epochs, this dilution is significant.

(C) Embedding resize without proper initialization regresses general capability
STATUS.md: ARC-Easy went 29.9% → 25.5% (−4.4%). Classic symptom.

train_sft.py calls model.resize_token_embeddings(len(tokenizer)) after adding <think> / </think>. HuggingFace's default initializes new rows from a small random normal. The well-known fix (Llama-3 multilingual extension, 2024) is to average existing embeddings as the init, or to use the Hewitt 2021 mean-init. Random init temporarily destabilizes the LM head distribution and hurts non-math evals until enough SFT steps recover it — which 2 epochs at lr=2e-5 don't.

(D) Wrong amount and direction of training
Two epochs is far too few for small models on SFT. Maximizing Accuracy with SFT, arxiv 2507.08267 finds extending SFT to 10 epochs is crucial for small models — the breakthrough comes well after typical large-model practice.
Stage 1 (Alpaca-only ~52k) is too narrow and uses a system prompt claiming "mathematical assistant" while the data is general — confuses the prior.
Stage 1 → Stage 2 ordering is backwards for a weak base. Strong-reasoning recipes (DeepSeek-R1, Tulu-3) do the opposite: rich math/CoT first, broad chat last as a polish.
lr 2e-5 is high for a 1B base whose math capability is essentially random. With a near-zero math prior, this lr causes the model to memorize SFT formatting noise rather than amplify any latent math signal.
No data filtering on the 660k examples. MathInstruct in particular has known quality issues (GPT-3.5/4 hallucinated solutions); training on uncurated data at 2 epochs reinforces wrong reasoning patterns. (See OpenMathInstruct paper for a clean comparison.)
1.3 GRPO: doomed by ~0% base correctness

train_grpo.py and grpo_report.md make this very clear.

(A) GRPO requires non-zero base accuracy to learn anything
GRPO computes per-group advantage from variance in rewards across G rollouts. With 1.4% empirical correctness rate and G=8:

Probability of a group containing ≥1 correct rollout ≈ 1 − (1−0.014)⁸ ≈ 10.6%.
Probability of a group with 0 or 8 correct (zero variance → zero advantage) ≈ 89%+.
So 89% of training steps produce zero correctness gradient. The only signal that consistently varies across rollouts is the format reward — which is exactly what the run shows: format reward saturated at 0.50, correctness near zero.

This is the DeepSeek-R1-Zero observation: pure RL works only when the policy can occasionally succeed by chance. For tiny models on hard math, that requires SFT cold-start with strong long-CoT data first. The repo skipped that.

(B) Reward shape lets format hack correctness
reward_functions = [correctness_reward_func, format_reward_func, repetition_penalty_func]
TRL sums the rewards. Max correctness = 1.0, max format = 1.0. Empirically: ~1.4% correctness, ~50% format → format dominates the gradient by 35×. The model's optimum is "produce a valid <think>...</think> answer shell with any content" — exactly the "discussing poverty when asked math" failure mode in grpo_report.md.

(C) max_completion_length=512 truncates real reasoning
Median GSM8K rationale is 100–250 tokens for grade-school problems, but G=8 rollouts at temperature 0.9 produce many that ramble past 512. The format regex requires both <think> AND </think> AND a non-empty answer; truncated rollouts get format=0 and correctness=0 and often trip the repetition penalty too. So long, exploratory generations — exactly the ones you want to encourage — are systematically punished.

(D) Single-dataset training causes exploration collapse
GSM8K train has ~7,473 unique problems. 22,419 steps × G=8 = 180k generations on the same 7.5k prompts — the policy memorizes prompt-specific completions rather than learning generalizable reasoning. The benchmark gain (2.2% on GSM8K, 2.0% on Minerva) is consistent with mild prompt-distribution memorization, not capability transfer.

(E) lr 5e-6 is conservative for a near-zero-reward regime
When the gradient is mostly noise (rare correctness signal + strong format hack), conservative lr means the policy moves slowly toward the wrong objective. This explains the high entropy (4.679) and low KL (2.975) — the policy barely changed substantively except to satisfy format.

1.4 Cross-cutting issue: evaluation protocol drift

The reported scores understate the SFT model and overstate the comparison apples-to-apples:

lm-eval-harness gsm8k task default uses raw few-shot text, not chat templates. The SFT model was trained with ChatML — so 8-shot eval prompts it out-of-distribution. STATUS.md already notes "0.00% under raw extraction → 1.00% with aligned ChatML template" — confirming the protocol issue.
Strict-match vs flexible-extract: SFT 8-shot uses strict, GRPO 8-shot uses flexible. Not directly comparable.
8-shot GSM8K eats ~5 KB of context, near the 4096 ctx limit. For a tiny model, in-context demos help less than they help large ones; 0-shot CoT is a more honest measure of intrinsic capability.
run_custom_eval.py uses temperature=0.7, do_sample=True with a single sample — high variance. Should use greedy (t=0) for benchmarking, or cons@8 if sampling.
future_scores.md projected GRPO at 45–55% GSM8K. The actual was 2.2%. The 25× gap should itself be a learning signal: it implies the assumed "GRPO unlocks latent reasoning" mechanism wasn't applicable here because the SFT cold-start was too weak.
Part 2: Optimization roadmap, tiered by ROI

I'll give you five tiers ordered by (gain per dollar/day spent). Tier 1–2 alone should get you 5–8× current scores with modest effort.

Tier 1: Free fixes — eval & inference (hours, no training)

Expected lift: 0 → realistic measurement, possibly +2–5% GSM8K just from protocol.

Re-eval everything with aligned protocols. For SFT/GRPO models: pass --apply_chat_template to lm-eval-harness, use gsm8k_cot (0-shot CoT) and gsm8k_cot_self_consistency tasks rather than 8-shot.
Self-consistency (cons@N). Sample N=8–32 completions at temp 0.7, take majority-vote final answer. [Wang et al. 2022] shows +10–20% on GSM8K free, especially for small models. With 1B + cons@32, Liu et al. 2502.06703 shows tiny models can reach surprising numbers.
Tokenizer audit. Run:
tok.tokenize("382 + 491 = 873  /  1/2 + 3/4  /  $\boxed{42}$")
If multi-digit numbers stay merged, that's a hard cap on math performance. Decide if it's worth re-tokenizing (Tier 4).
Greedy + parsed boxed answer for benchmark reporting; sampling only for cons@N.
Tier 2: Redo SFT properly (1–3 days, 1×MI300X or 4×A100)

Expected lift: GSM8K 1% → 8–15%, MATH 0% → 3–8%.

This is the highest-ROI bucket. Almost all the SFT problems are fixable.

Throw out the two-stage curriculum. Single-stage SFT on a curated math+chat blend:
70% NuminaMath-CoT (high-quality competition math, MIT-licensed)
15% OpenMathInstruct-2 filtered to verified-correct subset only
10% GSM8K train (rewritten with longer CoT — see #4)
5% UltraChat / Tulu-3-SFT-mixture (general chat preservation)
Use completion-only loss masking. Wrap with DataCollatorForCompletionOnlyLM (TRL) or set response_template="<|im_start|>assistant " so loss is only on the assistant turn. This is the single biggest gradient-efficiency win.
Fix CoT formatting deterministically per dataset. For GSM8K use ####; for NuminaMath use the existing <think> field if present; for everything else, treat the entire response as <think> and parse \boxed{} as the answer. Don't try to split on natural-language phrases.
Distill long CoT. Use a strong open teacher (Qwen2.5-Math-7B-Instruct, DeepSeek-Math-7B-RL, or Qwen2.5-72B-Instruct via API) to regenerate GSM8K solutions with explicit, verbose chain-of-thought. Filter teacher outputs by ground-truth correctness (rejection sampling). This produces clean <think> boundaries and high-quality reasoning patterns. ~30k examples is enough.
Embedding init for new tokens. When adding <think>/</think>:
with torch.no_grad():
    avg = model.get_input_embeddings().weight[:original_vocab].mean(dim=0)
    model.get_input_embeddings().weight[new_token_ids] = avg
    model.get_output_embeddings().weight[new_token_ids] = avg
Hyperparameters:
lr = 5e-6 (not 2e-5) with cosine, warmup 5%
5–10 epochs, not 2 (per arxiv 2507.08267)
Effective batch 64–128 (gradient accumulation)
Seq packing ON (packing=True in SFTConfig) with attention-mask reset
Validate every epoch on a held-out 200-problem GSM8K dev set with greedy + cons@8. Stop when dev plateaus; small models often peak at epoch 4–7 and degrade after.
Tier 3: Better RL — SFT first, then GRPO with redesigned rewards (2–4 days)

Expected lift over Tier-2 SFT: +3–8% GSM8K, +2–5% MATH.

Only worth doing after Tier 2 — RL needs Tier-2's higher base accuracy.

Gate, don't sum. Replace the additive reward with a gated one:
reward = format_pass * (1.0 if correct else -0.1) + repetition_penalty
No format → 0; bad answer → small negative; correct → +1. Format is now a precondition, not a competing objective. Eliminates the 35× format-hack imbalance.
max_completion_length = 2048 (4096 fits on MI300X 192GB with G=8). Truncation is a silent killer.
Multi-dataset GRPO: GSM8K + MATH-Algebra + MATH-Counting + ASDiv + SVAMP. Mix at sampling time. Eliminates the "memorize 7.5k prompts" trap.
Curriculum / difficulty-aware advantage (GRPO-LEAD, arxiv 2504.09696): up-weight advantage on harder problems. Concretely, bin problems by SFT-model pass rate; up-weight bins with 20–60% pass rate (the "learnable" zone).
If correctness rate stays < 5%: don't run GRPO yet — do RFT first. Rejection sampling fine-tuning: sample 16 CoTs per GSM8K problem with the SFT model, keep correct ones, SFT on them for 1–2 epochs, then GRPO. This is the DeepSeek-R1 cold-start recipe compressed.
Increase G to 16, lower temperature to 0.7. Lower temp gives more useful gradient when correctness is rare; G=16 doubles probability of seeing variance per group.
Tier 4: Fix the pretrain ceiling (1–4 weeks, real compute)

Expected lift: GSM8K → 25–45%, MATH → 8–20%. This is where the big gains live.

The honest answer is that no SFT/RL recipe will produce strong math out of a 57B-token base. Reference points:

TinyLlama-1.1B at 3T tokens: GSM8K 2.6% (8-shot, base).
TinyLlama-v1.1-Math-Code (continual pretrain on math): GSM8K 5.8%.
Rho-1B (Microsoft, arxiv 2404.07965): MATH 15.6% few-shot — and crucially, it gets there with only 15B tokens of continual pretraining via Selective Language Modeling (SLM). 30% absolute improvement over standard CLM on that same data.
The cheapest path to a real math model is continual pretraining of your existing checkpoint, not a redo:

+150B tokens of math-dense continual pretrain. Mix:
FineMath-4+ (HuggingFace, 2025): the new gold standard for math web data
Cosmopedia v2 synthetic math textbooks
Proof-Pile-2 (proofs, theorems — currently absent from your data!)
AutoMathText with high quality scores
OpenWebMath2 / TheoremQA
Apply Rho-1 SLM. Score each token with a small reference model (use your current base or a Qwen2.5-Math-1.5B) and only backprop on the top-60% highest-utility tokens. Reported +30% absolute on math benchmarks for ~same compute.
Document-boundary attention masks (modify the data packer or use FlashAttention's varlen mode). Stops cross-doc contamination.
Fix tokenizer if audit confirms multi-digit merges. Re-tokenize the data with a digit-split BPE (or just adopt Llama-3 / Qwen2.5 tokenizer). Continue pretrain with embedding-mean init for newly-mapped IDs.
Add QK-norm and z-loss. Both are 5-line changes, both stabilize long training.
Anneal with math-heavy mix in the last 10–15B tokens. Standard SmolLM2 / MiniCPM trick.
Tier 5: Test-time scaling (free, complements every tier)

Liu et al. 2025, "Can 1B LLM Surpass 405B" shows that with strong test-time scaling, a 1B model can beat 405B on MATH-500. Once Tier 2/3 lifts you above ~20% pass@1:

Best-of-N with a verifier. Train a tiny PRM (process reward model) on Math-Shepherd or PRM800K → use it to rerank N=64 samples.
Self-consistency over CoTs (cons@64) — purely free, +5–15% on GSM8K typically.
Step-level beam search with the PRM as score function.
Tool augmentation: <python> tag for arithmetic. Even simple regex-detected calculator handoff can take a 1B model from ~10% to ~30% on GSM8K because raw arithmetic is the failure mode you can verify is the problem.
Part 3: The 80/20 — what I'd do this week

If you have 5 days and one MI300X:

Day 1: Tokenizer audit + eval re-run with aligned ChatML protocols + cons@8 baseline. Cheap, shows true current capability.
Day 1: Distill 30k GSM8K solutions with Qwen2.5-Math-7B-Instruct (or 72B via API) — verify correctness, store as JSONL.
Day 2–3: Tier-2 SFT redo (single-stage, completion-only loss, mean-init embeddings, lr 5e-6, 5 epochs). Validate epoch-by-epoch.
Day 4: RFT — 16 samples per GSM8K problem with the new SFT, keep correct, SFT 1 more epoch.
Day 5: Tier-3 GRPO with gated reward, max_len=2048, multi-dataset.
Realistic target: GSM8K 12–20%, MATH 4–8%, without touching pretraining. That is genuinely competitive with TinyLlama-1.1B-Math at its weight class and would be the right place to stop the "post-training only" phase of the project.

For the next research arc, Tier 4 (continual math pretrain with Rho-1 SLM and FineMath) is where the model class transitions from "learning project that runs end-to-end" (already achieved — and that's a real accomplishment) to "competitive 1B math model" (currently out of reach without more pretrain compute).

Sources cited (rephrased for compliance with licensing)

Part 1: Why scores are low — root causes by stage

The headline numbers (Base ~1% / SFT ~1% / GRPO 2.2% on GSM8K) aren't a single failure — they're a cascade. Each stage's ceiling is set by the previous stage, and the pretrain ceiling is the binding constraint here.

1.1 Base model: severely undertrained on math, with weak data quality

This is the dominant cause of low math scores. Everything downstream inherits it.

(A) Token budget is 1–2 orders of magnitude below the modern small-LM regime
maxtext_config.yml shows steps: 54363 × per_device_batch_size: 2 × 32 chips × 4096 tokens ≈ 57B tokens.

Model	Params	Pretrain tokens	Tokens/param
TinyMathReason	1.12B	57B	~50
TinyLlama-1.1B	1.1B	3 T	~2,700
Llama-3.2-1B	1.2B	~9 T	~7,500
Qwen2.5-1.5B	1.5B	~18 T	~12,000
SmolLM2-1.7B	1.7B	11 T	~6,500
Chinchilla-optimal for compute is ~20 tokens/param, but for inference-deployable small models the literature now agrees on heavy overtraining — 1,000–10,000 tokens/param (Beyond Chinchilla-Optimal, ICML 2024; SmolLM2 paper). At 50 tokens/param, TinyMath simply hasn't seen enough math to memorize basic arithmetic identities, let alone compose them. This alone could explain ~80% of the GSM8K gap.

(B) Tokenizer almost certainly hurts arithmetic
STATUS.md says the 32k BPE was "trained on sample data". Modern math-capable models do one of two things:

Split digits at tokenizer time (Llama-3, DeepSeek-Math): each digit is its own token so addition/subtraction are linearly composable.
Reuse a pretrained 100k+ tokenizer (Qwen, Gemma).
A small custom BPE trained on web text typically merges frequent multi-digit numbers ("2024", "100", "50") into single tokens, while rarer numbers ("382391") become an unrecoverable mash. This makes arithmetic essentially unlearnable at small scale. Worth verifying: tokenize "1234567" and "382 + 491 = 873". If they aren't per-digit, this is a hard cap.

(C) Data pipeline is much weaker than advertised
Looking at b_clean_and_filter.py and c_mix_datasets.py:

Claim (docs)	Reality (code)
"MinHash dedup"	Only hash(text) exact-string dedup. No near-duplicate removal.
"Proof-Pile-2 15%"	Listed in mixer ratios (c_mix_datasets.py:23) but never downloaded (a_download_datasets.py:11–16 only has fineweb-edu, openwebmath, mathpile, stack-edu).
Quality filtering	Only len(words) > 20 and alpha_ratio > 0.3. No perplexity filter, no FineWeb-Edu-style classifier, no language detection, no PII scrubbing.
Sequence packing	d_tokenize_and_pack.py simply concatenates docs + EOS into 4096-token sequences. No document-attention mask → cross-document attention contamination, especially harmful for math where each problem must be self-contained.
Curriculum	None. Single-stage uniform mix. Modern recipes (SmolLM2, MiniCPM, Qwen2.5) use staged pretraining with math-heavy annealing in the final 10–20%.
Synthetic data	None. SmolLM2's main edge over TinyLlama is ~28B tokens of synthetic Cosmopedia textbooks.
(D) Hyperparameter / architecture nits that don't help
Warmup: warmup_steps_fraction: 0.0066 ≈ 360 steps out of 54k (~0.7%). Standard is 1–2%. Short warmup with bf16 + math-heavy data risks early-step instability.
No QK-norm, no z-loss, no MuP — the standard small-model stability stack now (Qwen2.5, SmolLM2). The model can train, but it leaves performance on the table.
scan_layers: False was used to fix the "zero-layers" bug, but it's also slower; that's why the run got cut short at 57B tokens of compute budget.
1.2 SFT: format is learned, reasoning isn't

prepare_sft_data.py, train_sft.py, and the configs reveal four independent problems.

(A) CoT extraction is heuristic and produces noisy <think> boundaries
extract_cot_and_answer splits responses by literal substrings like "The answer is", falling back to splitting on the last ". ". For:

GSM8K (#### separator): works correctly.
MetaMathQA: usually works (it has "The answer is X." reliably).
MathInstruct (260k examples): contains program-aided traces, GPT-4 explanations, multi-format outputs. The heuristic frequently puts the answer inside <think> and an empty / partial string outside it. The model learns the syntax of <think> without any consistent semantic boundary.
This is why GRPO's format reward saturates while correctness doesn't — the SFT model learned to open a <think> block but not to use it.

(B) No prompt-only loss masking
train_sft.py uses SFTTrainer with dataset_text_field="text" but does not pass DataCollatorForCompletionOnlyLM or set a response template. By default, SFTTrainer computes loss over the whole sequence — system prompt, user question, and assistant answer.

Effect: ~50–70% of the loss is the model "fitting" the system prompt and user questions, which is wasted gradient. For ~660k examples × 2 epochs, this dilution is significant.

(C) Embedding resize without proper initialization regresses general capability
STATUS.md: ARC-Easy went 29.9% → 25.5% (−4.4%). Classic symptom.

train_sft.py calls model.resize_token_embeddings(len(tokenizer)) after adding <think> / </think>. HuggingFace's default initializes new rows from a small random normal. The well-known fix (Llama-3 multilingual extension, 2024) is to average existing embeddings as the init, or to use the Hewitt 2021 mean-init. Random init temporarily destabilizes the LM head distribution and hurts non-math evals until enough SFT steps recover it — which 2 epochs at lr=2e-5 don't.

(D) Wrong amount and direction of training
Two epochs is far too few for small models on SFT. Maximizing Accuracy with SFT, arxiv 2507.08267 finds extending SFT to 10 epochs is crucial for small models — the breakthrough comes well after typical large-model practice.
Stage 1 (Alpaca-only ~52k) is too narrow and uses a system prompt claiming "mathematical assistant" while the data is general — confuses the prior.
Stage 1 → Stage 2 ordering is backwards for a weak base. Strong-reasoning recipes (DeepSeek-R1, Tulu-3) do the opposite: rich math/CoT first, broad chat last as a polish.
lr 2e-5 is high for a 1B base whose math capability is essentially random. With a near-zero math prior, this lr causes the model to memorize SFT formatting noise rather than amplify any latent math signal.
No data filtering on the 660k examples. MathInstruct in particular has known quality issues (GPT-3.5/4 hallucinated solutions); training on uncurated data at 2 epochs reinforces wrong reasoning patterns. (See OpenMathInstruct paper for a clean comparison.)
1.3 GRPO: doomed by ~0% base correctness

train_grpo.py and grpo_report.md make this very clear.

(A) GRPO requires non-zero base accuracy to learn anything
GRPO computes per-group advantage from variance in rewards across G rollouts. With 1.4% empirical correctness rate and G=8:

Probability of a group containing ≥1 correct rollout ≈ 1 − (1−0.014)⁸ ≈ 10.6%.
Probability of a group with 0 or 8 correct (zero variance → zero advantage) ≈ 89%+.
So 89% of training steps produce zero correctness gradient. The only signal that consistently varies across rollouts is the format reward — which is exactly what the run shows: format reward saturated at 0.50, correctness near zero.

This is the DeepSeek-R1-Zero observation: pure RL works only when the policy can occasionally succeed by chance. For tiny models on hard math, that requires SFT cold-start with strong long-CoT data first. The repo skipped that.

(B) Reward shape lets format hack correctness
reward_functions = [correctness_reward_func, format_reward_func, repetition_penalty_func]
TRL sums the rewards. Max correctness = 1.0, max format = 1.0. Empirically: ~1.4% correctness, ~50% format → format dominates the gradient by 35×. The model's optimum is "produce a valid <think>...</think> answer shell with any content" — exactly the "discussing poverty when asked math" failure mode in grpo_report.md.

(C) max_completion_length=512 truncates real reasoning
Median GSM8K rationale is 100–250 tokens for grade-school problems, but G=8 rollouts at temperature 0.9 produce many that ramble past 512. The format regex requires both <think> AND </think> AND a non-empty answer; truncated rollouts get format=0 and correctness=0 and often trip the repetition penalty too. So long, exploratory generations — exactly the ones you want to encourage — are systematically punished.

(D) Single-dataset training causes exploration collapse
GSM8K train has ~7,473 unique problems. 22,419 steps × G=8 = 180k generations on the same 7.5k prompts — the policy memorizes prompt-specific completions rather than learning generalizable reasoning. The benchmark gain (2.2% on GSM8K, 2.0% on Minerva) is consistent with mild prompt-distribution memorization, not capability transfer.

(E) lr 5e-6 is conservative for a near-zero-reward regime
When the gradient is mostly noise (rare correctness signal + strong format hack), conservative lr means the policy moves slowly toward the wrong objective. This explains the high entropy (4.679) and low KL (2.975) — the policy barely changed substantively except to satisfy format.

1.4 Cross-cutting issue: evaluation protocol drift

The reported scores understate the SFT model and overstate the comparison apples-to-apples:

lm-eval-harness gsm8k task default uses raw few-shot text, not chat templates. The SFT model was trained with ChatML — so 8-shot eval prompts it out-of-distribution. STATUS.md already notes "0.00% under raw extraction → 1.00% with aligned ChatML template" — confirming the protocol issue.
Strict-match vs flexible-extract: SFT 8-shot uses strict, GRPO 8-shot uses flexible. Not directly comparable.
8-shot GSM8K eats ~5 KB of context, near the 4096 ctx limit. For a tiny model, in-context demos help less than they help large ones; 0-shot CoT is a more honest measure of intrinsic capability.
run_custom_eval.py uses temperature=0.7, do_sample=True with a single sample — high variance. Should use greedy (t=0) for benchmarking, or cons@8 if sampling.
future_scores.md projected GRPO at 45–55% GSM8K. The actual was 2.2%. The 25× gap should itself be a learning signal: it implies the assumed "GRPO unlocks latent reasoning" mechanism wasn't applicable here because the SFT cold-start was too weak.
Part 2: Optimization roadmap, tiered by ROI

I'll give you five tiers ordered by (gain per dollar/day spent). Tier 1–2 alone should get you 5–8× current scores with modest effort.

Tier 1: Free fixes — eval & inference (hours, no training)

Expected lift: 0 → realistic measurement, possibly +2–5% GSM8K just from protocol.

Re-eval everything with aligned protocols. For SFT/GRPO models: pass --apply_chat_template to lm-eval-harness, use gsm8k_cot (0-shot CoT) and gsm8k_cot_self_consistency tasks rather than 8-shot.
Self-consistency (cons@N). Sample N=8–32 completions at temp 0.7, take majority-vote final answer. [Wang et al. 2022] shows +10–20% on GSM8K free, especially for small models. With 1B + cons@32, Liu et al. 2502.06703 shows tiny models can reach surprising numbers.
Tokenizer audit. Run:
tok.tokenize("382 + 491 = 873  /  1/2 + 3/4  /  $\boxed{42}$")
If multi-digit numbers stay merged, that's a hard cap on math performance. Decide if it's worth re-tokenizing (Tier 4).
Greedy + parsed boxed answer for benchmark reporting; sampling only for cons@N.
Tier 2: Redo SFT properly (1–3 days, 1×MI300X or 4×A100)

Expected lift: GSM8K 1% → 8–15%, MATH 0% → 3–8%.

This is the highest-ROI bucket. Almost all the SFT problems are fixable.

Throw out the two-stage curriculum. Single-stage SFT on a curated math+chat blend:
70% NuminaMath-CoT (high-quality competition math, MIT-licensed)
15% OpenMathInstruct-2 filtered to verified-correct subset only
10% GSM8K train (rewritten with longer CoT — see #4)
5% UltraChat / Tulu-3-SFT-mixture (general chat preservation)
Use completion-only loss masking. Wrap with DataCollatorForCompletionOnlyLM (TRL) or set response_template="<|im_start|>assistant " so loss is only on the assistant turn. This is the single biggest gradient-efficiency win.
Fix CoT formatting deterministically per dataset. For GSM8K use ####; for NuminaMath use the existing <think> field if present; for everything else, treat the entire response as <think> and parse \boxed{} as the answer. Don't try to split on natural-language phrases.
Distill long CoT. Use a strong open teacher (Qwen2.5-Math-7B-Instruct, DeepSeek-Math-7B-RL, or Qwen2.5-72B-Instruct via API) to regenerate GSM8K solutions with explicit, verbose chain-of-thought. Filter teacher outputs by ground-truth correctness (rejection sampling). This produces clean <think> boundaries and high-quality reasoning patterns. ~30k examples is enough.
Embedding init for new tokens. When adding <think>/</think>:
with torch.no_grad():
    avg = model.get_input_embeddings().weight[:original_vocab].mean(dim=0)
    model.get_input_embeddings().weight[new_token_ids] = avg
    model.get_output_embeddings().weight[new_token_ids] = avg
Hyperparameters:
lr = 5e-6 (not 2e-5) with cosine, warmup 5%
5–10 epochs, not 2 (per arxiv 2507.08267)
Effective batch 64–128 (gradient accumulation)
Seq packing ON (packing=True in SFTConfig) with attention-mask reset
Validate every epoch on a held-out 200-problem GSM8K dev set with greedy + cons@8. Stop when dev plateaus; small models often peak at epoch 4–7 and degrade after.
Tier 3: Better RL — SFT first, then GRPO with redesigned rewards (2–4 days)

Expected lift over Tier-2 SFT: +3–8% GSM8K, +2–5% MATH.

Only worth doing after Tier 2 — RL needs Tier-2's higher base accuracy.

Gate, don't sum. Replace the additive reward with a gated one:
reward = format_pass * (1.0 if correct else -0.1) + repetition_penalty
No format → 0; bad answer → small negative; correct → +1. Format is now a precondition, not a competing objective. Eliminates the 35× format-hack imbalance.
max_completion_length = 2048 (4096 fits on MI300X 192GB with G=8). Truncation is a silent killer.
Multi-dataset GRPO: GSM8K + MATH-Algebra + MATH-Counting + ASDiv + SVAMP. Mix at sampling time. Eliminates the "memorize 7.5k prompts" trap.
Curriculum / difficulty-aware advantage (GRPO-LEAD, arxiv 2504.09696): up-weight advantage on harder problems. Concretely, bin problems by SFT-model pass rate; up-weight bins with 20–60% pass rate (the "learnable" zone).
If correctness rate stays < 5%: don't run GRPO yet — do RFT first. Rejection sampling fine-tuning: sample 16 CoTs per GSM8K problem with the SFT model, keep correct ones, SFT on them for 1–2 epochs, then GRPO. This is the DeepSeek-R1 cold-start recipe compressed.
Increase G to 16, lower temperature to 0.7. Lower temp gives more useful gradient when correctness is rare; G=16 doubles probability of seeing variance per group.
Tier 4: Fix the pretrain ceiling (1–4 weeks, real compute)

Expected lift: GSM8K → 25–45%, MATH → 8–20%. This is where the big gains live.

The honest answer is that no SFT/RL recipe will produce strong math out of a 57B-token base. Reference points:

TinyLlama-1.1B at 3T tokens: GSM8K 2.6% (8-shot, base).
TinyLlama-v1.1-Math-Code (continual pretrain on math): GSM8K 5.8%.
Rho-1B (Microsoft, arxiv 2404.07965): MATH 15.6% few-shot — and crucially, it gets there with only 15B tokens of continual pretraining via Selective Language Modeling (SLM). 30% absolute improvement over standard CLM on that same data.
The cheapest path to a real math model is continual pretraining of your existing checkpoint, not a redo:

+150B tokens of math-dense continual pretrain. Mix:
FineMath-4+ (HuggingFace, 2025): the new gold standard for math web data
Cosmopedia v2 synthetic math textbooks
Proof-Pile-2 (proofs, theorems — currently absent from your data!)
AutoMathText with high quality scores
OpenWebMath2 / TheoremQA
Apply Rho-1 SLM. Score each token with a small reference model (use your current base or a Qwen2.5-Math-1.5B) and only backprop on the top-60% highest-utility tokens. Reported +30% absolute on math benchmarks for ~same compute.
Document-boundary attention masks (modify the data packer or use FlashAttention's varlen mode). Stops cross-doc contamination.
Fix tokenizer if audit confirms multi-digit merges. Re-tokenize the data with a digit-split BPE (or just adopt Llama-3 / Qwen2.5 tokenizer). Continue pretrain with embedding-mean init for newly-mapped IDs.
Add QK-norm and z-loss. Both are 5-line changes, both stabilize long training.
Anneal with math-heavy mix in the last 10–15B tokens. Standard SmolLM2 / MiniCPM trick.
Tier 5: Test-time scaling (free, complements every tier)

Liu et al. 2025, "Can 1B LLM Surpass 405B" shows that with strong test-time scaling, a 1B model can beat 405B on MATH-500. Once Tier 2/3 lifts you above ~20% pass@1:

Best-of-N with a verifier. Train a tiny PRM (process reward model) on Math-Shepherd or PRM800K → use it to rerank N=64 samples.
Self-consistency over CoTs (cons@64) — purely free, +5–15% on GSM8K typically.
Step-level beam search with the PRM as score function.
Tool augmentation: <python> tag for arithmetic. Even simple regex-detected calculator handoff can take a 1B model from ~10% to ~30% on GSM8K because raw arithmetic is the failure mode you can verify is the problem.
Part 3: The 80/20 — what I'd do this week

If you have 5 days and one MI300X:

Day 1: Tokenizer audit + eval re-run with aligned ChatML protocols + cons@8 baseline. Cheap, shows true current capability.
Day 1: Distill 30k GSM8K solutions with Qwen2.5-Math-7B-Instruct (or 72B via API) — verify correctness, store as JSONL.
Day 2–3: Tier-2 SFT redo (single-stage, completion-only loss, mean-init embeddings, lr 5e-6, 5 epochs). Validate epoch-by-epoch.
Day 4: RFT — 16 samples per GSM8K problem with the new SFT, keep correct, SFT 1 more epoch.
Day 5: Tier-3 GRPO with gated reward, max_len=2048, multi-dataset.
Realistic target: GSM8K 12–20%, MATH 4–8%, without touching pretraining. That is genuinely competitive with TinyLlama-1.1B-Math at its weight class and would be the right place to stop the "post-training only" phase of the project.

For the next research arc, Tier 4 (continual math pretrain with Rho-1 SLM and FineMath) is where the model class transitions from "learning project that runs end-to-end" (already achieved — and that's a real accomplishment) to "competitive 1B math model" (currently out of reach without more pretrain compute).

Sources cited (rephrased for compliance with licensing)

Rho-1: Not All Tokens Are What You Need (arxiv 2404.07965) — Selective Language Modeling, +30% absolute on math.
SmolLM2: Data-Centric Training (arxiv 2502.02737) — 11T tokens, multi-stage curriculum with synthetic data.
Beyond Chinchilla-Optimal (arxiv 2401.00448) — small models should be overtrained for inference-time efficiency.
Maximizing Accuracy with SFT (arxiv 2507.08267) — small models need ~10 SFT epochs for breakthroughs.
GRPO-LEAD: difficulty-aware GRPO (arxiv 2504.09696) — length-aware rewards and difficulty reweighting.
Can 1B LLM Surpass 405B LLM (arxiv 2502.06703) — test-time scaling enables tiny models to compete.
OpenMathInstruct-2 (arxiv 2410.01560) — diversity > polish for math SFT.
