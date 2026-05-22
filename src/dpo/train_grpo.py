import argparse
import logging
import re
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# TRL >=0.17 uses GRPOConfig; older versions used GRPOTrainingArguments
try:
    from trl import GRPOTrainer, GRPOConfig
except ImportError:
    try:
        from trl import GRPOTrainer, GRPOTrainingArguments as GRPOConfig
        logging.warning("Using GRPOTrainingArguments (older TRL). Upgrade to trl>=0.17 for GRPOConfig.")
    except ImportError:
        raise ImportError("GRPOTrainer not found. Install trl>=0.17: pip install trl>=0.17.0")

# AST-based mathematical verification (replaces string matching)
try:
    from math_verify import LatexExtractionConfig, parse, verify
    HAS_MATH_VERIFY = True
except ImportError:
    HAS_MATH_VERIFY = False
    logging.warning("math_verify not installed. Falling back to string-based correctness. "
                    "Install with: pip install math-verify")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# 1. REWARD FUNCTIONS
# ==========================================

def extract_answer_gsm8k(text: str) -> str:
    """Extract final answer from the text after the </think> tag."""
    # If the model produced a reasoning block, only look at the text after it
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    
    # GSM8K ground truth format: "... #### 42"
    if "####" in text:
        return text.split("####")[-1].strip().replace(",", "").replace(" ", "")
    # Boxed LaTeX
    boxed_match = re.search(r'\\boxed{([^}]+)}', text)
    if boxed_match:
        return boxed_match.group(1).strip().replace(",", "").replace(" ", "")
    # Fallback: extract the last number found in the answer section
    numbers = re.findall(r'-?\d[\d,]*\.?\d*', text)
    if numbers:
        return numbers[-1].replace(",", "").replace(" ", "")
    return text.strip()


def _get_completion_text(completion) -> str:
    """Extract text content from TRL completion format (dict or string)."""
    if isinstance(completion, dict):
        return completion.get("content", "")
    if isinstance(completion, list):
        # List of message dicts — take the last assistant message
        for msg in reversed(completion):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
        return str(completion)
    return str(completion)


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """
    Evaluates mathematical correctness using AST parsing (math_verify)
    on the text strictly after </think>, with a GSM8K numeric fallback.

    Returns 1.0 for correct, 0.0 for incorrect.
    """
    rewards = []
    for comp, gt in zip(completions, answer):
        content = _get_completion_text(comp)
        
        # Isolate candidate answer strictly after </think>
        if "</think>" in content:
            pred_ans_text = content.split("</think>")[-1].strip()
        else:
            pred_ans_text = content
            
        try:
            if HAS_MATH_VERIFY:
                # Try AST-based verification first (handles fractions, LaTeX, etc.)
                gold_parsed = parse(gt, extraction_mode="first_match",
                                    extraction_config=[LatexExtractionConfig()])
                if gold_parsed:
                    pred_parsed = parse(pred_ans_text, extraction_mode="first_match",
                                        extraction_config=[LatexExtractionConfig()])
                    if verify(pred_parsed, gold_parsed):
                        rewards.append(1.0)
                        continue

            # Fallback: GSM8K numeric extraction
            pred = extract_answer_gsm8k(content)
            truth = extract_answer_gsm8k(gt)
            if pred and truth and pred == truth:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except Exception:
            # Safely handle unparseable exceptions
            rewards.append(0.0)
    return rewards


def format_reward_func(completions, **kwargs) -> list[float]:
    """
    Enforces the actual SFT Stage 2 layout:
    Starts with <think>, has non-empty reasoning inside, closes with </think>,
    and has a non-empty answer following it. No <answer> tags are required.
    """
    import os
    os.makedirs("./logs", exist_ok=True)
    
    # Pattern: Must start with <think>, contain non-empty content,
    # close with </think>, followed by non-empty answer text.
    pattern = r"^\s*<think>\s*\S.*?</think>\s*\S.*"

    rewards = []
    for i, comp in enumerate(completions):
        content = _get_completion_text(comp)
        
        # Log first two completions of every batch to debug
        if i < 2:
            with open("./logs/debug_completions.txt", "a") as f:
                f.write(f"\n====================================\n")
                f.write(f"COMPLETION LENGTH: {len(content)}\n")
                f.write(f"COMPLETION CONTENT:\n{repr(content)}\n")
                f.write(f"====================================\n")

        if re.search(pattern, content, re.DOTALL):
            rewards.append(1.0)
        else:
            # Fallback/Step-stone: at minimum has <think>...</think> with some content
            fallback = r"<think>\s*\S.{10,}?</think>"
            if re.search(fallback, content, re.DOTALL):
                rewards.append(0.5)
            else:
                rewards.append(0.0)
    return rewards


def repetition_penalty_func(completions, **kwargs) -> list[float]:
    """
    Penalizes mode collapse loops via 3-gram uniqueness ratio.
    When the model enters infinite repetition (e.g., 'e^x + e^x + e^x...'),
    the unique n-gram ratio drops near zero, triggering a heavy negative penalty.

    This forces GRPO to update the policy away from absorbing Markov states.
    """
    ngram_size = 3
    max_penalty = -1.5
    collapse_threshold = 0.2  # Below 20% unique ratio = collapsed

    rewards = []
    for comp in completions:
        content = _get_completion_text(comp)
        words = content.split()

        if len(words) < ngram_size + 2:
            # Too short to evaluate — no penalty
            rewards.append(0.0)
            continue

        ngrams = [tuple(words[i:i + ngram_size]) for i in range(len(words) - ngram_size + 1)]
        unique_ratio = len(set(ngrams)) / len(ngrams)

        if unique_ratio < collapse_threshold:
            # Scale penalty by severity: worse collapse = harder penalty
            penalty = max_penalty * (1.0 - unique_ratio)
            rewards.append(penalty)
        else:
            rewards.append(0.0)

    return rewards


# ==========================================
# 2. DATASET FORMATTING
# ==========================================

SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. Solve problems step by step "
    "inside <think> tags, and then provide the final mathematical answer "
    "inside <answer> tags."
)


def format_prompt(example):
    """Format GSM8K examples for GRPOTrainer with ChatML structure."""
    example["prompt"] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["question"]}
    ]
    # Keep the raw 'answer' column — reward functions receive it via kwargs
    return example


# ==========================================
# 3. GRPO TRAINING ORCHESTRATION
# ==========================================

def train_grpo(model_path: str, output_dir: str, max_samples: int = None,
               num_epochs: int = 1, use_vllm: bool = False):
    """
    Trains using Group Relative Policy Optimization (GRPO).

    Implements all hardening from Phase 4 plan:
    - AST-based correctness verification
    - Strict regex format rewards
    - N-gram repetition penalty
    - Explicit stop token synchronization
    - Calibrated hyperparameters (G=8, β=0.01, lr=5e-6, cosine)
    """
    logging.info(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Enforce rigid padding/EOS mapping to prevent masking errors
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info(f"Set pad_token = eos_token (ID: {tokenizer.eos_token_id})")

    # Programmatically inject ChatML template if not set (base models/SFT checkpoint configs)
    if getattr(tokenizer, "chat_template", None) is None:
        logging.info("Tokenizer has no chat template configured. Injecting default ChatML template...")
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}"
            "{% elif message['role'] == 'system' %}"
            "{{ '<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ '<|im_start|>assistant\\n' + message['content'] + '<|im_end|>\\n' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\\n' }}"
            "{% endif %}"
        )

    # Remove token_type_ids to prevent Llama models from throwing an error during generate()
    if "token_type_ids" in tokenizer.model_input_names:
        tokenizer.model_input_names.remove("token_type_ids")
        logging.info("Removed token_type_ids from tokenizer.model_input_names")

    # Build explicit stop token list to eradicate conversation simulation
    stop_ids = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id > 0:
        stop_ids.append(im_end_id)
        logging.info(f"Added <|im_end|> (ID: {im_end_id}) to stop tokens")
    logging.info(f"Stop token IDs: {stop_ids}")

    # Load model in bfloat16
    logging.info(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa"
    )

    # Synchronize model config with tokenizer special tokens
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # Load and format dataset
    logging.info("Loading GSM8K train split for GRPO...")
    dataset = load_dataset("gsm8k", "main", split="train")

    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
        logging.info(f"Using {max_samples} samples (staged execution)")

    dataset = dataset.map(format_prompt)
    logging.info(f"Dataset size: {len(dataset)} examples")

    # Generation arguments with explicit termination
    generation_kwargs = {
        "max_new_tokens": 512,
        "do_sample": True,
        "temperature": 0.9,  # High temperature for group diversity
        "eos_token_id": stop_ids,  # Terminate generation at <|im_end|> or standard EOS
    }

    # Build GRPOConfig with hardened hyperparameters
    config_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,   # Effective batch = 8
        learning_rate=5e-6,              # Accelerated LR to escape 0% MATH baseline
        lr_scheduler_type="cosine",      # Smooth long-tail annealing
        warmup_ratio=0.05,               # Gentle ramp-up for high-variance GRPO gradients
        beta=0.01,                       # KL penalty protecting MMLU/ARC priors
        logging_steps=10,
        save_steps=100,
        save_total_limit=5,              # Keep last 5 checkpoints to save disk
        bf16=True,
        report_to="wandb",
        run_name="tinymath-1b-grpo",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        num_generations=8,               # G=8: minimum for stable advantage normalization
        max_prompt_length=256,
        max_completion_length=512,
        generation_kwargs=generation_kwargs,
    )

    # vLLM integration (optional — may not work on ROCm/MI300X)
    if use_vllm:
        logging.info("Enabling vLLM colocate mode for high-throughput generation")
        config_kwargs.update(
            use_vllm=True,
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=0.3,   # Conservative — PyTorch needs the rest
            vllm_enable_sleep_mode=True,        # Offload vLLM to CPU RAM during backward
        )
    else:
        logging.info("Using native HF generation (MI300X 192GB has ample headroom for G=8)")

    training_args = GRPOConfig(**config_kwargs)

    # Compile reward function pipeline (order: correctness, format, repetition)
    reward_functions = [
        correctness_reward_func,
        format_reward_func,
        repetition_penalty_func,
    ]
    logging.info(f"Reward functions: {[f.__name__ for f in reward_functions]}")

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_functions,
        processing_class=tokenizer,  # TRL >=0.17 uses processing_class, not tokenizer
    )

    # Execute GRPO training
    logging.info("=" * 60)
    logging.info("Starting GRPO training")
    logging.info(f"  Model: {model_path}")
    logging.info(f"  Dataset: {len(dataset)} examples")
    logging.info(f"  Group size (G): 8")
    logging.info(f"  KL penalty (β): 0.01")
    logging.info(f"  Learning rate: 5e-6 (cosine schedule)")
    logging.info(f"  vLLM: {'enabled' if use_vllm else 'disabled (native HF)'}")
    logging.info(f"  math_verify: {'available' if HAS_MATH_VERIFY else 'NOT available (string fallback)'}")
    logging.info("=" * 60)

    trainer.train()

    # Save final model and tokenizer
    final_path = f"{output_dir}/final"
    logging.info(f"Saving final model to {final_path}")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    logging.info("GRPO training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO training for TinyMathReason-1B")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to SFT model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./grpo_output",
                        help="Directory for checkpoints and final model")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit dataset size for staged execution (smoke/calibration)")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--use_vllm", action="store_true",
                        help="Enable vLLM colocate mode (requires vLLM with ROCm support)")
    args = parser.parse_args()

    train_grpo(
        model_path=args.model_path,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        num_epochs=args.num_train_epochs,
        use_vllm=args.use_vllm,
    )
