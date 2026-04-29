import argparse
import logging
import re
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
# In recent versions of TRL (April 2026), GRPOTrainer is available
try:
    from trl import GRPOTrainer, GRPOTrainingArguments
except ImportError:
    logging.error("GRPOTrainer not found. Ensure you have the latest TRL version.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def extract_answer(text: str) -> str:
    """Extracts answer for reward modeling."""
    if "####" in text:
        return text.split("####")[-1].strip()
    boxed_match = re.search(r'\\boxed{([^}]+)}', text)
    if boxed_match:
        return boxed_match.group(1).strip()
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]
    return ""

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """
    Reward function for GRPO.
    Returns 1.0 if correct, 0.0 if incorrect.
    """
    rewards = []
    for comp, gt in zip(completions, answer):
        pred = extract_answer(comp)
        truth = extract_answer(gt)
        if pred and truth and pred.replace(",", "").replace(" ", "") == truth.replace(",", "").replace(" ", ""):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def format_prompt(example):
    """Format for GRPOTrainer."""
    system_prompt = "You are a mathematical reasoning assistant. Solve problems step by step, showing all work clearly."
    example["prompt"] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example["question"]}
    ]
    # Keep the raw answer column for the reward function to use
    return example

def train_grpo(model_path: str, output_dir: str):
    """
    Trains using Group Relative Policy Optimization (GRPO),
    the RL method behind DeepSeek-R1.
    """
    logging.info(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    logging.info("Loading GSM8K train split for GRPO...")
    dataset = load_dataset("gsm8k", "main", split="train")
    dataset = dataset.map(format_prompt)
    
    training_args = GRPOTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1, # Very small per device, group size handles the variance
        gradient_accumulation_steps=8,
        learning_rate=1e-6,
        logging_steps=10,
        save_steps=100,
        bf16=True,
        report_to="wandb",
        run_name="tinymath-1b-grpo",
        gradient_checkpointing=True,
        num_generations=4, # Size of the group (G in GRPO paper)
        max_prompt_length=256,
        max_completion_length=512,
    )
    
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=[correctness_reward_func],
        tokenizer=tokenizer,
    )
    
    logging.info("Starting GRPO training...")
    trainer.train()
    
    logging.info(f"Saving final model to {output_dir}/final")
    trainer.save_model(f"{output_dir}/final")
    logging.info("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./grpo_output")
    args = parser.parse_args()
    
    train_grpo(args.model_path, args.output_dir)
