import logging
import argparse
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def train_dpo(model_path: str, dataset_path: str, output_dir: str):
    """
    Direct Preference Optimization (DPO) training.
    """
    logging.info(f"Loading SFT model as policy from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    
    # We load the reference model. DPOTrainer handles keeping it in eval mode and no_grad
    logging.info(f"Loading SFT model as reference from {model_path}...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    logging.info(f"Loading DPO dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    
    # DPOTrainer expects format: prompt, chosen, rejected (as lists of messages or strings)
    # Our generate_preferences script outputs them as raw strings (since we already applied chat template to prompt)
    # We must ensure the chosen/rejected are just the assistant completions.
    
    def format_dpo(example):
        return {
            "prompt": example["prompt"],
            "chosen": example["chosen"] + tokenizer.eos_token,
            "rejected": example["rejected"] + tokenizer.eos_token,
        }
        
    dataset = dataset.map(format_dpo)
    split_ds = dataset.train_test_split(test_size=0.05, seed=42)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2, # DPO is memory heavy (runs 2 models)
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16, # Effective batch size 32
        learning_rate=5e-7, # DPO LR is much lower than SFT
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=500,
        bf16=True,
        report_to="wandb",
        run_name="tinymath-1b-dpo",
        remove_unused_columns=False, # Required for DPO
        gradient_checkpointing=True,
    )
    
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        beta=0.1, # KL penalty coefficient
        train_dataset=split_ds["train"],
        eval_dataset=split_ds["test"],
        tokenizer=tokenizer,
        max_prompt_length=1024,
        max_length=2048,
    )
    
    logging.info("Starting DPO training...")
    trainer.train()
    
    logging.info(f"Saving final model to {output_dir}/final")
    trainer.save_model(f"{output_dir}/final")
    logging.info("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="./dpo_data")
    parser.add_argument("--output_dir", type=str, default="./dpo_output")
    args = parser.parse_args()
    
    train_dpo(args.model_path, args.dataset_path, args.output_dir)
