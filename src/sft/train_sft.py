import logging
import argparse
from pathlib import Path
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from trl import SFTTrainer
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def train_sft(config_path: str):
    """
    Supervised Fine-Tuning using TRL's SFTTrainer.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    logging.info(f"Loading HF model from {config['model_path']}...")
    model = AutoModelForCausalLM.from_pretrained(
        config['model_path'],
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if config.get('use_flash_attn', True) else "sdpa"
    )
    
    logging.info(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_path'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    logging.info(f"Loading dataset from {config['dataset_path']}...")
    dataset = load_from_disk(config['dataset_path'])
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    
    # Define formatting function for SFTTrainer
    # Since we saved data as "messages", we apply the chat template
    def format_prompts(example):
        texts = []
        for msgs in example['messages']:
            texts.append(tokenizer.apply_chat_template(msgs, tokenize=False))
        return {'text': texts}
        
    # Map the dataset to apply chat template natively
    train_dataset = train_dataset.map(format_prompts, batched=True)
    eval_dataset = eval_dataset.map(format_prompts, batched=True)

    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        num_train_epochs=config['epochs'],
        per_device_train_batch_size=config['per_device_train_batch_size'],
        per_device_eval_batch_size=config['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=float(config['learning_rate']),
        weight_decay=config['weight_decay'],
        warmup_ratio=config['warmup_ratio'],
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        bf16=True,
        report_to="wandb" if config.get('enable_wandb', True) else "none",
        run_name="tinymath-1b-sft",
        deepspeed=config.get('deepspeed_config', None),
        gradient_checkpointing=True,
    )
    
    # We use DataCollatorForSeq2Seq to ensure proper padding
    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=config['max_seq_length'],
        data_collator=data_collator,
    )
    
    logging.info("Starting SFT training...")
    trainer.train()
    
    logging.info(f"Saving final model to {config['output_dir']}/final")
    trainer.save_model(f"{config['output_dir']}/final")
    logging.info("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="sft_config.yaml")
    args = parser.parse_args()
    train_sft(args.config)
