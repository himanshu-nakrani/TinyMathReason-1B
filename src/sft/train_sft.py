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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_sft(
    config_path: str,
    override_model_path: str = None,
    override_dataset_path: str = None,
    override_output_dir: str = None,
    resize_token_embeddings: bool = False
):
    """
    Supervised Fine-Tuning using TRL's SFTTrainer, supporting command-line overrides.
    """
    # 1. Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Apply overrides if provided
    model_path = override_model_path or config.get('model_path')
    dataset_path = override_dataset_path or config.get('dataset_path')
    output_dir = override_output_dir or config.get('output_dir')
    
    logging.info(f"Final Model Path: {model_path}")
    logging.info(f"Final Dataset Path: {dataset_path}")
    logging.info(f"Final Output Directory: {output_dir}")
    
    # 2. Load tokenizer
    logging.info(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Programmatically inject ChatML template if not set (base models)
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

    # 3. Load model
    logging.info(f"Loading HF model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if config.get('use_flash_attn', True) else "sdpa"
    )
    
    # 4. Handle token resizing for <think> and </think>
    if resize_token_embeddings:
        logging.info("Resizing token embeddings to include <think> and </think> tags...")
        special_tokens_dict = {'additional_special_tokens': ['<think>', '</think>']}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        logging.info(f"Added {num_added_toks} new special tokens.")
        if num_added_toks > 0:
            model.resize_token_embeddings(len(tokenizer))
            logging.info(f"New model embedding size: {len(tokenizer)}")
            
    # 5. Load dataset
    logging.info(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
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
    logging.info("Mapping dataset with ChatML template...")
    train_dataset = train_dataset.map(format_prompts, batched=True)
    eval_dataset = eval_dataset.map(format_prompts, batched=True)

    # 6. Setup training args
    # TRL 0.12.0+ uses SFTConfig instead of TrainingArguments to handle SFT params
    use_sft_config = False
    try:
        from trl import SFTConfig
        use_sft_config = True
        logging.info("SFTConfig is available. Using SFTConfig for training arguments...")
    except ImportError:
        logging.info("SFTConfig is not available. Using standard TrainingArguments...")

    if use_sft_config:
        training_args = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=config.get('epochs', 2),
            per_device_train_batch_size=config.get('per_device_train_batch_size', 4),
            per_device_eval_batch_size=config.get('per_device_eval_batch_size', 4),
            gradient_accumulation_steps=config.get('gradient_accumulation_steps', 8),
            learning_rate=float(config.get('learning_rate', 2e-5)),
            weight_decay=config.get('weight_decay', 0.1),
            warmup_ratio=config.get('warmup_ratio', 0.05),
            lr_scheduler_type="cosine",
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            bf16=True,
            report_to="wandb" if config.get('enable_wandb', True) else "none",
            run_name="tinymath-1b-sft",
            deepspeed=config.get('deepspeed_config', None),
            gradient_checkpointing=True,
            dataset_text_field="text",
            max_seq_length=config.get('max_seq_length', 4096),
        )
    else:
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config.get('epochs', 2),
            per_device_train_batch_size=config.get('per_device_train_batch_size', 4),
            per_device_eval_batch_size=config.get('per_device_eval_batch_size', 4),
            gradient_accumulation_steps=config.get('gradient_accumulation_steps', 8),
            learning_rate=float(config.get('learning_rate', 2e-5)),
            weight_decay=config.get('weight_decay', 0.1),
            warmup_ratio=config.get('warmup_ratio', 0.05),
            lr_scheduler_type="cosine",
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            bf16=True,
            report_to="wandb" if config.get('enable_wandb', True) else "none",
            run_name="tinymath-1b-sft",
            deepspeed=config.get('deepspeed_config', None),
            gradient_checkpointing=True,
        )
    
    # Use DataCollatorForSeq2Seq to ensure proper padding
    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)

    # 7. SFT Trainer
    import inspect
    sig = inspect.signature(SFTTrainer.__init__)
    
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }
    
    # Add SFT-specific params directly to SFTTrainer only if not using SFTConfig
    if not use_sft_config:
        trainer_kwargs["dataset_text_field"] = "text"
        trainer_kwargs["max_seq_length"] = config.get('max_seq_length', 4096)
    
    if "processing_class" in sig.parameters:
        logging.info("SFTTrainer expects 'processing_class'. Passing tokenizer as processing_class.")
        trainer_kwargs["processing_class"] = tokenizer
    else:
        logging.info("SFTTrainer expects 'tokenizer'. Passing tokenizer directly.")
        trainer_kwargs["tokenizer"] = tokenizer
        
    trainer = SFTTrainer(**trainer_kwargs)
    
    logging.info("Starting SFT training...")
    trainer.train()
    
    final_output_path = Path(output_dir) / "final"
    logging.info(f"Saving final model and tokenizer to {final_output_path}...")
    trainer.save_model(str(final_output_path))
    logging.info("SFT Training complete! 🚀")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning for TinyMathReason-1B")
    parser.add_argument("--config", type=str, default="sft_config.yaml", help="Path to SFT configuration YAML file")
    parser.add_argument("--model_path", type=str, default=None, help="Override base model path")
    parser.add_argument("--dataset_path", type=str, default=None, help="Override SFT dataset path")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument(
        "--resize_token_embeddings", 
        action="store_true", 
        help="Resize token embeddings to accommodate special reasoning tags (<think>, </think>)"
    )
    args = parser.parse_args()
    
    train_sft(
        config_path=args.config,
        override_model_path=args.model_path,
        override_dataset_path=args.dataset_path,
        override_output_dir=args.output_dir,
        resize_token_embeddings=args.resize_token_embeddings
    )
