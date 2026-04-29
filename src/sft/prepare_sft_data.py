import argparse
import logging
from datasets import load_dataset, concatenate_datasets, Dataset
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def format_chatml(system: str, user: str, assistant: str) -> list:
    """Formats into ChatML message format."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant}
    ]

def prepare_sft_data(output_dir: str):
    """
    Downloads MathInstruct, MetaMathQA, and GSM8K, formats them into a unified
    ChatML representation, and saves the final HuggingFace Dataset.
    """
    system_prompt = "You are a mathematical reasoning assistant. Solve problems step by step, showing all work clearly."
    
    all_examples = []
    
    # 1. MathInstruct (260k)
    logging.info("Processing MathInstruct...")
    try:
        math_instruct = load_dataset("TIGER-Lab/MathInstruct", split="train")
        for row in math_instruct:
            all_examples.append({
                "messages": format_chatml(system_prompt, row["instruction"], row["output"])
            })
    except Exception as e:
        logging.error(f"Failed to load MathInstruct: {e}")

    # 2. MetaMathQA (395k)
    logging.info("Processing MetaMathQA...")
    try:
        metamath = load_dataset("meta-math/MetaMathQA", split="train")
        for row in metamath:
            all_examples.append({
                "messages": format_chatml(system_prompt, row["query"], row["response"])
            })
    except Exception as e:
        logging.error(f"Failed to load MetaMathQA: {e}")

    # 3. GSM8K (7.5k)
    logging.info("Processing GSM8K...")
    try:
        gsm8k = load_dataset("gsm8k", "main", split="train")
        for row in gsm8k:
            all_examples.append({
                "messages": format_chatml(system_prompt, row["question"], row["answer"])
            })
    except Exception as e:
        logging.error(f"Failed to load GSM8K: {e}")

    logging.info(f"Total SFT examples: {len(all_examples)}")
    
    # Convert to HuggingFace dataset
    final_ds = Dataset.from_list(all_examples)
    
    # Split into train/val (99/1)
    # We use a fixed seed for reproducibility
    split_ds = final_ds.train_test_split(test_size=0.01, seed=42)
    
    logging.info(f"Train size: {len(split_ds['train'])}, Val size: {len(split_ds['test'])}")
    
    logging.info(f"Saving to {output_dir}...")
    split_ds.save_to_disk(output_dir)
    logging.info("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./sft_data")
    args = parser.parse_args()
    
    prepare_sft_data(args.output_dir)
