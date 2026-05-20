import argparse
import logging
import sys
from datasets import load_dataset, Dataset

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def format_chatml(system: str, user: str, assistant: str) -> list:
    """Formats into ChatML message format."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant}
    ]

def extract_cot_and_answer(response: str):
    """
    Parses a math response into reasoning (CoT) and final answer.
    Splits at typical math explanation boundaries to separate thought steps from the final answer.
    """
    # Try splitting by typical final answer prefixes
    prefixes = [
        "The answer is", "Therefore, the answer is", "Hence, the answer is",
        "Thus, the answer is", "the answer is", "Therefore, the final answer is",
        "so the answer is"
    ]
    for prefix in prefixes:
        if prefix in response:
            parts = response.split(prefix, 1)
            reasoning = parts[0].strip()
            answer = prefix + parts[1]
            return reasoning, answer
            
    # Fallback: if no typical prefix is found, split by the last sentence
    sentences = response.split(". ")
    if len(sentences) > 1:
        reasoning = ". ".join(sentences[:-1]).strip() + "."
        answer = sentences[-1].strip()
        return reasoning, answer
        
    return "", response

def prepare_stage1_data(output_dir: str):
    """
    Prepares Stage 1 SFT data (Conversational Prior) using tatsu-lab/alpaca.
    Saves in ChatML format without CoT to train standard dialogue capability.
    """
    logging.info("Preparing Stage 1 SFT data (Conversational Prior)...")
    system_prompt = "You are a helpful and polite mathematical assistant. Provide clear, direct answers to the user's instructions."
    all_examples = []
    
    try:
        logging.info("Loading tatsu-lab/alpaca dataset...")
        alpaca = load_dataset("tatsu-lab/alpaca", split="train")
        for row in alpaca:
            user_content = row["instruction"]
            if row["input"]:
                user_content += "\n" + row["input"]
            all_examples.append({
                "messages": format_chatml(system_prompt, user_content, row["output"])
            })
    except Exception as e:
        logging.error(f"Failed to load or process Alpaca: {e}")
        sys.exit(1)
        
    save_dataset(all_examples, output_dir)

def prepare_stage2_data(output_dir: str):
    """
    Prepares Stage 2 SFT data (Reasoning Traces) using MathInstruct, MetaMathQA, and GSM8K.
    Saves in ChatML format with reasoning wrapped in <think> and </think> tags.
    """
    logging.info("Preparing Stage 2 SFT data (Reasoning Traces with <think>)...")
    system_prompt = "You are a mathematical reasoning assistant. Solve problems step by step inside <think> tags, and then provide the final answer."
    all_examples = []
    
    # 1. GSM8K (7.5k)
    logging.info("Processing GSM8K...")
    try:
        gsm8k = load_dataset("gsm8k", "main", split="train")
        for row in gsm8k:
            answer_field = row["answer"]
            if "####" in answer_field:
                parts = answer_field.split("####", 1)
                reasoning = parts[0].strip()
                answer = "The answer is " + parts[1].strip() + "."
            else:
                reasoning, answer = extract_cot_and_answer(answer_field)
                
            assistant_content = f"<think>\n{reasoning}\n</think>\n{answer}"
            all_examples.append({
                "messages": format_chatml(system_prompt, row["question"], assistant_content)
            })
    except Exception as e:
        logging.error(f"Failed to load GSM8K: {e}")
        
    # 2. MathInstruct (260k)
    logging.info("Processing MathInstruct...")
    try:
        math_instruct = load_dataset("TIGER-Lab/MathInstruct", split="train")
        for row in math_instruct:
            reasoning, answer = extract_cot_and_answer(row["output"])
            if reasoning:
                assistant_content = f"<think>\n{reasoning}\n</think>\n{answer}"
            else:
                assistant_content = answer
            all_examples.append({
                "messages": format_chatml(system_prompt, row["instruction"], assistant_content)
            })
    except Exception as e:
        logging.error(f"Failed to load MathInstruct: {e}")
        
    # 3. MetaMathQA (395k)
    logging.info("Processing MetaMathQA...")
    try:
        metamath = load_dataset("meta-math/MetaMathQA", split="train")
        for row in metamath:
            reasoning, answer = extract_cot_and_answer(row["response"])
            if reasoning:
                assistant_content = f"<think>\n{reasoning}\n</think>\n{answer}"
            else:
                assistant_content = answer
            all_examples.append({
                "messages": format_chatml(system_prompt, row["query"], assistant_content)
            })
    except Exception as e:
        logging.error(f"Failed to load MetaMathQA: {e}")
        
    save_dataset(all_examples, output_dir)

def save_dataset(examples, output_dir):
    logging.info(f"Total processed examples: {len(examples)}")
    if not examples:
        logging.error("No examples to save!")
        sys.exit(1)
        
    # Convert to HuggingFace dataset
    final_ds = Dataset.from_list(examples)
    
    # Split into train/val (99/1)
    split_ds = final_ds.train_test_split(test_size=0.01, seed=42)
    logging.info(f"Train size: {len(split_ds['train'])}, Val size: {len(split_ds['test'])}")
    
    logging.info(f"Saving dataset to disk at {output_dir}...")
    split_ds.save_to_disk(output_dir)
    logging.info("Dataset preparation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SFT datasets for Stage 1 or Stage 2")
    parser.add_argument(
        "--stage", 
        type=int, 
        choices=[1, 2], 
        required=True, 
        help="SFT Curriculum Stage (1: Conversational Prior, 2: Reasoning Traces)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Directory to save the formatted dataset"
    )
    args = parser.parse_args()
    
    if args.stage == 1:
        prepare_stage1_data(args.output_dir)
    else:
        prepare_stage2_data(args.output_dir)
