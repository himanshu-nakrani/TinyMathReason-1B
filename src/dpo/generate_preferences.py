import argparse
import logging
import torch
import re
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def extract_answer(text: str) -> str:
    """Basic extraction of the final answer from text."""
    # Often answers are after "####" in GSM8K or within \boxed{}
    if "####" in text:
        return text.split("####")[-1].strip()
    
    boxed_match = re.search(r'\\boxed{([^}]+)}', text)
    if boxed_match:
        return boxed_match.group(1).strip()
        
    # Fallback to the last number found
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]
    return ""

def check_correctness(prediction: str, ground_truth: str) -> bool:
    """Checks if predicted answer matches ground truth answer."""
    pred_ans = extract_answer(prediction)
    gt_ans = extract_answer(ground_truth)
    if not pred_ans or not gt_ans:
        return False
    # Simple string match after removing commas/spaces
    return pred_ans.replace(",", "").replace(" ", "") == gt_ans.replace(",", "").replace(" ", "")

def generate_preferences(model_path: str, output_path: str, num_problems: int = 10000):
    """
    Takes math problems, generates N candidates using temperature sampling,
    and constructs chosen/rejected pairs for DPO.
    """
    logging.info(f"Loading model and tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    logging.info("Loading GSM8K train split for generation...")
    dataset = load_dataset("gsm8k", "main", split="train")
    
    # Take a subset of problems
    subset = dataset.select(range(min(num_problems, len(dataset))))
    
    dpo_data = []
    
    logging.info("Generating candidate solutions...")
    system_prompt = "You are a mathematical reasoning assistant. Solve problems step by step, showing all work clearly."
    
    # Number of candidates per problem
    N_CANDIDATES = 4
    
    for row in tqdm(subset):
        question = row["question"]
        ground_truth = row["answer"]
        
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        candidates = []
        
        # Generate N candidates
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.8, # high temperature for diversity
                do_sample=True,
                num_return_sequences=N_CANDIDATES,
                pad_token_id=tokenizer.eos_token_id
            )
            
        for out in outputs:
            # Decode only the generated part
            generated_len = out.shape[0] - inputs.input_ids.shape[1]
            response = tokenizer.decode(out[-generated_len:], skip_special_tokens=True)
            
            is_correct = check_correctness(response, ground_truth)
            candidates.append({"response": response, "correct": is_correct})
            
        # We need at least one correct and one incorrect to form a pair
        correct_cands = [c for c in candidates if c["correct"]]
        incorrect_cands = [c for c in candidates if not c["correct"]]
        
        if correct_cands and incorrect_cands:
            # Pick the best correct and worst incorrect (here we just pick randomly)
            import random
            chosen = random.choice(correct_cands)["response"]
            rejected = random.choice(incorrect_cands)["response"]
            
            dpo_data.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "question": question
            })
            
    logging.info(f"Generated {len(dpo_data)} preference pairs.")
    
    out_ds = Dataset.from_list(dpo_data)
    out_ds.save_to_disk(output_path)
    logging.info(f"Saved DPO dataset to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to SFT model")
    parser.add_argument("--output_path", type=str, default="./dpo_data")
    parser.add_argument("--num_problems", type=int, default=5000)
    args = parser.parse_args()
    
    generate_preferences(args.model_path, args.output_path, args.num_problems)
