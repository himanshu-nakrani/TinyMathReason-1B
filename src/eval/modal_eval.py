import modal
import os

# 1. Define serverless image configuration
image = modal.Image.debian_slim().pip_install(
    "torch",
    "transformers",
    "accelerate"
)

# Initialize Modal App
app = modal.App("tinymath-base-eval")

# 2. Define the remote function running on a serverless GPU
# We mount our local converted model directory directly into the serverless container
@app.function(
    image=image,
    gpu="A10G",  # Lightweight serverless NVIDIA GPU
    mounts=[modal.Mount.from_local_dir("./hf_1b_model", remote_path="/model")]
)
def evaluate_base_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("🤖 Loading model and tokenizer on serverless A10G GPU...")
    tokenizer = AutoTokenizer.from_pretrained("/model")
    model = AutoModelForCausalLM.from_pretrained(
        "/model",
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    EVAL_PROBLEMS = [
        {"level": "Easy", "q": "What is 15% of 200?"},
        {"level": "Easy", "q": "Solve for x: 3x + 5 = 20"},
        {"level": "Easy", "q": "If a train travels 60 miles per hour, how far does it go in 2.5 hours?"},
        
        {"level": "Medium", "q": "Find the roots of the quadratic equation: x^2 - 5x + 6 = 0"},
        {"level": "Medium", "q": "What is the derivative of f(x) = 3x^4 - 2x^2 + x?"},
        {"level": "Medium", "q": "If A = {1, 2, 3} and B = {3, 4, 5}, what is the union of sets A and B?"},
        
        {"level": "Hard", "q": "Prove that the sum of the first n positive integers is n(n+1)/2."},
        {"level": "Hard", "q": "Evaluate the integral of x * e^x dx from 0 to 1."},
        {"level": "Hard", "q": "Find the eigenvalues of the matrix [[2, 1], [1, 2]]."}
    ]
    
    print("⚡ Running baseline math evaluation on GPU...")
    results = []
    
    for prob in EVAL_PROBLEMS:
        # Format input for base model (raw template)
        prompt = f"Below is a math problem. Solve it step-by-step.\nProblem: {prob['q']}\nSolution:"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.0,  # Greedy decoding for benchmark baseline
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id or 2
            )
            
        generated_len = outputs.shape[1] - inputs.input_ids.shape[1]
        response = tokenizer.decode(outputs[0][-generated_len:], skip_special_tokens=True)
        print(f"✅ Evaluated ({prob['level']}): {prob['q']}")
        
        results.append({
            "level": prob["level"],
            "question": prob["q"],
            "response": response.strip()
        })
        
    # Save a markdown report locally/remotely and print it
    print("\n" + "="*50)
    print("             BASE MODEL EVALUATION REPORT")
    print("="*50 + "\n")
    
    markdown_content = "# Custom Math Evaluation: TinyMathReason-1B (Base Model Baseline)\n\n"
    for res in results:
        markdown_content += f"### Level: {res['level']}\n"
        markdown_content += f"**Q: {res['question']}**\n\n"
        markdown_content += f"**A:**\n{res['response']}\n\n"
        markdown_content += "---\n\n"
        
    print(markdown_content)
    
    # Save the output file in a mapped directory or output it
    return markdown_content

@app.local_entrypoint()
def main():
    report = evaluate_base_model.remote()
    
    # Write the report to local disk
    os.makedirs("./eval_results", exist_ok=True)
    out_path = "./eval_results/base_modal_eval.md"
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\n📂 Saved local markdown report to: {out_path}")
