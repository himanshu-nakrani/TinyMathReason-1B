#!/bin/bash
# ============================================================================
# TinyMathReason-1B: Upload SFT Model & Run Evaluation Benchmarks
# ============================================================================
set -e

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
log_info()  { echo "[INFO] $TIMESTAMP - $1"; }
log_error() { echo "[ERROR] $TIMESTAMP - $1"; }

MODEL_DIR="src/sft/sft_output/stage2/final"
HF_REPO="himanshunakrani9/TinyMathReason-1B-sft"

# ---- Verify Model Exists ----
if [ ! -d "$MODEL_DIR" ]; then
    log_error "Model directory not found: $MODEL_DIR"
    log_error "Ensure SFT Stage 2 has completed successfully."
    exit 1
fi

log_info "Found trained model at: $MODEL_DIR"
ls -lh "$MODEL_DIR"

echo ""
echo "=============================================="
echo " TinyMathReason-1B Post-SFT Pipeline"
echo "=============================================="
echo "1. Upload model to Hugging Face Hub"
echo "2. Run benchmark evaluations (GSM8K, MATH, ARC, HellaSwag, MMLU)"
echo "3. Run BOTH (Upload then Evaluate)"
echo "4. Quick inference test (generate sample output)"
echo "5. Exit"
echo "Please select an option (1-5): "
read -r choice

# ============================================================================
# FUNCTION: Upload to Hugging Face Hub
# ============================================================================
upload_model() {
    log_info "Uploading model to Hugging Face Hub: $HF_REPO"
    
    # Check if logged in
    if ! huggingface-cli whoami &>/dev/null; then
        log_info "Not logged into Hugging Face. Please log in:"
        huggingface-cli login
    fi
    
    python3 -c "
from huggingface_hub import HfApi
import os

api = HfApi()

# Create repo if it doesn't exist
try:
    api.create_repo(repo_id='$HF_REPO', repo_type='model', exist_ok=True)
    print(f'Repository $HF_REPO is ready.')
except Exception as e:
    print(f'Repo creation note: {e}')

# Upload the entire model folder
print('Uploading model files...')
api.upload_folder(
    folder_path='$MODEL_DIR',
    repo_id='$HF_REPO',
    repo_type='model',
    commit_message='Upload TinyMathReason-1B SFT model (Stage 1 Chat + Stage 2 Reasoning)',
)
print('✅ Model uploaded successfully to https://huggingface.co/$HF_REPO')
"
    log_info "Upload complete!"
}

# ============================================================================
# FUNCTION: Run Benchmark Evaluations
# ============================================================================
run_benchmarks() {
    log_info "Installing lm-eval harness..."
    pip install lm-eval --quiet 2>/dev/null || pip install lm-eval
    
    EVAL_OUTPUT_DIR="eval_results/sft_model"
    mkdir -p "$EVAL_OUTPUT_DIR"
    
    log_info "Running benchmark evaluations on SFT model..."
    log_info "This will take approximately 30-60 minutes."
    echo ""
    
    # ---- GSM8K (8-shot) ----
    log_info "=== Running GSM8K (8-shot) ==="
    lm_eval --model hf \
        --model_args "pretrained=$MODEL_DIR,dtype=bfloat16,attn_implementation=flash_attention_2" \
        --tasks gsm8k \
        --num_fewshot 8 \
        --batch_size 256 \
        --output_path "$EVAL_OUTPUT_DIR/gsm8k" \
        2>&1 | tee "$EVAL_OUTPUT_DIR/gsm8k.log"
    echo ""
    
    # ---- MATH (Algebra, 4-shot) ----
    log_info "=== Running MATH Algebra (4-shot) ==="
    lm_eval --model hf \
        --model_args "pretrained=$MODEL_DIR,dtype=bfloat16,attn_implementation=flash_attention_2" \
        --tasks minerva_math_algebra \
        --num_fewshot 4 \
        --batch_size 256 \
        --output_path "$EVAL_OUTPUT_DIR/math_algebra" \
        2>&1 | tee "$EVAL_OUTPUT_DIR/math_algebra.log"
    echo ""
    
    # ---- ARC-Easy (0-shot) ----
    log_info "=== Running ARC-Easy (0-shot) ==="
    lm_eval --model hf \
        --model_args "pretrained=$MODEL_DIR,dtype=bfloat16,attn_implementation=flash_attention_2" \
        --tasks arc_easy \
        --num_fewshot 0 \
        --batch_size 256 \
        --output_path "$EVAL_OUTPUT_DIR/arc_easy" \
        2>&1 | tee "$EVAL_OUTPUT_DIR/arc_easy.log"
    echo ""
    
    # ---- ARC-Challenge (25-shot) ----
    log_info "=== Running ARC-Challenge (25-shot) ==="
    lm_eval --model hf \
        --model_args "pretrained=$MODEL_DIR,dtype=bfloat16,attn_implementation=flash_attention_2" \
        --tasks arc_challenge \
        --num_fewshot 25 \
        --batch_size 256 \
        --output_path "$EVAL_OUTPUT_DIR/arc_challenge" \
        2>&1 | tee "$EVAL_OUTPUT_DIR/arc_challenge.log"
    echo ""
    
    # ---- HellaSwag (10-shot) ----
    log_info "=== Running HellaSwag (10-shot) ==="
    lm_eval --model hf \
        --model_args "pretrained=$MODEL_DIR,dtype=bfloat16,attn_implementation=flash_attention_2" \
        --tasks hellaswag \
        --num_fewshot 10 \
        --batch_size 256 \
        --output_path "$EVAL_OUTPUT_DIR/hellaswag" \
        2>&1 | tee "$EVAL_OUTPUT_DIR/hellaswag.log"
    echo ""
    
    # ---- MMLU (5-shot) ----
    log_info "=== Running MMLU (5-shot) ==="
    lm_eval --model hf \
        --model_args "pretrained=$MODEL_DIR,dtype=bfloat16,attn_implementation=flash_attention_2" \
        --tasks mmlu \
        --num_fewshot 5 \
        --batch_size 256 \
        --output_path "$EVAL_OUTPUT_DIR/mmlu" \
        2>&1 | tee "$EVAL_OUTPUT_DIR/mmlu.log"
    echo ""
    
    log_info "=============================================="
    log_info "All benchmarks complete! Results saved to: $EVAL_OUTPUT_DIR/"
    log_info "=============================================="
    
    # Print summary
    echo ""
    echo "📊 Benchmark Results Summary"
    echo "=============================="
    echo "Check individual result files in: $EVAL_OUTPUT_DIR/"
    echo ""
    echo "Base Model Baselines (for comparison):"
    echo "  GSM8K (8-shot):          1.0%  Exact Match"
    echo "  MATH Algebra (4-shot):   0.0%  Exact Match"
    echo "  ARC-Easy (0-shot):      29.9%  Accuracy (Norm)"
    echo "  ARC-Challenge (25-shot): 21.7% Accuracy (Norm)"
    echo "  HellaSwag (10-shot):    25.8%  Accuracy (Norm)"
    echo "  MMLU (5-shot):          23.5%  Accuracy"
    echo ""
}

# ============================================================================
# FUNCTION: Quick Inference Test
# ============================================================================
quick_test() {
    log_info "Running quick inference test..."
    python3 -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print('Loading model...')
tokenizer = AutoTokenizer.from_pretrained('$MODEL_DIR')
model = AutoModelForCausalLM.from_pretrained(
    '$MODEL_DIR',
    torch_dtype=torch.bfloat16,
    device_map='auto'
)

# Test prompts
prompts = [
    'What is 15 + 27?',
    'Solve for x: 2x + 5 = 13',
    'A store sells apples for \$2 each. If John buys 5 apples and pays with a \$20 bill, how much change does he get?',
]

print('\n' + '='*60)
print('TinyMathReason-1B SFT Model — Inference Test')
print('='*60)

for prompt in prompts:
    messages = [
        {'role': 'system', 'content': 'You are a mathematical reasoning assistant. Solve problems step by step inside <think> tags, and then provide the final answer.'},
        {'role': 'user', 'content': prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f'\n📝 Prompt: {prompt}')
    print(f'🤖 Response: {response}')
    print('-'*60)

print('\n✅ Inference test complete!')
"
}

# ============================================================================
# Main Execution
# ============================================================================
case $choice in
    1) upload_model ;;
    2) run_benchmarks ;;
    3) upload_model && run_benchmarks ;;
    4) quick_test ;;
    5) echo "Exiting."; exit 0 ;;
    *) echo "Invalid option."; exit 1 ;;
esac
