#!/bin/bash
# ============================================================================
# eval_on_vm.sh — Runs on the GCP VM
#
# Installs dependencies, downloads model from HF Hub, runs the full
# lm-evaluation-harness benchmark suite, uploads results to GCS.
#
# Usage (called by run_eval_gcp.sh):
#   bash eval_on_vm.sh <HF_TOKEN> <HF_REPO_ID> <GCS_OUTPUT_PATH>
# ============================================================================
set -euo pipefail

HF_TOKEN="${1:?Usage: eval_on_vm.sh <HF_TOKEN> <HF_REPO_ID> <GCS_OUTPUT_PATH>}"
HF_REPO_ID="${2:?Usage: eval_on_vm.sh <HF_TOKEN> <HF_REPO_ID> <GCS_OUTPUT_PATH>}"
GCS_OUTPUT_PATH="${3:?Usage: eval_on_vm.sh <HF_TOKEN> <HF_REPO_ID> <GCS_OUTPUT_PATH>}"

MODEL_DIR="/tmp/hf_model"
RESULTS_DIR="/tmp/eval_results"
SUMMARY_FILE="${RESULTS_DIR}/benchmark_summary.md"

echo "============================================================"
echo " TinyMathReason-1B — Base Model Evaluation"
echo " Repo: ${HF_REPO_ID}"
echo " GCS Output: ${GCS_OUTPUT_PATH}"
echo " Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"

# -------------------------------------------------------
# 1. Install dependencies
# -------------------------------------------------------
echo ""
echo ">>> [1/5] Installing dependencies..."
pip install --break-system-packages --quiet --upgrade pip
pip install --break-system-packages --quiet \
    torch \
    transformers \
    accelerate \
    huggingface_hub \
    lm-eval==0.4.2 \
    sentencepiece \
    antlr4-python3-runtime==4.11 \
    protobuf

echo "    ✓ Dependencies installed"
echo "    Python: $(python3 --version)"
echo "    PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "    CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"

# Check GPU
if python3 -c 'import torch; assert torch.cuda.is_available()'; then
    DEVICE="cuda"
    echo "    GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
    echo "    VRAM: $(python3 -c 'import torch; print(f\"{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB\")')"
else
    DEVICE="cpu"
    echo "    WARNING: No GPU detected, falling back to CPU (will be slow)"
fi

# -------------------------------------------------------
# 2. Download model from HF Hub
# -------------------------------------------------------
echo ""
echo ">>> [2/5] Downloading model from HF Hub..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${HF_REPO_ID}',
    local_dir='${MODEL_DIR}',
    token='${HF_TOKEN}'
)
print('    ✓ Model downloaded to ${MODEL_DIR}')
"

echo "    Files:"
ls -lh "${MODEL_DIR}/"

# -------------------------------------------------------
# 3. Verify model loads correctly
# -------------------------------------------------------
echo ""
echo ">>> [3/5] Verifying model loads..."
python3 -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('${MODEL_DIR}')
print(f'    Tokenizer vocab size: {tokenizer.vocab_size}')

model = AutoModelForCausalLM.from_pretrained(
    '${MODEL_DIR}',
    torch_dtype=torch.bfloat16,
    device_map='${DEVICE}'
)
total_params = sum(p.numel() for p in model.parameters())
print(f'    Model parameters: {total_params / 1e9:.3f}B')
print(f'    Model dtype: {next(model.parameters()).dtype}')
print(f'    Model device: {next(model.parameters()).device}')

# Quick sanity check - generate a few tokens
inputs = tokenizer('2 + 2 =', return_tensors='pt').to(model.device)
outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f'    Sanity check: \"2 + 2 =\" → \"{generated}\"')
print('    ✓ Model verified')
del model
if '${DEVICE}' == 'cuda':
    torch.cuda.empty_cache()
"

# -------------------------------------------------------
# 4. Run benchmarks
# -------------------------------------------------------
echo ""
echo ">>> [4/5] Running benchmarks..."
mkdir -p "${RESULTS_DIR}"

# Define benchmark tasks
# Format: TASK_NAME:NUM_FEWSHOT
declare -a BENCHMARKS=(
    "gsm8k:8"
    "minerva_math_algebra:4"
    "arc_easy:0"
    "arc_challenge:25"
    "hellaswag:10"
    "mmlu:5"
)

# Run all benchmarks in a single lm_eval call for efficiency
# (shared model loading, single GPU allocation)
TASK_LIST=""
for benchmark in "${BENCHMARKS[@]}"; do
    TASK_NAME="${benchmark%%:*}"
    if [ -z "${TASK_LIST}" ]; then
        TASK_LIST="${TASK_NAME}"
    else
        TASK_LIST="${TASK_LIST},${TASK_NAME}"
    fi
done

echo "    Tasks: ${TASK_LIST}"
echo "    This will take approximately 1-2 hours..."
echo ""

# Run lm_eval with all tasks
# Note: num_fewshot is set per-task via the task config defaults when possible,
# but we run them separately for precise shot control
for benchmark in "${BENCHMARKS[@]}"; do
    TASK_NAME="${benchmark%%:*}"
    NUM_SHOTS="${benchmark##*:}"
    
    echo "    ── Running: ${TASK_NAME} (${NUM_SHOTS}-shot) ──"
    
    python3 -m lm_eval \
        --model hf \
        --model_args "pretrained=${MODEL_DIR},dtype=bfloat16" \
        --tasks "${TASK_NAME}" \
        --num_fewshot "${NUM_SHOTS}" \
        --batch_size auto \
        --device "${DEVICE}" \
        --output_path "${RESULTS_DIR}/${TASK_NAME}" \
        --log_samples \
        2>&1 | tee "${RESULTS_DIR}/${TASK_NAME}.log"
    
    echo "    ✓ ${TASK_NAME} complete"
    echo ""
done

echo "    ✓ All benchmarks complete"

# -------------------------------------------------------
# 5. Generate summary and upload results
# -------------------------------------------------------
echo ""
echo ">>> [5/5] Generating summary and uploading results..."

# Parse results and create summary
python3 << 'PYTHON_SCRIPT'
import json
import os
import glob
from datetime import datetime

results_dir = os.environ.get("RESULTS_DIR", "/tmp/eval_results")
summary_file = os.environ.get("SUMMARY_FILE", f"{results_dir}/benchmark_summary.md")

benchmarks = {
    "gsm8k": {"shots": 8, "metric": "exact_match,strict-match"},
    "minerva_math_algebra": {"shots": 4, "metric": "exact_match,strict-match"},
    "arc_easy": {"shots": 0, "metric": "acc_norm,none"},
    "arc_challenge": {"shots": 25, "metric": "acc_norm,none"},
    "hellaswag": {"shots": 10, "metric": "acc_norm,none"},
    "mmlu": {"shots": 5, "metric": "acc,none"},
}

results = {}
for task_name, config in benchmarks.items():
    # lm_eval saves results in a subdirectory with a timestamp
    result_files = glob.glob(f"{results_dir}/{task_name}/**/results.json", recursive=True)
    if not result_files:
        results[task_name] = "Error: No results file found"
        continue
    
    # Use the most recent result file
    result_file = sorted(result_files)[-1]
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        task_results = data.get("results", {})
        # Try to find the task results (might be under different keys)
        score = None
        for key in [task_name, task_name.replace("_", " ")]:
            if key in task_results:
                task_data = task_results[key]
                # Try multiple metric names
                for metric_name in config["metric"].split(","):
                    metric_key = f"{metric_name.strip()},none"
                    if metric_key in task_data:
                        score = task_data[metric_key]
                        break
                    # Also try without ",none"
                    if metric_name.strip() in task_data:
                        score = task_data[metric_name.strip()]
                        break
                if score is not None:
                    break
        
        if score is None:
            # Fallback: grab any numeric metric
            for key, task_data in task_results.items():
                if isinstance(task_data, dict):
                    for mk, mv in task_data.items():
                        if isinstance(mv, (int, float)) and not mk.startswith("alias"):
                            score = mv
                            break
                    if score is not None:
                        break
        
        results[task_name] = score if score is not None else "Parse error"
    except Exception as e:
        results[task_name] = f"Error: {e}"

# Write summary
with open(summary_file, 'w') as f:
    f.write(f"# TinyMathReason-1B Base Model — Benchmark Results\n\n")
    f.write(f"**Date:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n")
    f.write(f"**Model:** himanshu-nakrani/TinyMathReason-1B-base\n\n")
    f.write("| Benchmark | Shots | Score |\n")
    f.write("| :--- | :---: | :---: |\n")
    for task_name, config in benchmarks.items():
        score = results.get(task_name, "N/A")
        if isinstance(score, float):
            score_str = f"{score*100:.2f}%"
        else:
            score_str = str(score)
        f.write(f"| {task_name} | {config['shots']} | {score_str} |\n")
    f.write(f"\n*Evaluated using lm-evaluation-harness on NVIDIA L4 GPU with bfloat16 precision.*\n")

# Print summary to stdout
with open(summary_file, 'r') as f:
    print(f.read())

# Also save raw results as a single JSON
combined = {"results": results, "timestamp": datetime.utcnow().isoformat()}
with open(f"{results_dir}/combined_results.json", 'w') as f:
    json.dump(combined, f, indent=2, default=str)

print(f"\nResults saved to {results_dir}/")
PYTHON_SCRIPT

export RESULTS_DIR SUMMARY_FILE

# Upload to GCS
echo ""
echo "    Uploading results to GCS: ${GCS_OUTPUT_PATH}"
gsutil -m cp -r "${RESULTS_DIR}/"* "${GCS_OUTPUT_PATH}/"
echo "    ✓ Results uploaded to GCS"

echo ""
echo "============================================================"
echo " EVALUATION COMPLETE"
echo " Results: ${GCS_OUTPUT_PATH}/"
echo " Finished: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"
