#!/bin/bash
set -euo pipefail

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
VM_NAME="tinymath-eval-1779295614"
ZONE="us-west4-a"
PROJECT_ID="gen-lang-client-0318750942"
MODEL_DIR="/tmp/hf_model"
RESULTS_DIR="/tmp/eval_results"
GCS_OUTPUT_PATH="gs://tinymath-reason-data-himanshu/eval_results/base_model"

echo ">>> Resuming Evaluation on ${VM_NAME}..."

# Define remaining benchmarks
declare -a REMAINING=(
    "hellaswag:10"
    "mmlu:5"
)

for benchmark in "${REMAINING[@]}"; do
    TASK_NAME="${benchmark%%:*}"
    NUM_SHOTS="${benchmark##*:}"
    
    echo "    ── Running: ${TASK_NAME} (${NUM_SHOTS}-shot) ──"
    
    python3 -m lm_eval \
        --model hf \
        --model_args "pretrained=${MODEL_DIR},dtype=bfloat16" \
        --tasks "${TASK_NAME}" \
        --num_fewshot "${NUM_SHOTS}" \
        --batch_size auto \
        --device cuda \
        --output_path "${RESULTS_DIR}/${TASK_NAME}" \
        --log_samples 2>&1 | tee "${RESULTS_DIR}/${TASK_NAME}_resume.log"
    
    echo "    ✓ ${TASK_NAME} complete"
done

echo ">>> Uploading results to GCS..."
gsutil -m cp -r "${RESULTS_DIR}/*" "${GCS_OUTPUT_PATH}/"

echo ">>> Deleting VM..."
gcloud compute instances delete "${VM_NAME}" --project="${PROJECT_ID}" --zone="${ZONE}" --quiet
