#!/bin/bash
# ============================================================================
# run_eval_gcp.sh — Orchestrator script (runs on your Mac)
#
# Provisions a GCP VM with NVIDIA L4, runs the full benchmark suite,
# collects results, and deletes the VM.
#
# Usage:
#   bash scripts/run_eval_gcp.sh
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - HF token available (via huggingface-cli login or HF_TOKEN env var)
#   - Model already uploaded to HF Hub
# ============================================================================
set -euo pipefail

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
PROJECT_ID="gen-lang-client-0318750942"
ZONE="us-west1-b"
VM_NAME="tinymath-eval-$(date +%s)"
MACHINE_TYPE="g2-standard-4"
IMAGE_FAMILY="pytorch-2-9-cu129-ubuntu-2404-nvidia-580"
IMAGE_PROJECT="deeplearning-platform-release"
BOOT_DISK_SIZE="100GB"
HF_REPO_ID="himanshunakrani9/TinyMathReason-1B-base"
GCS_OUTPUT_PATH="gs://tinymath-reason-data-himanshu/eval_results/base_model"
LOCAL_RESULTS_DIR="./eval_results"

# Get HF token
if [ -n "${HF_TOKEN:-}" ]; then
    echo "Using HF_TOKEN from environment variable."
elif [ -f "${HOME}/.cache/huggingface/token" ]; then
    HF_TOKEN=$(cat "${HOME}/.cache/huggingface/token")
    echo "Using HF token from ~/.cache/huggingface/token"
else
    echo "ERROR: No Hugging Face token found."
    echo "Run: huggingface-cli login"
    echo "Or set: export HF_TOKEN=hf_..."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_on_vm.sh"

if [ ! -f "${EVAL_SCRIPT}" ]; then
    echo "ERROR: eval_on_vm.sh not found at ${EVAL_SCRIPT}"
    exit 1
fi

echo "============================================================"
echo " TinyMathReason-1B — GCP Evaluation Pipeline"
echo "============================================================"
echo " VM Name:     ${VM_NAME}"
echo " Machine:     ${MACHINE_TYPE} (NVIDIA L4)"
echo " Zone:        ${ZONE}"
echo " Model:       ${HF_REPO_ID}"
echo " GCS Output:  ${GCS_OUTPUT_PATH}"
echo "============================================================"
echo ""

# -------------------------------------------------------
# 1. Create VM
# -------------------------------------------------------
echo ">>> [1/6] Creating GCP VM..."

ZONES=("us-east4-a" "us-east4-b" "us-east4-c" "us-east5-a" "us-west4-a" "us-west4-b" "us-west4-c" "us-central1-a" "us-central1-b" "us-central1-c" "us-east1-b" "us-east1-c" "us-east1-d" "us-west1-a" "us-west1-b")

SUCCESS=0
for try_zone in "${ZONES[@]}"; do
    echo "Trying to create VM in zone: ${try_zone}..."
    if gcloud compute instances create "${VM_NAME}" \
        --project="${PROJECT_ID}" \
        --zone="${try_zone}" \
        --machine-type="${MACHINE_TYPE}" \
        --accelerator="type=nvidia-l4,count=1" \
        --image-family="${IMAGE_FAMILY}" \
        --image-project="${IMAGE_PROJECT}" \
        --boot-disk-size="${BOOT_DISK_SIZE}" \
        --boot-disk-type="pd-ssd" \
        --scopes="storage-rw" \
        --maintenance-policy=TERMINATE \
        --metadata="install-nvidia-driver=True"; then
        
        echo "    ✓ VM created successfully in ${try_zone}: ${VM_NAME}"
        ZONE="${try_zone}"
        SUCCESS=1
        break
    else
        echo "    ✗ Failed in ${try_zone} (likely stockout). Trying next..."
    fi
done

if [ "$SUCCESS" -eq 0 ]; then
    echo "ERROR: Failed to create VM in all attempted zones due to stockouts."
    exit 1
fi

# -------------------------------------------------------
# 2. Wait for VM to be SSH-ready
# -------------------------------------------------------
echo ""
echo ">>> [2/6] Waiting for VM to be SSH-ready..."
echo "    (This may take 2-3 minutes for GPU driver installation)"

MAX_RETRIES=30
RETRY_INTERVAL=20
for i in $(seq 1 ${MAX_RETRIES}); do
    if gcloud compute ssh "${VM_NAME}" \
        --project="${PROJECT_ID}" \
        --zone="${ZONE}" \
        --command="echo 'SSH ready'" \
        --ssh-flag="-o ConnectTimeout=10" \
        --ssh-flag="-o StrictHostKeyChecking=no" \
        2>/dev/null; then
        echo "    ✓ VM is SSH-ready"
        break
    fi
    if [ "${i}" -eq "${MAX_RETRIES}" ]; then
        echo "    ERROR: VM failed to become SSH-ready after $((MAX_RETRIES * RETRY_INTERVAL))s"
        echo "    Deleting VM..."
        gcloud compute instances delete "${VM_NAME}" --project="${PROJECT_ID}" --zone="${ZONE}" --quiet
        exit 1
    fi
    echo "    Retry ${i}/${MAX_RETRIES}... waiting ${RETRY_INTERVAL}s"
    sleep "${RETRY_INTERVAL}"
done

# Wait a bit more for GPU driver to finish initializing
echo "    Waiting 30s for GPU driver initialization..."
sleep 30

# Verify GPU is accessible
echo "    Checking GPU..."
gcloud compute ssh "${VM_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --command="nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" \
    --ssh-flag="-o StrictHostKeyChecking=no" \
    2>/dev/null || echo "    WARNING: nvidia-smi check failed, proceeding anyway"

# -------------------------------------------------------
# 3. Copy eval script to VM
# -------------------------------------------------------
echo ""
echo ">>> [3/6] Copying eval script to VM..."

gcloud compute scp "${EVAL_SCRIPT}" \
    "${VM_NAME}:/tmp/eval_on_vm.sh" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --scp-flag="-o StrictHostKeyChecking=no"

gcloud compute ssh "${VM_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --command="chmod +x /tmp/eval_on_vm.sh" \
    --ssh-flag="-o StrictHostKeyChecking=no"

echo "    ✓ Eval script copied"

# -------------------------------------------------------
# 4. Run evaluation on VM
# -------------------------------------------------------
echo ""
echo ">>> [4/6] Running evaluation on VM..."
echo "    This will take approximately 1-2 hours."
echo "    Streaming output below:"
echo "    ────────────────────────────────────────"

gcloud compute ssh "${VM_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --ssh-flag="-o StrictHostKeyChecking=no" \
    --ssh-flag="-o ServerAliveInterval=60" \
    --ssh-flag="-o ServerAliveCountMax=120" \
    --command="bash /tmp/eval_on_vm.sh '${HF_TOKEN}' '${HF_REPO_ID}' '${GCS_OUTPUT_PATH}'"

echo "    ────────────────────────────────────────"
echo "    ✓ Evaluation complete"

# -------------------------------------------------------
# 5. Download results from GCS
# -------------------------------------------------------
echo ""
echo ">>> [5/6] Downloading results from GCS..."

mkdir -p "${LOCAL_RESULTS_DIR}"
gsutil -m cp -r "${GCS_OUTPUT_PATH}/*" "${LOCAL_RESULTS_DIR}/" 2>/dev/null || true

if [ -f "${LOCAL_RESULTS_DIR}/benchmark_summary.md" ]; then
    echo ""
    echo "    ═══════════════════════════════════════"
    cat "${LOCAL_RESULTS_DIR}/benchmark_summary.md"
    echo "    ═══════════════════════════════════════"
else
    echo "    WARNING: Summary file not found locally. Check GCS: ${GCS_OUTPUT_PATH}"
fi

echo "    ✓ Results saved to ${LOCAL_RESULTS_DIR}/"

# -------------------------------------------------------
# 6. Delete VM
# -------------------------------------------------------
echo ""
echo ">>> [6/6] Deleting VM to save costs..."

gcloud compute instances delete "${VM_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --quiet

echo "    ✓ VM deleted: ${VM_NAME}"

echo ""
echo "============================================================"
echo " PIPELINE COMPLETE"
echo ""
echo " Results:"
echo "   Local: ${LOCAL_RESULTS_DIR}/"
echo "   GCS:   ${GCS_OUTPUT_PATH}/"
echo ""
echo " Next steps:"
echo "   1. Review benchmark_summary.md"
echo "   2. Update the HF model card with eval results"
echo "   3. Proceed to Phase 3 SFT when ready"
echo "============================================================"
