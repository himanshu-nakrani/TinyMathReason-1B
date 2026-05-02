#!/bin/bash
# Scripts to provision a GCP TPU v4-32 cluster for TinyMathReason-1B

# Configuration
PROJECT_ID="gen-lang-client-0318750942"
ZONE="us-central2-b"
TPU_NAME="node-v4-32-spot-uc2b"

echo "Provisioning TPU v4-32: ${TPU_NAME} in ${ZONE}..."

gcloud compute tpus tpu-vm create ${TPU_NAME} \
    --zone=${ZONE} \
    --accelerator-type=v4-32 \
    --version=tpu-vm-v4-base \
    --project=${PROJECT_ID} \
    --spot # Using spot for cost efficiency

if [ $? -eq 0 ]; then
    echo "TPU ${TPU_NAME} provisioned successfully."
    echo "You can now SSH into it using:"
    echo "gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --project=${PROJECT_ID}"
else
    echo "Failed to provision TPU. Check your quotas and availability."
fi
