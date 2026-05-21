#!/usr/bin/env bash
# ==============================================================================
# TinyMathReason-1B SFT Pipeline Automation (ROCm/AMD MI300X & NVIDIA GPU Compatible)
# This script handles environment verification, data prep, and SFT training runs.
# ==============================================================================

set -eo pipefail

# Style definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1${NC}"; }
log_warn() { echo -e "${YELLOW}[WARN] $(date '+%Y-%m-%d %H:%M:%S') - $1${NC}"; }
log_error() { echo -e "${RED}[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $1${NC}"; }
log_success() { echo -e "${GREEN}[SUCCESS] $(date '+%Y-%m-%d %H:%M:%S') - $1${NC}"; }

# --- 1. ENVIRONMENT VERIFICATION ---
log_info "Verifying Python environment and hardware..."

# Check virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    log_warn "You are not in a virtual environment. It is highly recommended to activate one."
    echo -n "Would you like to auto-create and activate a virtual environment? (y/n): "
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        python3 -m venv ~/sft_env
        source ~/sft_env/bin/activate
        log_success "Activated virtual environment: ~/sft_env"
    fi
fi

# Verify PyTorch installation and GPU availability (supporting both ROCm and CUDA)
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA/ROCm Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device Name: {torch.cuda.get_device_name(0)}')
    print(f'Device VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
else:
    raise RuntimeError('No GPU available! SFT training requires a GPU instance.')
" || { log_error "GPU check failed! Please ensure PyTorch is installed with appropriate ROCm/CUDA drivers."; exit 1; }

# Install SFT training dependencies if missing
log_info "Checking SFT dependencies..."
pip install --upgrade pip
pip install -r requirements/requirements-sft.txt || {
    log_warn "Failed to install dependencies from requirements file. Attempting manual installation..."
    pip install transformers datasets trl peft accelerate deepspeed wandb pyyaml
}
log_success "All dependencies satisfied."

# Ensure model directory is present
if [[ ! -d "hf_1b_model" ]]; then
    log_error "hf_1b_model directory not found in the current folder!"
    log_warn "Please ensure you have transferred the converted safetensors base model to this VM."
    exit 1
fi

# --- 2. DATA PREPARATION ---
log_info "Preparing dataset splits for both training stages..."

# Stage 1 dataset
if [[ ! -d "src/sft/sft_data/stage1_chat" ]]; then
    log_info "Creating SFT Stage 1 dataset (Conversational Prior)..."
    python3 src/sft/prepare_sft_data.py --stage 1 --output_dir src/sft/sft_data/stage1_chat
    log_success "Stage 1 dataset created."
else
    log_info "Stage 1 dataset already exists."
fi

# Stage 2 dataset
if [[ ! -d "src/sft/sft_data/stage2_reasoning" ]]; then
    log_info "Creating SFT Stage 2 dataset (Reasoning Traces with <think> tags)..."
    python3 src/sft/prepare_sft_data.py --stage 2 --output_dir src/sft/sft_data/stage2_reasoning
    log_success "Stage 2 dataset created."
else
    log_info "Stage 2 dataset already exists."
fi

# --- 3. TRAINING CHOICES ---
echo -e "\n=============================================="
echo -e "${YELLOW}TinyMathReason-1B SFT Execution Pipeline${NC}"
echo -e "=============================================="
echo "1. Run SFT Stage 1 (Conversational Prior)"
echo "2. Run SFT Stage 2 (Reasoning Traces - requires Stage 1 output)"
echo "3. Run FULL SFT Pipeline (Stage 1 then Stage 2 sequentially)"
echo "4. Exit"
echo -n "Please select an option (1-4): "
read -r choice

case $choice in
    1)
        log_info "Starting Stage 1 SFT..."
        cd src/sft
        accelerate launch train_sft.py --config sft_config_stage1.yaml
        log_success "Stage 1 SFT training completed successfully!"
        ;;
    2)
        log_info "Starting Stage 2 SFT (Reasoning Traces)..."
        if [[ ! -d "src/sft/sft_output/stage1/final" ]]; then
            log_warn "Stage 1 model output was not found at 'src/sft/sft_output/stage1/final'."
            echo -n "Do you want to override and start Stage 2 directly from the raw base model? (y/n): "
            read -r ans
            if [[ "$ans" =~ ^[Yy]$ ]]; then
                cd src/sft
                accelerate launch train_sft.py \
                    --config sft_config_stage2.yaml \
                    --model_path ../../hf_1b_model \
                    --resize_token_embeddings
            else
                log_info "Aborting Stage 2. Please train Stage 1 first."
                exit 0
            fi
        else
            cd src/sft
            accelerate launch train_sft.py \
                --config sft_config_stage2.yaml \
                --resize_token_embeddings
        fi
        log_success "Stage 2 SFT training completed successfully! 🚀"
        ;;
    3)
        log_info "Launching consecutive dual-stage SFT curriculum..."
        cd src/sft
        
        # Stage 1 SFT
        log_info "Executing Stage 1/2: Conversational Prior..."
        accelerate launch train_sft.py --config sft_config_stage1.yaml
        log_success "Stage 1/2 complete."
        
        # Stage 2 SFT
        log_info "Executing Stage 2/2: Reasoning Traces with custom <think> tags..."
        accelerate launch train_sft.py \
            --config sft_config_stage2.yaml \
            --resize_token_embeddings
            
        log_success "Complete SFT curriculum executed successfully! 🚀 Checkpoints saved in src/sft/sft_output/stage2/final"
        ;;
    *)
        log_info "Exiting."
        exit 0
        ;;
esac
