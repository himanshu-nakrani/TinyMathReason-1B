# Vultr Environment Setup for Data Processing

To process ~300B tokens (which is roughly ~1TB of raw text), we need a powerful CPU instance with large, fast storage.

## Instance Recommendation
- **Instance Type**: High Performance Compute (c2-standard-30)
- **vCPUs**: 30
- **RAM**: 120 GB
- **Storage**: 2 TB NVMe SSD (Add block storage if the default isn't enough)
- **OS**: Ubuntu 22.04 LTS

## Setup Commands

```bash
# 1. Update and install basic tools
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential python3-pip python3-venv git htop tmux zstd google-cloud-cli

# 2. Mount and format the NVMe volume (assuming it's at /dev/vdb)
# Note: Check actual device name using `lsblk`
sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/vdb
sudo mkdir -p /data
sudo mount -o discard,defaults /dev/vdb /data
sudo chown -R $USER:$USER /data

# 3. Setup Python Virtual Environment
python3 -m venv ~/tinymath_env
source ~/tinymath_env/bin/activate

# 4. Install requirements
pip install -U pip
pip install datasets pyarrow datasketch tokenizers transformers zstandard

# 5. Authenticate with Google Cloud (for uploading later)
gcloud auth login
# Follow instructions to authenticate
```

## Cost Estimate (Vultr)
- **Compute (c2-standard-30)**: ~$1.20 / hour
- **Additional 2TB Block Storage**: ~$0.15 / hour
- **Total per hour**: ~$1.35
- **Estimated time to process 300B tokens**: ~48 hours
- **Total Estimated Cost**: ~$65.00

This fits well within the $250 credit limit, leaving room for tokenizer training and other exploratory processing.
