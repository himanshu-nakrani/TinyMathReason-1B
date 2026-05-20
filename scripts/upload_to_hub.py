#!/usr/bin/env python3
"""
Upload TinyMathReason-1B base model to Hugging Face Hub.

Usage:
    python scripts/upload_to_hub.py --model_dir ./hf_1b_model --repo_id himanshu-nakrani/TinyMathReason-1B-base

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login  (or set HF_TOKEN env var)
"""

import argparse
import logging
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, get_token
except ImportError:
    print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def upload_model(
    model_dir: str,
    repo_id: str,
    private: bool = True,
    commit_message: str = "Upload TinyMathReason-1B base model"
):
    """Upload a local model directory to Hugging Face Hub."""
    model_path = Path(model_dir)
    if not model_path.exists():
        logger.error(f"Model directory not found: {model_path}")
        sys.exit(1)

    # Verify required files exist
    required_files = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]
    for fname in required_files:
        if not (model_path / fname).exists():
            logger.error(f"Missing required file: {fname}")
            sys.exit(1)

    # Check for authentication
    token = get_token()
    if token is None:
        logger.error(
            "No Hugging Face token found.\n"
            "Run: hf auth login\n"
            "Or set the HF_TOKEN environment variable."
        )
        sys.exit(1)

    api = HfApi()

    # Create repo if it doesn't exist
    logger.info(f"Creating repo '{repo_id}' (private={private})...")
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True
        )
        logger.info(f"Repo '{repo_id}' is ready.")
    except Exception as e:
        logger.error(f"Failed to create repo: {e}")
        sys.exit(1)

    # Upload the entire directory
    logger.info(f"Uploading {model_path} to {repo_id}...")
    logger.info(f"Files: {[f.name for f in model_path.iterdir() if f.is_file()]}")

    try:
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        sys.exit(1)

    repo_url = f"https://huggingface.co/{repo_id}"
    logger.info(f"✅ Upload complete!")
    logger.info(f"   Repo URL: {repo_url}")
    logger.info(f"   Visibility: {'Private' if private else 'Public'}")

    # Verify upload
    logger.info("Verifying uploaded files...")
    try:
        model_info = api.model_info(repo_id)
        siblings = [s.rfilename for s in model_info.siblings]
        logger.info(f"   Files on Hub: {siblings}")
        for fname in required_files:
            if fname in siblings:
                logger.info(f"   ✓ {fname}")
            else:
                logger.warning(f"   ✗ {fname} NOT FOUND on Hub!")
    except Exception as e:
        logger.warning(f"Could not verify: {e}")

    return repo_url


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload TinyMathReason-1B model to Hugging Face Hub"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./hf_1b_model",
        help="Path to the HuggingFace model directory"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="himanshunakrani9/TinyMathReason-1B-base",
        help="Hugging Face repo ID (e.g., username/model-name)"
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make the repo public (default: private)"
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Upload TinyMathReason-1B base model (1.126B params, bfloat16)",
        help="Commit message for the upload"
    )
    args = parser.parse_args()

    upload_model(
        model_dir=args.model_dir,
        repo_id=args.repo_id,
        private=not args.public,
        commit_message=args.commit_message
    )
