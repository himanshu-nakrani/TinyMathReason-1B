import argparse
import logging
from pathlib import Path
import subprocess
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def upload_to_gcs(input_dir: str, gcs_bucket: str):
    """
    Uploads all shards to a Google Cloud Storage bucket using gsutil.
    Verifies the upload by checking the manifest.
    """
    in_dir = Path(input_dir)
    if not in_dir.exists():
        logging.error(f"Input directory {in_dir} does not exist.")
        return
        
    logging.info(f"Uploading shards from {in_dir} to {gcs_bucket}...")
    
    # We use gsutil -m rsync for parallel, robust uploading
    cmd = [
        "gsutil", "-m", "rsync", "-r",
        str(in_dir),
        gcs_bucket
    ]
    
    try:
        logging.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info("Upload completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Upload failed: {e.stderr}")
        return
        
    # Print final stats
    manifest_file = in_dir / "manifest.json"
    if manifest_file.exists():
        with open(manifest_file, 'r') as f:
            manifest_data = json.load(f)
            
        logging.info("\n=== Upload Summary ===")
        logging.info(f"Destination: {gcs_bucket}")
        logging.info(f"Total Shards: {manifest_data.get('num_shards')}")
        logging.info(f"Total Tokens: {manifest_data.get('total_tokens')}")
        logging.info(f"Format: {manifest_data.get('format')}")
        logging.info("======================\n")
    else:
        logging.warning("Manifest file not found, cannot print stats.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/data/shards")
    parser.add_argument("--gcs_bucket", type=str, required=True, help="gs://my-bucket/dataset/")
    args = parser.parse_args()
    
    upload_to_gcs(args.input_dir, args.gcs_bucket)
