import argparse
import logging
from pathlib import Path
from datasets import load_dataset
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATASETS = {
    "fineweb-edu": {"path": "HuggingFaceFW/fineweb-edu", "split": "train", "name": "sample-10BT"},
    "openwebmath": {"path": "open-web-math/open-web-math", "split": "train", "name": None},
    "proof-pile-2": {"path": "EleutherAI/proof-pile-2", "split": "train", "name": "arxiv"},
    "stack-edu": {"path": "HuggingFaceTB/smollm-corpus", "split": "train", "name": "cosmopedia-v2"} # Approximation of Stack-Edu
}

def download_datasets(output_dir: str):
    """
    Downloads raw datasets and saves them in local Parquet format for faster subsequent processing.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for ds_id, info in DATASETS.items():
        logging.info(f"Downloading {ds_id}...")
        try:
            # We use streaming = False here to pull it locally if we have space,
            # but for 300B tokens, we would actually stream and write to local disk chunks.
            # To simulate downloading a huge dataset efficiently, we use the `load_dataset` 
            # to cache it locally in ~/.cache/huggingface, then we just symlink or rely on cache.
            # In Vultr, we have large SSDs.
            
            # Using streaming to save direct to chunked parquets locally
            ds = load_dataset(info["path"], name=info["name"], split=info["split"], streaming=True)
            
            ds_out_dir = out_dir / ds_id
            ds_out_dir.mkdir(exist_ok=True)
            
            # Write to parquet in 100k row chunks
            chunk_size = 100000
            current_chunk = []
            chunk_idx = 0
            
            # Limit for demonstration, remove limit in actual full run
            # MAX_DOCS removed for full production run
            
            for i, row in enumerate(ds):
                # Standardize column name to 'text'
                text = row.get('text', row.get('content', ''))
                current_chunk.append({"text": text})
                
                if len(current_chunk) >= chunk_size:
                    import pyarrow as pa
                    table = pa.Table.from_pylist(current_chunk)
                    pq.write_table(table, ds_out_dir / f"chunk_{chunk_idx:04d}.parquet")
                    logging.info(f"[{ds_id}] Wrote chunk {chunk_idx}")
                    chunk_idx += 1
                    current_chunk = []
            
            # write remaining
            if current_chunk:
                import pyarrow as pa
                table = pa.Table.from_pylist(current_chunk)
                pq.write_table(table, ds_out_dir / f"chunk_{chunk_idx:04d}.parquet")
                logging.info(f"[{ds_id}] Wrote final chunk {chunk_idx}")

        except Exception as e:
            logging.error(f"Failed to download {ds_id}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/data/raw", help="Output directory")
    parser.add_argument("--datasets", type=str, default="all", help="Comma separated list of datasets to download (e.g. fineweb-edu,openwebmath) or 'all'")
    args = parser.parse_args()
    
    if args.datasets != "all":
        selected = args.datasets.split(",")
        DATASETS = {k: v for k, v in DATASETS.items() if k in selected}
        
    download_datasets(args.output_dir)
