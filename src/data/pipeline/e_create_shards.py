import argparse
import logging
from pathlib import Path
import pyarrow.parquet as pq
import json
import zstandard as zstd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def create_shards(input_dir: str, output_dir: str):
    """
    Converts packed Parquet sequences into JSONL.ZST shards.
    MaxText's data loader (SeqIO/Grain) supports reading compressed jsonl directly.
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cctx = zstd.ZstdCompressor(level=3)
    
    shard_idx = 0
    total_tokens = 0
    
    logging.info("Creating jsonl.zst shards...")
    
    for f in in_dir.glob("*.parquet"):
        logging.info(f"Processing {f.name} into shard {shard_idx}")
        out_file = out_dir / f"shard_{shard_idx:06d}.jsonl.zst"
        
        try:
            table = pq.read_table(f)
            
            with open(out_file, 'wb') as out_f:
                with cctx.stream_writer(out_f) as compressor:
                    for row in table.to_pylist():
                        tokens = row['tokens']
                        total_tokens += len(tokens)
                        
                        # MaxText usually looks for a 'text' or 'targets' key. 
                        # If passing pre-tokenized, 'targets' containing integer IDs is standard.
                        record = {"targets": tokens}
                        
                        json_str = json.dumps(record) + "\n"
                        compressor.write(json_str.encode('utf-8'))
                        
            shard_idx += 1
            
        except Exception as e:
            logging.error(f"Error processing {f}: {e}")
            
    # Write manifest/index
    manifest_file = out_dir / "manifest.json"
    manifest_data = {
        "num_shards": shard_idx,
        "total_tokens": total_tokens,
        "format": "jsonl.zst"
    }
    with open(manifest_file, 'w') as f:
        json.dump(manifest_data, f, indent=4)
        
    logging.info(f"Successfully created {shard_idx} shards. Total tokens: {total_tokens}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/data/packed")
    parser.add_argument("--output_dir", type=str, default="/data/shards")
    args = parser.parse_args()
    
    create_shards(args.input_dir, args.output_dir)
