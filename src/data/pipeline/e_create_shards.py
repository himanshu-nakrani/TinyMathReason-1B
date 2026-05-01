import argparse
import logging
from pathlib import Path
import pyarrow.parquet as pq
import json
import zstandard as zstd
from multiprocessing import Pool, cpu_count
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def process_shard(args):
    f, out_dir, shard_idx = args
    out_file = out_dir / f"shard_{shard_idx:06d}.jsonl.zst"
    total_tokens = 0
    
    try:
        cctx = zstd.ZstdCompressor(level=3)
        table = pq.read_table(f)
        
        with open(out_file, 'wb') as out_f:
            with cctx.stream_writer(out_f) as compressor:
                for row in table.to_pylist():
                    tokens = row['tokens']
                    total_tokens += len(tokens)
                    
                    record = {"targets": tokens}
                    json_str = json.dumps(record) + "\n"
                    compressor.write(json_str.encode('utf-8'))
                    
        return shard_idx, total_tokens
    except Exception as e:
        logging.error(f"Error processing {f}: {e}")
        return shard_idx, 0

def create_shards(input_dir: str, output_dir: str):
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    files = list(in_dir.glob("*.parquet"))
    tasks = [(f, out_dir, i) for i, f in enumerate(files)]
    
    # We can use all cores here because zstd compression is CPU bound but memory light.
    num_cores = cpu_count()
    logging.info(f"Creating jsonl.zst shards using {num_cores} cores...")
    
    total_tokens = 0
    with Pool(processes=num_cores) as pool:
        for i, result in enumerate(pool.imap_unordered(process_shard, tasks)):
            shard_idx, shard_tokens = result
            total_tokens += shard_tokens
            if (i + 1) % 10 == 0:
                logging.info(f"Finished {i + 1}/{len(files)} shards")
                
    manifest_file = out_dir / "manifest.json"
    manifest_data = {
        "num_shards": len(files),
        "total_tokens": total_tokens,
        "format": "jsonl.zst"
    }
    with open(manifest_file, 'w') as f:
        json.dump(manifest_data, f, indent=4)
        
    logging.info(f"Successfully created {len(files)} shards. Total tokens: {total_tokens}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/data/packed")
    parser.add_argument("--output_dir", type=str, default="/data/shards")
    args = parser.parse_args()
    
    create_shards(args.input_dir, args.output_dir)
