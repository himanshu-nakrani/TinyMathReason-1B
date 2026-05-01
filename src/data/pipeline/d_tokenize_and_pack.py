import argparse
import logging
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
from transformers import PreTrainedTokenizerFast
from multiprocessing import Pool, cpu_count
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def process_file(args):
    f, out_dir, tokenizer_path, seq_len = args
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    eos_id = tokenizer.eos_token_id
    
    current_tokens = []
    chunk_size = 10000
    sequences = []
    
    out_chunk_idx = 0
    try:
        table = pq.read_table(f)
        for row in table.to_pylist():
            text = row['text']
            tokens = tokenizer.encode(text)
            current_tokens.extend(tokens)
            current_tokens.append(eos_id)
            
            while len(current_tokens) >= seq_len:
                seq = current_tokens[:seq_len]
                current_tokens = current_tokens[seq_len:]
                sequences.append({"tokens": seq})
                
                if len(sequences) >= chunk_size:
                    out_file = out_dir / f"packed_{f.stem}_{out_chunk_idx:06d}.parquet"
                    out_table = pa.Table.from_pylist(sequences)
                    pq.write_table(out_table, out_file)
                    out_chunk_idx += 1
                    sequences = []
                    
        # Handle remaining tokens (padding)
        if current_tokens:
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
            padding_len = seq_len - len(current_tokens)
            seq = current_tokens + [pad_id] * padding_len
            sequences.append({"tokens": seq})
            
        if sequences:
            out_file = out_dir / f"packed_{f.stem}_{out_chunk_idx:06d}.parquet"
            out_table = pa.Table.from_pylist(sequences)
            pq.write_table(out_table, out_file)
            
        return True
    except Exception as e:
        logging.error(f"Error processing {f}: {e}")
        return False

def tokenize_and_pack(input_dir: str, output_dir: str, tokenizer_path: str, seq_len: int = 4096):
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    files = list(in_dir.glob("*.parquet"))
    args_list = [(f, out_dir, tokenizer_path, seq_len) for f in files]
    
    # LIMIT CORES TO PREVENT OOM (Out of Memory) KILLS
    # 64GB RAM / 32 cores = 2GB per core. The chunks are heavy, so we limit to 14 cores to be safe.
    system_cores = cpu_count()
    num_cores = max(1, min(system_cores, 14))
    
    logging.info(f"Starting multiprocess tokenization with {num_cores} safe cores across {len(files)} files...")
    
    with Pool(processes=num_cores) as pool:
        for i, _ in enumerate(pool.imap_unordered(process_file, args_list)):
            if (i + 1) % 10 == 0:
                logging.info(f"Finished processing {i + 1}/{len(files)} files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/data/mixed")
    parser.add_argument("--output_dir", type=str, default="/data/packed")
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=4096)
    args = parser.parse_args()
    
    tokenize_and_pack(args.input_dir, args.output_dir, args.tokenizer_path, args.seq_len)
