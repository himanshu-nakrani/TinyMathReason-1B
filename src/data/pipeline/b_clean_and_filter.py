import argparse
import logging
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
from multiprocessing import Pool, cpu_count
import re
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def clean_text(text: str) -> str:
    """Basic text cleaning."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def is_quality_document(text: str) -> bool:
    """Filter out clearly low quality or too short documents."""
    words = text.split()
    if len(words) < 20: 
        return False
    if len(text) > 0:
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        if alpha_ratio < 0.3: 
            return False
    return True

def process_file(args):
    pq_file, ds_out_dir = args
    try:
        table = pq.read_table(pq_file)
        batch_records = []
        seen_hashes = set()
        
        for row in table.to_pylist():
            text = clean_text(row['text'])
            if not is_quality_document(text):
                continue
                
            exact_hash = hash(text)
            if exact_hash in seen_hashes:
                continue
            seen_hashes.add(exact_hash)
            
            batch_records.append({"text": text})
                
        if batch_records:
            out_table = pa.Table.from_pylist(batch_records)
            pq.write_table(out_table, ds_out_dir / pq_file.name)
        return True
    except Exception as e:
        logging.error(f"Error processing {pq_file}: {e}")
        return False

def clean_and_filter(input_dir: str, output_dir: str):
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    tasks = []
    for ds_dir in in_dir.iterdir():
        if not ds_dir.is_dir():
            continue
            
        ds_out_dir = out_dir / ds_dir.name
        ds_out_dir.mkdir(exist_ok=True)
        
        for pq_file in ds_dir.glob("*.parquet"):
            tasks.append((pq_file, ds_out_dir))
            
    # LIMIT CORES TO PREVENT OOM
    # 64GB RAM / 32 cores is not enough for large parquet reads. We limit to 14 cores max.
    system_cores = cpu_count()
    num_cores = max(1, min(system_cores, 14))
    
    logging.info(f"Starting multiprocessing filter on {len(tasks)} files using {num_cores} safe cores...")
    
    with Pool(processes=num_cores) as pool:
        for i, _ in enumerate(pool.imap_unordered(process_file, tasks)):
            if (i + 1) % 10 == 0:
                logging.info(f"Finished cleaning {i + 1}/{len(tasks)} files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/data/raw")
    parser.add_argument("--output_dir", type=str, default="/data/clean")
    args = parser.parse_args()
    
    clean_and_filter(args.input_dir, args.output_dir)
