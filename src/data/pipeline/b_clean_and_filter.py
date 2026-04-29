import argparse
import logging
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
from datasketch import MinHash, MinHashLSH
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def clean_text(text: str) -> str:
    """Basic text cleaning."""
    if not isinstance(text, str):
        return ""
    # Normalize whitespaces
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def is_quality_document(text: str) -> bool:
    """Filter out clearly low quality or too short documents."""
    words = text.split()
    if len(words) < 20: # Too short
        return False
    # Check for excessive repetition or non-alphanumeric
    if len(text) > 0:
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        if alpha_ratio < 0.3: # Might be pure code or junk, but we keep math. OpenWebMath usually has text too.
            return False
    return True

def clean_and_filter(input_dir: str, output_dir: str):
    """
    Cleans text, filters low quality, and performs exact/near dedup using MinHash.
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # We use a global LSH for all datasets to remove cross-dataset duplicates
    lsh = MinHashLSH(threshold=0.8, num_perm=128)
    
    global_doc_id = 0
    seen_hashes = set() # For exact exact deduplication
    
    for ds_dir in in_dir.iterdir():
        if not ds_dir.is_dir():
            continue
            
        logging.info(f"Processing {ds_dir.name}...")
        ds_out_dir = out_dir / ds_dir.name
        ds_out_dir.mkdir(exist_ok=True)
        
        for pq_file in ds_dir.glob("*.parquet"):
            try:
                table = pq.read_table(pq_file)
                batch_records = []
                
                for row in table.to_pylist():
                    text = clean_text(row['text'])
                    if not is_quality_document(text):
                        continue
                        
                    # Exact deduplication (hash the string)
                    exact_hash = hash(text)
                    if exact_hash in seen_hashes:
                        continue
                    seen_hashes.add(exact_hash)
                    
                    # Near deduplication using MinHash
                    m = MinHash(num_perm=128)
                    for word in text.split():
                        m.update(word.encode('utf-8'))
                        
                    result = lsh.query(m)
                    if not result:
                        lsh.insert(str(global_doc_id), m)
                        batch_records.append({"text": text})
                        global_doc_id += 1
                        
                if batch_records:
                    out_table = pa.Table.from_pylist(batch_records)
                    pq.write_table(out_table, ds_out_dir / pq_file.name)
                    
            except Exception as e:
                logging.error(f"Error processing {pq_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/data/raw")
    parser.add_argument("--output_dir", type=str, default="/data/clean")
    args = parser.parse_args()
    
    clean_and_filter(args.input_dir, args.output_dir)
