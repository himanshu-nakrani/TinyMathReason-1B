import argparse
import logging
import random
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def mix_datasets(input_dir: str, output_dir: str, chunk_size: int = 50000):
    """
    Mixes datasets according to target probabilities:
    - FineWeb-Edu: 40%
    - OpenWebMath: 35%
    - Proof-Pile-2: 15%
    - Stack-Edu: 10%
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Map ratios
    ratios = {
        "fineweb-edu": 0.40,
        "openwebmath": 0.35,
        "proof-pile-2": 0.15,
        "mathpile": 0.15,
        "stack-edu": 0.10
    }
    
    # Load iterators for all datasets
    iterators = {}
    for ds_name in ratios.keys():
        ds_path = in_dir / ds_name
        if ds_path.exists():
            files = list(ds_path.glob("*.parquet"))
            # Generator expression to yield rows from parquets
            def row_generator(files):
                for f in files:
                    try:
                        table = pq.read_table(f)
                        for row in table.to_pylist():
                            yield row
                    except Exception as e:
                        logging.error(f"Error reading {f}: {e}")
            iterators[ds_name] = row_generator(files)
        else:
            logging.warning(f"Dataset directory {ds_path} not found.")

    if not iterators:
        logging.error("No valid dataset directories found.")
        return

    # Normalize ratios based on available datasets
    total_ratio = sum(ratios[ds] for ds in iterators.keys())
    normalized_ratios = {ds: ratios[ds]/total_ratio for ds in iterators.keys()}
    
    # Create cumulative distribution for sampling
    datasets = list(normalized_ratios.keys())
    probs = [normalized_ratios[ds] for ds in datasets]
    
    out_chunk_idx = 0
    current_chunk = []
    
    logging.info("Starting mixing...")
    
    # Simple sampling approach
    while datasets:
        chosen_ds = random.choices(datasets, weights=probs, k=1)[0]
        try:
            row = next(iterators[chosen_ds])
            current_chunk.append(row)
            
            if len(current_chunk) >= chunk_size:
                out_file = out_dir / f"mixed_{out_chunk_idx:06d}.parquet"
                table = pa.Table.from_pylist(current_chunk)
                pq.write_table(table, out_file)
                if out_chunk_idx % 10 == 0:
                    logging.info(f"Wrote mixed chunk {out_chunk_idx}")
                out_chunk_idx += 1
                current_chunk = []
                
        except StopIteration:
            # Dataset is exhausted, remove it and rebalance probabilities
            logging.info(f"Dataset {chosen_ds} exhausted.")
            idx = datasets.index(chosen_ds)
            datasets.pop(idx)
            probs.pop(idx)
            if probs:
                total_prob = sum(probs)
                probs = [p/total_prob for p in probs]
                
    # Write remaining
    if current_chunk:
        out_file = out_dir / f"mixed_{out_chunk_idx:06d}.parquet"
        table = pa.Table.from_pylist(current_chunk)
        pq.write_table(table, out_file)
        logging.info(f"Wrote final chunk {out_chunk_idx}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/data/clean")
    parser.add_argument("--output_dir", type=str, default="/data/mixed")
    args = parser.parse_args()
    
    mix_datasets(args.input_dir, args.output_dir)
