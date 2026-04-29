import argparse
import logging
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
from transformers import PreTrainedTokenizerFast

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def tokenize_and_pack(input_dir: str, output_dir: str, tokenizer_path: str, seq_len: int = 4096):
    """
    Tokenizes mixed text and packs into dense sequences of max_seq_len.
    Documents are separated by the <|eos|> token.
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    eos_id = tokenizer.eos_token_id
    
    current_tokens = []
    out_chunk_idx = 0
    chunk_size = 10000 # sequences per file
    sequences = []
    
    for f in in_dir.glob("*.parquet"):
        logging.info(f"Tokenizing {f.name}...")
        try:
            table = pq.read_table(f)
            for row in table.to_pylist():
                text = row['text']
                # Encode text
                tokens = tokenizer.encode(text)
                
                # Append to buffer with eos token as document separator
                current_tokens.extend(tokens)
                current_tokens.append(eos_id)
                
                # While we have enough tokens for a full sequence, pack it
                while len(current_tokens) >= seq_len:
                    seq = current_tokens[:seq_len]
                    current_tokens = current_tokens[seq_len:]
                    
                    sequences.append({"tokens": seq})
                    
                    # If sequences hit chunk_size, save to disk
                    if len(sequences) >= chunk_size:
                        out_file = out_dir / f"packed_{out_chunk_idx:06d}.parquet"
                        out_table = pa.Table.from_pylist(sequences)
                        pq.write_table(out_table, out_file)
                        logging.info(f"Wrote packed chunk {out_chunk_idx}")
                        out_chunk_idx += 1
                        sequences = []
        except Exception as e:
            logging.error(f"Error processing {f}: {e}")
            
    # Handle remaining tokens (padding)
    if current_tokens:
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
        padding_len = seq_len - len(current_tokens)
        seq = current_tokens + [pad_id] * padding_len
        sequences.append({"tokens": seq})
        
    if sequences:
        out_file = out_dir / f"packed_{out_chunk_idx:06d}.parquet"
        out_table = pa.Table.from_pylist(sequences)
        pq.write_table(out_table, out_file)
        logging.info(f"Wrote final packed chunk {out_chunk_idx}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/data/mixed")
    parser.add_argument("--output_dir", type=str, default="/data/packed")
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=4096)
    args = parser.parse_args()
    
    tokenize_and_pack(args.input_dir, args.output_dir, args.tokenizer_path, args.seq_len)
