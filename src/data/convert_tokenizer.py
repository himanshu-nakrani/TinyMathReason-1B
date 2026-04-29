import argparse
import base64
import logging
from transformers import PreTrainedTokenizerFast

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def convert_to_tiktoken(tokenizer_dir: str, output_file: str):
    """
    Converts a HuggingFace BPE Tokenizer to TikToken format (.tiktoken)
    which is highly compatible with MaxText's tiktoken loader.
    """
    logging.info(f"Loading tokenizer from {tokenizer_dir}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    
    # Get the vocabulary
    vocab = tokenizer.get_vocab()
    
    logging.info(f"Converting {len(vocab)} tokens to TikToken format...")
    
    # Sort vocab by ID to ensure correct ordering
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for token, token_id in sorted_vocab:
            # Tiktoken format is: base64(token_bytes) <space> rank
            # HuggingFace tokens might be strings, we encode to utf-8 bytes
            
            # Special handling for ByteLevel BPE where tokens might have 'Ġ' (which is space)
            # We use the tokenizer's internal decoder to get the exact bytes
            # For special tokens, we just encode the special token string
            if token in tokenizer.all_special_tokens:
                token_bytes = token.encode('utf-8')
            else:
                # Convert back to bytes
                # This is a simplification; for a robust implementation we extract bytes from vocab
                token_bytes = tokenizer.backend_tokenizer.id_to_token(token_id).encode('utf-8')
                
            b64_token = base64.b64encode(token_bytes).decode('utf-8')
            f.write(f"{b64_token} {token_id}\n")
            
    logging.info(f"Saved TikToken vocab to {output_file}")
    
    # In MaxText config, you would set:
    # tokenizer_path: "path/to/tokenizer.tiktoken"
    # vocab_size: 32000

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Tokenizer for MaxText")
    parser.add_argument("--tokenizer_dir", type=str, default="./tokenizer", help="Path to HF tokenizer")
    parser.add_argument("--output_file", type=str, default="./tokenizer.tiktoken", help="Output file")
    args = parser.parse_args()
    
    convert_to_tiktoken(args.tokenizer_dir, args.output_file)
