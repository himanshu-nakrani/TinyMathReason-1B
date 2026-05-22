import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)

def run_diagnostics():
    paths = {
        "Base Model HF": "himanshunakrani9/TinyMathReason-1B-base",
        "SFT Model HF": "himanshunakrani9/TinyMathReason-1B-sft",
        "SFT Model Local": "src/sft/sft_output/stage2/final"
    }
    
    test_sentence = "Solve: 2 + 2 = 4 <think> simple math </think>"
    
    for name, path in paths.items():
        logging.info(f"\n=== Diagnostics for {name} ({path}) ===")
        try:
            tokenizer = AutoTokenizer.from_pretrained(path)
            logging.info(f"  Tokenizer class: {type(tokenizer).__name__}")
            logging.info(f"  Vocab size (len): {len(tokenizer)}")
            logging.info(f"  Tokenizer vocab_size attr: {tokenizer.vocab_size}")
            
            # Print specific token IDs
            for special in ["<|bos|>", "<|eos|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]:
                token_id = tokenizer.convert_tokens_to_ids(special)
                logging.info(f"  Special token '{special}' ID: {token_id}")
                
            # Tokenize sample sentence
            encoded = tokenizer.encode(test_sentence)
            logging.info(f"  Encoded sample ({len(encoded)} tokens): {encoded[:15]}...")
            decoded = tokenizer.decode(encoded)
            logging.info(f"  Decoded: {repr(decoded)}")
            
        except Exception as e:
            logging.error(f"  Failed to load/run diagnostics for {name}: {e}")

if __name__ == "__main__":
    run_diagnostics()
