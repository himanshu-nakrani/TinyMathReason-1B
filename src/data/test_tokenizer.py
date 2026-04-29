import argparse
import logging
from transformers import PreTrainedTokenizerFast, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def test_tokenizer(tokenizer_path: str):
    """
    Tests tokenizer quality by measuring compression ratio and fertility.
    Compares against Llama 3 tokenizer (or another standard tokenizer) if available.
    """
    logging.info(f"Loading custom tokenizer from {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    
    try:
        logging.info("Loading reference tokenizer (Qwen2.5-0.5B) for comparison...")
        ref_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    except Exception as e:
        logging.warning(f"Could not load reference tokenizer: {e}")
        ref_tokenizer = None

    test_texts = {
        "General English": "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.",
        "Math (LaTeX)": "Let $X$ be a topological space. A subset $A \subset X$ is closed if and only if its complement $X \setminus A$ is open. We can calculate the integral as follows: $\int_{0}^{\infty} e^{-x^2} dx = \frac{\sqrt{\pi}}{2}$.",
        "Code (Python)": "def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)\n\nprint(factorial(5))"
    }

    logging.info("\n--- Roundtrip Tests ---")
    for name, text in test_texts.items():
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text, f"Roundtrip failed for {name}!\nOriginal: {text}\nDecoded: {decoded}"
        logging.info(f"[{name}] Roundtrip passed.")

    logging.info("\n--- Fertility and Compression ---")
    for name, text in test_texts.items():
        num_chars = len(text)
        num_words = len(text.split())
        
        encoded = tokenizer.encode(text)
        num_tokens = len(encoded)
        
        fertility = num_tokens / max(1, num_words)
        compression = num_chars / max(1, num_tokens)
        
        logging.info(f"[{name}] Custom Tokenizer:")
        logging.info(f"  Tokens: {num_tokens}")
        logging.info(f"  Fertility (tokens/word): {fertility:.2f}")
        logging.info(f"  Compression (chars/token): {compression:.2f}")
        
        if ref_tokenizer:
            ref_encoded = ref_tokenizer.encode(text)
            ref_tokens = len(ref_encoded)
            ref_fertility = ref_tokens / max(1, num_words)
            ref_compression = num_chars / max(1, ref_tokens)
            
            logging.info(f"[{name}] Reference Tokenizer:")
            logging.info(f"  Tokens: {ref_tokens}")
            logging.info(f"  Fertility: {ref_fertility:.2f}")
            logging.info(f"  Compression: {ref_compression:.2f}")
            logging.info(f"  Difference (Custom - Ref): {num_tokens - ref_tokens} tokens")
        logging.info("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Tokenizer")
    parser.add_argument("--tokenizer_dir", type=str, default="./tokenizer", help="Path to trained tokenizer")
    args = parser.parse_args()
    
    test_tokenizer(args.tokenizer_dir)
