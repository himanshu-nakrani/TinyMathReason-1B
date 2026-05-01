import argparse
import logging
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, processors

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_tokenizer(output_dir: str, vocab_size: int = 32000):
    """
    Trains a BPE tokenizer optimized for mathematical reasoning.
    We sample ~10GB of text primarily from OpenWebMath and FineWeb-Edu.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer_file = output_path / "tinymath_tokenizer.json"

    logging.info("Initializing BPE Tokenizer...")
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    
    # We use ByteLevel pre-tokenizer which handles whitespace elegantly without needing a meta symbol
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    special_tokens = [
        "<|bos|>", 
        "<|eos|>", 
        "<|unk|>", 
        "<|pad|>",
        "<|im_start|>", 
        "<|im_end|>",
        "<think>",
        "</think>"
    ]
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    logging.info("Loading dataset samples...")
    # Sampling strategy:
    # 5M documents from FineWeb-Edu (~3-4GB)
    # 2M documents from OpenWebMath (~3-4GB)
    # 500k documents from Stack-Edu (~1GB)
    
    # In a real environment, we would stream this. For the tokenizer training script,
    # we yield strings directly to the trainer.
    
    def get_training_corpus():
        logging.info("Streaming FineWeb-Edu...")
        try:
            ds_fw = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
            for i, doc in enumerate(ds_fw):
                if i >= 50000: # Reduced limit for local testing resilience
                    break
                yield doc['text']
        except Exception as e:
            logging.warning(f"Error streaming FineWeb-Edu: {e}")
            
        logging.info("Streaming OpenWebMath...")
        try:
            ds_owm = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
            for i, doc in enumerate(ds_owm):
                if i >= 20000: # Reduced limit
                    break
                yield doc['text']
        except Exception as e:
            logging.warning(f"Error streaming OpenWebMath: {e}")

    logging.info("Training Tokenizer (this may take a while)...")
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    
    # Set post-processor for typical LM if needed, or leave to model (we typically leave raw for base model)
    
    logging.info(f"Saving tokenizer to {tokenizer_file}")
    tokenizer.save(str(tokenizer_file))
    
    # Save a basic config for HuggingFace compatibility
    from transformers import PreTrainedTokenizerFast
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        unk_token="<|unk|>",
        pad_token="<|pad|>",
        chat_template="{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )
    hf_tokenizer.save_pretrained(str(output_path))
    logging.info("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train tokenizer for TinyMathReason-1B")
    parser.add_argument("--output_dir", type=str, default="./tokenizer", help="Directory to save the tokenizer")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")
    args = parser.parse_args()
    
    train_tokenizer(args.output_dir, args.vocab_size)
