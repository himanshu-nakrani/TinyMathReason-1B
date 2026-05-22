import logging
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)

def test_template():
    model_path = "src/sft/sft_output/stage2/final"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Check if tokenizer has chat_template
    logging.info(f"chat_template: {repr(getattr(tokenizer, 'chat_template', None))}")
    
    messages = [
        {"role": "system", "content": "You are a mathematical reasoning assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ]
    
    try:
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        logging.info(f"Formatted text:\n{repr(formatted)}")
        
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        logging.info(f"Encoded token IDs: {encoded}")
        
        decoded = tokenizer.decode(encoded)
        logging.info(f"Decoded: {repr(decoded)}")
        
    except Exception as e:
        logging.error(f"apply_chat_template failed: {e}")

if __name__ == "__main__":
    test_template()
