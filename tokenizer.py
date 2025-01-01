from transformers import AutoTokenizer
import struct
import os

def export_tokenizer(model_path, output_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Using model vocab_size: {tokenizer.vocab_size}")
    
    # Calculate max token length and prepare tokens
    max_token_len = 0
    tokens = []
    
    # First pass: get actual tokens and max length
    for i in range(32000):  # Fixed vocab size
        token = tokenizer.convert_ids_to_tokens(i)
        token_bytes = token.encode('utf-8')
        max_token_len = max(max_token_len, len(token_bytes))
        tokens.append(token_bytes)
    
    print(f"Calculated max_token_len: {max_token_len}")
    
    with open(output_path, 'wb') as f:
        # Write max_token_len as int32
        f.write(struct.pack('<i', max_token_len))
        
        # Write each token
        for i, token_bytes in enumerate(tokens):
            # Write score (float32)
            f.write(struct.pack('<f', 0.0))
            
            # Write token length (int32)
            f.write(struct.pack('<i', len(token_bytes)))
            
            # Write token bytes
            f.write(token_bytes)
            
            # Debug first few tokens
            if i < 5:
                print(f"Token {i}: {token_bytes} (len={len(token_bytes)})")

if __name__ == "__main__":
    export_tokenizer("Llama-3.2-1B", "tokenizer-llama3.bin")