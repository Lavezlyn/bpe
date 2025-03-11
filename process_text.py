'''
Encodes given text.txt file into tokens, then decodes it back to text.
'''

import os
import re
from tokenizer import Tokenizer

def process_text(text_file, vocab_size=10000):
    # Read text file
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Initialize tokenizer
    tokenizer = Tokenizer()
    tokenizer.train(text, vocab_size)

    # Encode text
    ids = tokenizer.encode(text)

    # Decode text
    decoded_text = tokenizer.decode(ids)

    return ids, decoded_text

if __name__ == "__main__":
    text_file = "manual.txt"
    vocab_size = 1024
    ids, decoded_text = process_text(text_file, vocab_size)
    print(f"Encoded text: {ids}")
    print(f"Decoded text: {decoded_text}")

