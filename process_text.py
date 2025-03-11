'''
Encodes given text.txt file into tokens, then decodes it back to text.
'''

import os
import re
from tokenizer import Tokenizer

def process_text(tokenizer: Tokenizer, text_file, vocab_size=10000):
    # Read text file
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()

    tokenizer.train(text, vocab_size)

    # Encode text
    ids = tokenizer.encode(text)

    # Decode text
    decoded_text = tokenizer.decode(ids)

    return ids, decoded_text

def save_tokens(tokens, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for token in tokens:
            f.write(token + '\n')

def load_tokens(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        tokens = f.read().splitlines()
    return tokens

def save_ids(ids, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for id in ids:
            f.write(str(id) + '\n')

def load_ids(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        ids = f.read().splitlines()
    return ids

def save_decoded_text(decoded_text, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(decoded_text)

def load_decoded_text(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        decoded_text = f.read()
    return decoded_text

if __name__ == "__main__":
    text_file = "manual.txt"
    vocab_size = 1024
    tokenizer = Tokenizer()
    ids, decoded_text = process_text(tokenizer, text_file, vocab_size)
    save_ids(ids, "ids.txt")
    save_tokens(tokenizer.vocab, "tokens.txt")
    save_decoded_text(decoded_text, "decoded_text.txt")


