## Usage

```python
from tokenizer import Tokenizer

# Initialize tokenizer
tokenizer = Tokenizer()

# Train the tokenizer
text = "Your training text here..."
tokenizer.train(text, vocab_size=5000)

# Encode text to tokens
encoded = tokenizer.encode("Text to encode")

# Decode tokens back to text
decoded = tokenizer.decode(encoded)
```

## Implementation Details

The tokenizer implements three main functions:

1. `train(text, vocab_size)`: Trains the tokenizer using the BPE algorithm
2. `encode(text)`: Converts input text into token IDs
3. `decode(ids)`: Converts token IDs back into text

## BPE Algorithm Overview

The implementation follows these steps:

1. Initialize vocabulary with base characters
2. Count pair frequencies
3. Merge most frequent pairs
4. Update vocabulary and merge rules
5. Repeat until desired vocabulary size is reached

## Limitations

- Not optimized for production use
- Limited vocabulary size compared to professional implementations
- Basic handling of special characters and whitespace
- No pre-trained vocabularies included

## Educational Resources

The repository includes detailed documentation about:
- BPE algorithm implementation
- Tokenizer training process
- Common LLM tokenization challenges
- Comparisons with professional tokenizers (e.g., GPT-2)

