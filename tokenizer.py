'''
Implements a tokenizer class based on BPE algorithm.
'''

from collections import defaultdict, Counter
import re

class Tokenizer:
    def __init__(self):
        self.vocab = {}  # vocabulary: token -> id mapping
        self.inv_vocab = {}  # inverse vocabulary: id -> token mapping
        self.merges = {}  # merge rules: pair -> token mapping
        self.special_tokens = {
            '<SPACE>': ' ',
            '<NEWLINE>': '\n',
            '<TAB>': '\t'
        }
        
    def _preprocess_text(self, text):
        """
        Preprocess text by replacing special characters with special tokens
        """
        # Replace special characters with special tokens
        for token, char in self.special_tokens.items():
            text = text.replace(char, f' {token} ')
        return text
    
    def _postprocess_text(self, text):
        """
        Postprocess text by replacing special tokens with original characters
        """
        # Replace special tokens with original characters
        for token, char in self.special_tokens.items():
            text = text.replace(token, char)
        return text

    def _get_stats(self, tokens):
        """Count frequencies of adjacent symbol pairs"""
        pairs = defaultdict(int)
        for word, freq in tokens.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[symbols[i], symbols[i+1]] += freq
        return pairs
    
    def _merge_pair(self, pair, tokens):
        """Merge a specific pair of symbols in all tokens"""
        new_tokens = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word, freq in tokens.items():
            parts = word.split()
            i = 0
            new_parts = []
            while i < len(parts):
                if i < len(parts) - 1 and parts[i] == pair[0] and parts[i+1] == pair[1]:
                    new_parts.append(replacement)
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            new_tokens[' '.join(new_parts)] = freq
        return new_tokens

    def train(self, text, vocab_size):
        """
        Train the tokenizer using BPE algorithm
        Args:
            text (str): Training text
            vocab_size (int): Target vocabulary size
        """
        # Preprocess text
        processed_text = self._preprocess_text(text)
        
        # Initialize: split text into words, preserving special tokens
        words = processed_text.split()
        word_freqs = Counter(words)
        
        # Split each word into characters
        tokens = {}
        for word, freq in word_freqs.items():
            # Don't split special tokens
            if word in self.special_tokens:
                tokens[word] = freq
            else:
                chars = ' '.join(list(word))
                tokens[chars] = freq
            
        # Initialize base vocabulary with special tokens and characters
        base_vocab = set(self.special_tokens.keys())
        for word in tokens.keys():
            if word not in self.special_tokens:
                base_vocab.update(word.split())
        
        # Assign IDs to base vocabulary
        for i, token in enumerate(sorted(base_vocab)):
            self.vocab[token] = i
            self.inv_vocab[i] = token
            
        num_merges = vocab_size - len(self.vocab)
        
        # Iteratively perform BPE merges
        for i in range(num_merges):
            pairs = self._get_stats(tokens)
            if not pairs:
                break
                
            # Find the most frequent pair
            best_pair = max(pairs.items(), key=lambda x: x[1])[0]
            
            # Record merge rule
            new_token = ''.join(best_pair)
            self.merges[best_pair] = new_token
            
            # Add new token to vocabulary
            self.vocab[new_token] = len(self.vocab)
            self.inv_vocab[len(self.vocab)-1] = new_token
            
            # Perform the merge
            tokens = self._merge_pair(best_pair, tokens)

    def encode(self, text):
        """
        Encode input text into token IDs
        Args:
            text (str): Input text to encode
        Returns:
            list: List of token IDs
        """
        if not self.vocab:
            raise ValueError("Tokenizer needs to be trained first")
            
        # Preprocess text
        processed_text = self._preprocess_text(text)
        words = processed_text.split()
        encoded = []
        
        for word in words:
            # Handle special tokens
            if word in self.special_tokens:
                encoded.append(self.vocab[word])
                continue
                
            chars = ' '.join(list(word))
            subwords = chars.split()
            
            # Apply all possible merge rules
            while True:
                # Find mergeable pairs
                can_merge = False
                for i in range(len(subwords)-1):
                    pair = (subwords[i], subwords[i+1])
                    if pair in self.merges:
                        # Perform merge
                        new_token = self.merges[pair]
                        subwords[i] = new_token
                        del subwords[i+1]
                        can_merge = True
                        break
                
                if not can_merge:
                    break
            
            # Convert subwords to token IDs
            encoded.extend([self.vocab[token] for token in subwords])
            
        return encoded

    def decode(self, ids):
        """
        Decode token IDs back to text
        Args:
            ids (list): List of token IDs
        Returns:
            str: Decoded text
        """
        if not self.inv_vocab:
            raise ValueError("Tokenizer needs to be trained first")
            
        # Convert IDs back to tokens
        tokens = [self.inv_vocab[id] for id in ids]
        # Join tokens
        text = ''.join(tokens)
        # Postprocess to restore special characters
        text = self._postprocess_text(text)
        return text


