import torch
import numpy as np
from typing import List, Optional, Union
import json
import regex as re

class WhisperTokenizer:
    def __init__(self, vocab_path: str = None):
        if vocab_path is None:
            # Default Mongolian vocabulary
            self.vocab = {
                "<|startoftranscript|>": 50257,
                "<|endoftext|>": 50256,
                "<|notimestamps|>": 50258,
                " ": 220,
                "а": 288,
                "б": 289,
                "в": 290,
                "г": 291,
                "д": 292,
                "е": 293,
                "ё": 294,
                "ж": 295,
                "з": 296,
                "и": 297,
                "й": 298,
                "к": 299,
                "л": 300,
                "м": 301,
                "н": 302,
                "о": 303,
                "ө": 304,
                "п": 305,
                "р": 306,
                "с": 307,
                "т": 308,
                "у": 309,
                "ү": 310,
                "ф": 311,
                "х": 312,
                "ц": 313,
                "ч": 314,
                "ш": 315,
                "щ": 316,
                "ъ": 317,
                "ы": 318,
                "ь": 319,
                "э": 320,
                "ю": 321,
                "я": 322,
                ".": 323,
                ",": 324,
                "?": 325,
                "!": 326,
            }
        else:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.vocab = json.load(f)
                
        self.tokens = {v: k for k, v in self.vocab.items()}
        self.n_vocab = max(self.vocab.values()) + 1
        
        self.bos_token = "<|startoftranscript|>"
        self.eos_token = "<|endoftext|>"
        self.bos_token_id = self.vocab[self.bos_token]
        self.eos_token_id = self.vocab[self.eos_token]
        
    def encode(self, text: str) -> List[int]:
        """Convert text to token ids."""
        if not text:
            return []
            
        # Add special tokens
        text = f"{self.bos_token}{text}{self.eos_token}"
        
        # Simple character-based tokenization
        tokens = []
        for char in text:
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                # Skip unknown characters
                continue
                
        return tokens
        
    def decode(self, token_ids: List[int]) -> str:
        """Convert token ids back to text."""
        text = ""
        for tid in token_ids:
            if tid in self.tokens:
                text += self.tokens[tid]
                
        # Remove special tokens
        text = text.replace(self.bos_token, "").replace(self.eos_token, "")
        return text.strip()
        
    def batch_encode(self, texts: List[str], max_length: Optional[int] = None) -> torch.Tensor:
        """Encode a batch of texts."""
        batch_tokens = [self.encode(text) for text in texts]
        
        if max_length is None:
            max_length = max(len(tokens) for tokens in batch_tokens)
            
        # Pad sequences
        padded_tokens = []
        attention_mask = []
        
        for tokens in batch_tokens:
            padding = [self.eos_token_id] * (max_length - len(tokens))
            padded_tokens.append(tokens + padding)
            attention_mask.append([1] * len(tokens) + [0] * len(padding))
            
        return (
            torch.tensor(padded_tokens),
            torch.tensor(attention_mask)
        )
        
    def batch_decode(self, token_ids: torch.Tensor) -> List[str]:
        """Decode a batch of token ids."""
        if token_ids.ndim == 1:
            return self.decode(token_ids.tolist())
            
        return [self.decode(ids.tolist()) for ids in token_ids]