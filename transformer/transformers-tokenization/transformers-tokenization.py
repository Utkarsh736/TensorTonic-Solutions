import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        spl_toks = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for tok in spl_toks:
            if tok not in self.word_to_id:
                self.word_to_id[tok] = self.vocab_size
                self.id_to_word[self.vocab_size] = tok
                self.vocab_size += 1
                
        unique_words = set()
        for text in texts:
            words = text.lower().split()
            for word in words:
                unique_words.add(word)

        for i in sorted(unique_words):
            if i not in self.word_to_id:
                self.word_to_id[i] = self.vocab_size
                self.id_to_word[self.vocab_size] = i
                self.vocab_size += 1
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        words = text.lower().split()
        unk_id = self.word_to_id.get(self.unk_token, 1)
        return[self.word_to_id.get(word, unk_id)for word in words]
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        words = [self.id_to_word.get(idx, self.unk_token)for idx in ids]
        return " ".join(words)
