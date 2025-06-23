"""Text to phoneme sequence conversion module

This module provides the interface between text processing and phoneme sequences
"""

import torch
from typing import List, Dict, Optional, Tuple, Union
import numpy as np


class Text2PhonemeSequence:
    """Convert text to phoneme sequences for TTS models"""
    
    def __init__(
        self,
        vocab_size: int = 256,
        pad_token: int = 0,
        unk_token: int = 1,
        bos_token: int = 2,
        eos_token: int = 3,
        use_g2p: bool = True,
    ):
        """
        Initialize text to phoneme sequence converter
        
        Args:
            vocab_size: Size of phoneme vocabulary
            pad_token: Padding token ID
            unk_token: Unknown token ID
            bos_token: Beginning of sequence token ID
            eos_token: End of sequence token ID
            use_g2p: Whether to use G2P conversion
        """
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.use_g2p = use_g2p
        
        # Build phoneme vocabulary
        self._build_vocab()
        
    def _build_vocab(self):
        """Build phoneme vocabulary"""
        # Basic phoneme set (simplified)
        # In practice, this should be loaded from a comprehensive phoneme list
        basic_phonemes = [
            'pau', 'cl', 'a', 'i', 'u', 'e', 'o',  # Japanese vowels
            'k', 'g', 's', 'z', 't', 'd', 'n', 'h', 'b', 'p', 'm', 'y', 'r', 'w',  # Consonants
            'ky', 'gy', 'sh', 'ch', 'ny', 'hy', 'by', 'py', 'my', 'ry',  # Palatalized
            'N', 'q',  # Special phonemes
            'A', 'I', 'U', 'E', 'O',  # Accented versions
        ]
        
        # Add prosody markers
        prosody_markers = [
            '[', ']',  # Phrase boundaries
            '{', '}',  # Accent phrase boundaries
            '^',  # Accent nucleus
            '_',  # Word boundary
            '#',  # Utterance boundary
        ]
        
        # Build vocab
        self.phoneme2id = {
            '<pad>': self.pad_token,
            '<unk>': self.unk_token,
            '<bos>': self.bos_token,
            '<eos>': self.eos_token,
        }
        
        idx = len(self.phoneme2id)
        for phoneme in basic_phonemes + prosody_markers:
            if phoneme not in self.phoneme2id:
                self.phoneme2id[phoneme] = idx
                idx += 1
                
        # Add remaining slots up to vocab_size
        for i in range(idx, self.vocab_size):
            self.phoneme2id[f'<reserved_{i}>'] = i
            
        self.id2phoneme = {v: k for k, v in self.phoneme2id.items()}
        
    def text_to_phonemes(self, text: str, language: str = 'ja') -> List[str]:
        """
        Convert text to phoneme sequence
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            List of phonemes
        """
        if not self.use_g2p:
            # Simple character-based fallback
            phonemes = list(text.lower())
        else:
            # This should use the actual G2P from japanese_g2p.py
            # For now, we'll use a simplified version
            if language == 'ja':
                from ..frontend.japanese_g2p import JapaneseG2P
                g2p = JapaneseG2P()
                phonemes = g2p.g2p(text)
            else:
                # Simple phoneme mapping for other languages
                phonemes = self._simple_g2p(text)
                
        return phonemes
        
    def _simple_g2p(self, text: str) -> List[str]:
        """Simple G2P for non-Japanese languages"""
        # Very basic mapping - in practice, use proper G2P
        char_to_phoneme = {
            'a': 'a', 'e': 'e', 'i': 'i', 'o': 'o', 'u': 'u',
            'b': 'b', 'c': 'k', 'd': 'd', 'f': 'h', 'g': 'g',
            'h': 'h', 'j': 'y', 'k': 'k', 'l': 'r', 'm': 'm',
            'n': 'n', 'p': 'p', 'q': 'k', 'r': 'r', 's': 's',
            't': 't', 'v': 'b', 'w': 'w', 'x': 'k', 'y': 'y',
            'z': 'z', ' ': 'pau', '.': 'pau', ',': 'pau',
        }
        
        phonemes = []
        for char in text.lower():
            if char in char_to_phoneme:
                phonemes.append(char_to_phoneme[char])
            else:
                phonemes.append('pau')
                
        return phonemes
        
    def phonemes_to_ids(self, phonemes: List[str]) -> List[int]:
        """Convert phonemes to IDs"""
        ids = []
        for phoneme in phonemes:
            if phoneme in self.phoneme2id:
                ids.append(self.phoneme2id[phoneme])
            else:
                ids.append(self.unk_token)
        return ids
        
    def ids_to_phonemes(self, ids: List[int]) -> List[str]:
        """Convert IDs back to phonemes"""
        phonemes = []
        for id in ids:
            if id in self.id2phoneme:
                phonemes.append(self.id2phoneme[id])
            else:
                phonemes.append('<unk>')
        return phonemes
        
    def __call__(
        self,
        text: Union[str, List[str]],
        language: str = 'ja',
        add_bos_eos: bool = True,
        pad_to_length: Optional[int] = None,
        return_tensors: bool = True,
    ) -> Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Convert text to phoneme sequence IDs
        
        Args:
            text: Input text or list of texts
            language: Language code
            add_bos_eos: Whether to add BOS/EOS tokens
            pad_to_length: Pad sequences to this length
            return_tensors: Whether to return PyTorch tensors
            
        Returns:
            Phoneme sequence IDs
        """
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
            
        all_ids = []
        
        for t in texts:
            # Convert to phonemes
            phonemes = self.text_to_phonemes(t, language)
            
            # Convert to IDs
            ids = self.phonemes_to_ids(phonemes)
            
            # Add BOS/EOS
            if add_bos_eos:
                ids = [self.bos_token] + ids + [self.eos_token]
                
            all_ids.append(ids)
            
        # Pad sequences
        if pad_to_length is not None or len(texts) > 1:
            max_len = pad_to_length or max(len(ids) for ids in all_ids)
            padded_ids = []
            attention_mask = []
            
            for ids in all_ids:
                pad_len = max_len - len(ids)
                padded = ids + [self.pad_token] * pad_len
                mask = [1] * len(ids) + [0] * pad_len
                
                padded_ids.append(padded[:max_len])
                attention_mask.append(mask[:max_len])
                
            all_ids = padded_ids
            
            if return_tensors:
                return {
                    'input_ids': torch.tensor(all_ids, dtype=torch.long),
                    'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                }
            else:
                return {'input_ids': all_ids, 'attention_mask': attention_mask}
        else:
            if return_tensors:
                return torch.tensor(all_ids[0], dtype=torch.long)
            else:
                return all_ids[0]
                
    def decode(self, ids: Union[List[int], torch.Tensor]) -> str:
        """Decode phoneme IDs back to text representation"""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
            
        phonemes = self.ids_to_phonemes(ids)
        
        # Remove special tokens
        phonemes = [p for p in phonemes if p not in ['<pad>', '<bos>', '<eos>', '<unk>']]
        
        # Join with spaces
        return ' '.join(phonemes)
        
    def batch_decode(self, ids: Union[List[List[int]], torch.Tensor]) -> List[str]:
        """Decode batch of phoneme IDs"""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
            
        return [self.decode(seq) for seq in ids]


# Global instance for compatibility
text2phonemesequence = Text2PhonemeSequence()


# Convenience functions
def text_to_sequence(text: str, language: str = 'ja') -> torch.Tensor:
    """Convert text to phoneme sequence tensor"""
    return text2phonemesequence(text, language=language, return_tensors=True)


def sequence_to_text(sequence: Union[List[int], torch.Tensor]) -> str:
    """Convert phoneme sequence back to text representation"""
    return text2phonemesequence.decode(sequence)