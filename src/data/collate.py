"""
Collate functions for TTS data loading.
"""

import torch
from typing import Dict, List, Optional, Union
import numpy as np


def tts_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for TTS training.
    
    Args:
        batch: List of dictionaries from TsukuyomiDataset
        
    Returns:
        Dictionary with batched tensors
    """
    # Separate different fields
    texts = [item['text'] for item in batch]
    speaker_ids = [item['speaker_id'] for item in batch]
    audio_paths = [item['audio_path'] for item in batch]
    
    # Handle audio - find max length and pad
    audios = [item['audio'] for item in batch]
    max_audio_len = max(audio.shape[0] for audio in audios)
    
    # Pad audios to same length
    padded_audios = []
    audio_lengths = []
    for audio in audios:
        audio_len = audio.shape[0]
        audio_lengths.append(audio_len)
        if audio_len < max_audio_len:
            padding = max_audio_len - audio_len
            padded_audio = torch.nn.functional.pad(audio, (0, padding), value=0.0)
            padded_audios.append(padded_audio)
        else:
            padded_audios.append(audio)
    
    # Stack into batch tensors
    audio_batch = torch.stack(padded_audios, dim=0)
    audio_lengths = torch.tensor(audio_lengths, dtype=torch.long)
    
    # Map speaker IDs to indices
    unique_speakers = sorted(list(set(speaker_ids)))
    speaker_id_to_idx = {sid: idx for idx, sid in enumerate(unique_speakers)}
    speaker_indices = torch.tensor([speaker_id_to_idx[sid] for sid in speaker_ids], dtype=torch.long)
    
    return {
        'audio': audio_batch,
        'audio_lengths': audio_lengths,
        'text': texts,
        'speaker_ids': speaker_indices,
        'speaker_names': speaker_ids,
        'audio_paths': audio_paths,
    }