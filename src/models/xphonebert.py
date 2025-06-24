"""
XPhoneBERT wrapper for multilingual phoneme embeddings

This module provides a wrapper around the XPhoneBERT model for extracting
high-quality phoneme embeddings from text, with support for Japanese and
other languages.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Optional, Union
from collections.abc import Sequence
import numpy as np

# Import text2phonemesequence from frontend directory
from ..frontend.text2phonemesequence import Text2PhonemeSequence


class XPhoneBERTEncoder(nn.Module):
    """
    XPhoneBERT encoder for extracting phoneme embeddings.

    Uses the vinai/xphonebert-base model to create 768-dimensional
    embeddings from phonemized text input.
    """

    def __init__(
        self,
        model_name: str = "vinai/xphonebert-base",
        device: Optional[str] = None,
        use_bf16: bool = True,
        max_length: int = 512,
    ):
        """
        Initialize XPhoneBERT encoder.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run the model on (None for auto-detect)
            use_bf16: Whether to use BF16 precision (for H100 optimization)
            max_length: Maximum sequence length for tokenization
        """
        super().__init__()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_bf16 = use_bf16 and self.device == "cuda"
        self.max_length = max_length

        # Load tokenizer and model
        # Try to load without authentication first
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, use_auth_token=False
            )
            self.model = AutoModel.from_pretrained(model_name, use_auth_token=False)
        except Exception as e:
            # If that fails, try with trust_remote_code
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=True
                )
                self.model = AutoModel.from_pretrained(
                    model_name, trust_remote_code=True
                )
            except Exception as e2:
                # If both fail, raise informative error
                raise RuntimeError(
                    f"Failed to load XPhoneBERT model '{model_name}'. "
                    f"Error: {e}. "
                    f"The model should be publicly available. "
                    f"You can try: pip install --upgrade transformers"
                )

        # Move to device and set precision
        self.model = self.model.to(self.device)
        if self.use_bf16:
            self.model = self.model.to(dtype=torch.bfloat16)

        # Freeze parameters (we're using it as a feature extractor)
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

    def forward(
        self, phoneme_sequences: str | Sequence[str], return_pooled: bool = True
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Extract phoneme embeddings from input sequences.

        Args:
            phoneme_sequences: Single phoneme sequence or list of sequences
            return_pooled: If True, return pooled output; if False, return all token embeddings

        Returns:
            If return_pooled=True: Tensor of shape (batch_size, 768)
            If return_pooled=False: Tuple of (all_embeddings, attention_mask)
                where all_embeddings has shape (batch_size, seq_len, 768)
        """
        if isinstance(phoneme_sequences, str):
            phoneme_sequences = [phoneme_sequences]

        # Tokenize input
        inputs = self.tokenizer(
            phoneme_sequences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        # Extract embeddings
        with torch.no_grad():
            if self.use_bf16:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)

        if return_pooled:
            # Use [CLS] token embedding as pooled representation
            return outputs.last_hidden_state[:, 0, :]
        else:
            # Return all token embeddings with attention mask
            return outputs.last_hidden_state, inputs.attention_mask

    def extract_frame_level_features(
        self,
        phoneme_sequences: str | Sequence[str],
        phoneme_durations: Sequence[Sequence[int]] | None = None,
    ) -> torch.Tensor:
        """
        Extract frame-level features by expanding phoneme embeddings
        according to their durations.

        Args:
            phoneme_sequences: Phoneme sequences
            phoneme_durations: Duration (in frames) for each phoneme.
                              If None, returns token-level features.

        Returns:
            Frame-level features of shape (batch_size, n_frames, 768)
        """
        # Get token-level embeddings
        embeddings, attention_mask = self.forward(
            phoneme_sequences, return_pooled=False
        )

        if phoneme_durations is None:
            return embeddings

        batch_size = embeddings.shape[0]
        frame_features = []

        for b in range(batch_size):
            # Get valid embeddings (excluding padding)
            valid_length = attention_mask[b].sum().item()
            valid_embeddings = embeddings[
                b, 1 : valid_length - 1
            ]  # Exclude [CLS] and [SEP]

            if phoneme_durations[b] is not None:
                # Expand each phoneme embedding by its duration
                expanded_features = []
                for emb, dur in zip(valid_embeddings, phoneme_durations[b]):
                    expanded_features.append(emb.unsqueeze(0).expand(dur, -1))

                frame_feature = torch.cat(expanded_features, dim=0)
            else:
                frame_feature = valid_embeddings

            frame_features.append(frame_feature)

        # Pad to same length
        max_frames = max(f.shape[0] for f in frame_features)
        padded_features = []

        for f in frame_features:
            if f.shape[0] < max_frames:
                padding = torch.zeros(
                    max_frames - f.shape[0], f.shape[1], device=f.device, dtype=f.dtype
                )
                f = torch.cat([f, padding], dim=0)
            padded_features.append(f)

        return torch.stack(padded_features)

    def get_embedding_dim(self) -> int:
        """Return the dimension of the embeddings."""
        return 768
