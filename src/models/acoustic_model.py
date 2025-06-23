"""
Base acoustic model with BF16 support for Tsukuyomi TTS

This module provides the core acoustic modeling functionality with
H100-optimized BF16 support and efficient training/inference capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class AcousticModelConfig:
    """Configuration for the acoustic model."""
    
    # Model dimensions
    hidden_dim: int = 512
    n_layers: int = 6
    n_heads: int = 8
    filter_channels: int = 2048
    n_mel_channels: int = 80
    
    # Phoneme embedding
    phoneme_embedding_dim: int = 768  # XPhoneBERT output dimension
    
    # Speaker embedding
    n_speakers: int = 1
    speaker_embedding_dim: int = 256
    
    # Training settings
    use_bf16: bool = True
    use_flash_attention: bool = True
    gradient_checkpointing: bool = False
    
    # Decoder settings
    decoder_type: str = "flow"  # "flow" or "diffusion"
    n_flow_layers: int = 4
    
    # Loss weights
    mel_loss_weight: float = 45.0
    duration_loss_weight: float = 1.0
    kl_loss_weight: float = 1.0


class ConformerBlock(nn.Module):
    """
    Conformer block for acoustic modeling with BF16 optimization.
    
    Combines convolution and self-attention for better acoustic modeling.
    """
    
    def __init__(self, config: AcousticModelConfig):
        super().__init__()
        self.config = config
        
        # Feed-forward module 1
        self.ff1 = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.filter_channels),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(config.filter_channels, config.hidden_dim),
            nn.Dropout(0.1)
        )
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.n_heads,
            dropout=0.1,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(config.hidden_dim)
        
        # Convolution module
        self.conv = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Conv1d(config.hidden_dim, config.hidden_dim * 2, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(config.hidden_dim, config.hidden_dim, kernel_size=9, padding=4, groups=config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.SiLU(),
            nn.Conv1d(config.hidden_dim, config.hidden_dim, kernel_size=1),
            nn.Dropout(0.1)
        )
        
        # Feed-forward module 2
        self.ff2 = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.filter_channels),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(config.filter_channels, config.hidden_dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass through Conformer block.
        
        Args:
            x: Input tensor of shape (batch, time, channels)
            mask: Optional attention mask
            
        Returns:
            Output tensor of same shape as input
        """
        # First feed-forward
        x = x + 0.5 * self.ff1(x)
        
        # Self-attention with BF16 optimization
        if self.config.use_bf16 and x.device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                attn_out, _ = self.self_attn(
                    self.attn_norm(x), 
                    self.attn_norm(x), 
                    self.attn_norm(x),
                    key_padding_mask=mask
                )
        else:
            attn_out, _ = self.self_attn(
                self.attn_norm(x), 
                self.attn_norm(x), 
                self.attn_norm(x),
                key_padding_mask=mask
            )
        x = x + attn_out
        
        # Convolution (need to transpose for Conv1d)
        conv_in = x.transpose(1, 2)  # (B, C, T)
        conv_out = self.conv(conv_in)
        x = x + conv_out.transpose(1, 2)
        
        # Second feed-forward
        x = x + 0.5 * self.ff2(x)
        
        return x


class AcousticModel(nn.Module):
    """
    Base acoustic model for Tsukuyomi TTS with BF16 support.
    
    This model takes phoneme embeddings and produces mel-spectrograms
    or other acoustic features for vocoding.
    """
    
    def __init__(self, config: AcousticModelConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_projection = nn.Linear(
            config.phoneme_embedding_dim, 
            config.hidden_dim
        )
        
        # Speaker embedding
        if config.n_speakers > 1:
            self.speaker_embedding = nn.Embedding(
                config.n_speakers, 
                config.speaker_embedding_dim
            )
            self.speaker_projection = nn.Linear(
                config.speaker_embedding_dim,
                config.hidden_dim
            )
        else:
            self.speaker_embedding = None
            self.speaker_projection = None
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.hidden_dim)
        
        # Conformer encoder stack
        self.encoder_blocks = nn.ModuleList([
            ConformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Duration predictor
        self.duration_predictor = DurationPredictor(config.hidden_dim)
        
        # Decoder projection
        self.decoder_projection = nn.Linear(
            config.hidden_dim,
            config.n_mel_channels
        )
        
        # Initialize weights with BF16-friendly values
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights for stable BF16 training."""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # Use smaller initialization for BF16 stability
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(
        self,
        phoneme_embeddings: torch.Tensor,
        phoneme_mask: Optional[torch.Tensor] = None,
        speaker_ids: Optional[torch.Tensor] = None,
        target_durations: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the acoustic model.
        
        Args:
            phoneme_embeddings: Phoneme embeddings from XPhoneBERT (B, T, 768)
            phoneme_mask: Mask for valid phonemes (B, T)
            speaker_ids: Speaker IDs for multi-speaker (B,)
            target_durations: Target durations for training (B, T)
            
        Returns:
            Dictionary containing:
                - mel_output: Predicted mel-spectrogram (B, T', n_mel)
                - duration_output: Predicted durations (B, T)
                - encoder_output: Encoder hidden states (B, T, hidden_dim)
        """
        batch_size, seq_len, _ = phoneme_embeddings.shape
        
        # Input projection
        hidden = self.input_projection(phoneme_embeddings)
        
        # Add speaker embedding if multi-speaker
        if self.speaker_embedding is not None and speaker_ids is not None:
            speaker_emb = self.speaker_embedding(speaker_ids)  # (B, speaker_dim)
            speaker_emb = self.speaker_projection(speaker_emb)  # (B, hidden_dim)
            speaker_emb = speaker_emb.unsqueeze(1).expand(-1, seq_len, -1)  # (B, T, hidden_dim)
            hidden = hidden + speaker_emb
        
        # Add positional encoding
        hidden = self.pos_encoding(hidden)
        
        # Pass through encoder blocks with optional gradient checkpointing
        for block in self.encoder_blocks:
            if self.config.gradient_checkpointing and self.training:
                hidden = torch.utils.checkpoint.checkpoint(
                    block, hidden, phoneme_mask
                )
            else:
                hidden = block(hidden, phoneme_mask)
        
        encoder_output = hidden
        
        # Predict durations
        duration_output = self.duration_predictor(hidden, phoneme_mask)
        
        # Expand hidden states according to durations
        if target_durations is not None:
            # Training: use target durations
            expanded_hidden = self._expand_states(hidden, target_durations)
        else:
            # Inference: use predicted durations
            predicted_durations = torch.round(torch.exp(duration_output) - 1).long()
            predicted_durations = predicted_durations.clamp(min=0, max=100)  # Prevent extreme values
            expanded_hidden = self._expand_states(hidden, predicted_durations)
        
        # Decode to mel-spectrogram
        mel_output = self.decoder_projection(expanded_hidden)
        
        return {
            "mel_output": mel_output,
            "duration_output": duration_output,
            "encoder_output": encoder_output
        }
    
    def _expand_states(
        self, 
        hidden_states: torch.Tensor, 
        durations: torch.Tensor
    ) -> torch.Tensor:
        """
        Expand hidden states according to durations.
        
        Args:
            hidden_states: Encoder outputs (B, T, hidden_dim)
            durations: Duration for each phoneme (B, T)
            
        Returns:
            Expanded states (B, T', hidden_dim)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Calculate output length for each sequence
        out_lengths = durations.sum(dim=1)
        max_out_len = out_lengths.max().item()
        
        # Expand each sequence
        expanded_states = []
        for b in range(batch_size):
            expanded = []
            for t in range(seq_len):
                duration = durations[b, t].item()
                if duration > 0:
                    expanded.append(
                        hidden_states[b, t:t+1].expand(duration, -1)
                    )
            
            if expanded:
                expanded = torch.cat(expanded, dim=0)
                # Pad to max length
                if expanded.shape[0] < max_out_len:
                    padding = torch.zeros(
                        max_out_len - expanded.shape[0],
                        hidden_dim,
                        device=expanded.device,
                        dtype=expanded.dtype
                    )
                    expanded = torch.cat([expanded, padding], dim=0)
                else:
                    expanded = expanded[:max_out_len]
            else:
                expanded = torch.zeros(
                    max_out_len, hidden_dim,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype
                )
            
            expanded_states.append(expanded)
        
        return torch.stack(expanded_states)


class DurationPredictor(nn.Module):
    """Duration predictor module."""
    
    def __init__(self, hidden_dim: int, filter_size: int = 256, n_layers: int = 2):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_dim if i == 0 else filter_size,
                        filter_size,
                        kernel_size=3,
                        padding=1
                    ),
                    nn.ReLU(),
                    nn.LayerNorm(filter_size),
                    nn.Dropout(0.1)
                )
            )
        
        self.projection = nn.Linear(filter_size, 1)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Predict log-durations for input sequence.
        
        Args:
            x: Input features (B, T, C)
            mask: Padding mask (B, T)
            
        Returns:
            Log-durations (B, T)
        """
        # Transpose for conv layers
        x = x.transpose(1, 2)  # (B, C, T)
        
        for layer in self.layers:
            x = layer(x)
        
        x = x.transpose(1, 2)  # (B, T, C)
        duration = self.projection(x).squeeze(-1)  # (B, T)
        
        if mask is not None:
            duration = duration.masked_fill(mask, 0.0)
        
        return duration


class PositionalEncoding(nn.Module):
    """Positional encoding module."""
    
    def __init__(self, hidden_dim: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() *
            -(np.log(10000.0) / hidden_dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:, :x.size(1)]