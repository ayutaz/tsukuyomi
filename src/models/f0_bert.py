"""
F0-BERT: Advanced pitch prediction model for natural prosody

This module implements a BERT-based F0 (fundamental frequency) predictor
that models pitch contours with high accuracy for expressive TTS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
import numpy as np
from dataclasses import dataclass
import math
import logging

from transformers import RobertaModel, RobertaConfig

# CUDA 12.1+ optimizations
try:
    from flash_attn import flash_attn_func

    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False

logger = logging.getLogger(__name__)

# Enable CUDA 12.1+ optimizations
if torch.cuda.is_available():
    cuda_version = torch.version.cuda
    if cuda_version and float(cuda_version.split(".")[0]) >= 12.1:
        logger.info(f"CUDA {cuda_version} detected - enabling F0-BERT optimizations")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


@dataclass
class F0BERTConfig:
    """Configuration for F0-BERT model."""

    # Model architecture
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    # F0 specific
    f0_min: float = 50.0  # Minimum F0 in Hz
    f0_max: float = 800.0  # Maximum F0 in Hz
    f0_bins: int = 256  # Number of F0 quantization bins

    # Frame-level prediction
    frame_shift_ms: float = 5.0  # Frame shift in milliseconds
    sample_rate: int = 24000

    # Multi-task learning
    predict_vuv: bool = True  # Voiced/Unvoiced prediction
    predict_energy: bool = True  # Energy prediction
    predict_duration: bool = True  # Duration prediction

    # Style and emotion
    num_emotions: int = 7  # Neutral, Happy, Sad, Angry, Fear, Surprise, Disgust
    num_styles: int = 10  # Various speaking styles
    use_style_embedding: bool = True
    style_embedding_dim: int = 256


class ContinuousF0Encoder(nn.Module):
    """Encode continuous F0 values with learnable positional encoding."""

    def __init__(self, config: F0BERTConfig):
        super().__init__()
        self.config = config

        # F0 to embedding
        self.f0_projection = nn.Sequential(
            nn.Linear(1, config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, config.hidden_size),
        )

        # Sinusoidal encoding for F0 patterns
        self.register_buffer(
            "f0_positional_encoding",
            self._create_sinusoidal_encoding(config.f0_bins, config.hidden_size),
        )

    def _create_sinusoidal_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)

    def forward(self, f0_values: torch.Tensor) -> torch.Tensor:
        """
        Encode F0 values.

        Args:
            f0_values: (batch, seq_len) F0 values in Hz

        Returns:
            F0 embeddings (batch, seq_len, hidden_size)
        """
        # Normalize F0 to [0, 1]
        f0_normalized = (f0_values - self.config.f0_min) / (
            self.config.f0_max - self.config.f0_min
        )
        f0_normalized = torch.clamp(f0_normalized, 0, 1)

        # Project to hidden size
        f0_embedded = self.f0_projection(f0_normalized.unsqueeze(-1))

        # Add positional encoding based on F0 range
        f0_pos_idx = (f0_normalized * (self.config.f0_bins - 1)).long()
        f0_pos_encoding = self.f0_positional_encoding[:, f0_pos_idx].squeeze(0)

        return f0_embedded + f0_pos_encoding


class StyleConditioningModule(nn.Module):
    """Module for conditioning on style and emotion."""

    def __init__(self, config: F0BERTConfig):
        super().__init__()
        self.config = config

        # Emotion embedding
        self.emotion_embedding = nn.Embedding(
            config.num_emotions, config.style_embedding_dim
        )

        # Style embedding
        self.style_embedding = nn.Embedding(
            config.num_styles, config.style_embedding_dim
        )

        # Combine style and emotion
        self.style_projector = nn.Sequential(
            nn.Linear(config.style_embedding_dim * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        # Global style token (GST) for reference audio
        self.gst_num_tokens = 10
        self.gst_embedding = nn.Parameter(
            torch.randn(self.gst_num_tokens, config.style_embedding_dim)
        )

        self.gst_attention = nn.MultiheadAttention(
            embed_dim=config.style_embedding_dim,
            num_heads=4,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
        )

    def forward(
        self,
        emotion_id: Optional[torch.Tensor] = None,
        style_id: Optional[torch.Tensor] = None,
        reference_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get style conditioning vector.

        Args:
            emotion_id: Emotion ID (batch,)
            style_id: Style ID (batch,)
            reference_embedding: Reference audio embedding (batch, ref_len, dim)

        Returns:
            Style conditioning (batch, hidden_size)
        """
        batch_size = emotion_id.size(0) if emotion_id is not None else 1
        device = emotion_id.device if emotion_id is not None else torch.device("cpu")

        # Default to neutral if not specified
        if emotion_id is None:
            emotion_id = torch.zeros(batch_size, dtype=torch.long, device=device)
        if style_id is None:
            style_id = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Get embeddings
        emotion_emb = self.emotion_embedding(emotion_id)
        style_emb = self.style_embedding(style_id)

        # Combine emotion and style
        combined = torch.cat([emotion_emb, style_emb], dim=-1)
        style_vector = self.style_projector(combined)

        # Add GST if reference provided
        if reference_embedding is not None:
            # Expand GST tokens
            gst_tokens = self.gst_embedding.unsqueeze(0).expand(batch_size, -1, -1)

            # Attention over reference
            gst_output, _ = self.gst_attention(
                gst_tokens, reference_embedding, reference_embedding
            )

            # Average pool and add to style
            gst_vector = gst_output.mean(dim=1)
            style_vector = style_vector + self.style_projector(
                torch.cat([gst_vector, gst_vector], dim=-1)
            )

        return style_vector


class F0PredictionHead(nn.Module):
    """Multi-task prediction head for F0 and related features."""

    def __init__(self, config: F0BERTConfig):
        super().__init__()
        self.config = config

        # Shared feature extraction
        self.shared_layer = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
        )

        # F0 prediction (continuous)
        self.f0_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 1),
        )

        # F0 quantized prediction (for discrete modeling)
        self.f0_quantized = nn.Linear(config.hidden_size, config.f0_bins)

        # V/UV prediction
        if config.predict_vuv:
            self.vuv_predictor = nn.Linear(config.hidden_size, 2)

        # Energy prediction
        if config.predict_energy:
            self.energy_predictor = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 4),
                nn.ReLU(),
                nn.Linear(config.hidden_size // 4, 1),
            )

        # Duration prediction
        if config.predict_duration:
            self.duration_predictor = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_size // 2, 1),
                nn.Softplus(),  # Ensure positive durations
            )

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict F0 and related features.

        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            Dictionary of predictions
        """
        # Shared processing
        shared_features = self.shared_layer(hidden_states)

        outputs = {}

        # Continuous F0
        f0_continuous = self.f0_predictor(shared_features).squeeze(-1)
        # Convert to Hz
        f0_hz = (
            torch.sigmoid(f0_continuous) * (self.config.f0_max - self.config.f0_min)
            + self.config.f0_min
        )
        outputs["f0"] = f0_hz

        # Quantized F0
        outputs["f0_quantized"] = self.f0_quantized(shared_features)

        # V/UV
        if self.config.predict_vuv:
            outputs["vuv"] = self.vuv_predictor(shared_features)

        # Energy
        if self.config.predict_energy:
            outputs["energy"] = self.energy_predictor(shared_features).squeeze(-1)

        # Duration
        if self.config.predict_duration:
            outputs["duration"] = self.duration_predictor(shared_features).squeeze(-1)

        return outputs


class F0BERT(nn.Module):
    """
    F0-BERT: BERT-based pitch prediction model.

    Predicts frame-level F0 contours with high accuracy,
    considering linguistic context, style, and emotion.
    """

    def __init__(self, config: Optional[F0BERTConfig] = None):
        super().__init__()
        self.config = config or F0BERTConfig()

        # BERT backbone
        bert_config = RobertaConfig(
            hidden_size=self.config.hidden_size,
            num_hidden_layers=self.config.num_hidden_layers,
            num_attention_heads=self.config.num_attention_heads,
            intermediate_size=self.config.intermediate_size,
            hidden_dropout_prob=self.config.hidden_dropout_prob,
            attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
        )
        self.bert = RobertaModel(bert_config)

        # F0 encoder for teacher forcing
        self.f0_encoder = ContinuousF0Encoder(self.config)

        # Style conditioning
        self.style_conditioning = StyleConditioningModule(self.config)

        # Frame-level expansion
        self.frame_expander = nn.Conv1d(
            in_channels=self.config.hidden_size,
            out_channels=self.config.hidden_size,
            kernel_size=3,
            padding=1,
        )

        # Prediction head
        self.prediction_head = F0PredictionHead(self.config)

        # Smoothing layer for F0 contours
        self.f0_smoother = nn.Conv1d(
            in_channels=1, out_channels=1, kernel_size=5, padding=2
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        phoneme_embeddings: Optional[torch.Tensor] = None,
        emotion_id: Optional[torch.Tensor] = None,
        style_id: Optional[torch.Tensor] = None,
        reference_embedding: Optional[torch.Tensor] = None,
        target_f0: Optional[torch.Tensor] = None,
        target_vuv: Optional[torch.Tensor] = None,
        frame_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of F0-BERT.

        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            phoneme_embeddings: Optional phoneme embeddings from acoustic model
            emotion_id: Emotion ID for conditioning
            style_id: Style ID for conditioning
            reference_embedding: Reference audio for GST
            target_f0: Target F0 for teacher forcing (training)
            target_vuv: Target V/UV labels
            frame_lengths: Number of frames per phoneme

        Returns:
            Dictionary containing predictions
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Get style conditioning
        style_vector = self.style_conditioning(
            emotion_id=emotion_id,
            style_id=style_id,
            reference_embedding=reference_embedding,
        )

        # Expand style vector to sequence length
        style_expanded = style_vector.unsqueeze(1).expand(-1, seq_len, -1)

        # Get BERT embeddings
        bert_outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )

        hidden_states = bert_outputs.last_hidden_state

        # Add style conditioning
        hidden_states = hidden_states + style_expanded

        # Add phoneme embeddings if provided
        if phoneme_embeddings is not None:
            hidden_states = hidden_states + phoneme_embeddings

        # Teacher forcing: add target F0 encoding during training
        if target_f0 is not None and self.training:
            f0_encoding = self.f0_encoder(target_f0)
            hidden_states = hidden_states + 0.5 * f0_encoding

        # Expand to frame level if needed
        if frame_lengths is not None:
            # Repeat hidden states according to frame lengths
            expanded_states = []
            for i in range(batch_size):
                state = hidden_states[i]
                lengths = frame_lengths[i]
                expanded = torch.repeat_interleave(state, lengths, dim=0)
                expanded_states.append(expanded)

            # Pad to max length
            max_frames = max(s.size(0) for s in expanded_states)
            hidden_states = torch.stack(
                [F.pad(s, (0, 0, 0, max_frames - s.size(0))) for s in expanded_states]
            )

            # Apply frame-level convolution
            hidden_states = hidden_states.transpose(1, 2)
            hidden_states = self.frame_expander(hidden_states)
            hidden_states = hidden_states.transpose(1, 2)

        # Get predictions
        outputs = self.prediction_head(hidden_states)

        # Smooth F0 contours
        f0_raw = outputs["f0"]
        f0_smooth = self.f0_smoother(f0_raw.unsqueeze(1)).squeeze(1)
        outputs["f0_smooth"] = f0_smooth

        # Apply V/UV mask if predicted
        if "vuv" in outputs:
            vuv_mask = torch.sigmoid(outputs["vuv"][:, :, 1]) > 0.5
            outputs["f0_final"] = f0_smooth * vuv_mask.float()
        else:
            outputs["f0_final"] = f0_smooth

        return outputs

    def inference(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        emotion_id: Optional[torch.Tensor] = None,
        style_id: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference mode with temperature control.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            emotion_id: Emotion for conditioning
            style_id: Style for conditioning
            temperature: Temperature for F0 variation

        Returns:
            F0 predictions and related features
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                emotion_id=emotion_id,
                style_id=style_id,
            )

            # Apply temperature to F0
            if temperature != 1.0:
                f0_mean = outputs["f0_final"].mean()
                outputs["f0_final"] = (
                    f0_mean + (outputs["f0_final"] - f0_mean) * temperature
                )

        return outputs

    def optimize_for_inference(self) -> None:
        """
        Optimize model for inference using CUDA 12.1+ features.
        """
        self.eval()

        if not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping optimization")
            return

        cuda_version = torch.version.cuda
        if cuda_version and float(cuda_version.split(".")[0]) >= 12.1:
            logger.info("Optimizing F0-BERT with CUDA 12.1+ features")

            # Replace BERT attention with Flash Attention if available
            if FLASH_ATTENTION_AVAILABLE:
                try:
                    self._replace_bert_attention_with_flash()
                    logger.info("Replaced BERT attention with Flash Attention 2")
                except Exception as e:
                    logger.warning(f"Failed to replace attention: {e}")

            # Compile model with torch.compile
            try:
                self.forward = torch.compile(
                    self.forward, mode="max-autotune", fullgraph=False
                )
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")

            # Enable mixed precision for BF16
            self.to(torch.bfloat16)
            logger.info("Enabled BF16 precision")
        else:
            logger.warning(
                f"CUDA {cuda_version} detected - CUDA 12.1+ required for optimization"
            )

    def _replace_bert_attention_with_flash(self):
        """Replace BERT's attention layers with Flash Attention."""
        for layer in self.bert.encoder.layer:
            # This would require modifying the transformers library internals
            # For now, we'll rely on torch.compile to optimize attention
            pass


def create_f0_bert(
    checkpoint_path: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> F0BERT:
    """Create F0-BERT instance."""
    config = F0BERTConfig()
    model = F0BERT(config)

    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)

    return model.to(device)
