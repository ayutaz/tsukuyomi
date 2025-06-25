"""VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech

Based on: https://arxiv.org/abs/2106.06103
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .commons import generate_path, monotonic_align, rand_slice_segments, sequence_mask
from .f0_bert import F0BERT, F0BERTConfig
from .hifigan import Generator as HiFiGANGenerator
from .modules import (
    DurationPredictor,
    PosteriorEncoder,
    ResidualCouplingBlock,
    StochasticDurationPredictor,
    TextEncoder,
)


class VITS(nn.Module):
    """VITS: Variational Inference with adversarial learning for end-to-end Text-to-Speech"""

    def __init__(
        self,
        n_vocab: int = 256,
        n_speakers: int = 1,
        speaker_embedding_dim: int = 256,
        n_feats: int = 80,
        sample_rate: int = 22050,
        hop_length: int = 256,
        win_length: int = 1024,
        n_fft: int = 1024,
        filter_length: int = 1024,
        # Text encoder
        hidden_channels: int = 192,
        filter_channels: int = 768,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        # Flow
        n_flows: int = 4,
        # Duration predictor
        duration_predictor_filter_channels: int = 256,
        duration_predictor_kernel_size: int = 3,
        duration_predictor_p_dropout: float = 0.5,
        # Decoder
        upsample_rates: List[int] = [8, 8, 2, 2],
        upsample_initial_channel: int = 512,
        upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    ):
        super().__init__()

        self.n_vocab = n_vocab
        self.n_speakers = n_speakers
        self.n_feats = n_feats
        self.sample_rate = sample_rate
        self.hop_length = hop_length

        # Speaker embedding
        if n_speakers > 1:
            self.speaker_embedding = nn.Embedding(n_speakers, speaker_embedding_dim)
        else:
            self.speaker_embedding = None

        # Text encoder
        self.text_encoder = TextEncoder(
            n_vocab=n_vocab,
            n_feats=n_feats,
            n_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            speaker_embedding_dim=speaker_embedding_dim if n_speakers > 1 else 0,
        )

        # Posterior encoder
        self.posterior_encoder = PosteriorEncoder(
            in_channels=n_feats,
            out_channels=n_feats,
            hidden_channels=hidden_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=16,
            speaker_embedding_dim=speaker_embedding_dim if n_speakers > 1 else 0,
        )

        # Flow
        self.flow = ResidualCouplingBlock(
            channels=n_feats,
            hidden_channels=hidden_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=n_flows,
            speaker_embedding_dim=speaker_embedding_dim if n_speakers > 1 else 0,
        )

        # Duration predictor
        self.duration_predictor = StochasticDurationPredictor(
            in_channels=hidden_channels,
            filter_channels=duration_predictor_filter_channels,
            kernel_size=duration_predictor_kernel_size,
            p_dropout=duration_predictor_p_dropout,
            n_flows=4,
            speaker_embedding_dim=speaker_embedding_dim if n_speakers > 1 else 0,
        )

        # Decoder (HiFi-GAN)
        self.decoder = HiFiGANGenerator(
            in_channels=n_feats,
            upsample_initial_channel=upsample_initial_channel,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
        )

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        mel: Optional[torch.Tensor] = None,
        mel_lengths: Optional[torch.Tensor] = None,
        speaker_ids: Optional[torch.Tensor] = None,
        durations: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            text: [B, T_text]
            text_lengths: [B]
            mel: [B, C, T_mel] (for training)
            mel_lengths: [B] (for training)
            speaker_ids: [B] (if multi-speaker)
            durations: [B, T_text] (optional, for training)

        Returns:
            Dictionary containing:
                - audio: Generated audio [B, 1, T_audio]
                - mel_outputs: Predicted mel-spectrogram [B, C, T_mel]
                - log_duration_prediction: [B, T_text]
                - attention: [B, T_mel, T_text]
                - losses: Dictionary of losses (if training)
        """
        # Get speaker embeddings
        if self.speaker_embedding is not None and speaker_ids is not None:
            g = self.speaker_embedding(speaker_ids).unsqueeze(-1)  # [B, C, 1]
        else:
            g = None

        # Text encoding
        x, m_p, logs_p, x_mask = self.text_encoder(text, text_lengths, g)

        if mel is not None:  # Training mode
            # Posterior encoding
            z, m_q, logs_q, y_mask = self.posterior_encoder(mel, mel_lengths, g)

            # Flow
            z_p = self.flow(z, y_mask, g=g)

            # Duration prediction
            with torch.no_grad():
                # Negative cross-entropy
                s_p_sq_r = torch.exp(-2 * logs_p)  # [B, C, T_text]
                neg_cent1 = torch.sum(
                    -0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True
                )  # [B, 1, T_text]
                neg_cent2 = torch.matmul(
                    -0.5 * (z_p**2).transpose(1, 2), s_p_sq_r
                )  # [B, T_mel, T_text]
                neg_cent3 = torch.matmul(
                    z_p.transpose(1, 2), (m_p * s_p_sq_r)
                )  # [B, T_mel, T_text]
                neg_cent4 = torch.sum(
                    -0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True
                )  # [B, 1, T_text]
                neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

                attention = monotonic_align.maximum_path(
                    neg_cent, x_mask.squeeze(1), y_mask.squeeze(1)
                ).detach()

            # Duration from attention
            w = attention.sum(1, keepdim=True)

            # Duration loss
            log_w, log_w_std = self.duration_predictor(x, x_mask, g=g)
            log_duration_targets = torch.log(w.float() + 1e-6) * x_mask

            # Expand prior
            m_p = torch.matmul(attention.squeeze(1), m_p.transpose(1, 2)).transpose(
                1, 2
            )
            logs_p = torch.matmul(
                attention.squeeze(1), logs_p.transpose(1, 2)
            ).transpose(1, 2)

            # Generate audio
            z_slice, ids_slice = rand_slice_segments(z, mel_lengths, self.segment_size)
            audio = self.decoder(z_slice)

            # Compute losses
            losses = self.compute_losses(
                z_p,
                logs_q,
                m_q,
                logs_p,
                m_p,
                z_mask=y_mask,
                log_duration_prediction=log_w,
                log_duration_targets=log_duration_targets,
                duration_mask=x_mask,
            )

            return {
                "audio": audio,
                "mel_outputs": z_slice,
                "log_duration_prediction": log_w,
                "attention": attention,
                "losses": losses,
            }

        else:  # Inference mode
            # Duration prediction
            log_w, log_w_std = self.duration_predictor(x, x_mask, g=g)
            w = torch.exp(log_w) * x_mask

            # Round durations
            w_ceil = torch.ceil(w)
            y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
            y_mask = sequence_mask(y_lengths).unsqueeze(1).to(x_mask.dtype)

            # Generate path
            attention = generate_path(
                w_ceil.squeeze(1), x_mask.squeeze(1), y_mask.squeeze(1)
            )

            # Expand prior
            m_p = torch.matmul(attention.squeeze(1), m_p.transpose(1, 2)).transpose(
                1, 2
            )
            logs_p = torch.matmul(
                attention.squeeze(1), logs_p.transpose(1, 2)
            ).transpose(1, 2)

            # Sample from prior
            z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * 0.66

            # Inverse flow
            z = self.flow(z_p, y_mask, g=g, reverse=True)

            # Generate audio
            audio = self.decoder(z * y_mask)

            return {
                "audio": audio,
                "mel_outputs": z,
                "log_duration_prediction": log_w,
                "attention": attention,
            }

    def compute_losses(
        self,
        z_p: torch.Tensor,
        logs_q: torch.Tensor,
        m_q: torch.Tensor,
        logs_p: torch.Tensor,
        m_p: torch.Tensor,
        z_mask: torch.Tensor,
        log_duration_prediction: torch.Tensor,
        log_duration_targets: torch.Tensor,
        duration_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute VITS losses"""
        # KL divergence loss
        kl_loss = kl_divergence(logs_q, m_q, logs_p, m_p, z_mask)

        # Duration loss
        duration_loss = F.mse_loss(
            log_duration_prediction * duration_mask,
            log_duration_targets * duration_mask,
        )

        return {
            "kl": kl_loss,
            "duration": duration_loss,
        }

    @torch.no_grad()
    def inference(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        speaker_ids: Optional[torch.Tensor] = None,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Optimized inference"""
        self.eval()

        # Get speaker embeddings
        if self.speaker_embedding is not None and speaker_ids is not None:
            g = self.speaker_embedding(speaker_ids).unsqueeze(-1)
        else:
            g = None

        # Text encoding
        x, m_p, logs_p, x_mask = self.text_encoder(text, text_lengths, g)

        # Duration prediction
        log_w, log_w_std = self.duration_predictor(x, x_mask, g=g)
        w = torch.exp(log_w) * x_mask * length_scale

        # Generate path
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = sequence_mask(y_lengths).unsqueeze(1).to(x_mask.dtype)

        attention = generate_path(
            w_ceil.squeeze(1), x_mask.squeeze(1), y_mask.squeeze(1)
        )

        # Expand prior
        m_p = torch.matmul(attention.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attention.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )

        # Sample from prior
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        # Inverse flow
        z = self.flow(z_p, y_mask, g=g, reverse=True)

        # Generate audio
        audio = self.decoder(z * y_mask)

        return {
            "audio": audio.squeeze(1),
            "attention": attention,
            "mel": z,
        }


def kl_divergence(
    logs_q: torch.Tensor,
    m_q: torch.Tensor,
    logs_p: torch.Tensor,
    m_p: torch.Tensor,
    z_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute KL divergence between two distributions"""
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((m_p - m_q) ** 2) * torch.exp(-2.0 * logs_p)
    kl += 0.5 * (torch.exp(2.0 * logs_q) - 1.0) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    kl = kl / torch.sum(z_mask)
    return kl
