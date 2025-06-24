"""Matcha-TTS: Fast and High-Quality TTS with Conditional Flow Matching

Based on the paper: https://arxiv.org/abs/2309.03199
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .commons import generate_path, rand_slice_segments, sequence_mask
from .modules import (
    DurationPredictor,
    Encoder,
    LayerNorm,
    PosteriorEncoder,
    TextEncoder,
)


class FlowMatchingDecoder(nn.Module):
    """Flow matching decoder for mel-spectrogram generation"""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_blocks: int,
        n_layers: int,
        n_flows: int,
        p_dropout: float = 0.0,
        n_split: int = 4,
        speaker_embedding_dim: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.p_dropout = p_dropout
        self.n_split = n_split

        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # Flow blocks
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                FlowBlock(
                    in_channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    n_split,
                    p_dropout,
                    speaker_embedding_dim,
                )
            )

        # Final projection
        self.proj = nn.Conv1d(in_channels, in_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of flow matching decoder

        Args:
            x: Input noise [B, C, T]
            x_mask: Mask [B, 1, T]
            mu: Conditioning features [B, C, T]
            t: Time steps [B, 1]
            g: Speaker embedding [B, spk_dim, 1]

        Returns:
            Velocity field [B, C, T]
        """
        # Add time embedding
        t_emb = self.time_embedding(t.unsqueeze(-1))  # [B, hidden_channels]
        t_emb = t_emb.unsqueeze(-1)  # [B, hidden_channels, 1]

        # Interpolate between noise and data
        h = (1 - t.unsqueeze(-1)) * x + t.unsqueeze(-1) * mu
        h = h * x_mask

        # Pass through flow blocks
        for flow in self.flows:
            h = flow(h, x_mask, mu, t_emb, g)

        # Final projection
        v = self.proj(h) * x_mask

        return v

    def sample(
        self,
        mu: torch.Tensor,
        x_mask: torch.Tensor,
        n_timesteps: int = 100,
        temperature: float = 1.0,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample from the flow matching decoder using ODE integration

        Args:
            mu: Conditioning features [B, C, T]
            x_mask: Mask [B, 1, T]
            n_timesteps: Number of integration steps
            temperature: Sampling temperature
            g: Speaker embedding [B, spk_dim, 1]

        Returns:
            Generated mel-spectrogram [B, C, T]
        """
        B, C, T = mu.shape
        device = mu.device
        dtype = mu.dtype

        # Start from noise
        x = torch.randn(B, C, T, device=device, dtype=dtype) * temperature
        x = x * x_mask

        # Time steps from 0 to 1
        dt = 1.0 / n_timesteps
        
        # Euler integration
        for i in range(n_timesteps):
            t = torch.full((B, 1), i * dt, device=device, dtype=dtype)
            v = self.forward(x, x_mask, mu, t, g)
            x = x + v * dt
            x = x * x_mask

        return x


class FlowBlock(nn.Module):
    """Single flow block with WaveNet backbone"""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_split: int = 4,
        p_dropout: float = 0.0,
        speaker_embedding_dim: int = 0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_split = n_split
        self.p_dropout = p_dropout

        # Split dimensions
        self.split_channels = channels // n_split

        # Pre-projection
        self.pre = nn.Conv1d(channels + channels, hidden_channels, 1)

        # WaveNet layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.drops = nn.ModuleList()

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            self.convs.append(
                nn.Conv1d(
                    hidden_channels,
                    hidden_channels * 2,
                    kernel_size,
                    dilation=dilation,
                    padding=padding,
                )
            )
            self.norms.append(LayerNorm(hidden_channels))
            self.drops.append(nn.Dropout(p_dropout))

        # Post-projection
        self.post = nn.Conv1d(hidden_channels, channels, 1)

        # Time conditioning
        self.time_layers = nn.ModuleList(
            [nn.Linear(hidden_channels, hidden_channels) for _ in range(n_layers)]
        )

        # Speaker conditioning
        if speaker_embedding_dim > 0:
            self.cond_layer = nn.Conv1d(
                speaker_embedding_dim, hidden_channels * n_layers, 1
            )

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        mu: torch.Tensor,
        t_emb: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of flow block

        Args:
            x: Input [B, C, T]
            x_mask: Mask [B, 1, T]
            mu: Conditioning [B, C, T]
            t_emb: Time embedding [B, hidden_channels, 1]
            g: Speaker embedding [B, spk_dim, 1]

        Returns:
            Output [B, C, T]
        """
        # Concatenate input and conditioning
        h = torch.cat([x, mu], dim=1)
        h = self.pre(h) * x_mask

        # Speaker conditioning
        if g is not None and hasattr(self, "cond_layer"):
            g_layers = self.cond_layer(g)

        # WaveNet layers
        skip = 0
        for i, (conv, norm, drop, time_layer) in enumerate(
            zip(self.convs, self.norms, self.drops, self.time_layers)
        ):
            # Convolution
            h_conv = conv(h)

            # Add time conditioning
            h_time = time_layer(t_emb.transpose(1, 2)).transpose(1, 2)
            h_conv = h_conv + h_time

            # Add speaker conditioning
            if g is not None and hasattr(self, "cond_layer"):
                offset = i * self.hidden_channels
                g_layer = g_layers[:, offset : offset + self.hidden_channels * 2, :]
                h_conv = h_conv + g_layer

            # Gated activation
            h_a, h_b = torch.split(h_conv, self.hidden_channels, dim=1)
            acts = h_a.tanh() * h_b.sigmoid()

            # Residual and skip connections
            acts = drop(acts)
            h = (h + acts) * x_mask
            skip = skip + acts

            # Layer norm
            h = norm(h)

        # Post-projection
        out = self.post(skip) * x_mask

        return x + out


class MatchaTTS(nn.Module):
    """Matcha-TTS: Fast TTS with Conditional Flow Matching"""

    def __init__(
        self,
        n_vocab: int,
        n_speakers: int = 1,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        filter_channels_dp: int = 256,
        n_heads: int = 2,
        n_layers_enc: int = 6,
        n_layers_dec: int = 6,
        n_layers_flow: int = 4,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        window_size: int = 4,
        n_split: int = 4,
        n_timesteps: int = 100,
        segment_size: int = 8192,
        use_sdp: bool = True,
        encoder_hidden_channels: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.n_speakers = n_speakers
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.n_heads = n_heads
        self.n_layers_enc = n_layers_enc
        self.n_layers_dec = n_layers_dec
        self.n_layers_flow = n_layers_flow
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.n_split = n_split
        self.n_timesteps = n_timesteps
        self.segment_size = segment_size
        self.use_sdp = use_sdp

        # Calculate final encoder output dimension
        self.encoder_out_channels = encoder_hidden_channels or hidden_channels

        # Speaker embedding
        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, hidden_channels)
            self.speaker_embedding_dim = hidden_channels
        else:
            self.speaker_embedding_dim = 0

        # Text encoder
        self.encoder = TextEncoder(
            n_vocab,
            hidden_channels,
            self.encoder_out_channels,
            filter_channels,
            n_heads,
            n_layers_enc,
            kernel_size,
            p_dropout,
            speaker_embedding_dim=self.speaker_embedding_dim,
        )

        # Duration predictor
        self.dp = DurationPredictor(
            self.encoder_out_channels,
            filter_channels_dp,
            kernel_size,
            p_dropout,
            speaker_embedding_dim=self.speaker_embedding_dim,
        )

        # Mel-spectrogram projection
        self.proj_m = nn.Conv1d(self.encoder_out_channels, hidden_channels, 1)

        # Flow matching decoder
        self.decoder = FlowMatchingDecoder(
            hidden_channels,
            hidden_channels,
            kernel_size,
            1,  # dilation_rate
            n_blocks=1,
            n_layers=n_layers_dec,
            n_flows=n_layers_flow,
            p_dropout=p_dropout,
            n_split=n_split,
            speaker_embedding_dim=self.speaker_embedding_dim,
        )

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        mel: torch.Tensor,
        mel_lengths: torch.Tensor,
        speaker_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training

        Args:
            text: Input text tokens [B, T_text]
            text_lengths: Text lengths [B]
            mel: Target mel-spectrogram [B, n_mel, T_mel]
            mel_lengths: Mel lengths [B]
            speaker_ids: Speaker IDs [B]

        Returns:
            Dictionary containing losses
        """
        # Get speaker embedding
        g = None
        if self.n_speakers > 1 and speaker_ids is not None:
            g = self.emb_g(speaker_ids).unsqueeze(-1)  # [B, hidden_channels, 1]

        # Encode text
        x, m_p, logs_p, x_mask = self.encoder(text, text_lengths, g)

        # Duration prediction
        logw = self.dp(x, x_mask, g=g)
        w = torch.exp(logw) * x_mask

        # Align text and mel using monotonic alignment
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(
            sequence_mask(mel_lengths, mel.size(2)), 1
        )
        attn = generate_path(w.squeeze(1), attn_mask.squeeze(1))

        # Aggregate encoded text to mel length
        m_p = torch.matmul(attn.squeeze(1).transpose(1, 2), m_p.transpose(1, 2))
        m_p = m_p.transpose(1, 2)

        # Project to hidden channels
        mu = self.proj_m(m_p) * sequence_mask(mel_lengths).unsqueeze(1)

        # Flow matching loss
        t = torch.rand(mel.size(0), 1, device=mel.device)
        z_0 = torch.randn_like(mel)
        z_t = (1 - t.unsqueeze(-1)) * z_0 + t.unsqueeze(-1) * mel
        
        # Compute velocity
        v_pred = self.decoder(
            z_t,
            sequence_mask(mel_lengths).unsqueeze(1),
            mu,
            t,
            g,
        )
        
        # Target velocity
        v_target = mel - z_0

        # Flow matching loss
        flow_loss = F.mse_loss(v_pred, v_target)

        # Duration loss
        l_length = ((logw - torch.log(attn.sum(2))).pow(2) * x_mask).sum() / x_mask.sum()

        losses = {
            "flow_loss": flow_loss,
            "duration_loss": l_length,
            "loss": flow_loss + l_length,
        }

        return losses

    def infer(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        speaker_ids: Optional[torch.Tensor] = None,
        length_scale: float = 1.0,
        temperature: float = 1.0,
        max_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inference

        Args:
            text: Input text tokens [B, T_text]
            text_lengths: Text lengths [B]
            speaker_ids: Speaker IDs [B]
            length_scale: Length scale factor
            temperature: Sampling temperature
            max_len: Maximum output length

        Returns:
            mel: Generated mel-spectrogram [B, n_mel, T_mel]
            w: Predicted durations [B, 1, T_text]
            attn: Attention weights [B, T_mel, T_text]
        """
        # Get speaker embedding
        g = None
        if self.n_speakers > 1 and speaker_ids is not None:
            g = self.emb_g(speaker_ids).unsqueeze(-1)

        # Encode text
        x, m_p, logs_p, x_mask = self.encoder(text, text_lengths, g)

        # Predict duration
        logw = self.dp(x, x_mask, g=g)
        w = torch.exp(logw) * x_mask * length_scale

        # Round durations
        w_ceil = torch.ceil(w)
        mel_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        mel_mask = sequence_mask(mel_lengths).unsqueeze(1)

        # Generate alignment path
        attn_mask = torch.unsqueeze(x_mask, 2) * mel_mask.unsqueeze(1)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1))

        # Aggregate encoded text to mel length
        m_p = torch.matmul(attn.squeeze(1).transpose(1, 2), m_p.transpose(1, 2))
        m_p = m_p.transpose(1, 2)

        # Project to hidden channels
        mu = self.proj_m(m_p) * mel_mask

        # Generate mel-spectrogram using flow matching
        mel = self.decoder.sample(
            mu,
            mel_mask,
            n_timesteps=self.n_timesteps,
            temperature=temperature,
            g=g,
        )

        return mel, w, attn