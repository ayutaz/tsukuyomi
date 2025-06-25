"""Common modules for TTS models"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """Layer normalization with optional speaker conditioning"""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class ConvReluNorm(nn.Module):
    """Convolution + ReLU + Normalization block"""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int,
        n_layers: int,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for i in range(n_layers):
            in_c = in_channels if i == 0 else hidden_channels
            out_c = out_channels if i == n_layers - 1 else hidden_channels
            self.conv_layers.append(
                nn.Conv1d(in_c, out_c, kernel_size, padding=kernel_size // 2)
            )
            self.norm_layers.append(LayerNorm(out_c))

        self.relu_drop = nn.Sequential(nn.ReLU(), nn.Dropout(p_dropout))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            if i < self.n_layers - 1:
                x = self.relu_drop(x)
        return x * x_mask


class DurationPredictor(nn.Module):
    """Duration predictor module"""

    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float = 0.5,
        n_layers: int = 3,
        speaker_embedding_dim: int = 0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.drop = nn.Dropout(p_dropout)
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for i in range(n_layers):
            in_c = in_channels + speaker_embedding_dim if i == 0 else filter_channels
            self.conv_layers.append(
                nn.Conv1d(in_c, filter_channels, kernel_size, padding=kernel_size // 2)
            )
            self.norm_layers.append(LayerNorm(filter_channels))

        self.proj = nn.Conv1d(filter_channels, 1, 1)

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, g: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if g is not None:
            g = g.expand(-1, -1, x.size(2))
            x = torch.cat([x, g], dim=1)

        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = F.relu(x)
            x = self.drop(x)

        x = self.proj(x * x_mask)
        return x * x_mask


class StochasticDurationPredictor(nn.Module):
    """Stochastic duration predictor with flow"""

    def __init__(
        self,
        in_channels: int,
        filter_channels: int = 256,
        kernel_size: int = 3,
        p_dropout: float = 0.5,
        n_flows: int = 4,
        speaker_embedding_dim: int = 0,
    ):
        super().__init__()

        # Condition encoder
        self.encoder = DurationPredictor(
            in_channels,
            filter_channels,
            kernel_size,
            p_dropout,
            n_layers=3,
            speaker_embedding_dim=speaker_embedding_dim,
        )

        # Projection layers
        self.proj = nn.Conv1d(filter_channels, filter_channels * 2, 1)

        # Normalizing flow
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(ConvFlow(filter_channels, kernel_size, n_layers=3))

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
        noise_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode
        h = self.encoder(x, x_mask, g)

        # Project to mean and log_std
        stats = self.proj(h) * x_mask
        m, logs = torch.split(stats, stats.size(1) // 2, dim=1)

        if not reverse:
            # Sample duration
            z = (m + torch.randn_like(m) * torch.exp(logs) * noise_scale) * x_mask

            # Flow transform
            log_det_tot = 0
            for flow in self.flows:
                z, log_det = flow(z, x_mask, g=g)
                log_det_tot += log_det

            return z, log_det_tot
        else:
            # Reverse flow
            z = m * x_mask
            for flow in reversed(self.flows):
                z = flow(z, x_mask, g=g, reverse=True)

            return z, logs


class TextEncoder(nn.Module):
    """Text encoder with self-attention"""

    def __init__(
        self,
        n_vocab: int,
        n_feats: int,
        n_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        speaker_embedding_dim: int = 0,
    ):
        super().__init__()

        self.n_vocab = n_vocab
        self.n_feats = n_feats
        self.n_channels = n_channels

        # Token embedding
        self.emb = nn.Embedding(n_vocab, n_channels)
        nn.init.normal_(self.emb.weight, 0.0, n_channels**-0.5)

        # Encoder
        self.encoder = Encoder(
            n_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            speaker_embedding_dim=speaker_embedding_dim,
        )

        # Projection
        self.proj = nn.Conv1d(n_channels, n_feats * 2, 1)

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Embed text
        x = self.emb(text) * math.sqrt(self.n_channels)
        x = x.transpose(1, 2)  # [B, C, T]

        # Create mask
        x_mask = sequence_mask(text_lengths).unsqueeze(1).to(x.dtype)

        # Encode
        x = self.encoder(x * x_mask, x_mask, g=g)

        # Project to mean and log_std
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.n_feats, dim=1)

        return x, m, logs, x_mask


class PosteriorEncoder(nn.Module):
    """Posterior encoder for mel-spectrogram"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        speaker_embedding_dim: int = 0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        # Pre-convolution
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)

        # WaveNet encoder
        self.enc = WaveNet(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            speaker_embedding_dim=speaker_embedding_dim,
        )

        # Projection
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self, x: torch.Tensor, x_lengths: torch.Tensor, g: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Create mask
        x_mask = sequence_mask(x_lengths).unsqueeze(1).to(x.dtype)

        # Encode
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)

        # Project to mean and log_std
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)

        # Sample
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask

        return z, m, logs, x_mask


class ResidualCouplingBlock(nn.Module):
    """Residual coupling block for normalizing flow"""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        speaker_embedding_dim: int = 0,
    ):
        super().__init__()

        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    speaker_embedding_dim=speaker_embedding_dim,
                    mean_only=True,
                )
            )
            self.flows.append(Flip())

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class ResidualCouplingLayer(nn.Module):
    """Single residual coupling layer"""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        speaker_embedding_dim: int = 0,
        mean_only: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.mean_only = mean_only

        self.half_channels = channels // 2

        # Pre-convolution
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)

        # WaveNet
        self.enc = WaveNet(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            speaker_embedding_dim=speaker_embedding_dim,
        )

        # Post-convolution
        if mean_only:
            self.post = nn.Conv1d(hidden_channels, self.half_channels, 1)
        else:
            self.post = nn.Conv1d(hidden_channels, self.half_channels * 2, 1)

        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)

        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask

        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 * torch.exp(logs) + m) * x_mask
            x = torch.cat([x0, x1], 1)
            return x


class Flip(nn.Module):
    """Flip operation for flow"""

    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(x)
            return x, logdet
        else:
            return x


class WaveNet(nn.Module):
    """WaveNet residual blocks"""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        speaker_embedding_dim: int = 0,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.speaker_embedding_dim = speaker_embedding_dim
        self.p_dropout = p_dropout

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if speaker_embedding_dim > 0:
            self.cond_layer = nn.Conv1d(
                speaker_embedding_dim, channels * 2 * n_layers, 1
            )

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = nn.Conv1d(
                channels, channels * 2, kernel_size, dilation=dilation, padding=padding
            )
            self.in_layers.append(in_layer)

            res_skip_layer = nn.Conv1d(channels, channels * 2, 1)
            self.res_skip_layers.append(res_skip_layer)

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, g: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        output = torch.zeros_like(x)

        if g is not None and hasattr(self, "cond_layer"):
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)

            if g is not None:
                cond_offset = i * 2 * self.channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.channels, :]
                x_in = x_in + g_l

            acts = fused_add_tanh_sigmoid_multiply(x_in, self.channels)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            res_acts, skip_acts = torch.split(res_skip_acts, [self.channels] * 2, 1)

            x = (x + res_acts) * x_mask
            output = output + skip_acts

        return output * x_mask


class Encoder(nn.Module):
    """Transformer encoder"""

    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        window_size: int = 4,
        speaker_embedding_dim: int = 0,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()

        for i in range(n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

        if speaker_embedding_dim > 0:
            self.cond_layer = nn.Conv1d(speaker_embedding_dim, hidden_channels, 1)

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, g: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if g is not None and hasattr(self, "cond_layer"):
            print(f"DEBUG Encoder: x shape: {x.shape}, g shape before cond_layer: {g.shape}")
            g = self.cond_layer(g)
            print(f"DEBUG Encoder: g shape after cond_layer: {g.shape}")
            # Expand g to match x's time dimension
            g = g.expand(-1, -1, x.size(2))
            print(f"DEBUG Encoder: g shape after expand: {g.shape}")
            x = x + g

        for i in range(self.n_layers):
            y = self.attn_layers[i](x, x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)

        return x * x_mask


class MultiHeadAttention(nn.Module):
    """Multi-head attention module"""

    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
        window_size: Optional[int] = None,
    ):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.window_size = window_size

        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x, _ = self.attention(q, k, v, mask=attn_mask)

        x = self.conv_o(x)
        return x

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, d, t_s = query.size()
        t_t = key.size(2)

        query = query.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.k_channels)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)

        p_attn = torch.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)

        output = torch.matmul(p_attn, value)
        output = output.transpose(2, 3).contiguous().view(b, d, t_s)

        return output, p_attn


class FFN(nn.Module):
    """Feed-forward network"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.conv_2 = nn.Conv1d(
            filter_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x * x_mask)
        x = F.relu(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask


class ConvFlow(nn.Module):
    """Convolutional flow for duration predictor"""

    def __init__(
        self,
        in_channels: int,
        filter_channels: int = 256,
        kernel_size: int = 3,
        n_layers: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers

        self.in_proj = nn.Conv1d(in_channels, filter_channels, 1)
        self.convs = nn.ModuleList()

        for i in range(n_layers):
            self.convs.append(
                nn.Conv1d(
                    filter_channels,
                    filter_channels,
                    kernel_size,
                    padding=kernel_size // 2,
                )
            )

        self.out_proj = nn.Conv1d(filter_channels, in_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not reverse:
            h = self.in_proj(x)
            for conv in self.convs:
                h = F.relu(conv(h))
            h = self.out_proj(h)

            x = x + h * x_mask
            logdet = 0
            return x, logdet
        else:
            h = self.in_proj(x)
            for conv in self.convs:
                h = F.relu(conv(h))
            h = self.out_proj(h)

            x = x - h * x_mask
            return x


# Helper functions


def sequence_mask(
    lengths: torch.Tensor, max_length: Optional[int] = None
) -> torch.Tensor:
    """Generate sequence mask"""
    if max_length is None:
        max_length = lengths.max()

    x = torch.arange(max_length, dtype=lengths.dtype, device=lengths.device)
    return x.unsqueeze(0) < lengths.unsqueeze(1)


def fused_add_tanh_sigmoid_multiply(x: torch.Tensor, n_channels: int) -> torch.Tensor:
    """Fused activation function"""
    t_act, s_act = torch.split(x, n_channels, dim=1)
    acts = t_act.tanh() * s_act.sigmoid()
    return acts
