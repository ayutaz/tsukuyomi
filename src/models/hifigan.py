"""HiFi-GAN Vocoder implementation

Based on: https://arxiv.org/abs/2010.05646
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block with dilated convolutions"""

    def __init__(
        self, channels: int, kernel_size: int = 3, dilation: Tuple[int, ...] = (1, 3, 5)
    ):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=d,
                    padding=(kernel_size * d - d) // 2,
                )
                for d in dilation
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=(kernel_size - 1) // 2,
                )
                for _ in dilation
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x


class Generator(nn.Module):
    """HiFi-GAN Generator"""

    def __init__(
        self,
        in_channels: int = 80,
        upsample_initial_channel: int = 512,
        upsample_rates: List[int] = [8, 8, 2, 2],
        upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[Tuple[int, ...]] = [
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        ],
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        # Initial convolution
        self.conv_pre = nn.Conv1d(
            in_channels, upsample_initial_channel, 7, 1, padding=3
        )

        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(
                    upsample_initial_channel // (2**i),
                    upsample_initial_channel // (2 ** (i + 1)),
                    k,
                    u,
                    padding=(k - u) // 2,
                )
            )

        # Residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(ResBlock(ch, k, d))

        # Post convolution
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: mel-spectrogram tensor [B, C, T]
        Returns:
            audio waveform tensor [B, 1, T']
        """
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)

            xs = 0
            for j in range(self.num_kernels):
                xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x


class MultiPeriodDiscriminator(nn.Module):
    """Multi-Period Discriminator"""

    def __init__(self, periods: List[int] = [2, 3, 5, 7, 11]):
        super().__init__()
        self.discriminators = nn.ModuleList([PeriodDiscriminator(p) for p in periods])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class PeriodDiscriminator(nn.Module):
    """Period Discriminator"""

    def __init__(self, period: int):
        super().__init__()
        self.period = period

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(1, 32, (5, 1), (3, 1), (2, 0)),
                nn.Conv2d(32, 128, (5, 1), (3, 1), (2, 0)),
                nn.Conv2d(128, 512, (5, 1), (3, 1), (2, 0)),
                nn.Conv2d(512, 1024, (5, 1), (3, 1), (2, 0)),
                nn.Conv2d(1024, 1024, (5, 1), 1, (2, 0)),
            ]
        )
        self.conv_post = nn.Conv2d(1024, 1, (3, 1), 1, (1, 0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []

        # Reshape to 2D
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    """Multi-Scale Discriminator"""

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                ScaleDiscriminator(use_spectral_norm=True),
                ScaleDiscriminator(),
                ScaleDiscriminator(),
            ]
        )
        self.meanpools = nn.ModuleList(
            [nn.AvgPool1d(4, 2, padding=2), nn.AvgPool1d(4, 2, padding=2)]
        )

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)

            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class ScaleDiscriminator(nn.Module):
    """Scale Discriminator"""

    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        norm_f = nn.utils.spectral_norm if use_spectral_norm else lambda x: x

        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv1d(1, 128, 15, 1, 7)),
                norm_f(nn.Conv1d(128, 128, 41, 2, 20)),
                norm_f(nn.Conv1d(128, 256, 41, 2, 20)),
                norm_f(nn.Conv1d(256, 512, 41, 4, 20)),
                norm_f(nn.Conv1d(512, 1024, 41, 4, 20)),
                norm_f(nn.Conv1d(1024, 1024, 41, 1, 20)),
                norm_f(nn.Conv1d(1024, 1024, 5, 1, 2)),
            ]
        )
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class HiFiGAN(nn.Module):
    """Complete HiFi-GAN model with generator and discriminators"""

    def __init__(
        self,
        in_channels: int = 80,
        upsample_initial_channel: int = 512,
        upsample_rates: List[int] = [8, 8, 2, 2],
        upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[Tuple[int, ...]] = [
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        ],
    ):
        super().__init__()

        self.generator = Generator(
            in_channels,
            upsample_initial_channel,
            upsample_rates,
            upsample_kernel_sizes,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
        )

        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Generate audio from mel-spectrogram (inference mode)"""
        return self.generator(mel)

    @torch.no_grad()
    def inference(self, mel: torch.Tensor) -> torch.Tensor:
        """Optimized inference"""
        self.generator.eval()
        return self.generator(mel)
