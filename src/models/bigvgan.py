"""BigVGAN: A Universal Neural Vocoder with Large-Scale Training

Based on: https://arxiv.org/abs/2206.04658
This is a wrapper around HiFi-GAN with BigVGAN improvements
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hifigan import Generator as HiFiGANGenerator
from .hifigan import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
)


class SnakeBeta(nn.Module):
    """
    Snake activation function with learnable parameters
    https://arxiv.org/abs/2006.08195
    """

    def __init__(self, in_features: int, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
        self.beta = nn.Parameter(torch.ones(in_features) * beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (1.0 / self.beta) * torch.sin(self.alpha * x) ** 2


class BigVGANResBlock(nn.Module):
    """Enhanced residual block with Snake activation and anti-aliasing"""

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

        # Snake activation
        self.activations = nn.ModuleList(
            [SnakeBeta(channels) for _ in range(len(dilation))]
        )

        # Anti-aliasing
        self.antialias = nn.ModuleList(
            [nn.AvgPool1d(2, stride=1, padding=1) for _ in range(len(dilation))]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, (c1, c2, act, aa) in enumerate(
            zip(self.convs1, self.convs2, self.activations, self.antialias)
        ):
            xt = act(x)
            xt = c1(xt)
            xt = act(xt)
            xt = c2(xt)

            # Anti-aliasing
            if xt.size(-1) > 1:
                xt = aa(xt)[:, :, : x.size(-1)]

            x = xt + x

        return x


class BigVGANGenerator(nn.Module):
    """BigVGAN Generator with enhanced architecture"""

    def __init__(
        self,
        num_mels: int = 80,
        upsample_initial_channel: int = 1536,
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[Tuple[int, ...]] = [
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        ],
        upsample_rates: List[int] = [8, 8, 2, 2],
        upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
        use_snake_activation: bool = True,
        use_anti_aliasing: bool = True,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.use_snake_activation = use_snake_activation

        # Initial convolution
        self.conv_pre = nn.Conv1d(num_mels, upsample_initial_channel, 7, 1, padding=3)

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
                if use_snake_activation and use_anti_aliasing:
                    self.resblocks.append(BigVGANResBlock(ch, k, d))
                else:
                    # Fallback to standard HiFi-GAN ResBlock
                    from .hifigan import ResBlock

                    self.resblocks.append(ResBlock(ch, k, d))

        # Activation
        if use_snake_activation:
            self.activation = SnakeBeta(ch)
        else:
            self.activation = nn.LeakyReLU(0.1)

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
            x: mel-spectrogram tensor [B, n_mels, T]
        Returns:
            audio waveform tensor [B, 1, T']
        """
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            if isinstance(self.activation, SnakeBeta):
                x = self.activation(x)
            else:
                x = F.leaky_relu(x, 0.1)

            x = self.ups[i](x)

            xs = 0
            for j in range(self.num_kernels):
                xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        if isinstance(self.activation, SnakeBeta):
            x = self.activation(x)
        else:
            x = F.leaky_relu(x, 0.1)

        x = self.conv_post(x)
        x = torch.tanh(x)

        return x


class BigVGAN(nn.Module):
    """
    Complete BigVGAN model with generator and discriminators

    This is the main class to use for BigVGAN vocoding
    """

    def __init__(
        self,
        num_mels: int = 80,
        upsample_initial_channel: int = 1536,
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[Tuple[int, ...]] = [
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        ],
        upsample_rates: List[int] = [8, 8, 2, 2],
        upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
        use_snake_activation: bool = True,
        use_anti_aliasing: bool = True,
    ):
        super().__init__()

        # Generator
        self.generator = BigVGANGenerator(
            num_mels=num_mels,
            upsample_initial_channel=upsample_initial_channel,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            use_snake_activation=use_snake_activation,
            use_anti_aliasing=use_anti_aliasing,
        )

        # Discriminators (from HiFi-GAN)
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()

    def forward(
        self, mel: torch.Tensor, return_discriminator: bool = False
    ) -> torch.Tensor:
        """Generate audio from mel-spectrogram"""
        audio = self.generator(mel)

        if return_discriminator:
            return audio, self.mpd, self.msd
        else:
            return audio

    @torch.no_grad()
    def inference(self, mel: torch.Tensor) -> torch.Tensor:
        """Optimized inference"""
        self.generator.eval()
        return self.generator(mel)

    def remove_weight_norm(self):
        """Remove weight normalization for inference"""
        # BigVGAN doesn't use weight norm by default
        pass


class Generator(BigVGANGenerator):
    """Alias for compatibility"""

    pass
