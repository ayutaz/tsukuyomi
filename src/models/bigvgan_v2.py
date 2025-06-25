"""
BigVGAN-v2: State-of-the-art neural vocoder for 48kHz audio synthesis

Implementation of BigVGAN-v2 with:
- Anti-aliased activation functions (Snake-Beta)
- Multi-scale and multi-resolution discriminators
- 48kHz high-fidelity audio generation
- Optimized for H100 GPUs
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# CUDA 12.1+ optimizations
try:
    from apex import amp

    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

logger = logging.getLogger(__name__)

# Enable CUDA 12.1+ optimizations
if torch.cuda.is_available():
    cuda_version = torch.version.cuda
    if cuda_version and float(cuda_version.split(".")[0]) >= 12.1:
        logger.info(f"CUDA {cuda_version} detected - enabling BigVGAN-v2 optimizations")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


@dataclass
class BigVGANv2Config:
    """Configuration for BigVGAN-v2."""

    # Model architecture
    hidden_channels: int = 1536
    upsample_rates: List[int] = None
    upsample_kernel_sizes: List[int] = None
    resblock_kernel_sizes: List[int] = None
    resblock_dilations: List[List[int]] = None

    # Audio parameters
    sampling_rate: int = 48000
    hop_length: int = 480
    n_mel_channels: int = 128

    # Anti-aliased activation
    use_snake_activation: bool = True
    snake_logscale: bool = True

    # Discriminator
    mpd_periods: List[int] = None
    mrd_resolutions: List[Tuple[int, int, int]] = None

    # Training
    use_spectral_norm: bool = False
    use_weight_norm: bool = True

    def __post_init__(self):
        if self.upsample_rates is None:
            # For 48kHz with hop_length=480: 480 = 8 * 6 * 5 * 2
            self.upsample_rates = [8, 6, 5, 2]

        if self.upsample_kernel_sizes is None:
            self.upsample_kernel_sizes = [16, 12, 10, 4]

        if self.resblock_kernel_sizes is None:
            self.resblock_kernel_sizes = [3, 7, 11, 15]

        if self.resblock_dilations is None:
            self.resblock_dilations = [[1, 3, 5], [1, 3, 5], [1, 3, 5], [1, 3, 5]]

        if self.mpd_periods is None:
            self.mpd_periods = [2, 3, 5, 7, 11]

        if self.mrd_resolutions is None:
            # (filter_length, hop_length, win_length)
            self.mrd_resolutions = [(2048, 240, 1200), (1024, 120, 600), (512, 50, 240)]


# Triton kernel for Snake-Beta activation (CUDA 12.1+ optimization)
if TRITON_AVAILABLE:

    @triton.jit
    def snake_beta_kernel(
        x_ptr,
        alpha_ptr,
        beta_ptr,
        output_ptr,
        n_elements,
        alpha_logscale: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel for Snake-Beta activation."""
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load inputs
        x = tl.load(x_ptr + offsets, mask=mask)
        alpha = tl.load(alpha_ptr + offsets % BLOCK_SIZE, mask=mask)
        beta = tl.load(beta_ptr + offsets % BLOCK_SIZE, mask=mask)

        # Apply exp if needed
        if alpha_logscale:
            alpha = tl.exp(alpha)
            beta = tl.exp(beta)

        # Snake-Beta computation
        sin_beta_x = tl.sin(beta * x)
        output = x + (1.0 / beta) * sin_beta_x * sin_beta_x / alpha

        # Store result
        tl.store(output_ptr + offsets, output, mask=mask)


class SnakeBeta(nn.Module):
    """
    Snake-Beta activation function for anti-aliasing.

    Implementation of the trainable activation function from BigVGAN-v2
    that provides inherent anti-aliasing properties.
    Includes Triton optimization for CUDA 12.1+.
    """

    def __init__(
        self, in_features: int, alpha_logscale: bool = True, use_triton: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        self.use_triton = use_triton and TRITON_AVAILABLE and torch.cuda.is_available()

        # Initialize alpha and beta
        if alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(1, in_features, 1))
            self.beta = nn.Parameter(torch.zeros(1, in_features, 1))
        else:
            self.alpha = nn.Parameter(torch.ones(1, in_features, 1) * 0.5)
            self.beta = nn.Parameter(torch.ones(1, in_features, 1) * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Snake-Beta activation.

        Args:
            x: Input tensor (B, C, T)

        Returns:
            Activated tensor
        """
        if self.use_triton and x.is_cuda:
            # Use Triton kernel for CUDA 12.1+ optimization
            B, C, T = x.shape
            x_flat = x.reshape(-1)
            output = torch.empty_like(x_flat)

            # Broadcast alpha and beta
            alpha_broadcast = self.alpha.expand(B, C, T).reshape(-1)
            beta_broadcast = self.beta.expand(B, C, T).reshape(-1)

            # Launch kernel
            n_elements = x_flat.numel()
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

            snake_beta_kernel[grid](
                x_flat,
                alpha_broadcast,
                beta_broadcast,
                output,
                n_elements,
                self.alpha_logscale,
                BLOCK_SIZE,
            )

            return output.reshape(B, C, T)
        else:
            # Standard PyTorch implementation
            if self.alpha_logscale:
                alpha = torch.exp(self.alpha)
                beta = torch.exp(self.beta)
            else:
                alpha = self.alpha
                beta = self.beta

            # Snake-Beta: x + (1/beta) * sin^2(beta * x) / alpha
            return x + (1.0 / beta) * torch.pow(torch.sin(beta * x), 2) / alpha


class ResBlock(nn.Module):
    """
    Residual block with anti-aliased activation.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: List[int],
        use_snake: bool = True,
    ):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.activations1 = nn.ModuleList()
        self.activations2 = nn.ModuleList()

        for dilation in dilations:
            padding = (kernel_size * dilation - dilation) // 2

            self.convs1.append(
                weight_norm(
                    nn.Conv1d(
                        channels, channels, kernel_size, 1, padding, dilation=dilation
                    )
                )
            )
            self.convs2.append(
                weight_norm(
                    nn.Conv1d(
                        channels, channels, kernel_size, 1, padding, dilation=dilation
                    )
                )
            )

            if use_snake:
                self.activations1.append(SnakeBeta(channels))
                self.activations2.append(SnakeBeta(channels))
            else:
                self.activations1.append(nn.LeakyReLU(0.1))
                self.activations2.append(nn.LeakyReLU(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual block."""
        for c1, c2, a1, a2 in zip(
            self.convs1, self.convs2, self.activations1, self.activations2
        ):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        """Remove weight normalization."""
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class GeneratorBlock(nn.Module):
    """
    Generator block with transposed convolution and residual connections.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        resblock_kernel_sizes: List[int],
        resblock_dilations: List[List[int]],
        use_snake: bool = True,
    ):
        super().__init__()

        # Transposed convolution for upsampling
        padding = (kernel_size - stride) // 2
        self.conv = weight_norm(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)
        )

        # Multiple residual blocks
        self.resblocks = nn.ModuleList()
        for kernel_size, dilations in zip(resblock_kernel_sizes, resblock_dilations):
            self.resblocks.append(
                ResBlock(out_channels, kernel_size, dilations, use_snake)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply generator block."""
        x = self.conv(x)

        # Apply residual blocks in parallel and sum
        xs = 0
        for resblock in self.resblocks:
            xs += resblock(x)
        x = xs / len(self.resblocks)

        return x

    def remove_weight_norm(self):
        """Remove weight normalization."""
        remove_weight_norm(self.conv)
        for resblock in self.resblocks:
            resblock.remove_weight_norm()


class BigVGANv2Generator(nn.Module):
    """
    BigVGAN-v2 generator for high-fidelity waveform synthesis.
    """

    def __init__(self, config: BigVGANv2Config):
        super().__init__()
        self.config = config
        self.num_kernels = len(config.resblock_kernel_sizes)

        # Pre-convolution
        self.conv_pre = weight_norm(
            nn.Conv1d(
                config.n_mel_channels,
                config.hidden_channels,
                kernel_size=7,
                stride=1,
                padding=3,
            )
        )

        # Generator blocks
        self.blocks = nn.ModuleList()
        channels = config.hidden_channels

        for i, (rate, kernel_size) in enumerate(
            zip(config.upsample_rates, config.upsample_kernel_sizes)
        ):
            # Reduce channels progressively
            out_channels = channels // (2 ** (i + 1))
            out_channels = max(out_channels, 128)  # Minimum channels

            self.blocks.append(
                GeneratorBlock(
                    channels,
                    out_channels,
                    kernel_size,
                    rate,
                    config.resblock_kernel_sizes,
                    config.resblock_dilations,
                    config.use_snake_activation,
                )
            )
            channels = out_channels

        # Activation
        if config.use_snake_activation:
            self.activation_post = SnakeBeta(channels)
        else:
            self.activation_post = nn.LeakyReLU(0.1)

        # Post-convolution
        self.conv_post = weight_norm(
            nn.Conv1d(channels, 1, kernel_size=7, stride=1, padding=3)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            m.weight.data.normal_(0.0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Generate waveform from mel-spectrogram.

        Args:
            mel: Mel-spectrogram (B, n_mel_channels, T)

        Returns:
            Waveform (B, 1, T * hop_length)
        """
        x = self.conv_pre(mel)

        for block in self.blocks:
            x = block(x)

        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        """Remove weight normalization for inference."""
        logger.info("Removing weight normalization from generator")
        remove_weight_norm(self.conv_pre)
        for block in self.blocks:
            block.remove_weight_norm()
        remove_weight_norm(self.conv_post)

    @torch.no_grad()
    def inference(self, mel: torch.Tensor, use_cuda_graph: bool = True) -> torch.Tensor:
        """
        Optimized inference with CUDA 12.1+ features.

        Args:
            mel: Mel-spectrogram (B, n_mel_channels, T)
            use_cuda_graph: Whether to use CUDA graphs for static shapes

        Returns:
            Waveform (B, 1, T * hop_length)
        """
        if (
            use_cuda_graph
            and torch.cuda.is_available()
            and hasattr(self, "_cuda_graph")
        ):
            # Use CUDA graph for static shapes
            if self._static_mel.shape == mel.shape:
                self._static_mel.copy_(mel)
                self._cuda_graph.replay()
                return self._static_output.clone()

        # Standard inference
        return self.forward(mel)

    def compile_for_inference(self, example_mel: torch.Tensor) -> None:
        """
        Compile model for faster inference using CUDA 12.1+ features.

        Args:
            example_mel: Example mel-spectrogram for shape inference
        """
        self.eval()
        self.remove_weight_norm()

        if not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping compilation")
            return

        cuda_version = torch.version.cuda
        if cuda_version and float(cuda_version.split(".")[0]) >= 12.1:
            logger.info("Compiling BigVGAN-v2 with CUDA 12.1+ optimizations")

            # Compile with torch.compile
            try:
                self.forward = torch.compile(
                    self.forward,
                    mode="reduce-overhead",
                    fullgraph=True,
                    backend="inductor",
                )
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")

            # Setup CUDA graphs for static shapes
            if hasattr(torch.cuda, "graph"):
                try:
                    # Warmup
                    with torch.cuda.amp.autocast(enabled=False):
                        _ = self.forward(example_mel)

                    # Create static tensors
                    self._static_mel = example_mel.clone()
                    self._cuda_graph = torch.cuda.CUDAGraph()

                    # Capture graph
                    with torch.cuda.graph(self._cuda_graph):
                        self._static_output = self.forward(self._static_mel)

                    logger.info("CUDA graph captured for static inference")
                except Exception as e:
                    logger.warning(f"CUDA graph capture failed: {e}")
        else:
            logger.warning(
                f"CUDA {cuda_version} detected - CUDA 12.1+ required for optimal compilation"
            )


class MultiPeriodDiscriminator(nn.Module):
    """
    Multi-Period Discriminator for BigVGAN-v2.
    """

    def __init__(self, config: BigVGANv2Config):
        super().__init__()
        self.discriminators = nn.ModuleList()

        for period in config.mpd_periods:
            self.discriminators.append(PeriodDiscriminator(period))

    def forward(self, real: torch.Tensor, fake: torch.Tensor) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:
        """
        Forward pass of MPD.

        Returns:
            (real_outputs, fake_outputs, real_features, fake_features)
        """
        real_outputs = []
        fake_outputs = []
        real_features = []
        fake_features = []

        for disc in self.discriminators:
            real_out, real_feat = disc(real)
            fake_out, fake_feat = disc(fake)

            real_outputs.append(real_out)
            fake_outputs.append(fake_out)
            real_features.append(real_feat)
            fake_features.append(fake_feat)

        return real_outputs, fake_outputs, real_features, fake_features


class PeriodDiscriminator(nn.Module):
    """
    Single period discriminator.
    """

    def __init__(self, period: int):
        super().__init__()
        self.period = period

        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(1, 32, (5, 1), (3, 1), (2, 0))),
                weight_norm(nn.Conv2d(32, 128, (5, 1), (3, 1), (2, 0))),
                weight_norm(nn.Conv2d(128, 512, (5, 1), (3, 1), (2, 0))),
                weight_norm(nn.Conv2d(512, 1024, (5, 1), (3, 1), (2, 0))),
                weight_norm(nn.Conv2d(1024, 1024, (5, 1), 1, (2, 0))),
            ]
        )

        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, (1, 0)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass.

        Returns:
            (output, features)
        """
        features = []

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
            features.append(x)

        x = self.conv_post(x)
        features.append(x)

        x = torch.flatten(x, 1, -1)

        return x, features


class MultiResolutionDiscriminator(nn.Module):
    """
    Multi-Resolution Discriminator for BigVGAN-v2.
    """

    def __init__(self, config: BigVGANv2Config):
        super().__init__()
        self.discriminators = nn.ModuleList()

        for filter_length, hop_length, win_length in config.mrd_resolutions:
            self.discriminators.append(
                ResolutionDiscriminator(filter_length, hop_length, win_length)
            )

    def forward(self, real: torch.Tensor, fake: torch.Tensor) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:
        """Forward pass of MRD."""
        real_outputs = []
        fake_outputs = []
        real_features = []
        fake_features = []

        for disc in self.discriminators:
            real_out, real_feat = disc(real)
            fake_out, fake_feat = disc(fake)

            real_outputs.append(real_out)
            fake_outputs.append(fake_out)
            real_features.append(real_feat)
            fake_features.append(fake_feat)

        return real_outputs, fake_outputs, real_features, fake_features


class ResolutionDiscriminator(nn.Module):
    """
    Single resolution discriminator using spectrogram.
    """

    def __init__(self, filter_length: int, hop_length: int, win_length: int):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length

        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
                weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
            ]
        )

        self.conv_post = weight_norm(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass."""
        features = []

        # Compute spectrogram
        x = self.spectrogram(x)
        x = x.unsqueeze(1)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            features.append(x)

        x = self.conv_post(x)
        features.append(x)

        x = torch.flatten(x, 1, -1)

        return x, features

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        """Compute magnitude spectrogram."""
        x = x.squeeze(1)

        # STFT
        x_stft = torch.stft(
            x,
            self.filter_length,
            self.hop_length,
            self.win_length,
            window=torch.hann_window(self.win_length, device=x.device),
            return_complex=True,
        )

        # Magnitude
        x_mag = torch.abs(x_stft)

        return x_mag


class BigVGANv2(nn.Module):
    """
    Complete BigVGAN-v2 model with generator and discriminators.
    """

    def __init__(self, config: Optional[BigVGANv2Config] = None):
        super().__init__()
        self.config = config or BigVGANv2Config()

        # Generator
        self.generator = BigVGANv2Generator(self.config)

        # Discriminators
        self.mpd = MultiPeriodDiscriminator(self.config)
        self.mrd = MultiResolutionDiscriminator(self.config)

        logger.info("Initialized BigVGAN-v2")

    def forward(
        self, mel: torch.Tensor, audio: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of BigVGAN-v2.

        Args:
            mel: Mel-spectrogram (B, n_mel_channels, T)
            audio: Target audio for training (B, 1, T * hop_length)

        Returns:
            Dictionary containing generated audio and losses
        """
        # Generate audio
        audio_fake = self.generator(mel)

        outputs = {"audio": audio_fake}

        if audio is not None and self.training:
            # Discriminator outputs
            mpd_real, mpd_fake, mpd_feat_real, mpd_feat_fake = self.mpd(
                audio, audio_fake.detach()
            )
            mrd_real, mrd_fake, mrd_feat_real, mrd_feat_fake = self.mrd(
                audio, audio_fake.detach()
            )

            # Losses
            outputs["mpd_real"] = mpd_real
            outputs["mpd_fake"] = mpd_fake
            outputs["mrd_real"] = mrd_real
            outputs["mrd_fake"] = mrd_fake
            outputs["mpd_features"] = (mpd_feat_real, mpd_feat_fake)
            outputs["mrd_features"] = (mrd_feat_real, mrd_feat_fake)

        return outputs

    @torch.inference_mode()
    def inference(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Inference mode.

        Args:
            mel: Mel-spectrogram

        Returns:
            Generated waveform
        """
        self.generator.eval()
        return self.generator(mel)

    def remove_weight_norm(self):
        """Remove weight normalization from generator for faster inference."""
        self.generator.remove_weight_norm()


# Helper functions
def weight_norm(module: nn.Module) -> nn.Module:
    """Apply weight normalization."""
    try:
        from torch.nn.utils.parametrizations import weight_norm as wn
    except ImportError:
        # Fallback to old API for older PyTorch versions
        from torch.nn.utils import weight_norm as wn

    return wn(module)


def remove_weight_norm(module: nn.Module) -> None:
    """Remove weight normalization."""
    try:
        from torch.nn.utils.parametrize import remove_parametrizations
        # New API uses remove_parametrizations
        remove_parametrizations(module, "weight")
    except (ImportError, ValueError):
        # Fallback to old API or already removed
        try:
            from torch.nn.utils import remove_weight_norm as rwn
            rwn(module)
        except ValueError:
            # Already removed
            pass


def compute_gan_loss(
    disc_real: List[torch.Tensor],
    disc_fake: List[torch.Tensor],
    loss_type: str = "hinge",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GAN losses.

    Args:
        disc_real: Real discriminator outputs
        disc_fake: Fake discriminator outputs
        loss_type: Type of GAN loss

    Returns:
        (discriminator_loss, generator_loss)
    """
    if loss_type == "hinge":
        # Discriminator loss
        disc_loss = 0
        for dr, df in zip(disc_real, disc_fake):
            disc_loss += torch.mean(torch.relu(1 - dr))
            disc_loss += torch.mean(torch.relu(1 + df))

        # Generator loss
        gen_loss = 0
        for df in disc_fake:
            gen_loss += -torch.mean(df)

    elif loss_type == "lsgan":
        # Discriminator loss
        disc_loss = 0
        for dr, df in zip(disc_real, disc_fake):
            disc_loss += torch.mean((dr - 1) ** 2)
            disc_loss += torch.mean(df**2)

        # Generator loss
        gen_loss = 0
        for df in disc_fake:
            gen_loss += torch.mean((df - 1) ** 2)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    return disc_loss, gen_loss


def compute_feature_matching_loss(
    feat_real: List[List[torch.Tensor]], feat_fake: List[List[torch.Tensor]]
) -> torch.Tensor:
    """
    Compute feature matching loss.

    Args:
        feat_real: Real features from discriminator
        feat_fake: Fake features from discriminator

    Returns:
        Feature matching loss
    """
    loss = 0
    for fr, ff in zip(feat_real, feat_fake):
        for r, f in zip(fr, ff):
            loss += torch.mean(torch.abs(r - f))

    return loss


def compute_mel_loss(
    audio_real: torch.Tensor,
    audio_fake: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int = 480,
    win_length: int = 1920,
    n_mels: int = 128,
) -> torch.Tensor:
    """
    Compute mel-spectrogram loss between real and generated audio.

    Args:
        audio_real: Real audio
        audio_fake: Generated audio
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
        n_mels: Number of mel channels

    Returns:
        Mel loss
    """
    # Create mel filter
    mel_filter = librosa.filters.mel(
        sr=48000, n_fft=n_fft, n_mels=n_mels, fmin=0, fmax=24000
    )
    mel_filter = torch.from_numpy(mel_filter).to(audio_real.device).float()

    # Compute spectrograms
    spec_real = torch.stft(
        audio_real.squeeze(1),
        n_fft,
        hop_length,
        win_length,
        window=torch.hann_window(win_length, device=audio_real.device),
        return_complex=True,
    )
    spec_fake = torch.stft(
        audio_fake.squeeze(1),
        n_fft,
        hop_length,
        win_length,
        window=torch.hann_window(win_length, device=audio_fake.device),
        return_complex=True,
    )

    # Convert to magnitude
    spec_real = torch.abs(spec_real)
    spec_fake = torch.abs(spec_fake)

    # Apply mel filter
    mel_real = torch.matmul(mel_filter.unsqueeze(0), spec_real)
    mel_fake = torch.matmul(mel_filter.unsqueeze(0), spec_fake)

    # Log scale
    mel_real = torch.log(torch.clamp(mel_real, min=1e-5))
    mel_fake = torch.log(torch.clamp(mel_fake, min=1e-5))

    # L1 loss
    return F.l1_loss(mel_real, mel_fake)


def create_bigvgan_v2(
    checkpoint_path: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> BigVGANv2:
    """Create BigVGAN-v2 instance."""
    config = BigVGANv2Config()
    model = BigVGANv2(config)

    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model.remove_weight_norm()  # Remove for inference

    return model.to(device)
