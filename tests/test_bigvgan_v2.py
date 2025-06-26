"""
Tests for BigVGAN-v2 vocoder
"""

import numpy as np
import pytest
import torch

from src.models.bigvgan_v2 import (
    BigVGANv2,
    BigVGANv2Config,
    BigVGANv2Generator,
    GeneratorBlock,
    MultiPeriodDiscriminator,
    MultiResolutionDiscriminator,
    ResBlock,
    SnakeBeta,
    compute_feature_matching_loss,
    compute_gan_loss,
    compute_mel_loss,
    create_bigvgan_v2,
)


class TestBigVGANv2Config:
    """Test configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BigVGANv2Config()

        assert config.hidden_channels == 1536
        assert config.sampling_rate == 48000
        assert config.hop_length == 480
        assert config.n_mel_channels == 128
        assert config.use_snake_activation == True

        # Check default upsample rates
        assert config.upsample_rates == [8, 6, 5, 2]
        assert np.prod(config.upsample_rates) == 240  # Should match hop_length/2

    def test_custom_config(self):
        """Test custom configuration."""
        config = BigVGANv2Config(
            hidden_channels=1024, sampling_rate=24000, hop_length=240
        )

        assert config.hidden_channels == 1024
        assert config.sampling_rate == 24000
        assert config.hop_length == 240


class TestSnakeBeta:
    """Test Snake-Beta activation function."""

    def test_initialization(self):
        """Test SnakeBeta initialization."""
        channels = 256
        snake = SnakeBeta(channels, alpha_logscale=True)

        assert snake.alpha.shape == (1, channels, 1)
        assert snake.beta.shape == (1, channels, 1)

    def test_forward(self):
        """Test forward pass."""
        batch_size = 2
        channels = 256
        time_steps = 100

        snake = SnakeBeta(channels)
        x = torch.randn(batch_size, channels, time_steps)

        output = snake(x)

        # Check shape preserved
        assert output.shape == x.shape

        # Check that it's not just identity
        assert not torch.allclose(output, x)

    def test_alpha_logscale(self):
        """Test alpha logscale mode."""
        channels = 128

        # With logscale
        snake_log = SnakeBeta(channels, alpha_logscale=True)
        # Without logscale
        snake_linear = SnakeBeta(channels, alpha_logscale=False)

        x = torch.randn(1, channels, 50)

        out_log = snake_log(x)
        out_linear = snake_linear(x)

        # Outputs should be different due to different parameterization
        assert not torch.allclose(out_log, out_linear)


class TestResBlock:
    """Test residual block."""

    def test_forward(self):
        """Test forward pass."""
        channels = 512
        kernel_size = 3
        dilations = [1, 3, 5]

        resblock = ResBlock(channels, kernel_size, dilations, use_snake=True)

        x = torch.randn(2, channels, 100)
        output = resblock(x)

        # Check shape preserved
        assert output.shape == x.shape

    def test_remove_weight_norm(self):
        """Test weight norm removal."""
        resblock = ResBlock(256, 3, [1, 3, 5])

        # Should not raise error
        resblock.remove_weight_norm()


class TestGeneratorBlock:
    """Test generator block."""

    def test_forward(self):
        """Test forward pass with upsampling."""
        in_channels = 512
        out_channels = 256
        kernel_size = 16
        stride = 8  # Upsample by 8

        block = GeneratorBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilations=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        )

        x = torch.randn(2, in_channels, 50)
        output = block(x)

        # Check upsampling worked
        assert output.shape == (2, out_channels, 50 * stride)

    def test_multiple_resblocks(self):
        """Test that multiple residual blocks are averaged."""
        block = GeneratorBlock(
            in_channels=256,
            out_channels=128,
            kernel_size=8,
            stride=4,
            resblock_kernel_sizes=[3, 5],
            resblock_dilations=[[1, 3], [1, 5]],
        )

        x = torch.randn(1, 256, 25)
        output = block(x)

        assert output.shape == (1, 128, 100)


class TestBigVGANv2Generator:
    """Test BigVGAN-v2 generator."""

    @pytest.fixture()
    def generator(self):
        """Create test generator."""
        config = BigVGANv2Config()
        # Reduce size for testing
        config.hidden_channels = 512
        return BigVGANv2Generator(config)

    def test_initialization(self, generator):
        """Test generator initialization."""
        assert hasattr(generator, "conv_pre")
        assert hasattr(generator, "blocks")
        assert hasattr(generator, "activation_post")
        assert hasattr(generator, "conv_post")

        # Check number of blocks matches upsample rates
        assert len(generator.blocks) == len(generator.config.upsample_rates)

    def test_forward(self, generator):
        """Test forward pass."""
        batch_size = 2
        mel_channels = 128
        mel_frames = 100

        mel = torch.randn(batch_size, mel_channels, mel_frames)

        audio = generator(mel)

        # Check output shape
        expected_length = mel_frames * np.prod(generator.config.upsample_rates)
        assert audio.shape == (batch_size, 1, expected_length)

        # Check output range (tanh activation)
        assert torch.all(audio >= -1)
        assert torch.all(audio <= 1)

    def test_remove_weight_norm(self, generator):
        """Test weight norm removal."""
        generator.remove_weight_norm()

        # Should complete without error

    def test_48khz_generation(self):
        """Test 48kHz audio generation."""
        config = BigVGANv2Config(sampling_rate=48000, hop_length=480)
        generator = BigVGANv2Generator(config)

        mel = torch.randn(1, 128, 100)
        audio = generator(mel)

        # For 100 mel frames with hop_length=480
        # Expected audio length = 100 * 480 = 48000 samples = 1 second @ 48kHz
        assert audio.shape[2] == 48000


class TestMultiPeriodDiscriminator:
    """Test Multi-Period Discriminator."""

    @pytest.fixture()
    def mpd(self):
        """Create test MPD."""
        config = BigVGANv2Config()
        return MultiPeriodDiscriminator(config)

    def test_forward(self, mpd):
        """Test forward pass."""
        batch_size = 2
        audio_length = 8000

        real_audio = torch.randn(batch_size, 1, audio_length)
        fake_audio = torch.randn(batch_size, 1, audio_length)

        real_outs, fake_outs, real_feats, fake_feats = mpd(real_audio, fake_audio)

        # Check number of discriminators
        assert len(real_outs) == 5  # Default 5 periods
        assert len(fake_outs) == 5
        assert len(real_feats) == 5
        assert len(fake_feats) == 5

        # Check that outputs are different for real/fake
        for real_out, fake_out in zip(real_outs, fake_outs):
            assert not torch.allclose(real_out, fake_out)


class TestMultiResolutionDiscriminator:
    """Test Multi-Resolution Discriminator."""

    @pytest.fixture()
    def mrd(self):
        """Create test MRD."""
        config = BigVGANv2Config()
        return MultiResolutionDiscriminator(config)

    def test_forward(self, mrd):
        """Test forward pass."""
        batch_size = 2
        audio_length = 16000

        real_audio = torch.randn(batch_size, 1, audio_length)
        fake_audio = torch.randn(batch_size, 1, audio_length)

        real_outs, fake_outs, real_feats, fake_feats = mrd(real_audio, fake_audio)

        # Check number of discriminators
        assert len(real_outs) == 3  # Default 3 resolutions
        assert len(fake_outs) == 3
        assert len(real_feats) == 3
        assert len(fake_feats) == 3


class TestBigVGANv2:
    """Test complete BigVGAN-v2 model."""

    @pytest.fixture()
    def model(self):
        """Create test model."""
        config = BigVGANv2Config()
        config.hidden_channels = 512  # Smaller for testing
        return BigVGANv2(config)

    def test_initialization(self, model):
        """Test model initialization."""
        assert hasattr(model, "generator")
        assert hasattr(model, "mpd")
        assert hasattr(model, "mrd")

    def test_forward_inference(self, model):
        """Test forward pass in inference mode."""
        model.eval()

        batch_size = 2
        mel_frames = 50

        mel = torch.randn(batch_size, 128, mel_frames)

        outputs = model(mel)

        assert "audio" in outputs
        assert outputs["audio"].shape[2] == mel_frames * np.prod(
            model.config.upsample_rates
        )

    def test_forward_training(self, model):
        """Test forward pass in training mode."""
        model.train()

        batch_size = 2
        mel_frames = 50
        expected_audio_length = mel_frames * np.prod(model.config.upsample_rates)

        mel = torch.randn(batch_size, 128, mel_frames)
        real_audio = torch.randn(batch_size, 1, expected_audio_length)

        outputs = model(mel, audio=real_audio)

        # Check all outputs present
        assert "audio" in outputs
        assert "mpd_real" in outputs
        assert "mpd_fake" in outputs
        assert "mrd_real" in outputs
        assert "mrd_fake" in outputs
        assert "mpd_features" in outputs
        assert "mrd_features" in outputs

    def test_inference_method(self, model):
        """Test dedicated inference method."""
        mel = torch.randn(1, 128, 100)

        audio = model.inference(mel)

        assert audio.shape == (1, 1, 100 * np.prod(model.config.upsample_rates))
        assert torch.all(audio >= -1)
        assert torch.all(audio <= 1)

    def test_remove_weight_norm(self, model):
        """Test weight norm removal."""
        model.remove_weight_norm()

        # Should complete without error


class TestLossFunctions:
    """Test loss computation functions."""

    def test_compute_gan_loss_hinge(self):
        """Test hinge GAN loss."""
        disc_real = [torch.randn(2, 10) + 1]  # Tend positive
        disc_fake = [torch.randn(2, 10) - 1]  # Tend negative

        disc_loss, gen_loss = compute_gan_loss(disc_real, disc_fake, "hinge")

        assert disc_loss.item() > 0
        assert gen_loss.item() != 0

    def test_compute_gan_loss_lsgan(self):
        """Test LSGAN loss."""
        disc_real = [torch.randn(2, 10)]
        disc_fake = [torch.randn(2, 10)]

        disc_loss, gen_loss = compute_gan_loss(disc_real, disc_fake, "lsgan")

        assert disc_loss.item() > 0
        assert gen_loss.item() > 0

    def test_compute_feature_matching_loss(self):
        """Test feature matching loss."""
        # Simulate discriminator features
        feat_real = [[torch.randn(2, 64, 50), torch.randn(2, 128, 25)]]
        feat_fake = [[torch.randn(2, 64, 50), torch.randn(2, 128, 25)]]

        loss = compute_feature_matching_loss(feat_real, feat_fake)

        assert loss.item() > 0

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="Requires CUDA for librosa mel filters"
    )
    def test_compute_mel_loss(self):
        """Test mel-spectrogram loss."""
        audio_real = torch.randn(2, 1, 8000)
        audio_fake = torch.randn(2, 1, 8000)

        loss = compute_mel_loss(audio_real, audio_fake)

        assert loss.item() > 0


class TestCreateBigVGANv2:
    """Test factory function."""

    def test_create_default(self):
        """Test creating with default settings."""
        model = create_bigvgan_v2()

        assert isinstance(model, BigVGANv2)

    def test_create_with_checkpoint(self, tmp_path):
        """Test creating with checkpoint."""
        # Create dummy checkpoint
        checkpoint_path = tmp_path / "bigvgan_checkpoint.pt"

        # Save dummy state dict
        dummy_model = create_bigvgan_v2()
        torch.save(dummy_model.state_dict(), checkpoint_path)

        # Load model
        model = create_bigvgan_v2(checkpoint_path=str(checkpoint_path))

        assert isinstance(model, BigVGANv2)


class TestIntegration:
    """Integration tests for BigVGAN-v2."""

    @pytest.fixture()
    def model(self):
        """Create model for integration tests."""
        config = BigVGANv2Config()
        config.hidden_channels = 256  # Smaller for faster tests
        return create_bigvgan_v2(device="cpu")

    def test_variable_length_generation(self, model):
        """Test generation with different mel lengths."""
        model.eval()

        for mel_frames in [10, 50, 100, 200]:
            mel = torch.randn(1, 128, mel_frames)
            audio = model.inference(mel)

            expected_length = mel_frames * np.prod(model.config.upsample_rates)
            assert audio.shape == (1, 1, expected_length)

    def test_batch_generation(self, model):
        """Test batch generation."""
        model.eval()

        batch_size = 4
        mel_frames = 50

        mel = torch.randn(batch_size, 128, mel_frames)
        audio = model.inference(mel)

        assert audio.shape[0] == batch_size

    def test_deterministic_generation(self, model):
        """Test that generation is deterministic in eval mode."""
        model.eval()

        mel = torch.randn(1, 128, 50)

        # Generate twice
        audio1 = model.inference(mel)
        audio2 = model.inference(mel)

        # Should be identical
        assert torch.allclose(audio1, audio2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
