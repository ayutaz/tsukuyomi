"""
Tests for core TTS models
"""

import pytest
import torch
import numpy as np

from src.models.vits import VITS
from src.models.hifigan import HiFiGAN, Generator
from src.models.f0_bert import F0BERT
from src.models.commons import monotonic_align
from src.models.modules import TextEncoder, PosteriorEncoder, ResidualCouplingBlock


class TestVITS:
    """Test VITS model"""

    def test_vits_creation(self):
        """Test VITS model creation"""
        model = VITS(
            n_vocab=100,
            n_speakers=1,
            hidden_channels=32,
            filter_channels=64,
            n_heads=2,
            n_layers=2,
            kernel_size=3,
            p_dropout=0.1,
        )
        assert model is not None
        assert model.n_vocab == 100
        assert model.n_speakers == 1

    def test_vits_forward(self):
        """Test VITS forward pass"""
        model = VITS(
            n_vocab=100,
            n_speakers=1,
            hidden_channels=32,
            filter_channels=64,
            n_heads=2,
            n_layers=2,
        )
        model.eval()

        # Create dummy inputs
        text = torch.randint(0, 100, (1, 10))
        text_lengths = torch.tensor([10])

        with torch.no_grad():
            outputs = model.infer(text, text_lengths)

        assert outputs is not None
        assert outputs.dim() == 3  # [B, 1, T]
        assert outputs.shape[0] == 1

    def test_vits_multispeaker(self):
        """Test VITS with multiple speakers"""
        model = VITS(
            n_vocab=100,
            n_speakers=5,
            hidden_channels=32,
            filter_channels=64,
            n_heads=2,
            n_layers=2,
        )
        model.eval()

        # Test with speaker ID
        text = torch.randint(0, 100, (1, 10))
        text_lengths = torch.tensor([10])
        speaker_ids = torch.tensor([2])

        with torch.no_grad():
            outputs = model.infer(text, text_lengths, speaker_ids)

        assert outputs is not None
        assert outputs.shape[0] == 1


class TestHiFiGAN:
    """Test HiFi-GAN vocoder"""

    def test_generator_creation(self):
        """Test HiFi-GAN generator creation"""
        generator = Generator(
            in_channels=80,
            upsample_initial_channel=128,
            upsample_rates=[8, 8, 2, 2],
            upsample_kernel_sizes=[16, 16, 4, 4],
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        )
        assert generator is not None

    def test_generator_forward(self):
        """Test generator forward pass"""
        generator = Generator(
            in_channels=80,
            upsample_initial_channel=128,
            upsample_rates=[8, 8, 2, 2],
            upsample_kernel_sizes=[16, 16, 4, 4],
        )
        generator.eval()

        # Create dummy mel spectrogram
        mel = torch.randn(1, 80, 10)

        with torch.no_grad():
            audio = generator(mel)

        assert audio is not None
        assert audio.dim() == 3  # [B, 1, T]
        assert audio.shape[0] == 1
        assert audio.shape[1] == 1
        # Check upsampling ratio
        expected_length = 10 * 8 * 8 * 2 * 2  # mel_length * product(upsample_rates)
        assert audio.shape[2] == expected_length

    def test_hifigan_full(self):
        """Test full HiFi-GAN model"""
        model = HiFiGAN(
            in_channels=80,
            upsample_initial_channel=128,
            upsample_rates=[8, 8, 2, 2],
            upsample_kernel_sizes=[16, 16, 4, 4],
        )
        model.eval()

        # Test inference
        mel = torch.randn(1, 80, 10)
        with torch.no_grad():
            audio = model.infer(mel)

        assert audio is not None
        assert audio.dim() == 3


class TestF0BERT:
    """Test F0-BERT model"""

    def test_f0bert_creation(self):
        """Test F0-BERT creation"""
        model = F0BERT(hidden_size=256, num_layers=4, num_heads=4, pitch_bins=256)
        assert model is not None
        assert model.hidden_size == 256
        assert model.pitch_bins == 256

    def test_f0bert_forward(self):
        """Test F0-BERT forward pass"""
        model = F0BERT(hidden_size=256, num_layers=4, num_heads=4, pitch_bins=256)
        model.eval()

        # Create dummy input
        hidden_states = torch.randn(2, 50, 256)
        attention_mask = torch.ones(2, 50)

        with torch.no_grad():
            outputs = model(hidden_states, attention_mask)

        assert "pitch_logits" in outputs
        assert "pitch_embedding" in outputs
        assert outputs["pitch_logits"].shape == (2, 50, 256)
        assert outputs["pitch_embedding"].shape == (2, 50, 256)


class TestCommons:
    """Test common utilities"""

    def test_monotonic_align(self):
        """Test monotonic alignment search"""
        # Create dummy cost matrix
        neg_cent = torch.randn(2, 20, 10)  # [B, T_y, T_x]
        x_mask = torch.ones(2, 10)
        y_mask = torch.ones(2, 20)

        # Test alignment
        path = monotonic_align.maximum_path(neg_cent, x_mask, y_mask)

        assert path is not None
        assert path.shape == (2, 20, 10)
        # Check that each text position has at least one aligned mel frame
        assert (path.sum(dim=1) > 0).all()


class TestModules:
    """Test individual modules"""

    def test_text_encoder(self):
        """Test text encoder module"""
        encoder = TextEncoder(
            n_vocab=100,
            n_feats=80,
            n_channels=192,
            filter_channels=768,
            n_heads=2,
            n_layers=2,
            kernel_size=3,
            p_dropout=0.1,
        )

        # Test forward
        x = torch.randint(0, 100, (2, 20))
        x_lengths = torch.tensor([20, 18])

        z, m, logs, x_mask = encoder(x, x_lengths)

        assert z.shape[0] == 2
        assert m.shape[0] == 2
        assert logs.shape[0] == 2
        assert x_mask.shape == (2, 1, 20)

    def test_posterior_encoder(self):
        """Test posterior encoder"""
        encoder = PosteriorEncoder(
            in_channels=80,
            out_channels=192,
            hidden_channels=192,
            kernel_size=5,
            dilation_rate=1,
            n_layers=16,
        )

        # Test forward
        x = torch.randn(2, 80, 100)
        x_lengths = torch.tensor([100, 90])

        z, m, logs, x_mask = encoder(x, x_lengths)

        assert z.shape[0] == 2
        assert z.shape[1] == 192
        assert m.shape == logs.shape

    def test_residual_coupling_block(self):
        """Test residual coupling block"""
        block = ResidualCouplingBlock(
            channels=192,
            hidden_channels=192,
            kernel_size=5,
            dilation_rate=1,
            n_layers=4,
        )

        # Test forward
        x = torch.randn(2, 192, 100)
        x_mask = torch.ones(2, 1, 100)

        output = block(x, x_mask)

        assert output.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
