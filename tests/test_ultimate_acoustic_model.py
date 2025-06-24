"""
Tests for Ultimate Acoustic Model (Matcha-TTS/VITS integration)
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.models.ultimate_acoustic_model import (
    UltimateAcousticModel,
    UltimateAcousticConfig,
    ConditionalFlowMatching,
    StochasticDurationPredictor,
    MultiSpeakerEncoder,
    PosteriorEncoder,
    create_ultimate_acoustic_model,
    sequence_mask,
    generate_path,
    expand_durations,
    kl_divergence,
)


class TestUltimateAcousticConfig:
    """Test configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = UltimateAcousticConfig()

        assert config.hidden_channels == 512
        assert config.n_mel_channels == 128
        assert config.sampling_rate == 48000
        assert config.n_flows == 12
        assert config.n_speakers == 1000
        assert config.use_stochastic_duration == True
        assert config.use_bf16 == True

    def test_custom_config(self):
        """Test custom configuration."""
        config = UltimateAcousticConfig(
            hidden_channels=256, n_speakers=500, sampling_rate=24000
        )

        assert config.hidden_channels == 256
        assert config.n_speakers == 500
        assert config.sampling_rate == 24000


class TestConditionalFlowMatching:
    """Test Conditional Flow Matching module."""

    @pytest.fixture
    def flow_module(self):
        """Create test flow module."""
        config = UltimateAcousticConfig()
        return ConditionalFlowMatching(config)

    def test_forward(self, flow_module):
        """Test forward pass."""
        batch_size = 2
        channels = 512
        time_steps = 100

        x = torch.randn(batch_size, channels, time_steps)
        mask = torch.ones(batch_size, 1, time_steps)
        mu = torch.randn(batch_size, channels, time_steps)

        mel = flow_module(x, mask, mu)

        # Check output shape
        assert mel.shape == (batch_size, 128, time_steps)  # n_mel_channels

    def test_reverse(self, flow_module):
        """Test reverse flow for inference."""
        batch_size = 1
        channels = 512
        time_steps = 50

        x = torch.randn(batch_size, channels, time_steps)
        mask = torch.ones(batch_size, 1, time_steps)
        mu = torch.randn(batch_size, channels, time_steps)

        mel = flow_module(x, mask, mu, reverse=True)

        assert mel.shape == (batch_size, 128, time_steps)

    def test_time_conditioning(self, flow_module):
        """Test with explicit time steps."""
        batch_size = 2
        channels = 512
        time_steps = 100

        x = torch.randn(batch_size, channels, time_steps)
        mask = torch.ones(batch_size, 1, time_steps)
        mu = torch.randn(batch_size, channels, time_steps)
        t = torch.linspace(0, 1, batch_size).unsqueeze(1)

        mel = flow_module(x, mask, mu, t=t)

        assert mel.shape == (batch_size, 128, time_steps)


class TestStochasticDurationPredictor:
    """Test Stochastic Duration Predictor."""

    @pytest.fixture
    def duration_predictor(self):
        """Create test duration predictor."""
        config = UltimateAcousticConfig()
        return StochasticDurationPredictor(config)

    def test_training_mode(self, duration_predictor):
        """Test forward in training mode."""
        duration_predictor.train()

        batch_size = 2
        channels = 512
        time_steps = 50

        x = torch.randn(batch_size, channels, time_steps)
        mask = torch.ones(batch_size, 1, time_steps)
        w = torch.randint(
            1, 10, (batch_size, 1, time_steps)
        ).float()  # Target durations

        durations, loss = duration_predictor(x, mask, w=w, reverse=False)

        assert durations.shape == w.shape
        assert loss.item() > 0  # Should have positive loss

    def test_inference_mode(self, duration_predictor):
        """Test forward in inference mode."""
        duration_predictor.eval()

        batch_size = 2
        channels = 512
        time_steps = 50

        x = torch.randn(batch_size, channels, time_steps)
        mask = torch.ones(batch_size, 1, time_steps)

        durations, _ = duration_predictor(x, mask, reverse=True)

        assert durations.shape == (batch_size, 1, time_steps)
        # Durations should be non-negative
        assert torch.all(durations >= 0)


class TestMultiSpeakerEncoder:
    """Test Multi-Speaker Encoder."""

    @pytest.fixture
    def speaker_encoder(self):
        """Create test speaker encoder."""
        config = UltimateAcousticConfig()
        return MultiSpeakerEncoder(config)

    def test_speaker_embedding(self, speaker_encoder):
        """Test basic speaker embedding."""
        batch_size = 2
        speaker_ids = torch.tensor([0, 10])

        encoding = speaker_encoder(speaker_ids=speaker_ids)

        assert encoding.shape == (batch_size, 512)  # hidden_channels

    def test_reference_encoder(self, speaker_encoder):
        """Test reference encoder for voice cloning."""
        if not speaker_encoder.config.use_speaker_encoder:
            pytest.skip("Reference encoder not enabled")

        batch_size = 2
        mel_channels = 128
        time_steps = 200

        reference_mel = torch.randn(batch_size, mel_channels, time_steps)

        encoding = speaker_encoder(reference_mel=reference_mel)

        assert encoding.shape == (batch_size, 512)

    def test_emotion_conditioning(self, speaker_encoder):
        """Test emotion conditioning."""
        batch_size = 2
        speaker_ids = torch.tensor([0, 1])
        emotion_ids = torch.tensor([1, 2])  # Happy, Sad

        encoding = speaker_encoder(speaker_ids=speaker_ids, emotion_ids=emotion_ids)

        assert encoding.shape == (batch_size, 512)

    def test_style_tokens(self, speaker_encoder):
        """Test style token attention."""
        batch_size = 2
        speaker_ids = torch.tensor([0, 1])

        encoding_with_gst = speaker_encoder(
            speaker_ids=speaker_ids, use_style_tokens=True
        )

        encoding_without_gst = speaker_encoder(
            speaker_ids=speaker_ids, use_style_tokens=False
        )

        # Both should have same shape
        assert encoding_with_gst.shape == encoding_without_gst.shape

        # But different values (due to GST)
        assert not torch.allclose(encoding_with_gst, encoding_without_gst)


class TestPosteriorEncoder:
    """Test Posterior Encoder (VAE)."""

    @pytest.fixture
    def posterior_encoder(self):
        """Create test posterior encoder."""
        return PosteriorEncoder(
            in_channels=128,  # n_mel_channels
            out_channels=192,  # z_channels
            hidden_channels=384,
            n_layers=16,
        )

    def test_forward(self, posterior_encoder):
        """Test forward pass."""
        batch_size = 2
        mel_channels = 128
        time_steps = 100

        x = torch.randn(batch_size, mel_channels, time_steps)
        mask = torch.ones(batch_size, 1, time_steps)

        z, m, logs = posterior_encoder(x, mask)

        # Check shapes
        assert z.shape == (batch_size, 192, time_steps)
        assert m.shape == (batch_size, 192, time_steps)
        assert logs.shape == (batch_size, 192, time_steps)

        # Check that sampling worked
        assert not torch.allclose(z, m)  # z should be sampled, not just mean


class TestUltimateAcousticModel:
    """Test complete Ultimate Acoustic Model."""

    @pytest.fixture
    def model(self):
        """Create test model."""
        config = UltimateAcousticConfig()
        # Reduce size for testing
        config.n_flows = 4
        config.vae_layers = 4
        return UltimateAcousticModel(config)

    def test_initialization(self, model):
        """Test model initialization."""
        assert hasattr(model, "text_encoder")
        assert hasattr(model, "duration_predictor")
        assert hasattr(model, "flow_matching")
        assert hasattr(model, "posterior_encoder")
        assert hasattr(model, "speaker_encoder")

    def test_forward_inference(self, model):
        """Test forward pass in inference mode."""
        model.eval()

        batch_size = 2
        phoneme_channels = 512
        phoneme_length = 30

        phoneme_embeddings = torch.randn(batch_size, phoneme_channels, phoneme_length)
        phoneme_lengths = torch.tensor([30, 28])
        speaker_ids = torch.tensor([0, 1])

        outputs = model(
            phoneme_embeddings=phoneme_embeddings,
            phoneme_lengths=phoneme_lengths,
            speaker_ids=speaker_ids,
        )

        assert "mel" in outputs
        assert "mel_mask" in outputs
        assert "durations" in outputs
        assert outputs["mel"].dim() == 3

    def test_forward_training(self, model):
        """Test forward pass in training mode."""
        model.train()

        batch_size = 2
        phoneme_channels = 512
        phoneme_length = 30
        mel_length = 200

        phoneme_embeddings = torch.randn(batch_size, phoneme_channels, phoneme_length)
        phoneme_lengths = torch.tensor([30, 28])
        speaker_ids = torch.tensor([0, 1])
        mel_targets = torch.randn(batch_size, 128, mel_length)
        mel_lengths = torch.tensor([200, 180])

        outputs = model(
            phoneme_embeddings=phoneme_embeddings,
            phoneme_lengths=phoneme_lengths,
            speaker_ids=speaker_ids,
            mel_targets=mel_targets,
            mel_lengths=mel_lengths,
        )

        # Check outputs
        assert "mel" in outputs
        assert "recon_loss" in outputs
        assert "kl_loss" in outputs
        assert "duration_loss" in outputs
        assert "total_loss" in outputs

        # Losses should be positive
        assert outputs["recon_loss"].item() > 0
        assert outputs["kl_loss"].item() >= 0  # KL can be 0
        assert outputs["total_loss"].item() > 0

    def test_multi_speaker(self, model):
        """Test with multiple speakers."""
        model.eval()

        batch_size = 4
        phoneme_embeddings = torch.randn(batch_size, 512, 20)
        phoneme_lengths = torch.tensor([20, 19, 20, 18])
        speaker_ids = torch.tensor([0, 100, 500, 999])  # Different speakers

        outputs = model(
            phoneme_embeddings=phoneme_embeddings,
            phoneme_lengths=phoneme_lengths,
            speaker_ids=speaker_ids,
        )

        assert outputs["mel"].shape[0] == batch_size

    def test_emotion_conditioning(self, model):
        """Test with emotion conditioning."""
        model.eval()

        batch_size = 2
        phoneme_embeddings = torch.randn(batch_size, 512, 25)
        phoneme_lengths = torch.tensor([25, 24])
        speaker_ids = torch.tensor([0, 1])
        emotion_ids = torch.tensor([1, 2])  # Happy, Sad

        outputs = model(
            phoneme_embeddings=phoneme_embeddings,
            phoneme_lengths=phoneme_lengths,
            speaker_ids=speaker_ids,
            emotion_ids=emotion_ids,
        )

        assert outputs["mel"].shape[0] == batch_size

    def test_voice_cloning(self, model):
        """Test voice cloning with reference mel."""
        model.eval()

        batch_size = 1
        phoneme_embeddings = torch.randn(batch_size, 512, 30)
        phoneme_lengths = torch.tensor([30])
        reference_mel = torch.randn(batch_size, 128, 300)  # Reference audio

        outputs = model(
            phoneme_embeddings=phoneme_embeddings,
            phoneme_lengths=phoneme_lengths,
            reference_mel=reference_mel,
        )

        assert outputs["mel"].shape[0] == batch_size

    def test_inference_method(self, model):
        """Test dedicated inference method."""
        batch_size = 2
        phoneme_embeddings = torch.randn(batch_size, 512, 20)
        phoneme_lengths = torch.tensor([20, 19])
        speaker_ids = torch.tensor([0, 1])

        mel = model.inference(
            phoneme_embeddings=phoneme_embeddings,
            phoneme_lengths=phoneme_lengths,
            speaker_ids=speaker_ids,
            length_scale=1.2,  # Slower
            temperature=0.8,  # Less variation
        )

        assert mel.dim() == 3
        assert mel.shape[0] == batch_size
        # Length should be scaled
        assert mel.shape[2] > 20  # Expanded from phonemes


class TestUtilityFunctions:
    """Test utility functions."""

    def test_sequence_mask(self):
        """Test sequence mask generation."""
        lengths = torch.tensor([5, 3, 4])
        mask = sequence_mask(lengths, max_len=6)

        expected = (
            torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0]])
            .unsqueeze(1)
            .bool()
        )

        assert torch.all(mask == expected)

    def test_generate_path(self):
        """Test alignment path generation."""
        durations = torch.tensor([[2, 3, 2], [1, 4, 2]]).float()

        mask_x = torch.ones(2, 1, 3)
        mask_y = torch.ones(2, 1, 7)

        path = generate_path(durations, mask_x, mask_y)

        assert path.shape == (2, 3, 7)
        # Check that each phoneme maps to correct frames
        assert path[0, 0, :2].sum() == 2  # First phoneme -> 2 frames
        assert path[0, 1, 2:5].sum() == 3  # Second phoneme -> 3 frames

    def test_expand_durations(self):
        """Test duration-based expansion."""
        x = torch.randn(2, 256, 3)  # 2 batch, 256 channels, 3 phonemes
        durations = torch.tensor([[2, 3, 2], [1, 4, 2]]).float()

        expanded = expand_durations(x, durations)

        assert expanded.shape == (2, 256, 7)  # Max total duration

    def test_kl_divergence(self):
        """Test KL divergence calculation."""
        batch_size = 2
        channels = 192
        time_steps = 100

        m_p = torch.randn(batch_size, channels, time_steps)
        logs_p = torch.randn(batch_size, channels, time_steps)
        m_q = torch.randn(batch_size, channels, time_steps)
        logs_q = torch.randn(batch_size, channels, time_steps)
        mask = torch.ones(batch_size, 1, time_steps)

        kl = kl_divergence(m_p, logs_p, m_q, logs_q, mask)

        assert kl.item() >= 0  # KL divergence is non-negative


class TestCreateUltimateAcousticModel:
    """Test factory function."""

    def test_create_default(self):
        """Test creating with default settings."""
        model = create_ultimate_acoustic_model()

        assert isinstance(model, UltimateAcousticModel)

    def test_create_with_checkpoint(self, tmp_path):
        """Test creating with checkpoint."""
        # Create dummy checkpoint
        checkpoint_path = tmp_path / "acoustic_checkpoint.pt"

        # Save dummy state dict
        dummy_model = create_ultimate_acoustic_model()
        torch.save(dummy_model.state_dict(), checkpoint_path)

        # Load model
        model = create_ultimate_acoustic_model(checkpoint_path=str(checkpoint_path))

        assert isinstance(model, UltimateAcousticModel)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
