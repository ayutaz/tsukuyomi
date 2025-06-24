"""Test voice morphing capabilities in Ultimate Acoustic Model."""

import numpy as np
import pytest
import torch

from src.models.ultimate_acoustic_model import (
    UltimateAcousticConfig,
    UltimateAcousticModel,
)


class TestVoiceMorphing:
    """Test suite for voice morphing functionality."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return UltimateAcousticConfig(
            hidden_channels=192,
            n_speakers=100,
            speaker_embed_dim=256,
            emotion_embed_dim=128,
            n_emotions=7,
            n_flows=4,
            vae_layers=8,
        )

    @pytest.fixture
    def model(self, config):
        """Create acoustic model instance."""
        return UltimateAcousticModel(config)

    def test_basic_voice_morphing(self, model):
        """Test basic voice morphing with two speakers."""
        batch_size = 2
        seq_len = 50
        hidden_size = model.config.hidden_channels

        # Create dummy inputs
        phoneme_embeddings = torch.randn(batch_size, hidden_size, seq_len)
        phoneme_lengths = torch.tensor([50, 48])

        # Two speakers per batch
        speaker_ids = torch.tensor([[0, 5], [10, 15]])
        speaker_weights = torch.tensor([[0.7, 0.3], [0.5, 0.5]])

        # Test morphing
        mel = model.morph_voices(
            phoneme_embeddings=phoneme_embeddings,
            phoneme_lengths=phoneme_lengths,
            speaker_ids=speaker_ids,
            speaker_weights=speaker_weights,
        )

        assert mel.shape[0] == batch_size
        assert mel.shape[1] == model.config.n_mel_channels
        assert mel.dim() == 3

    def test_multi_speaker_morphing(self, model):
        """Test morphing with multiple speakers."""
        batch_size = 1
        seq_len = 30
        hidden_size = model.config.hidden_channels

        # Create inputs
        phoneme_embeddings = torch.randn(batch_size, hidden_size, seq_len)
        phoneme_lengths = torch.tensor([30])

        # Four speakers
        speaker_ids = torch.tensor([[0, 5, 10, 15]])
        speaker_weights = torch.tensor([[0.4, 0.3, 0.2, 0.1]])

        mel = model.morph_voices(
            phoneme_embeddings=phoneme_embeddings,
            phoneme_lengths=phoneme_lengths,
            speaker_ids=speaker_ids,
            speaker_weights=speaker_weights,
        )

        assert mel.shape[0] == batch_size
        assert mel.shape[1] == model.config.n_mel_channels

    def test_emotion_morphing(self, model):
        """Test morphing with emotions."""
        batch_size = 2
        seq_len = 40
        hidden_size = model.config.hidden_channels

        # Create inputs
        phoneme_embeddings = torch.randn(batch_size, hidden_size, seq_len)
        phoneme_lengths = torch.tensor([40, 38])

        # Speakers and emotions
        speaker_ids = torch.tensor([[0, 5], [10, 15]])
        speaker_weights = torch.tensor([[0.6, 0.4], [0.8, 0.2]])

        emotion_ids = torch.tensor([[0, 1], [2, 3]])  # neutral+happy, sad+angry
        emotion_weights = torch.tensor([[0.7, 0.3], [0.5, 0.5]])

        mel = model.morph_voices(
            phoneme_embeddings=phoneme_embeddings,
            phoneme_lengths=phoneme_lengths,
            speaker_ids=speaker_ids,
            speaker_weights=speaker_weights,
            emotion_ids=emotion_ids,
            emotion_weights=emotion_weights,
        )

        assert mel.shape[0] == batch_size
        assert mel.shape[1] == model.config.n_mel_channels

    def test_weight_normalization(self, model):
        """Test automatic weight normalization."""
        batch_size = 1
        seq_len = 30
        hidden_size = model.config.hidden_channels

        # Create inputs
        phoneme_embeddings = torch.randn(batch_size, hidden_size, seq_len)
        phoneme_lengths = torch.tensor([30])

        # Unnormalized weights
        speaker_ids = torch.tensor([[0, 5]])
        speaker_weights = torch.tensor([[2.0, 3.0]])  # Sum != 1

        mel = model.morph_voices(
            phoneme_embeddings=phoneme_embeddings,
            phoneme_lengths=phoneme_lengths,
            speaker_ids=speaker_ids,
            speaker_weights=speaker_weights,
        )

        # Should still work with automatic softmax normalization
        assert mel.shape[0] == batch_size
        assert mel.shape[1] == model.config.n_mel_channels

    def test_length_scale_with_morphing(self, model):
        """Test length scaling with voice morphing."""
        batch_size = 1
        seq_len = 30
        hidden_size = model.config.hidden_channels

        # Create inputs
        phoneme_embeddings = torch.randn(batch_size, hidden_size, seq_len)
        phoneme_lengths = torch.tensor([30])

        speaker_ids = torch.tensor([[0, 5]])
        speaker_weights = torch.tensor([[0.5, 0.5]])

        # Test different length scales
        mel_normal = model.morph_voices(
            phoneme_embeddings=phoneme_embeddings,
            phoneme_lengths=phoneme_lengths,
            speaker_ids=speaker_ids,
            speaker_weights=speaker_weights,
            length_scale=1.0,
        )

        mel_fast = model.morph_voices(
            phoneme_embeddings=phoneme_embeddings,
            phoneme_lengths=phoneme_lengths,
            speaker_ids=speaker_ids,
            speaker_weights=speaker_weights,
            length_scale=0.8,
        )

        mel_slow = model.morph_voices(
            phoneme_embeddings=phoneme_embeddings,
            phoneme_lengths=phoneme_lengths,
            speaker_ids=speaker_ids,
            speaker_weights=speaker_weights,
            length_scale=1.2,
        )

        # Check length relationships
        assert mel_fast.shape[2] < mel_normal.shape[2]
        assert mel_slow.shape[2] > mel_normal.shape[2]

    def test_temperature_with_morphing(self, model):
        """Test temperature control with voice morphing."""
        batch_size = 1
        seq_len = 30
        hidden_size = model.config.hidden_channels

        # Create inputs
        phoneme_embeddings = torch.randn(batch_size, hidden_size, seq_len)
        phoneme_lengths = torch.tensor([30])

        speaker_ids = torch.tensor([[0, 5]])
        speaker_weights = torch.tensor([[0.5, 0.5]])

        # Test different temperatures
        mel_normal = model.morph_voices(
            phoneme_embeddings=phoneme_embeddings,
            phoneme_lengths=phoneme_lengths,
            speaker_ids=speaker_ids,
            speaker_weights=speaker_weights,
            temperature=1.0,
        )

        mel_low_temp = model.morph_voices(
            phoneme_embeddings=phoneme_embeddings,
            phoneme_lengths=phoneme_lengths,
            speaker_ids=speaker_ids,
            speaker_weights=speaker_weights,
            temperature=0.5,
        )

        # Low temperature should have smaller values
        assert torch.allclose(mel_low_temp, mel_normal * 0.5, rtol=1e-5)

    def test_gradient_morphing(self, model):
        """Test gradual morphing between speakers."""
        batch_size = 1
        seq_len = 30
        hidden_size = model.config.hidden_channels

        # Create inputs
        phoneme_embeddings = torch.randn(batch_size, hidden_size, seq_len)
        phoneme_lengths = torch.tensor([30])

        speaker_ids = torch.tensor([[0, 5]])

        # Test gradual morphing from speaker 0 to speaker 5
        morphing_steps = 5
        mels = []

        for i in range(morphing_steps):
            weight = i / (morphing_steps - 1)
            speaker_weights = torch.tensor([[1 - weight, weight]])

            mel = model.morph_voices(
                phoneme_embeddings=phoneme_embeddings,
                phoneme_lengths=phoneme_lengths,
                speaker_ids=speaker_ids,
                speaker_weights=speaker_weights,
            )
            mels.append(mel)

        # All outputs should have same shape
        for mel in mels:
            assert mel.shape == mels[0].shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_morphing(self, model):
        """Test voice morphing on CUDA."""
        model = model.cuda()

        batch_size = 2
        seq_len = 30
        hidden_size = model.config.hidden_channels

        # Create CUDA inputs
        phoneme_embeddings = torch.randn(batch_size, hidden_size, seq_len).cuda()
        phoneme_lengths = torch.tensor([30, 28]).cuda()

        speaker_ids = torch.tensor([[0, 5], [10, 15]]).cuda()
        speaker_weights = torch.tensor([[0.7, 0.3], [0.5, 0.5]]).cuda()

        mel = model.morph_voices(
            phoneme_embeddings=phoneme_embeddings,
            phoneme_lengths=phoneme_lengths,
            speaker_ids=speaker_ids,
            speaker_weights=speaker_weights,
        )

        assert mel.is_cuda
        assert mel.shape[0] == batch_size
