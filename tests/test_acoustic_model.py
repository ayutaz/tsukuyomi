"""
Unit tests for acoustic model components.

Tests the acoustic model architecture including Conformer blocks,
attention mechanisms, and flow-based generation.
"""

from typing import Dict, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.acoustic_model import AcousticModel, AcousticModelConfig, ConformerBlock


class TestAcousticModelConfig:
    """Test suite for AcousticModelConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = AcousticModelConfig()

        assert config.hidden_dim == 512
        assert config.n_layers == 6
        assert config.n_heads == 8
        assert config.n_mel_channels == 80
        assert config.use_bf16 is True
        assert config.decoder_type == "flow"

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = AcousticModelConfig(
            hidden_dim=256, n_layers=4, use_bf16=False, n_speakers=100
        )

        assert config.hidden_dim == 256
        assert config.n_layers == 4
        assert config.use_bf16 is False
        assert config.n_speakers == 100


class TestConformerBlock:
    """Test suite for ConformerBlock."""

    @pytest.fixture
    def config(self) -> AcousticModelConfig:
        """Create test configuration."""
        return AcousticModelConfig(hidden_dim=256, n_heads=4)

    @pytest.fixture
    def conformer_block(self, config: AcousticModelConfig) -> ConformerBlock:
        """Create ConformerBlock instance."""
        return ConformerBlock(config)

    def test_initialization(self, conformer_block: ConformerBlock) -> None:
        """Test ConformerBlock initialization."""
        assert isinstance(conformer_block.ff1, nn.Sequential)
        assert isinstance(conformer_block.self_attn, nn.MultiheadAttention)
        assert isinstance(conformer_block.conv, nn.Sequential)
        assert isinstance(conformer_block.ff2, nn.Sequential)

    def test_forward_pass(
        self, conformer_block: ConformerBlock, config: AcousticModelConfig
    ) -> None:
        """Test forward pass through ConformerBlock."""
        batch_size = 2
        seq_len = 100
        hidden_dim = config.hidden_dim

        # Create input tensor
        x = torch.randn(batch_size, seq_len, hidden_dim)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        output = conformer_block(x, mask)

        assert output.shape == (batch_size, seq_len, hidden_dim)
        assert not torch.isnan(output).any()

    def test_gradient_flow(
        self, conformer_block: ConformerBlock, config: AcousticModelConfig
    ) -> None:
        """Test gradient flow through ConformerBlock."""
        x = torch.randn(1, 50, config.hidden_dim, requires_grad=True)
        mask = torch.ones(1, 50, dtype=torch.bool)

        output = conformer_block(x, mask)
        loss = output.mean()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    @pytest.mark.parametrize("seq_len", [10, 50, 200])
    def test_variable_sequence_length(
        self, conformer_block: ConformerBlock, config: AcousticModelConfig, seq_len: int
    ) -> None:
        """Test ConformerBlock with variable sequence lengths."""
        batch_size = 2
        x = torch.randn(batch_size, seq_len, config.hidden_dim)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        output = conformer_block(x, mask)
        assert output.shape == (batch_size, seq_len, config.hidden_dim)


class TestDurationPredictor:
    """Test suite for DurationPredictor."""

    @pytest.fixture
    def duration_predictor(self) -> nn.Module:
        """Create DurationPredictor instance."""
        # Import here to avoid circular imports
        from src.models.acoustic_model import DurationPredictor

        return DurationPredictor(hidden_dim=256)

    def test_forward_pass(self, duration_predictor: nn.Module) -> None:
        """Test forward pass of duration predictor."""
        batch_size = 2
        seq_len = 50
        hidden_dim = 256

        # Input should be (B, T, hidden_dim)
        x = torch.randn(batch_size, seq_len, hidden_dim)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # Forward expects (B, hidden_dim, T) so transpose
        x_transposed = x.transpose(1, 2)
        log_durations = duration_predictor(x_transposed, mask)

        assert log_durations.shape == (batch_size, seq_len)
        assert not torch.isnan(log_durations).any()


class TestAcousticModel:
    """Test suite for complete AcousticModel."""

    @pytest.fixture
    def config(self) -> AcousticModelConfig:
        """Create test configuration."""
        return AcousticModelConfig(
            hidden_dim=128, n_layers=2, n_heads=4, use_bf16=False
        )

    @pytest.fixture
    def acoustic_model(self, config: AcousticModelConfig) -> AcousticModel:
        """Create AcousticModel instance."""
        return AcousticModel(config)

    def test_initialization(
        self, acoustic_model: AcousticModel, config: AcousticModelConfig
    ) -> None:
        """Test acoustic model initialization."""
        assert len(acoustic_model.encoder_layers) == config.n_layers
        assert acoustic_model.config == config

    def test_forward_pass_training(self, acoustic_model: AcousticModel) -> None:
        """Test forward pass in training mode."""
        batch_size = 2
        seq_len = 100
        phoneme_dim = 768
        mel_len = 500
        n_mels = 80

        # Create inputs
        phoneme_features = torch.randn(batch_size, seq_len, phoneme_dim)
        phoneme_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        mel_targets = torch.randn(batch_size, mel_len, n_mels)
        mel_mask = torch.ones(batch_size, mel_len, dtype=torch.bool)

        outputs = acoustic_model(
            phoneme_features=phoneme_features,
            phoneme_mask=phoneme_mask,
            mel_targets=mel_targets,
            mel_mask=mel_mask,
        )

        assert "mel_outputs" in outputs
        assert "log_durations" in outputs
        assert "losses" in outputs
        assert outputs["mel_outputs"].shape == (batch_size, mel_len, n_mels)

    def test_forward_pass_inference(self, acoustic_model: AcousticModel) -> None:
        """Test forward pass in inference mode."""
        batch_size = 1
        seq_len = 50
        phoneme_dim = 768

        phoneme_features = torch.randn(batch_size, seq_len, phoneme_dim)
        phoneme_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        with torch.no_grad():
            outputs = acoustic_model.inference(
                phoneme_features=phoneme_features, phoneme_mask=phoneme_mask
            )

        assert "mel_outputs" in outputs
        assert "durations" in outputs
        assert outputs["mel_outputs"].ndim == 3
        assert outputs["mel_outputs"].shape[2] == 80

    def test_loss_computation(self, acoustic_model: AcousticModel) -> None:
        """Test loss computation."""
        batch_size = 2
        seq_len = 30
        phoneme_dim = 768
        mel_len = 150
        n_mels = 80

        phoneme_features = torch.randn(batch_size, seq_len, phoneme_dim)
        phoneme_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        mel_targets = torch.randn(batch_size, mel_len, n_mels)
        mel_mask = torch.ones(batch_size, mel_len, dtype=torch.bool)

        outputs = acoustic_model(
            phoneme_features=phoneme_features,
            phoneme_mask=phoneme_mask,
            mel_targets=mel_targets,
            mel_mask=mel_mask,
        )

        losses = outputs["losses"]
        assert "mel_loss" in losses
        assert "duration_loss" in losses
        assert "total_loss" in losses

        assert losses["total_loss"].requires_grad
        assert losses["total_loss"].item() > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_bf16_support(self, config: AcousticModelConfig) -> None:
        """Test BF16 support on CUDA."""
        config.use_bf16 = True
        model = AcousticModel(config).cuda()

        batch_size = 1
        seq_len = 20
        phoneme_dim = 768

        phoneme_features = torch.randn(
            batch_size, seq_len, phoneme_dim, device="cuda", dtype=torch.bfloat16
        )
        phoneme_mask = torch.ones(batch_size, seq_len, device="cuda", dtype=torch.bool)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model.inference(
                phoneme_features=phoneme_features, phoneme_mask=phoneme_mask
            )

        assert outputs["mel_outputs"].dtype == torch.bfloat16


@pytest.mark.integration
class TestAcousticModelIntegration:
    """Integration tests for acoustic model with other components."""

    def test_with_xphonebert_features(self) -> None:
        """Test acoustic model with XPhoneBERT features."""
        config = AcousticModelConfig(hidden_dim=256, n_layers=2)
        model = AcousticModel(config)

        # Simulate XPhoneBERT output
        batch_size = 2
        seq_len = 50
        phoneme_features = torch.randn(batch_size, seq_len, 768)  # XPhoneBERT dim
        phoneme_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        with torch.no_grad():
            outputs = model.inference(
                phoneme_features=phoneme_features, phoneme_mask=phoneme_mask
            )

        assert outputs["mel_outputs"].shape[0] == batch_size
        assert outputs["mel_outputs"].shape[2] == 80
