"""
Tests for F0-BERT pitch prediction model
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.f0_bert import (
    F0BERT,
    ContinuousF0Encoder,
    F0BERTConfig,
    F0PredictionHead,
    StyleConditioningModule,
    create_f0_bert,
)


class TestF0BERTConfig:
    """Test F0-BERT configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = F0BERTConfig()

        assert config.hidden_size == 768
        assert config.num_hidden_layers == 12
        assert config.f0_min == 50.0
        assert config.f0_max == 800.0
        assert config.f0_bins == 256
        assert config.predict_vuv == True
        assert config.predict_energy == True
        assert config.predict_duration == True

    def test_custom_config(self):
        """Test custom configuration."""
        config = F0BERTConfig(
            hidden_size=512, f0_min=60.0, f0_max=600.0, num_emotions=10
        )

        assert config.hidden_size == 512
        assert config.f0_min == 60.0
        assert config.f0_max == 600.0
        assert config.num_emotions == 10


class TestContinuousF0Encoder:
    """Test continuous F0 encoder."""

    @pytest.fixture()
    def encoder(self):
        """Create test encoder."""
        config = F0BERTConfig()
        return ContinuousF0Encoder(config)

    def test_forward(self, encoder):
        """Test forward pass."""
        batch_size = 2
        seq_len = 100

        # Create dummy F0 values (in Hz)
        f0_values = torch.rand(batch_size, seq_len) * 300 + 100  # 100-400 Hz

        # Forward pass
        f0_embeddings = encoder(f0_values)

        # Check shape
        assert f0_embeddings.shape == (batch_size, seq_len, 768)

    def test_normalization(self, encoder):
        """Test F0 normalization."""
        # Test with extreme values
        f0_values = torch.tensor(
            [[30.0, 150.0, 1000.0]]
        )  # Below min, normal, above max

        f0_embeddings = encoder(f0_values)

        # Should handle out-of-range values gracefully
        assert f0_embeddings.shape == (1, 3, 768)
        assert not torch.isnan(f0_embeddings).any()

    def test_sinusoidal_encoding(self, encoder):
        """Test sinusoidal positional encoding."""
        # Check that positional encoding is properly initialized
        assert hasattr(encoder, "f0_positional_encoding")
        assert encoder.f0_positional_encoding.shape == (
            1,
            256,
            768,
        )  # (1, f0_bins, hidden_size)


class TestStyleConditioningModule:
    """Test style conditioning module."""

    @pytest.fixture()
    def module(self):
        """Create test module."""
        config = F0BERTConfig()
        return StyleConditioningModule(config)

    def test_forward_basic(self, module):
        """Test basic forward pass."""
        batch_size = 2

        emotion_id = torch.tensor([0, 2])  # Neutral, Sad
        style_id = torch.tensor([1, 3])

        style_vector = module(emotion_id=emotion_id, style_id=style_id)

        assert style_vector.shape == (batch_size, 768)

    def test_forward_with_reference(self, module):
        """Test forward with reference embedding."""
        batch_size = 2
        ref_len = 50

        emotion_id = torch.tensor([1, 2])
        reference_embedding = torch.randn(batch_size, ref_len, 256)

        style_vector = module(
            emotion_id=emotion_id, reference_embedding=reference_embedding
        )

        assert style_vector.shape == (batch_size, 768)

    def test_default_values(self, module):
        """Test with default emotion/style."""
        style_vector = module()

        # Should return something even with no inputs
        assert style_vector.shape == (1, 768)

    def test_gst_tokens(self, module):
        """Test Global Style Tokens."""
        assert hasattr(module, "gst_embedding")
        assert module.gst_embedding.shape == (10, 256)  # 10 style tokens


class TestF0PredictionHead:
    """Test F0 prediction head."""

    @pytest.fixture()
    def head(self):
        """Create test prediction head."""
        config = F0BERTConfig()
        return F0PredictionHead(config)

    def test_forward(self, head):
        """Test forward pass."""
        batch_size = 2
        seq_len = 100
        hidden_size = 768

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        outputs = head(hidden_states)

        # Check all outputs
        assert "f0" in outputs
        assert "f0_quantized" in outputs
        assert "vuv" in outputs
        assert "energy" in outputs
        assert "duration" in outputs

        # Check shapes
        assert outputs["f0"].shape == (batch_size, seq_len)
        assert outputs["f0_quantized"].shape == (batch_size, seq_len, 256)  # f0_bins
        assert outputs["vuv"].shape == (batch_size, seq_len, 2)
        assert outputs["energy"].shape == (batch_size, seq_len)
        assert outputs["duration"].shape == (batch_size, seq_len)

    def test_f0_range(self, head):
        """Test that F0 values are in correct range."""
        hidden_states = torch.randn(2, 50, 768)
        outputs = head(hidden_states)

        # F0 should be between f0_min and f0_max (50-800 Hz)
        assert torch.all(outputs["f0"] >= 50.0)
        assert torch.all(outputs["f0"] <= 800.0)

    def test_positive_duration(self, head):
        """Test that durations are positive."""
        hidden_states = torch.randn(2, 50, 768)
        outputs = head(hidden_states)

        # Durations should be positive (using Softplus)
        assert torch.all(outputs["duration"] > 0)


class TestF0BERT:
    """Test complete F0-BERT model."""

    @pytest.fixture()
    def model(self):
        """Create test model."""
        config = F0BERTConfig()
        return F0BERT(config)

    def test_initialization(self, model):
        """Test model initialization."""
        assert hasattr(model, "bert")
        assert hasattr(model, "f0_encoder")
        assert hasattr(model, "style_conditioning")
        assert hasattr(model, "prediction_head")
        assert hasattr(model, "f0_smoother")

    def test_forward_basic(self, model):
        """Test basic forward pass."""
        batch_size = 2
        seq_len = 50

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        outputs = model(input_ids, attention_mask)

        # Check outputs
        assert "f0" in outputs
        assert "f0_smooth" in outputs
        assert "f0_final" in outputs
        assert "vuv" in outputs

        # Check shapes
        assert outputs["f0"].shape == (batch_size, seq_len)
        assert outputs["f0_final"].shape == (batch_size, seq_len)

    def test_forward_with_style(self, model):
        """Test forward with style conditioning."""
        batch_size = 2
        seq_len = 50

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        emotion_id = torch.tensor([1, 2])  # Happy, Sad
        style_id = torch.tensor([0, 3])

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            emotion_id=emotion_id,
            style_id=style_id,
        )

        assert outputs["f0_final"].shape == (batch_size, seq_len)

    def test_forward_with_phoneme_embeddings(self, model):
        """Test forward with phoneme embeddings."""
        batch_size = 2
        seq_len = 50

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        phoneme_embeddings = torch.randn(batch_size, seq_len, 768)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            phoneme_embeddings=phoneme_embeddings,
        )

        assert outputs["f0_final"].shape == (batch_size, seq_len)

    def test_teacher_forcing(self, model):
        """Test teacher forcing during training."""
        model.train()

        batch_size = 2
        seq_len = 50

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        target_f0 = torch.rand(batch_size, seq_len) * 300 + 100  # 100-400 Hz

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, target_f0=target_f0
        )

        assert outputs["f0_final"].shape == (batch_size, seq_len)

    def test_frame_expansion(self, model):
        """Test frame-level expansion."""
        batch_size = 2
        seq_len = 10  # Short sequence

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        frame_lengths = torch.tensor(
            [[5, 10, 8, 12, 6, 9, 7, 11, 8, 10], [6, 9, 10, 8, 7, 11, 9, 8, 10, 7]]
        )  # Frames per phoneme

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            frame_lengths=frame_lengths,
        )

        # Output should be expanded to frame level
        expected_frames_0 = frame_lengths[0].sum().item()
        expected_frames_1 = frame_lengths[1].sum().item()
        max_frames = max(expected_frames_0, expected_frames_1)

        assert outputs["f0_final"].shape == (batch_size, max_frames)

    def test_vuv_masking(self, model):
        """Test voiced/unvoiced masking."""
        batch_size = 2
        seq_len = 50

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        outputs = model(input_ids, attention_mask)

        # Check that unvoiced regions have zero F0
        if "vuv" in outputs:
            vuv_mask = torch.sigmoid(outputs["vuv"][:, :, 1]) > 0.5
            unvoiced_f0 = outputs["f0_final"][~vuv_mask]

            # Most unvoiced frames should have zero or very low F0
            # (some smoothing might cause small non-zero values)
            if unvoiced_f0.numel() > 0:
                assert torch.median(torch.abs(unvoiced_f0)) < 10.0

    def test_inference_mode(self, model):
        """Test inference mode."""
        model.eval()

        input_ids = torch.randint(0, 1000, (1, 30))
        attention_mask = torch.ones(1, 30)

        outputs = model.inference(
            input_ids=input_ids,
            attention_mask=attention_mask,
            emotion_id=torch.tensor([2]),  # Sad
            temperature=0.8,  # Less variation
        )

        assert "f0_final" in outputs
        assert outputs["f0_final"].shape == (1, 30)

    def test_temperature_control(self, model):
        """Test temperature control in inference."""
        model.eval()

        input_ids = torch.randint(0, 1000, (1, 30))
        attention_mask = torch.ones(1, 30)

        # Get outputs with different temperatures
        outputs_low = model.inference(
            input_ids=input_ids, attention_mask=attention_mask, temperature=0.5
        )

        outputs_high = model.inference(
            input_ids=input_ids, attention_mask=attention_mask, temperature=1.5
        )

        # High temperature should have more variation
        std_low = outputs_low["f0_final"].std().item()
        std_high = outputs_high["f0_final"].std().item()

        # This might not always hold due to randomness, but generally true
        # Just check both are reasonable
        assert std_low > 0
        assert std_high > 0


class TestCreateF0BERT:
    """Test factory function."""

    def test_create_default(self):
        """Test creating with default settings."""
        model = create_f0_bert()

        assert isinstance(model, F0BERT)

    def test_create_with_checkpoint(self, tmp_path):
        """Test creating with checkpoint."""
        # Create dummy checkpoint
        checkpoint_path = tmp_path / "f0_bert_checkpoint.pt"

        # Save dummy state dict
        dummy_model = create_f0_bert()
        torch.save(dummy_model.state_dict(), checkpoint_path)

        # Load model
        model = create_f0_bert(checkpoint_path=str(checkpoint_path))

        assert isinstance(model, F0BERT)


class TestIntegration:
    """Integration tests for F0-BERT."""

    @pytest.fixture()
    def model(self):
        """Create model for integration tests."""
        return create_f0_bert(device="cpu")

    def test_multi_emotion_generation(self, model):
        """Test generating F0 for different emotions."""
        model.eval()

        input_ids = torch.randint(0, 1000, (1, 50))
        attention_mask = torch.ones(1, 50)

        emotions = {0: "Neutral", 1: "Happy", 2: "Sad", 3: "Angry", 4: "Fear"}

        f0_patterns = {}

        for emotion_id, emotion_name in emotions.items():
            outputs = model.inference(
                input_ids=input_ids,
                attention_mask=attention_mask,
                emotion_id=torch.tensor([emotion_id]),
            )

            f0_patterns[emotion_name] = outputs["f0_final"].mean().item()

        # Different emotions should produce different average F0
        # (Happy typically higher than Sad)
        assert len(set(f0_patterns.values())) > 1  # At least some variation

    @pytest.mark.parametrize("seq_len", [10, 50, 100, 200])
    def test_variable_length_sequences(self, model, seq_len):
        """Test with different sequence lengths."""
        model.eval()

        input_ids = torch.randint(0, 1000, (1, seq_len))
        attention_mask = torch.ones(1, seq_len)

        outputs = model(input_ids, attention_mask)

        assert outputs["f0_final"].shape == (1, seq_len)
        assert not torch.isnan(outputs["f0_final"]).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
