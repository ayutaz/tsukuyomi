"""
Tests for XPhoneBERT Japanese optimization
"""

import pytest
import torch

from src.models.xphonebert_japanese import (
    JapanesePhonemeAdapter,
    LoRALayer,
    XPhoneBERTJapanese,
    XPhoneBERTJapaneseConfig,
    create_xphonebert_japanese,
)


class TestXPhoneBERTJapaneseConfig:
    """Test configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = XPhoneBERTJapaneseConfig()

        assert config.base_model == "vinai/xphonebert-base"
        assert config.hidden_size == 768
        assert config.num_accent_types == 4
        assert config.num_dialects == 47
        assert config.use_lora == True
        assert config.lora_rank == 64
        assert config.lora_target_modules == ["query", "value"]

    def test_custom_config(self):
        """Test custom configuration."""
        config = XPhoneBERTJapaneseConfig(
            hidden_size=512, lora_rank=32, num_dialects=10, use_lora=False
        )

        assert config.hidden_size == 512
        assert config.lora_rank == 32
        assert config.num_dialects == 10
        assert config.use_lora == False


class TestLoRALayer:
    """Test LoRA layer implementation."""

    def test_initialization(self):
        """Test LoRA layer initialization."""
        lora = LoRALayer(in_features=768, out_features=768, rank=64, alpha=128)

        assert lora.rank == 64
        assert lora.alpha == 128
        assert lora.scaling == 2.0  # alpha / rank
        assert lora.lora_A.shape == (768, 64)
        assert lora.lora_B.shape == (64, 768)

    def test_forward(self):
        """Test LoRA forward pass."""
        batch_size = 2
        seq_len = 10
        hidden_size = 768

        lora = LoRALayer(hidden_size, hidden_size, rank=32)

        # Set dummy weight
        lora.weight = torch.eye(hidden_size)

        # Input
        x = torch.randn(batch_size, seq_len, hidden_size)

        # Forward
        output = lora(x)

        # Check shape
        assert output.shape == x.shape

    def test_low_rank_adaptation(self):
        """Test that LoRA actually adds low-rank updates."""
        hidden_size = 256
        rank = 16

        lora = LoRALayer(hidden_size, hidden_size, rank=rank)

        # Set identity weight
        lora.weight = torch.eye(hidden_size)

        # Set non-zero LoRA weights
        lora.lora_A.data = torch.randn_like(lora.lora_A) * 0.1
        lora.lora_B.data = torch.randn_like(lora.lora_B) * 0.1

        # Test input
        x = torch.randn(1, hidden_size)

        # Forward
        output = lora(x)

        # Output should be different from input (due to LoRA)
        assert not torch.allclose(output, x, rtol=1e-5)


class TestJapanesePhonemeAdapter:
    """Test Japanese phoneme adapter."""

    @pytest.fixture()
    def adapter(self):
        """Create test adapter."""
        config = XPhoneBERTJapaneseConfig()
        return JapanesePhonemeAdapter(config)

    def test_initialization(self, adapter):
        """Test adapter initialization."""
        assert hasattr(adapter, "accent_embedding")
        assert hasattr(adapter, "dialect_embedding")
        assert hasattr(adapter, "pitch_encoder")
        assert hasattr(adapter, "phoneme_transform")

        # Check embedding sizes
        assert adapter.accent_embedding.num_embeddings == 4
        assert adapter.dialect_embedding.num_embeddings == 47

    def test_forward_basic(self, adapter):
        """Test basic forward pass."""
        batch_size = 2
        seq_len = 20
        hidden_size = 768

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        output = adapter(hidden_states)

        # Check shape preserved
        assert output.shape == hidden_states.shape

    def test_forward_with_accent(self, adapter):
        """Test forward with accent information."""
        batch_size = 2
        seq_len = 20
        hidden_size = 768

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        accent_ids = torch.randint(0, 4, (batch_size, seq_len))

        output = adapter(hidden_states, accent_ids=accent_ids)

        assert output.shape == hidden_states.shape

    def test_forward_with_dialect(self, adapter):
        """Test forward with dialect information."""
        batch_size = 2
        seq_len = 20
        hidden_size = 768

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        dialect_ids = torch.tensor([0, 27])  # Tokyo, Osaka

        output = adapter(hidden_states, dialect_ids=dialect_ids)

        assert output.shape == hidden_states.shape

    def test_forward_with_phoneme_types(self, adapter):
        """Test forward with phoneme type information."""
        batch_size = 2
        seq_len = 20
        hidden_size = 768

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        phoneme_types = torch.randint(0, 4, (batch_size, seq_len))

        output = adapter(hidden_states, phoneme_types=phoneme_types)

        assert output.shape == hidden_states.shape

    def test_forward_complete(self, adapter):
        """Test forward with all features."""
        batch_size = 2
        seq_len = 20
        hidden_size = 768
        pitch_dim = 256

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        accent_ids = torch.randint(0, 4, (batch_size, seq_len))
        dialect_ids = torch.tensor([0, 1])
        pitch_features = torch.randn(batch_size, seq_len, pitch_dim)
        phoneme_types = torch.randint(0, 4, (batch_size, seq_len))

        output = adapter(
            hidden_states,
            accent_ids=accent_ids,
            dialect_ids=dialect_ids,
            pitch_features=pitch_features,
            phoneme_types=phoneme_types,
        )

        assert output.shape == hidden_states.shape


class TestXPhoneBERTJapanese:
    """Test complete XPhoneBERT Japanese model."""

    @pytest.fixture()
    def model(self):
        """Create test model."""
        config = XPhoneBERTJapaneseConfig()
        return XPhoneBERTJapanese(config)

    def test_initialization(self, model):
        """Test model initialization."""
        assert hasattr(model, "base_model")
        assert hasattr(model, "japanese_adapter")
        assert hasattr(model, "output_projection")
        assert hasattr(model, "japanese_phoneme_embeddings")
        assert hasattr(model, "mora_boundary_predictor")

        if model.config.use_lora:
            assert hasattr(model, "lora_layers")

    def test_forward_with_ids(self, model):
        """Test forward pass with phoneme IDs."""
        batch_size = 2
        seq_len = 30

        phoneme_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        outputs = model(
            phoneme_ids=phoneme_ids, attention_mask=attention_mask, return_dict=True
        )

        assert "hidden_states" in outputs
        assert outputs["hidden_states"].shape == (batch_size, seq_len, 768)

    def test_forward_with_strings(self, model):
        """Test forward pass with phoneme strings."""
        # This test might fail if tokenizer not available
        if model.tokenizer is None:
            pytest.skip("Tokenizer not available")

        phoneme_strings = ["k o N n i ch i w a", "a r i g a t o o"]

        outputs = model(phoneme_strings=phoneme_strings, return_dict=True)

        assert "hidden_states" in outputs

    def test_forward_with_japanese_features(self, model):
        """Test forward with Japanese-specific features."""
        batch_size = 2
        seq_len = 30

        phoneme_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        accent_ids = torch.randint(0, 4, (batch_size, seq_len))
        dialect_ids = torch.tensor([0, 1])  # Tokyo, Hokkaido

        outputs = model(
            phoneme_ids=phoneme_ids,
            attention_mask=attention_mask,
            accent_ids=accent_ids,
            dialect_ids=dialect_ids,
            return_dict=True,
        )

        assert outputs["hidden_states"].shape == (batch_size, seq_len, 768)

    def test_mora_boundary_prediction(self, model):
        """Test mora boundary prediction."""
        batch_size = 2
        seq_len = 30

        phoneme_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        outputs = model(
            phoneme_ids=phoneme_ids,
            attention_mask=attention_mask,
            return_mora_boundaries=True,
            return_dict=True,
        )

        assert "mora_boundaries" in outputs
        assert outputs["mora_boundaries"].shape == (batch_size, seq_len, 2)

    def test_encode_japanese_phonemes(self, model):
        """Test convenience encoding method."""
        phonemes = "k o N n i ch i w a"

        embeddings = model.encode_japanese_phonemes(phonemes)

        assert embeddings.dim() == 3  # (batch, seq, hidden)
        assert embeddings.shape[0] == 1  # Batch size

    def test_encode_with_accent_pattern(self, model):
        """Test encoding with accent pattern."""
        phonemes = "k o N n i ch i w a"
        accent_pattern = "LHHHLLLLL"  # Low-High pattern

        embeddings = model.encode_japanese_phonemes(
            phonemes, accent_pattern=accent_pattern
        )

        assert embeddings.shape[0] == 1

    def test_encode_with_dialect(self, model):
        """Test encoding with dialect."""
        phonemes = "k o N n i ch i w a"

        # Test different dialects
        for dialect in ["tokyo", "osaka", "kyoto", "tohoku"]:
            embeddings = model.encode_japanese_phonemes(phonemes, dialect=dialect)
            assert embeddings.shape[0] == 1

    def test_save_and_load(self, model, tmp_path):
        """Test saving and loading model."""
        save_path = tmp_path / "xphonebert_japanese"

        # Save model
        model.save_pretrained(save_path)

        # Check files exist
        assert (save_path / "config.json").exists()
        if model.config.use_lora:
            assert (save_path / "lora_weights.pt").exists()
        assert (save_path / "adapter_weights.pt").exists()

        # Load model
        loaded_model = XPhoneBERTJapanese.from_pretrained(save_path)

        assert isinstance(loaded_model, XPhoneBERTJapanese)
        assert loaded_model.config.hidden_size == model.config.hidden_size


class TestCreateXPhoneBERTJapanese:
    """Test factory function."""

    def test_create_default(self):
        """Test creating with default settings."""
        model = create_xphonebert_japanese()

        assert isinstance(model, XPhoneBERTJapanese)

    def test_create_with_checkpoint(self, tmp_path):
        """Test creating with checkpoint."""
        # Create and save dummy model
        dummy_model = create_xphonebert_japanese()
        save_path = tmp_path / "checkpoint"
        dummy_model.save_pretrained(save_path)

        # Load model
        model = create_xphonebert_japanese(checkpoint_path=save_path)

        assert isinstance(model, XPhoneBERTJapanese)


class TestIntegration:
    """Integration tests."""

    @pytest.fixture()
    def model(self):
        """Create model for integration tests."""
        return create_xphonebert_japanese(device="cpu")

    def test_japanese_phoneme_sequences(self, model):
        """Test various Japanese phoneme sequences."""
        test_sequences = [
            "k o N n i ch i w a",  # こんにちは
            "a r i g a t o o g o z a i m a s u",  # ありがとうございます
            "t s u k u y o m i",  # 月読
            "o h a y o o",  # おはよう
            "s a y o o n a r a",  # さようなら
        ]

        for seq in test_sequences:
            embeddings = model.encode_japanese_phonemes(seq)

            # Basic checks
            assert embeddings.shape[0] == 1
            assert embeddings.shape[2] == 768
            assert not torch.isnan(embeddings).any()

    def test_special_phonemes(self, model):
        """Test special Japanese phonemes."""
        # Test geminate consonant (っ)
        embeddings1 = model.encode_japanese_phonemes("k i q t e")  # きって
        assert embeddings1.shape[0] == 1

        # Test long vowel
        embeddings2 = model.encode_japanese_phonemes("t o o ky o o")  # とうきょう
        assert embeddings2.shape[0] == 1

        # Test nasal ん
        embeddings3 = model.encode_japanese_phonemes("h o N")  # ほん
        assert embeddings3.shape[0] == 1

    @pytest.mark.parametrize(
        "dialect,expected_shape",
        [
            ("tokyo", (1, 9, 768)),
            ("osaka", (1, 9, 768)),
            ("kyoto", (1, 9, 768)),
            ("unknown", (1, 9, 768)),  # Should default to 0
        ],
    )
    def test_dialect_variations(self, model, dialect, expected_shape):
        """Test dialect variations."""
        phonemes = "k o N n i ch i w a"

        embeddings = model.encode_japanese_phonemes(phonemes, dialect=dialect)

        assert embeddings.shape == expected_shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
