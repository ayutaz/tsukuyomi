"""
Tests for Ultimate Japanese G2P with 97%+ accuracy target
"""


import pytest
import torch

from src.models.ultimate_g2p import (
    AccentBERT,
    ContextualPhonemeEncoder,
    NeuralPhonemeCorrectorRNN,
    UltimateG2PConfig,
    UltimateJapaneseG2P,
    create_ultimate_g2p,
)


class TestUltimateG2PConfig:
    """Test configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = UltimateG2PConfig()

        assert config.bert_model == "tohoku-nlp/bert-large-japanese-v3"
        assert config.hidden_size == 1024
        assert config.num_accent_types == 4
        assert config.use_lora == True
        assert config.lora_rank == 64

    def test_custom_config(self):
        """Test custom configuration."""
        config = UltimateG2PConfig(hidden_size=768, lora_rank=32, dropout_rate=0.2)

        assert config.hidden_size == 768
        assert config.lora_rank == 32
        assert config.dropout_rate == 0.2


class TestAccentBERT:
    """Test AccentBERT module."""

    @pytest.fixture()
    def model(self):
        """Create test model."""
        config = UltimateG2PConfig()
        return AccentBERT(config)

    def test_forward(self, model):
        """Test forward pass."""
        batch_size = 2
        seq_len = 10

        # Create dummy inputs
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        # Forward pass
        accent_logits, accent_positions = model(input_ids, attention_mask)

        # Check shapes
        assert accent_logits.shape == (batch_size, seq_len, 4)  # 4 accent types
        assert accent_positions.shape == (batch_size, seq_len, 1)

    def test_with_word_boundaries(self, model):
        """Test with word boundary information."""
        batch_size = 2
        seq_len = 10

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        word_boundaries = torch.randint(0, 2, (batch_size, seq_len))

        accent_logits, accent_positions = model(
            input_ids, attention_mask, word_boundaries
        )

        assert accent_logits.shape == (batch_size, seq_len, 4)
        assert accent_positions.shape == (batch_size, seq_len, 1)


class TestNeuralPhonemeCorrectorRNN:
    """Test neural phoneme corrector."""

    @pytest.fixture()
    def model(self):
        """Create test model."""
        config = UltimateG2PConfig()
        return NeuralPhonemeCorrectorRNN(config)

    def test_forward(self, model):
        """Test forward pass."""
        batch_size = 2
        seq_len = 20
        vocab_size = 150

        # Create dummy phoneme IDs
        phoneme_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        phoneme_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # Forward pass
        corrected_logits = model(phoneme_ids, phoneme_mask)

        # Check shape
        assert corrected_logits.shape == (batch_size, seq_len, vocab_size)

    def test_lstm_bidirectional(self, model):
        """Test that LSTM is properly bidirectional."""
        # Check LSTM configuration
        assert model.lstm.bidirectional == True
        assert model.lstm.num_layers == 3


class TestContextualPhonemeEncoder:
    """Test contextual phoneme encoder."""

    @pytest.fixture()
    def model(self):
        """Create test model."""
        config = UltimateG2PConfig()
        return ContextualPhonemeEncoder(config)

    def test_forward(self, model):
        """Test forward pass."""
        batch_size = 2
        seq_len = 15
        hidden_size = 1024

        # Create dummy features
        phoneme_features = torch.randn(batch_size, seq_len, hidden_size)

        # Forward pass
        context_features = model(phoneme_features)

        # Check shape preserved
        assert context_features.shape == phoneme_features.shape

    def test_with_context_mask(self, model):
        """Test with context mask."""
        batch_size = 2
        seq_len = 15
        hidden_size = 1024

        phoneme_features = torch.randn(batch_size, seq_len, hidden_size)
        context_mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.bool)

        context_features = model(phoneme_features, context_mask)

        assert context_features.shape == phoneme_features.shape


class TestUltimateJapaneseG2P:
    """Test complete Ultimate G2P system."""

    @pytest.fixture()
    def model(self):
        """Create test model."""
        config = UltimateG2PConfig()
        # Use smaller model for testing
        config.bert_model = "tohoku-nlp/bert-base-japanese"
        config.hidden_size = 768
        return UltimateJapaneseG2P(config)

    def test_initialization(self, model):
        """Test model initialization."""
        assert model.config.hidden_size == 768
        assert hasattr(model, "accent_predictor")
        assert hasattr(model, "phoneme_corrector")
        assert hasattr(model, "context_encoder")
        assert len(model.phoneme_vocab) > 50  # Should have reasonable vocab

    def test_phoneme_vocab(self, model):
        """Test phoneme vocabulary construction."""
        vocab = model.phoneme_vocab

        # Check essential phonemes
        assert "a" in vocab
        assert "ka" in vocab
        assert "N" in vocab  # ん
        assert "sil" in vocab  # silence
        assert "<PAD>" in vocab
        assert "<UNK>" in vocab

    def test_forward_basic(self, model):
        """Test basic forward pass."""
        text = "こんにちは"

        # Forward pass
        outputs = model.forward(text, return_accent=False, return_confidence=False)

        # Check outputs
        assert "phonemes" in outputs
        assert "phoneme_logits" in outputs
        assert outputs["phonemes"].dim() == 2  # (batch, seq)

    def test_forward_with_accent(self, model):
        """Test forward with accent prediction."""
        text = "今日はいい天気ですね"

        outputs = model.forward(text, return_accent=True, return_confidence=False)

        assert "phonemes" in outputs
        assert "accent_types" in outputs
        assert "accent_positions" in outputs

    def test_forward_with_confidence(self, model):
        """Test forward with confidence scores."""
        text = "月読の音声合成システム"

        outputs = model.forward(text, return_accent=True, return_confidence=True)

        assert "confidence" in outputs
        assert outputs["confidence"].shape == outputs["phonemes"].shape
        assert torch.all(outputs["confidence"] >= 0)
        assert torch.all(outputs["confidence"] <= 1)

    def test_phonemes_to_ids(self, model):
        """Test phoneme to ID conversion."""
        phonemes = "k o N n i ch i w a"
        ids = model._phonemes_to_ids(phonemes)

        assert isinstance(ids, list)
        assert len(ids) == len(phonemes.split())
        assert all(isinstance(i, int) for i in ids)

    def test_ids_to_logits(self, model):
        """Test ID to logits conversion."""
        ids = torch.tensor([[4, 5, 6, 7]])  # Dummy IDs
        logits = model._ids_to_logits(ids)

        assert logits.shape == (1, 4, model.config.phoneme_vocab_size)
        # Check it's one-hot like (scaled)
        assert torch.allclose(logits.max(dim=-1)[0], torch.tensor(10.0))


class TestCreateUltimateG2P:
    """Test factory function."""

    def test_create_default(self):
        """Test creating with default settings."""
        model = create_ultimate_g2p()

        assert isinstance(model, UltimateJapaneseG2P)
        assert model.training == False  # Should be in eval mode

    def test_create_with_checkpoint(self, tmp_path):
        """Test creating with checkpoint."""
        # Create dummy checkpoint
        checkpoint_path = tmp_path / "g2p_checkpoint.pt"

        # Save dummy state dict
        dummy_model = create_ultimate_g2p()
        torch.save(dummy_model.state_dict(), checkpoint_path)

        # Load model
        model = create_ultimate_g2p(checkpoint_path=checkpoint_path)

        assert isinstance(model, UltimateJapaneseG2P)


class TestIntegration:
    """Integration tests for complete pipeline."""

    @pytest.fixture()
    def model(self):
        """Create model for integration tests."""
        return create_ultimate_g2p(device="cpu")

    def test_japanese_sentences(self, model):
        """Test various Japanese sentences."""
        test_cases = [
            "こんにちは",
            "ありがとうございます",
            "今日はいい天気ですね",
            "月読の音声合成システムです",
            "東京タワーに行きました",
            "AIの研究をしています",
        ]

        for text in test_cases:
            outputs = model.forward(text, return_accent=True)

            # Basic checks
            assert outputs["phonemes"].shape[0] == 1  # Batch size
            assert outputs["phonemes"].shape[1] > 0  # Has phonemes
            assert outputs["accent_types"].shape == outputs["phonemes"].shape

    def test_special_cases(self, model):
        """Test special Japanese cases."""
        # Test with numbers
        outputs1 = model.forward("123個のりんご", return_accent=True)
        assert outputs1["phonemes"].shape[1] > 0

        # Test with English mixed
        outputs2 = model.forward("AIとMLの違い", return_accent=True)
        assert outputs2["phonemes"].shape[1] > 0

        # Test with punctuation
        outputs3 = model.forward("え？本当に！", return_accent=True)
        assert outputs3["phonemes"].shape[1] > 0

    @pytest.mark.parametrize(
        "text,min_confidence",
        [
            ("こんにちは", 0.8),  # Simple greeting - high confidence
            ("月読", 0.7),  # Proper noun - medium confidence
            ("最新のAI技術", 0.6),  # Mixed content - lower confidence
        ],
    )
    def test_confidence_levels(self, model, text, min_confidence):
        """Test confidence scores for different text types."""
        outputs = model.forward(text, return_confidence=True)

        avg_confidence = outputs["confidence"].mean().item()
        assert avg_confidence >= min_confidence


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
