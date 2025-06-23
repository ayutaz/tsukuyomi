"""
Unit tests for XPhoneBERT module.

Tests the XPhoneBERT wrapper for multilingual phoneme embeddings,
including forward pass, feature extraction, and error handling.
"""

from typing import List, Tuple

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from src.models.xphonebert import XPhoneBERTEncoder


class TestXPhoneBERTEncoder:
    """Test suite for XPhoneBERTEncoder."""
    
    @pytest.fixture
    def mock_model_and_tokenizer(self) -> Tuple[MagicMock, MagicMock]:
        """Create mock model and tokenizer for testing."""
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            'input_ids': torch.randint(0, 1000, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }
        mock_tokenizer.return_value.to = lambda device: mock_tokenizer.return_value
        
        # Mock model
        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.randn(2, 10, 768)
        mock_model.return_value = mock_outputs
        mock_model.to = lambda *args, **kwargs: mock_model
        mock_model.parameters = lambda: []
        mock_model.eval = lambda: None
        
        return mock_model, mock_tokenizer
    
    @pytest.fixture
    def encoder(self, mock_model_and_tokenizer: Tuple[MagicMock, MagicMock]) -> XPhoneBERTEncoder:
        """Create XPhoneBERTEncoder instance with mocked dependencies."""
        mock_model, mock_tokenizer = mock_model_and_tokenizer
        
        with patch('src.models.xphonebert.AutoModel') as mock_auto_model:
            with patch('src.models.xphonebert.AutoTokenizer') as mock_auto_tokenizer:
                mock_auto_model.from_pretrained.return_value = mock_model
                mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
                
                encoder = XPhoneBERTEncoder(
                    model_name="vinai/xphonebert-base",
                    device="cpu",
                    use_bf16=False
                )
                
        return encoder
    
    def test_initialization(self, encoder: XPhoneBERTEncoder) -> None:
        """Test encoder initialization."""
        assert encoder.device == "cpu"
        assert not encoder.use_bf16
        assert encoder.max_length == 512
        assert encoder.get_embedding_dim() == 768
    
    def test_forward_single_sequence(self, encoder: XPhoneBERTEncoder) -> None:
        """Test forward pass with single phoneme sequence."""
        phoneme_seq = "h ɛ l oʊ w ɜ˞ l d"
        output = encoder.forward(phoneme_seq, return_pooled=True)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 768)
    
    def test_forward_batch_sequences(self, encoder: XPhoneBERTEncoder) -> None:
        """Test forward pass with batch of phoneme sequences."""
        phoneme_seqs = ["h ɛ l oʊ", "w ɜ˞ l d"]
        output = encoder.forward(phoneme_seqs, return_pooled=True)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape == (2, 768)
    
    def test_forward_unpooled_output(self, encoder: XPhoneBERTEncoder) -> None:
        """Test forward pass returning all token embeddings."""
        phoneme_seq = "h ɛ l oʊ w ɜ˞ l d"
        embeddings, attention_mask = encoder.forward(phoneme_seq, return_pooled=False)
        
        assert isinstance(embeddings, torch.Tensor)
        assert isinstance(attention_mask, torch.Tensor)
        assert embeddings.shape[0] == 1
        assert embeddings.shape[2] == 768
        assert attention_mask.shape == (1, embeddings.shape[1])
    
    def test_extract_frame_level_features_no_duration(self, encoder: XPhoneBERTEncoder) -> None:
        """Test frame-level feature extraction without durations."""
        phoneme_seq = "h ɛ l oʊ"
        features = encoder.extract_frame_level_features(phoneme_seq)
        
        assert isinstance(features, torch.Tensor)
        assert features.ndim == 3
        assert features.shape[2] == 768
    
    def test_extract_frame_level_features_with_duration(self, encoder: XPhoneBERTEncoder) -> None:
        """Test frame-level feature extraction with durations."""
        phoneme_seqs = ["h ɛ l oʊ", "w ɜ˞ l d"]
        # Mock durations for each phoneme
        phoneme_durations = [[5, 10, 8, 12], [7, 9, 11, 8]]
        
        features = encoder.extract_frame_level_features(phoneme_seqs, phoneme_durations)
        
        assert isinstance(features, torch.Tensor)
        assert features.shape[0] == 2
        assert features.shape[2] == 768
        # Total frames should be max of sum of durations
        expected_frames = max(sum(phoneme_durations[0]), sum(phoneme_durations[1]))
        assert features.shape[1] >= expected_frames
    
    @pytest.mark.parametrize("device,use_bf16", [
        ("cpu", False),
        ("cpu", True),  # Should be False even if requested on CPU
    ])
    def test_device_and_precision_settings(
        self, 
        mock_model_and_tokenizer: Tuple[MagicMock, MagicMock],
        device: str, 
        use_bf16: bool
    ) -> None:
        """Test different device and precision settings."""
        mock_model, mock_tokenizer = mock_model_and_tokenizer
        
        with patch('src.models.xphonebert.AutoModel') as mock_auto_model:
            with patch('src.models.xphonebert.AutoTokenizer') as mock_auto_tokenizer:
                mock_auto_model.from_pretrained.return_value = mock_model
                mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
                
                encoder = XPhoneBERTEncoder(
                    device=device,
                    use_bf16=use_bf16
                )
                
                assert encoder.device == device
                if device == "cpu":
                    assert not encoder.use_bf16
    
    def test_empty_sequence_handling(self, encoder: XPhoneBERTEncoder) -> None:
        """Test handling of empty sequences."""
        with pytest.raises(Exception):
            encoder.forward("")
    
    def test_long_sequence_truncation(self, encoder: XPhoneBERTEncoder) -> None:
        """Test that long sequences are properly truncated."""
        # Create a very long phoneme sequence
        long_seq = " ".join(["p h oʊ n iː m"] * 200)
        
        output = encoder.forward(long_seq)
        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 768)


class TestXPhoneBERTIntegration:
    """Integration tests for XPhoneBERT with other components."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_support(self) -> None:
        """Test CUDA support if available."""
        with patch('src.models.xphonebert.AutoModel') as mock_model:
            with patch('src.models.xphonebert.AutoTokenizer') as mock_tokenizer:
                # Setup mocks
                mock_model.from_pretrained.return_value = MagicMock()
                mock_tokenizer.from_pretrained.return_value = MagicMock()
                
                encoder = XPhoneBERTEncoder(device="cuda", use_bf16=True)
                assert encoder.device == "cuda"
                assert encoder.use_bf16
    
    def test_gradient_computation_disabled(self, encoder: XPhoneBERTEncoder) -> None:
        """Test that gradients are not computed during forward pass."""
        phoneme_seq = "t ɛ s t"
        
        # Check that no gradients are computed
        output = encoder.forward(phoneme_seq)
        assert not output.requires_grad
    
    def test_deterministic_output(self, encoder: XPhoneBERTEncoder) -> None:
        """Test that outputs are deterministic for same input."""
        phoneme_seq = "d ɪ t ɜ˞ m ɪ n ɪ s t ɪ k"
        
        output1 = encoder.forward(phoneme_seq)
        output2 = encoder.forward(phoneme_seq)
        
        assert torch.allclose(output1, output2)


@pytest.mark.benchmark
class TestXPhoneBERTPerformance:
    """Performance benchmarks for XPhoneBERT."""
    
    def test_inference_speed(self, encoder: XPhoneBERTEncoder, benchmark) -> None:
        """Benchmark inference speed."""
        phoneme_seqs = ["h ɛ l oʊ w ɜ˞ l d"] * 10
        
        def run_inference():
            return encoder.forward(phoneme_seqs)
        
        result = benchmark(run_inference)
        assert result.shape == (10, 768)
    
    def test_memory_usage(self, encoder: XPhoneBERTEncoder) -> None:
        """Test memory usage stays within bounds."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process multiple sequences
        for _ in range(100):
            phoneme_seq = "m ɛ m ə ɹ i t ɛ s t"
            _ = encoder.forward(phoneme_seq)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100