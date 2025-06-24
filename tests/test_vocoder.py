"""
Tests for vocoder modules
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.vocoder.bigvgan import BigVGANVocoder


class TestBigVGANVocoder:
    """Test BigVGAN vocoder wrapper"""

    @pytest.fixture
    def mock_bigvgan(self):
        """Mock bigvgan module"""
        with patch("src.vocoder.bigvgan.bigvgan") as mock_bg:
            # Mock model instance
            mock_model = MagicMock()
            mock_model.eval.return_value = mock_model
            mock_model.to.return_value = mock_model
            mock_model.remove_weight_norm.return_value = None

            # Mock model loading
            mock_bg.BigVGAN.from_pretrained.return_value = mock_model

            yield {
                "module": mock_bg,
                "model": mock_model,
            }

    def test_initialization(self, mock_bigvgan):
        """Test vocoder initialization"""
        vocoder = BigVGANVocoder(device="cpu")

        assert vocoder.device == torch.device("cpu")
        assert vocoder.model is not None
        mock_bigvgan["module"].BigVGAN.from_pretrained.assert_called_once()
        mock_bigvgan["model"].remove_weight_norm.assert_called_once()

    def test_inference(self, mock_bigvgan):
        """Test vocoder inference"""
        vocoder = BigVGANVocoder(device="cpu")

        # Mock model output
        expected_waveform = torch.randn(1, 22050)  # 1 second at 22.05kHz
        mock_bigvgan["model"].return_value = expected_waveform

        # Test inference
        mel = torch.randn(1, 80, 100)  # 80 mel bins, 100 frames
        waveform = vocoder.inference(mel)

        assert isinstance(waveform, np.ndarray)
        assert waveform.shape == (22050,)
        assert not np.isnan(waveform).any()

    def test_batch_inference(self, mock_bigvgan):
        """Test batch inference"""
        vocoder = BigVGANVocoder(device="cpu")

        # Mock batch output
        batch_size = 3
        expected_waveforms = torch.randn(batch_size, 22050)
        mock_bigvgan["model"].return_value = expected_waveforms

        # Test batch inference
        mel_batch = torch.randn(batch_size, 80, 100)
        waveforms = vocoder.batch_inference(mel_batch)

        assert len(waveforms) == batch_size
        for waveform in waveforms:
            assert isinstance(waveform, np.ndarray)
            assert not np.isnan(waveform).any()

    def test_different_mel_sizes(self, mock_bigvgan):
        """Test with different mel spectrogram sizes"""
        vocoder = BigVGANVocoder(device="cpu")

        mel_sizes = [
            (1, 80, 50),  # Short
            (1, 80, 200),  # Medium
            (1, 80, 500),  # Long
        ]

        for mel_size in mel_sizes:
            mel = torch.randn(*mel_size)
            # Mock appropriate output size
            output_samples = mel_size[2] * 256  # hop_size = 256
            mock_bigvgan["model"].return_value = torch.randn(1, output_samples)

            waveform = vocoder.inference(mel)
            assert isinstance(waveform, np.ndarray)
            assert len(waveform) == output_samples

    @pytest.mark.gpu
    def test_gpu_inference(self, mock_bigvgan):
        """Test GPU inference"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        vocoder = BigVGANVocoder(device="cuda")
        assert vocoder.device.type == "cuda"

        # GPU tensors
        mel = torch.randn(1, 80, 100).cuda()
        mock_bigvgan["model"].return_value = torch.randn(1, 22050).cuda()

        waveform = vocoder.inference(mel)
        assert isinstance(waveform, np.ndarray)
        assert waveform.dtype == np.float32

    @pytest.mark.bf16
    def test_bf16_inference(self, mock_bigvgan):
        """Test BF16 inference"""
        if not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
            pytest.skip("BF16 not supported")

        vocoder = BigVGANVocoder(device="cuda")

        # BF16 input
        mel = torch.randn(1, 80, 100, dtype=torch.bfloat16).cuda()
        # Vocoder converts to FP32 internally
        mock_bigvgan["model"].return_value = torch.randn(1, 22050).cuda()

        waveform = vocoder.inference(mel)
        assert isinstance(waveform, np.ndarray)
        assert waveform.dtype == np.float32

    def test_inference_with_invalid_input(self, mock_bigvgan):
        """Test inference with invalid inputs"""
        vocoder = BigVGANVocoder(device="cpu")

        # Wrong number of dimensions
        with pytest.raises(ValueError):
            mel = torch.randn(80, 100)  # Missing batch dimension
            vocoder.inference(mel)

        # Wrong mel dimension
        with pytest.raises(ValueError):
            mel = torch.randn(1, 128, 100)  # Wrong mel bins (128 instead of 80)
            vocoder.inference(mel)

    def test_to_method(self, mock_bigvgan):
        """Test device movement"""
        vocoder = BigVGANVocoder(device="cpu")

        # Test to() method
        vocoder.to("cpu")  # Should not raise error
        assert vocoder.device == torch.device("cpu")

        if torch.cuda.is_available():
            vocoder.to("cuda")
            assert vocoder.device.type == "cuda"
            mock_bigvgan["model"].to.assert_called()

    def test_eval_mode(self, mock_bigvgan):
        """Test that model is in eval mode"""
        vocoder = BigVGANVocoder(device="cpu")

        # Model should be in eval mode
        mock_bigvgan["model"].eval.assert_called()

        # Ensure eval mode is maintained
        vocoder.model.eval()
        mock_bigvgan["model"].eval.assert_called()

    @pytest.mark.parametrize("sample_rate", [22050, 24000, 48000])
    def test_different_sample_rates(self, mock_bigvgan, sample_rate):
        """Test with different sample rates"""
        # Mock different model variants
        model_name = f"nvidia/bigvgan_{sample_rate//1000}khz_80band"
        vocoder = BigVGANVocoder(model_name=model_name, device="cpu")

        mel = torch.randn(1, 80, 100)
        output_samples = 100 * (sample_rate // 100)  # Approximate
        mock_bigvgan["model"].return_value = torch.randn(1, output_samples)

        waveform = vocoder.inference(mel)
        assert len(waveform) == output_samples

    def test_memory_efficiency(self, mock_bigvgan):
        """Test memory efficiency with no_grad"""
        vocoder = BigVGANVocoder(device="cpu")

        mel = torch.randn(1, 80, 100, requires_grad=True)
        mock_bigvgan["model"].return_value = torch.randn(1, 22050)

        # Should not create gradients
        waveform = vocoder.inference(mel)
        assert not hasattr(waveform, "grad_fn")

    def test_deterministic_output(self, mock_bigvgan):
        """Test deterministic output with same input"""
        vocoder = BigVGANVocoder(device="cpu")

        mel = torch.randn(1, 80, 100)
        fixed_output = torch.randn(1, 22050)
        mock_bigvgan["model"].return_value = fixed_output.clone()

        # Multiple inferences should give same result
        waveform1 = vocoder.inference(mel)
        mock_bigvgan["model"].return_value = fixed_output.clone()
        waveform2 = vocoder.inference(mel)

        np.testing.assert_array_almost_equal(waveform1, waveform2)

    @pytest.mark.slow
    def test_large_batch_inference(self, mock_bigvgan):
        """Test inference with large batches"""
        vocoder = BigVGANVocoder(device="cpu")

        # Large batch
        batch_size = 32
        mel_batch = torch.randn(batch_size, 80, 100)
        mock_bigvgan["model"].return_value = torch.randn(batch_size, 22050)

        waveforms = vocoder.batch_inference(mel_batch)
        assert len(waveforms) == batch_size
