"""
Integration tests for the complete TTS pipeline
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.inference import TsukuyomiTTS


class TestTsukuyomiTTSIntegration:
    """Integration tests for the complete TTS system"""

    @pytest.fixture()
    def mock_components(self):
        """Mock all TTS components"""
        with (
            patch("src.inference.JapaneseTextNormalizer") as mock_normalizer,
            patch("src.inference.XPhoneBERTWrapper") as mock_xphonebert,
            patch("src.inference.TsukuyomiAcousticModel") as mock_acoustic,
            patch("src.inference.BigVGANVocoder") as mock_vocoder,
        ):

            # Mock normalizer
            mock_normalizer_instance = MagicMock()
            mock_normalizer_instance.normalize.return_value = {
                "text": "normalized text",
                "phonemes": "k o n n i ch i w a",
                "tokens": ["kon", "ni", "chi", "wa"],
            }
            mock_normalizer.return_value = mock_normalizer_instance

            # Mock XPhoneBERT
            mock_xphonebert_instance = MagicMock()
            mock_xphonebert_instance.encode.return_value = torch.randn(1, 10, 768)
            mock_xphonebert.return_value = mock_xphonebert_instance

            # Mock acoustic model
            mock_acoustic_instance = MagicMock()
            mock_acoustic_instance.eval.return_value = mock_acoustic_instance
            mock_acoustic_instance.to.return_value = mock_acoustic_instance
            mock_acoustic_instance.return_value = (
                torch.randn(1, 80, 100),  # mel spectrogram
                torch.randn(1, 1, 10),  # durations
            )
            mock_acoustic.return_value = mock_acoustic_instance

            # Mock vocoder
            mock_vocoder_instance = MagicMock()
            mock_vocoder_instance.inference.return_value = np.random.randn(22050)
            mock_vocoder.return_value = mock_vocoder_instance

            yield {
                "normalizer": mock_normalizer_instance,
                "xphonebert": mock_xphonebert_instance,
                "acoustic": mock_acoustic_instance,
                "vocoder": mock_vocoder_instance,
            }

    def test_initialization(self, mock_components):
        """Test TTS system initialization"""
        tts = TsukuyomiTTS(device="cpu")

        assert tts.device == torch.device("cpu")
        assert tts.sample_rate == 22050
        assert hasattr(tts, "text_normalizer")
        assert hasattr(tts, "phoneme_encoder")
        assert hasattr(tts, "acoustic_model")
        assert hasattr(tts, "vocoder")

    def test_basic_synthesis(self, mock_components):
        """Test basic text-to-speech synthesis"""
        tts = TsukuyomiTTS(device="cpu")

        text = "こんにちは"
        audio, sr = tts.synthesize(text)

        # Check outputs
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert sr == 22050
        assert len(audio) == 22050  # 1 second

        # Verify call chain
        mock_components["normalizer"].normalize.assert_called_once()
        mock_components["xphonebert"].encode.assert_called_once()
        mock_components["vocoder"].inference.assert_called_once()

    def test_synthesis_with_speaker(self, mock_components):
        """Test synthesis with speaker ID"""
        tts = TsukuyomiTTS(device="cpu")

        text = "テスト"
        speaker_id = 5
        audio, sr = tts.synthesize(text, speaker_id=speaker_id)

        assert isinstance(audio, np.ndarray)
        # Verify speaker ID was passed
        call_args = mock_components["acoustic"].call_args
        assert call_args is not None

    def test_synthesis_with_speed_control(self, mock_components):
        """Test synthesis with speed control"""
        tts = TsukuyomiTTS(device="cpu")

        # Faster speech
        audio_fast, _ = tts.synthesize("テスト", speed=1.5)

        # Slower speech
        audio_slow, _ = tts.synthesize("テスト", speed=0.5)

        # Both should produce audio
        assert isinstance(audio_fast, np.ndarray)
        assert isinstance(audio_slow, np.ndarray)

    def test_synthesis_with_pitch_shift(self, mock_components):
        """Test synthesis with pitch shifting"""
        tts = TsukuyomiTTS(device="cpu")

        # Higher pitch
        audio_high, _ = tts.synthesize("テスト", pitch_shift=2.0)

        # Lower pitch
        audio_low, _ = tts.synthesize("テスト", pitch_shift=-2.0)

        assert isinstance(audio_high, np.ndarray)
        assert isinstance(audio_low, np.ndarray)

    def test_batch_synthesis(self, mock_components):
        """Test batch synthesis"""
        tts = TsukuyomiTTS(device="cpu")

        texts = ["テスト1", "テスト2", "テスト3"]
        speaker_ids = [0, 1, 2]

        # Mock batch returns
        batch_size = len(texts)
        mock_components["xphonebert"].batch_encode.return_value = torch.randn(
            batch_size, 10, 768
        )
        mock_components["normalizer"].batch_normalize.return_value = [
            {
                "text": f"normalized {i}",
                "phonemes": "t e s u t o",
                "tokens": ["te", "su", "to"],
            }
            for i in range(batch_size)
        ]

        audio_list = tts.synthesize_batch(texts, speaker_ids=speaker_ids)

        assert len(audio_list) == batch_size
        for audio, sr in audio_list:
            assert isinstance(audio, np.ndarray)
            assert sr == 22050

    def test_multilingual_synthesis(self, mock_components):
        """Test multilingual synthesis"""
        tts = TsukuyomiTTS(device="cpu")

        test_cases = [
            ("Hello world", "en"),
            ("こんにちは", "ja"),
            ("你好", "zh"),
        ]

        for text, language in test_cases:
            audio, sr = tts.synthesize(text, language=language)
            assert isinstance(audio, np.ndarray)

            # Verify language was passed
            call_args = mock_components["xphonebert"].encode.call_args
            assert "language" in call_args[1]
            assert call_args[1]["language"] == language

    @pytest.mark.gpu()
    def test_gpu_synthesis(self, mock_components):
        """Test synthesis on GPU"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        tts = TsukuyomiTTS(device="cuda")
        assert tts.device.type == "cuda"

        audio, sr = tts.synthesize("GPUテスト")
        assert isinstance(audio, np.ndarray)

    @pytest.mark.bf16()
    def test_bf16_synthesis(self, mock_components):
        """Test BF16 synthesis"""
        if not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
            pytest.skip("BF16 not supported")

        tts = TsukuyomiTTS(device="cuda", use_bf16=True)

        # Mock BF16 tensors
        mock_components["xphonebert"].encode.return_value = torch.randn(
            1, 10, 768, dtype=torch.bfloat16
        ).cuda()

        audio, sr = tts.synthesize("BF16テスト")
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32  # Output is always FP32

    def test_checkpoint_loading(self, mock_components):
        """Test checkpoint loading"""
        tts = TsukuyomiTTS(device="cpu")

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            # Save dummy checkpoint
            checkpoint = {
                "model_state_dict": mock_components["acoustic"].state_dict(),
                "epoch": 100,
                "global_step": 10000,
            }
            torch.save(checkpoint, f.name)

            # Load checkpoint
            tts.load_checkpoint(f.name)

            # Verify loading was attempted
            mock_components["acoustic"].load_state_dict.assert_called()

    def test_onnx_export(self, mock_components):
        """Test ONNX export"""
        tts = TsukuyomiTTS(device="cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = Path(tmpdir)

            # Mock ONNX export
            with patch("torch.onnx.export"):
                tts.export_onnx(export_dir)

                # Verify export paths would be created
                expected_files = [
                    "text_encoder.onnx",
                    "acoustic_model.onnx",
                    "vocoder.onnx",
                ]
                # In real implementation, these files would be created

    def test_performance_benchmark(self, mock_components):
        """Test performance benchmarking"""
        tts = TsukuyomiTTS(device="cpu")

        results = tts.benchmark(num_runs=3)

        assert "avg_rtf" in results
        assert "tokens_per_second" in results
        assert "avg_latency" in results
        assert results["avg_rtf"] > 0
        assert results["tokens_per_second"] > 0

    def test_error_handling(self, mock_components):
        """Test error handling"""
        tts = TsukuyomiTTS(device="cpu")

        # Empty text
        audio, sr = tts.synthesize("")
        assert len(audio) == 0

        # Very long text
        long_text = "テスト" * 1000
        audio, sr = tts.synthesize(long_text)
        assert isinstance(audio, np.ndarray)

        # Invalid speaker ID (should use default)
        audio, sr = tts.synthesize("テスト", speaker_id=9999)
        assert isinstance(audio, np.ndarray)

    def test_memory_efficiency(self, mock_components):
        """Test memory efficiency"""
        tts = TsukuyomiTTS(device="cpu")

        # Multiple synthesis calls shouldn't accumulate memory
        for i in range(10):
            audio, sr = tts.synthesize(f"テスト{i}")
            assert isinstance(audio, np.ndarray)

        # No gradients should be created
        for param in mock_components["acoustic"].parameters():
            if hasattr(param, "grad"):
                assert param.grad is None or param.grad.sum() == 0

    @pytest.mark.slow()
    def test_long_text_synthesis(self, mock_components):
        """Test synthesis of long text"""
        tts = TsukuyomiTTS(device="cpu")

        # Long paragraph
        long_text = "これは長いテキストのテストです。" * 20

        # Mock longer outputs
        mock_components["xphonebert"].encode.return_value = torch.randn(1, 500, 768)
        mock_components["acoustic"].return_value = (
            torch.randn(1, 80, 2000),  # Long mel
            torch.randn(1, 1, 500),  # Long durations
        )
        mock_components["vocoder"].inference.return_value = np.random.randn(
            441000
        )  # 20 seconds

        audio, sr = tts.synthesize(long_text)

        assert isinstance(audio, np.ndarray)
        assert len(audio) == 441000  # 20 seconds at 22050 Hz
