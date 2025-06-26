"""Tests for evaluation metrics and benchmarking."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch

from src.evaluation.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    PerformanceBenchmark,
)
from src.evaluation.metrics import (
    ComprehensiveEvaluator,
    EvaluationResult,
    MelCepstralDistortion,
    PitchEvaluator,
    PronunciationAccuracy,
    SpeakerSimilarity,
    calculate_rtf,
)


class TestEvaluationMetrics:
    """Test evaluation metrics."""

    @pytest.fixture()
    def sample_audio(self):
        """Create sample audio for testing."""
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        # Create synthetic audio with harmonic content
        audio = np.sin(2 * np.pi * 440 * t)  # A4 note
        audio += 0.5 * np.sin(2 * np.pi * 880 * t)  # A5
        audio += 0.25 * np.sin(2 * np.pi * 1320 * t)  # E6

        return audio.astype(np.float32), sr

    def test_mel_cepstral_distortion(self, sample_audio):
        """Test MCD calculation."""
        audio, sr = sample_audio

        # Add slight noise to create second audio
        noise = np.random.normal(0, 0.01, len(audio))
        audio_noisy = audio + noise

        mcd_calc = MelCepstralDistortion()
        mcd = mcd_calc.calculate(audio, audio_noisy, sr)

        assert isinstance(mcd, float)
        assert mcd > 0  # Should have some distortion
        assert mcd < 10  # But not too much

    def test_pitch_evaluator(self, sample_audio):
        """Test pitch evaluation."""
        audio, sr = sample_audio

        # Create pitch-shifted version
        audio_shifted = audio  # Simplified for test

        with patch("crepe.predict") as mock_crepe:
            # Mock CREPE output
            mock_crepe.return_value = (
                None,  # time
                np.full(100, 440.0),  # frequency
                np.ones(100),  # confidence
                None,  # activation
            )

            pitch_eval = PitchEvaluator()
            results = pitch_eval.evaluate(audio, audio_shifted, sr)

            assert "pitch_correlation" in results
            assert "pitch_rmse" in results
            assert "vuv_error" in results

    def test_speaker_similarity(self, sample_audio):
        """Test speaker similarity calculation."""
        audio1, sr = sample_audio
        audio2 = audio1 * 0.9  # Slightly different amplitude

        # Mock the model
        with patch("transformers.AutoProcessor.from_pretrained") as mock_proc:
            with patch("transformers.AutoModel.from_pretrained") as mock_model:
                mock_proc.return_value = MagicMock()
                mock_model_instance = MagicMock()
                mock_model.return_value = mock_model_instance

                # Mock model output
                mock_embeddings = MagicMock()
                mock_embeddings.embeddings = torch.randn(1, 10, 768)
                mock_model_instance.return_value = mock_embeddings

                speaker_eval = SpeakerSimilarity()

                # Test with mock
                if speaker_eval.model is None:
                    # If model loading failed, should return default
                    similarity = speaker_eval.calculate_similarity(audio1, audio2, sr)
                    assert similarity == 0.5
                else:
                    similarity = speaker_eval.calculate_similarity(audio1, audio2, sr)
                    assert 0 <= similarity <= 1

    def test_pronunciation_accuracy(self, sample_audio):
        """Test pronunciation accuracy evaluation."""
        audio, sr = sample_audio
        reference_text = "こんにちは"

        with patch("whisper.load_model") as mock_whisper:
            mock_model = MagicMock()
            mock_model.transcribe.return_value = {"text": "こんにちは"}
            mock_whisper.return_value = mock_model

            pron_eval = PronunciationAccuracy()

            if pron_eval.model is None:
                # Default values if model not loaded
                results = pron_eval.evaluate(audio, reference_text, sr)
                assert results["pronunciation_accuracy"] == 0.5
            else:
                with patch("jiwer.cer", return_value=0.1):
                    with patch("jiwer.wer", return_value=0.2):
                        results = pron_eval.evaluate(audio, reference_text, sr)

                        assert "pronunciation_accuracy" in results
                        assert "cer" in results
                        assert "wer" in results
                        assert results["pronunciation_accuracy"] == 0.9  # 1 - cer

    def test_evaluation_result(self):
        """Test EvaluationResult dataclass."""
        result = EvaluationResult(
            mel_cepstral_distortion=3.5,
            pitch_correlation=0.85,
            voice_similarity=0.9,
            pronunciation_accuracy=0.95,
            speaker_similarity=0.88,
        )

        # Test overall score calculation
        score = result.overall_score
        assert 1.0 <= score <= 5.0

        # Test to_dict
        result_dict = result.to_dict()
        assert "mel_cepstral_distortion" in result_dict
        assert result_dict["pitch_correlation"] == 0.85

    def test_comprehensive_evaluator(self, sample_audio):
        """Test comprehensive evaluation."""
        audio, sr = sample_audio

        with patch.object(MelCepstralDistortion, "calculate", return_value=3.5):
            with patch.object(
                PitchEvaluator, "evaluate", return_value={"pitch_correlation": 0.9}
            ):
                with patch.object(
                    SpeakerSimilarity, "calculate_similarity", return_value=0.85
                ):
                    evaluator = ComprehensiveEvaluator(use_gpu=False)

                    result = evaluator.evaluate(
                        synthesized_audio=audio,
                        reference_audio=audio,
                        sr=sr,
                        compute_all=False,
                    )

                    assert isinstance(result, EvaluationResult)
                    assert result.mel_cepstral_distortion == 3.5
                    assert result.pitch_correlation == 0.9

    def test_calculate_rtf(self):
        """Test RTF calculation."""
        model_time = 0.5  # 500ms
        audio_duration = 2.0  # 2 seconds

        rtf = calculate_rtf(model_time, audio_duration)
        assert rtf == 0.25  # 4x faster than real-time


class TestBenchmark:
    """Test benchmarking functionality."""

    @pytest.fixture()
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        import shutil

        shutil.rmtree(temp_dir)

    @pytest.fixture()
    def benchmark_config(self, temp_dir):
        """Create benchmark configuration."""
        return BenchmarkConfig(
            model_path="dummy_model.pt",
            test_dataset="dummy_dataset",
            output_dir=str(temp_dir),
            batch_sizes=[1, 2],
            sequence_lengths=[50, 100],
            num_warmup_runs=2,
            num_benchmark_runs=5,
            compute_quality_metrics=False,
        )

    @pytest.fixture()
    def mock_model(self):
        """Create mock model."""
        model = MagicMock()
        model.eval.return_value = model
        model.to.return_value = model

        # Mock forward pass
        def forward(**kwargs):
            batch_size = kwargs["phoneme_ids"].shape[0]
            seq_len = kwargs["phoneme_ids"].shape[1]
            return torch.randn(batch_size, 80, seq_len * 2)

        model.forward = forward
        model.__call__ = forward

        return model

    def test_benchmark_result(self):
        """Test BenchmarkResult dataclass."""
        result = BenchmarkResult(
            batch_size=16,
            sequence_length=100,
            num_speakers=10,
            device="cuda:0",
            model_type="pytorch",
            inference_time_mean=0.05,
            inference_time_std=0.01,
            rtf_mean=0.1,
            rtf_std=0.02,
            throughput=320.0,
        )

        result_dict = result.to_dict()
        assert result_dict["batch_size"] == 16
        assert result_dict["rtf_mean"] == 0.1

    @patch("torch.cuda.is_available", return_value=False)
    def test_performance_benchmark(
        self, mock_cuda, benchmark_config, mock_model, temp_dir
    ):
        """Test performance benchmarking."""
        benchmark = PerformanceBenchmark(benchmark_config)

        # Run benchmark
        results = benchmark.benchmark_model(mock_model, "pytorch")

        assert len(results) > 0
        assert all(isinstance(r, BenchmarkResult) for r in results)

        # Check results have expected fields
        result = results[0]
        assert result.device == "cpu"
        assert result.inference_time_mean > 0
        assert result.rtf_mean > 0

    def test_benchmark_report_generation(self, benchmark_config, temp_dir):
        """Test report generation."""
        benchmark = PerformanceBenchmark(benchmark_config)

        # Add mock results
        benchmark.results = [
            BenchmarkResult(
                batch_size=1,
                sequence_length=100,
                num_speakers=1,
                device="cpu",
                model_type="pytorch",
                inference_time_mean=0.1,
                inference_time_std=0.01,
                rtf_mean=0.2,
                rtf_std=0.02,
                throughput=10.0,
                peak_memory_mb=100.0,
            )
        ]

        # Generate report
        benchmark.generate_report()

        # Check files were created
        assert (temp_dir / "benchmark_results.csv").exists()
        assert (temp_dir / "benchmark_report.md").exists()

    @patch("psutil.cpu_percent", return_value=50.0)
    @patch("GPUtil.getGPUs", return_value=[])
    def test_system_metrics(self, mock_gpu, mock_cpu, benchmark_config, mock_model):
        """Test system metrics collection."""
        benchmark = PerformanceBenchmark(benchmark_config)

        result = benchmark._benchmark_configuration(
            mock_model,
            batch_size=1,
            sequence_length=50,
            device="cpu",
            model_type="pytorch",
        )

        assert result.cpu_percent == 50.0
