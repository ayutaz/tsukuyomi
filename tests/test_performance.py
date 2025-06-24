"""
Performance and benchmark tests for Tsukuyomi TTS
"""

import pytest
import torch
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock

from src.inference import TsukuyomiTTS


class TestPerformance:
    """Performance-related tests"""

    @pytest.fixture
    def mock_fast_components(self):
        """Mock components with controlled timing"""
        with (
            patch("src.inference.JapaneseTextNormalizer") as mock_normalizer,
            patch("src.inference.XPhoneBERTWrapper") as mock_xphonebert,
            patch("src.inference.TsukuyomiAcousticModel") as mock_acoustic,
            patch("src.inference.BigVGANVocoder") as mock_vocoder,
        ):

            # Mock with controlled execution times
            mock_normalizer_instance = MagicMock()
            mock_normalizer_instance.normalize.return_value = {
                "text": "normalized",
                "phonemes": "p h o n e m e s",
                "tokens": ["pho", "ne", "mes"],
            }
            mock_normalizer_instance.normalize.side_effect = lambda x: (
                time.sleep(0.01),
                mock_normalizer_instance.normalize.return_value,
            )[1]

            mock_xphonebert_instance = MagicMock()
            mock_xphonebert_instance.encode.return_value = torch.randn(1, 10, 768)
            mock_xphonebert_instance.encode.side_effect = lambda *args, **kwargs: (
                time.sleep(0.02),
                mock_xphonebert_instance.encode.return_value,
            )[1]

            mock_acoustic_instance = MagicMock()
            mock_acoustic_instance.eval.return_value = mock_acoustic_instance
            mock_acoustic_instance.to.return_value = mock_acoustic_instance
            mock_acoustic_instance.return_value = (
                torch.randn(1, 80, 100),
                torch.randn(1, 1, 10),
            )
            mock_acoustic_instance.side_effect = lambda *args, **kwargs: (
                time.sleep(0.03),
                mock_acoustic_instance.return_value,
            )[1]

            mock_vocoder_instance = MagicMock()
            mock_vocoder_instance.inference.return_value = np.random.randn(22050)
            mock_vocoder_instance.inference.side_effect = lambda x: (
                time.sleep(0.02),
                mock_vocoder_instance.inference.return_value,
            )[1]

            mock_normalizer.return_value = mock_normalizer_instance
            mock_xphonebert.return_value = mock_xphonebert_instance
            mock_acoustic.return_value = mock_acoustic_instance
            mock_vocoder.return_value = mock_vocoder_instance

            yield {
                "normalizer": mock_normalizer_instance,
                "xphonebert": mock_xphonebert_instance,
                "acoustic": mock_acoustic_instance,
                "vocoder": mock_vocoder_instance,
            }

    def test_inference_speed(self, mock_fast_components):
        """Test inference speed meets requirements"""
        tts = TsukuyomiTTS(device="cpu")

        # Measure synthesis time
        text = "これは推論速度のテストです。"
        start_time = time.time()
        audio, sr = tts.synthesize(text)
        inference_time = time.time() - start_time

        # Check that inference completed
        assert isinstance(audio, np.ndarray)

        # Calculate real-time factor
        audio_duration = len(audio) / sr
        rtf = inference_time / audio_duration

        # Should be faster than real-time (RTF < 1 for real-time)
        # With mocked components, should be very fast
        assert rtf < 10  # Generous limit for mocked components
        assert inference_time < 1.0  # Should complete within 1 second

    @pytest.mark.benchmark
    def test_component_timing(self, mock_fast_components):
        """Benchmark individual components"""
        tts = TsukuyomiTTS(device="cpu")

        # Time each component
        timings = {}

        # Text normalization
        start = time.time()
        result = tts.text_normalizer.normalize("テスト")
        timings["normalization"] = time.time() - start

        # Phoneme encoding
        start = time.time()
        embeddings = tts.phoneme_encoder.encode("test phonemes")
        timings["phoneme_encoding"] = time.time() - start

        # Acoustic model
        start = time.time()
        with torch.no_grad():
            mel, durations = tts.acoustic_model(embeddings, torch.tensor([0]))
        timings["acoustic_model"] = time.time() - start

        # Vocoder
        start = time.time()
        audio = tts.vocoder.inference(mel)
        timings["vocoder"] = time.time() - start

        # All components should be reasonably fast
        for component, timing in timings.items():
            assert timing < 0.5, f"{component} took too long: {timing}s"

        # Total should be under 1 second
        total_time = sum(timings.values())
        assert total_time < 1.0

    def test_batch_performance(self, mock_fast_components):
        """Test batch processing performance"""
        tts = TsukuyomiTTS(device="cpu")

        # Mock batch processing
        batch_size = 8
        texts = [f"バッチテスト{i}" for i in range(batch_size)]

        mock_fast_components["normalizer"].batch_normalize.return_value = [
            {"text": f"norm{i}", "phonemes": "t e s t", "tokens": ["test"]}
            for i in range(batch_size)
        ]
        mock_fast_components["xphonebert"].batch_encode.return_value = torch.randn(
            batch_size, 10, 768
        )

        # Time batch processing
        start_time = time.time()
        audio_list = tts.synthesize_batch(texts)
        batch_time = time.time() - start_time

        # Batch should be more efficient than individual
        assert len(audio_list) == batch_size

        # Rough estimate: batch should be faster than sequential
        expected_sequential_time = 0.08 * batch_size  # Based on mock timings
        assert batch_time < expected_sequential_time

    @pytest.mark.slow
    def test_memory_usage(self, mock_fast_components):
        """Test memory usage during synthesis"""
        tts = TsukuyomiTTS(device="cpu")

        # Track memory usage
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()

        # Synthesize multiple times
        for i in range(10):
            audio, sr = tts.synthesize(f"メモリテスト{i}")
            assert isinstance(audio, np.ndarray)

        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()

            # Memory should not grow significantly
            memory_growth = final_memory - initial_memory
            assert memory_growth < 100 * 1024 * 1024  # Less than 100MB growth

            # Peak should be reasonable
            assert peak_memory < 2 * 1024 * 1024 * 1024  # Less than 2GB peak

    def test_long_text_performance(self, mock_fast_components):
        """Test performance with long texts"""
        tts = TsukuyomiTTS(device="cpu")

        # Create texts of different lengths
        text_lengths = [10, 50, 100, 500]
        timings = []

        for length in text_lengths:
            text = "あ" * length

            # Mock appropriate output sizes
            mock_fast_components["xphonebert"].encode.return_value = torch.randn(
                1, length, 768
            )
            mock_fast_components["acoustic"].return_value = (
                torch.randn(1, 80, length * 10),
                torch.randn(1, 1, length),
            )
            mock_fast_components["vocoder"].inference.return_value = np.random.randn(
                length * 220
            )

            start_time = time.time()
            audio, sr = tts.synthesize(text)
            synthesis_time = time.time() - start_time

            timings.append((length, synthesis_time))

        # Check that timing scales reasonably with length
        # Should be roughly linear or sub-linear
        for i in range(1, len(timings)):
            length_ratio = timings[i][0] / timings[i - 1][0]
            time_ratio = timings[i][1] / timings[i - 1][1]

            # Time should not scale super-linearly
            assert time_ratio < length_ratio * 1.5

    @pytest.mark.gpu
    def test_gpu_performance(self, mock_fast_components):
        """Test GPU performance improvements"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        # CPU timing
        tts_cpu = TsukuyomiTTS(device="cpu")
        start = time.time()
        audio_cpu, _ = tts_cpu.synthesize("GPUテスト")
        cpu_time = time.time() - start

        # GPU timing
        tts_gpu = TsukuyomiTTS(device="cuda")

        # Warm up GPU
        _ = tts_gpu.synthesize("ウォームアップ")

        start = time.time()
        audio_gpu, _ = tts_gpu.synthesize("GPUテスト")
        gpu_time = time.time() - start

        # GPU should not be significantly slower than CPU for single inference
        # (Due to overhead, GPU might be slower for single small inputs)
        assert gpu_time < cpu_time * 2

    def test_compile_performance(self, mock_fast_components):
        """Test torch.compile performance"""
        if not hasattr(torch, "compile"):
            pytest.skip("torch.compile not available")

        tts = TsukuyomiTTS(device="cpu", compile_model=True)

        # First run (compilation)
        start = time.time()
        audio1, _ = tts.synthesize("コンパイルテスト")
        first_run_time = time.time() - start

        # Subsequent runs (should be faster)
        times = []
        for i in range(5):
            start = time.time()
            audio, _ = tts.synthesize(f"テスト{i}")
            times.append(time.time() - start)

        avg_time = np.mean(times)

        # Subsequent runs should be faster (or at least not slower)
        assert avg_time <= first_run_time * 1.2

    def test_realtime_factor(self, mock_fast_components):
        """Test real-time factor calculation"""
        tts = TsukuyomiTTS(device="cpu")

        # Run benchmark
        results = tts.benchmark(num_runs=5)

        # Check results
        assert "avg_rtf" in results
        assert "tokens_per_second" in results
        assert "avg_latency" in results

        # RTF should be reasonable (< 1 for real-time)
        assert results["avg_rtf"] < 50  # Generous limit for mocked components
        assert results["tokens_per_second"] > 10
        assert results["avg_latency"] < 1.0

    def test_concurrent_synthesis(self, mock_fast_components):
        """Test concurrent synthesis requests"""
        import threading
        import queue

        tts = TsukuyomiTTS(device="cpu")

        # Queue for results
        result_queue = queue.Queue()

        def synthesize_worker(text, queue):
            try:
                audio, sr = tts.synthesize(text)
                queue.put((True, len(audio)))
            except Exception as e:
                queue.put((False, str(e)))

        # Launch concurrent threads
        threads = []
        num_threads = 4
        for i in range(num_threads):
            t = threading.Thread(
                target=synthesize_worker, args=(f"並行テスト{i}", result_queue)
            )
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join(timeout=5.0)

        # Check results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        assert len(results) == num_threads
        for success, data in results:
            assert success, f"Synthesis failed: {data}"
            assert data > 0  # Audio was generated
