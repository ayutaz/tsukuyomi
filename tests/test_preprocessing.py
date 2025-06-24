"""
Tests for preprocessing components
"""

import struct
import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

from src.tools.audio_preprocessing import (
    AudioAugmentation,
    AudioNormalizer,
    AudioPreprocessor,
    AudioSegmenter,
    NoiseReduction,
    PreprocessingConfig,
    VoiceActivityDetector,
)


class TestAudioPreprocessor:
    """Test audio preprocessing pipeline"""

    @pytest.fixture
    def audio_data(self):
        """Create dummy audio data"""
        # Generate 1 second of audio at 22050 Hz
        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Create audio with multiple frequency components
        audio = np.sin(2 * np.pi * 440 * t)  # A4
        audio += 0.5 * np.sin(2 * np.pi * 880 * t)  # A5
        audio += 0.1 * np.random.randn(len(t))  # Add noise

        # Normalize
        audio = audio / np.abs(audio).max()

        return audio.astype(np.float32), sample_rate

    @pytest.fixture
    def temp_audio_file(self, audio_data):
        """Create temporary audio file"""
        audio, sample_rate = audio_data

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # Write WAV file
            with wave.open(f.name, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)

                # Convert to 16-bit PCM
                audio_int16 = (audio * 32767).astype(np.int16)
                wav_file.writeframes(audio_int16.tobytes())

            yield f.name

            # Cleanup
            Path(f.name).unlink(missing_ok=True)

    def test_preprocessor_creation(self):
        """Test preprocessor creation"""
        config = PreprocessingConfig()
        preprocessor = AudioPreprocessor(config)

        assert preprocessor is not None
        assert preprocessor.config == config
        assert preprocessor.noise_reducer is not None
        assert preprocessor.normalizer is not None
        assert preprocessor.vad is not None
        assert preprocessor.segmenter is not None
        assert preprocessor.augmenter is not None

    def test_preprocess_audio(self, audio_data):
        """Test audio preprocessing"""
        audio, sample_rate = audio_data

        config = PreprocessingConfig(
            target_sample_rate=22050,
            enable_noise_reduction=True,
            enable_normalization=True,
            enable_vad=True,
        )

        preprocessor = AudioPreprocessor(config)
        processed_audio = preprocessor.preprocess(audio, sample_rate)

        assert isinstance(processed_audio, np.ndarray)
        assert processed_audio.dtype == np.float32
        assert len(processed_audio) > 0
        assert np.abs(processed_audio).max() <= 1.0

    def test_preprocess_file(self, temp_audio_file):
        """Test file preprocessing"""
        config = PreprocessingConfig()
        preprocessor = AudioPreprocessor(config)

        # Process file
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "processed.wav"
            result = preprocessor.preprocess_file(temp_audio_file, output_path)

            assert output_path.exists()
            assert result["duration"] > 0
            assert result["sample_rate"] == config.target_sample_rate


class TestNoiseReduction:
    """Test noise reduction module"""

    def test_noise_reduction(self):
        """Test basic noise reduction"""
        nr = NoiseReduction()

        # Create noisy signal
        clean = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))
        noise = 0.1 * np.random.randn(22050)
        noisy = clean + noise

        # Apply noise reduction
        denoised = nr.reduce_noise(noisy, 22050)

        assert isinstance(denoised, np.ndarray)
        assert len(denoised) == len(noisy)

        # Check that noise is reduced
        noise_power_before = np.mean((noisy - clean) ** 2)
        noise_power_after = np.mean((denoised - clean) ** 2)
        assert noise_power_after <= noise_power_before


class TestAudioNormalizer:
    """Test audio normalization"""

    def test_peak_normalization(self):
        """Test peak normalization"""
        normalizer = AudioNormalizer(method="peak")

        # Create audio with varying amplitude
        audio = np.random.randn(1000) * 0.5

        # Normalize
        normalized = normalizer.normalize(audio)

        assert np.abs(normalized).max() == pytest.approx(1.0, abs=0.01)

    def test_rms_normalization(self):
        """Test RMS normalization"""
        normalizer = AudioNormalizer(method="rms", target_level=-20)

        # Create audio
        audio = np.random.randn(1000) * 0.5

        # Normalize
        normalized = normalizer.normalize(audio)

        # Check RMS level
        rms_db = 20 * np.log10(np.sqrt(np.mean(normalized**2)))
        assert rms_db == pytest.approx(-20, abs=1.0)

    def test_lufs_normalization(self):
        """Test LUFS normalization"""
        normalizer = AudioNormalizer(method="lufs", target_level=-16)

        # Create audio (longer for LUFS measurement)
        audio = np.random.randn(44100) * 0.5  # 1 second

        # Normalize
        normalized = normalizer.normalize(audio, sample_rate=44100)

        assert isinstance(normalized, np.ndarray)
        assert len(normalized) == len(audio)


class TestVoiceActivityDetector:
    """Test VAD module"""

    def test_vad_detection(self):
        """Test voice activity detection"""
        vad = VoiceActivityDetector(frame_duration=30, sample_rate=16000)

        # Create audio with speech and silence
        sample_rate = 16000
        speech = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, sample_rate // 2))
        silence = np.zeros(sample_rate // 2)
        audio = np.concatenate([silence, speech, silence])

        # Detect segments
        segments = vad.detect_segments(audio, sample_rate)

        assert len(segments) > 0
        assert all("start" in seg and "end" in seg for seg in segments)

        # Check that speech is detected in the middle
        speech_detected = False
        for seg in segments:
            if seg["start"] <= 0.5 <= seg["end"]:
                speech_detected = True
                break
        assert speech_detected


class TestAudioSegmenter:
    """Test audio segmentation"""

    def test_segmentation(self):
        """Test audio segmentation"""
        segmenter = AudioSegmenter(
            min_segment_duration=0.5, max_segment_duration=2.0, silence_threshold=-40
        )

        # Create audio with pauses
        sample_rate = 16000
        segment1 = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, sample_rate // 2))
        silence = np.zeros(sample_rate // 4)
        segment2 = np.sin(2 * np.pi * 880 * np.linspace(0, 0.5, sample_rate // 2))

        audio = np.concatenate([segment1, silence, segment2])

        # Segment audio
        segments = segmenter.segment(audio, sample_rate)

        assert len(segments) >= 2
        for seg in segments:
            assert "audio" in seg
            assert "start_time" in seg
            assert "end_time" in seg
            assert isinstance(seg["audio"], np.ndarray)


class TestAudioAugmentation:
    """Test audio augmentation"""

    def test_pitch_shift(self):
        """Test pitch shifting augmentation"""
        augmenter = AudioAugmentation()

        # Create test audio
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))

        # Apply pitch shift
        shifted = augmenter.pitch_shift(audio, 22050, semitones=2)

        assert isinstance(shifted, np.ndarray)
        assert len(shifted) == len(audio)

    def test_time_stretch(self):
        """Test time stretching"""
        augmenter = AudioAugmentation()

        # Create test audio
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))

        # Apply time stretch
        stretched = augmenter.time_stretch(audio, rate=1.2)

        assert isinstance(stretched, np.ndarray)
        # Should be shorter when rate > 1
        assert len(stretched) < len(audio)

    def test_add_noise(self):
        """Test noise addition"""
        augmenter = AudioAugmentation()

        # Create test audio
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))

        # Add noise
        noisy = augmenter.add_noise(audio, noise_level=0.05)

        assert isinstance(noisy, np.ndarray)
        assert len(noisy) == len(audio)

        # Check that noise was added
        difference = np.mean(np.abs(noisy - audio))
        assert difference > 0

    def test_apply_effects(self):
        """Test combined effects"""
        augmenter = AudioAugmentation()

        # Create test audio
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))

        # Apply multiple effects
        augmented = augmenter.apply_effects(
            audio, 22050, effects=["pitch_shift", "reverb", "noise"]
        )

        assert isinstance(augmented, np.ndarray)
        assert len(augmented) > 0

        # Should be different from original
        assert not np.allclose(augmented, audio[: len(augmented)])


class TestBatchProcessing:
    """Test batch processing functionality"""

    def test_batch_preprocess(self):
        """Test batch preprocessing"""
        config = PreprocessingConfig(
            enable_noise_reduction=False, enable_vad=False  # Faster for testing
        )
        preprocessor = AudioPreprocessor(config)

        # Create multiple audio samples
        audios = []
        sample_rate = 22050
        for freq in [440, 880, 1320]:
            audio = np.sin(2 * np.pi * freq * np.linspace(0, 0.1, 2205))
            audios.append(audio)

        # Process batch
        processed = preprocessor.preprocess_batch(audios, sample_rate)

        assert len(processed) == len(audios)
        for audio in processed:
            assert isinstance(audio, np.ndarray)
            assert audio.dtype == np.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
