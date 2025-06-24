"""
Tests for utility modules
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from src.utils.audio import (
    audio_to_mel,
    compute_mel_spectrogram,
    load_audio,
    normalize_audio,
    save_audio,
    trim_silence,
)


class TestAudioUtils:
    """Test audio utility functions"""

    @pytest.fixture
    def sample_audio_data(self):
        """Generate sample audio data"""
        sample_rate = 22050
        duration = 1.0  # 1 second
        t = np.linspace(0, duration, int(sample_rate * duration))
        # 440 Hz sine wave (A4 note)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        return audio, sample_rate

    @pytest.fixture
    def temp_audio_file(self, sample_audio_data):
        """Create a temporary audio file"""
        audio, sr = sample_audio_data
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sr)
            yield f.name
        # Cleanup
        Path(f.name).unlink()

    def test_load_audio(self, temp_audio_file, sample_audio_data):
        """Test audio loading"""
        expected_audio, expected_sr = sample_audio_data

        # Load with original sample rate
        audio, sr = load_audio(temp_audio_file, sr=None)
        assert sr == expected_sr
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        np.testing.assert_array_almost_equal(audio, expected_audio, decimal=4)

        # Load with resampling
        target_sr = 16000
        audio_resampled, sr_resampled = load_audio(temp_audio_file, sr=target_sr)
        assert sr_resampled == target_sr
        assert len(audio_resampled) == int(len(audio) * target_sr / expected_sr)

    def test_save_audio(self, sample_audio_data):
        """Test audio saving"""
        audio, sr = sample_audio_data

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name

        try:
            save_audio(output_path, audio, sr)
            assert Path(output_path).exists()

            # Verify saved audio
            loaded_audio, loaded_sr = load_audio(output_path)
            assert loaded_sr == sr
            np.testing.assert_array_almost_equal(loaded_audio, audio, decimal=4)
        finally:
            Path(output_path).unlink()

    def test_normalize_audio(self):
        """Test audio normalization"""
        # Test with various input ranges
        test_cases = [
            np.array([0.5, -0.5, 0.3, -0.3]),  # Already normalized
            np.array([2.0, -2.0, 1.0, -1.0]),  # Needs normalization
            np.array([0.1, -0.1, 0.05, -0.05]),  # Low amplitude
        ]

        for audio in test_cases:
            normalized = normalize_audio(audio.astype(np.float32))
            assert np.abs(normalized).max() <= 1.0
            assert normalized.dtype == np.float32

            # Check relative values are preserved
            if np.abs(audio).max() > 0:
                ratio = normalized[0] / audio[0]
                expected = audio * ratio
                np.testing.assert_array_almost_equal(normalized, expected)

    def test_normalize_audio_edge_cases(self):
        """Test audio normalization edge cases"""
        # Silent audio
        silent = np.zeros(1000, dtype=np.float32)
        normalized = normalize_audio(silent)
        assert np.all(normalized == 0)

        # Single spike
        spike = np.zeros(1000, dtype=np.float32)
        spike[500] = 10.0
        normalized = normalize_audio(spike)
        assert normalized[500] == pytest.approx(0.95, abs=0.01)

    def test_trim_silence(self, sample_audio_data):
        """Test silence trimming"""
        audio, sr = sample_audio_data

        # Add silence at beginning and end
        silence_duration = int(0.2 * sr)  # 200ms
        silence = np.zeros(silence_duration, dtype=np.float32)
        audio_with_silence = np.concatenate([silence, audio, silence])

        # Trim silence
        trimmed = trim_silence(audio_with_silence, threshold_db=-40)

        # Should be shorter than original
        assert len(trimmed) < len(audio_with_silence)
        # Should preserve most of the actual audio
        assert len(trimmed) >= len(audio) * 0.9

    def test_compute_mel_spectrogram(self, sample_audio_data):
        """Test mel spectrogram computation"""
        audio, sr = sample_audio_data

        # Default parameters
        mel = compute_mel_spectrogram(audio, sr)
        assert isinstance(mel, torch.Tensor)
        assert mel.dim() == 2
        assert mel.shape[0] == 80  # Default n_mels
        assert mel.shape[1] > 0  # Time frames
        assert not torch.isnan(mel).any()

        # Custom parameters
        mel_custom = compute_mel_spectrogram(
            audio, sr, n_mels=128, n_fft=2048, hop_length=512
        )
        assert mel_custom.shape[0] == 128
        assert mel_custom.shape[1] == (len(audio) // 512) + 1

    def test_audio_to_mel(self, sample_audio_data):
        """Test audio to mel conversion"""
        audio, sr = sample_audio_data

        mel = audio_to_mel(audio, sr)
        assert isinstance(mel, torch.Tensor)
        assert mel.shape[0] == 80
        assert not torch.isnan(mel).any()
        assert not torch.isinf(mel).any()

    def test_mel_spectrogram_properties(self, sample_audio_data):
        """Test mel spectrogram properties"""
        audio, sr = sample_audio_data

        mel = compute_mel_spectrogram(audio, sr)

        # Check dynamic range
        assert mel.min() >= -100  # Typical floor in dB
        assert mel.max() <= 20  # Typical ceiling in dB

        # Check that louder audio produces higher mel values
        loud_audio = audio * 2.0
        mel_loud = compute_mel_spectrogram(loud_audio, sr)
        assert mel_loud.mean() > mel.mean()

    @pytest.mark.parametrize(
        "sr,n_fft,hop_length",
        [
            (22050, 1024, 256),
            (16000, 512, 128),
            (48000, 2048, 480),
        ],
    )
    def test_different_audio_configs(self, sr, n_fft, hop_length):
        """Test with different audio configurations"""
        duration = 1.0
        samples = int(sr * duration)
        audio = np.random.randn(samples).astype(np.float32) * 0.1

        mel = compute_mel_spectrogram(audio, sr, n_fft=n_fft, hop_length=hop_length)

        expected_frames = (samples // hop_length) + 1
        assert mel.shape[1] == pytest.approx(expected_frames, abs=1)

    def test_batch_mel_computation(self, sample_audio_data):
        """Test batch mel spectrogram computation"""
        audio, sr = sample_audio_data

        # Create batch
        batch_size = 4
        audio_batch = np.stack([audio] * batch_size)

        # Process batch
        mel_batch = []
        for audio_single in audio_batch:
            mel = compute_mel_spectrogram(audio_single, sr)
            mel_batch.append(mel)

        mel_batch = torch.stack(mel_batch)
        assert mel_batch.shape[0] == batch_size
        assert mel_batch.shape[1] == 80

    def test_audio_loading_errors(self):
        """Test error handling in audio loading"""
        # Non-existent file
        with pytest.raises(FileNotFoundError):
            load_audio("non_existent_file.wav")

        # Invalid file
        with tempfile.NamedTemporaryFile(suffix=".txt") as f:
            f.write(b"This is not an audio file")
            f.flush()
            with pytest.raises(Exception):
                load_audio(f.name)

    def test_extreme_audio_values(self):
        """Test handling of extreme audio values"""
        # Very loud audio
        loud_audio = np.full(1000, 100.0, dtype=np.float32)
        normalized = normalize_audio(loud_audio)
        assert np.abs(normalized).max() <= 1.0

        # Very quiet audio
        quiet_audio = np.full(1000, 1e-6, dtype=np.float32)
        mel = audio_to_mel(quiet_audio, 22050)
        assert not torch.isnan(mel).any()
        assert not torch.isinf(mel).any()
