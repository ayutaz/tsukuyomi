"""
Pytest configuration and shared fixtures for Tsukuyomi tests
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@pytest.fixture
def sample_text_ja():
    """Sample Japanese text for testing"""
    return "こんにちは、テストです。"


@pytest.fixture
def sample_text_en():
    """Sample English text for testing"""
    return "Hello, this is a test."


@pytest.fixture
def sample_texts():
    """Sample texts in multiple languages"""
    return {
        "ja": ["こんにちは", "テストです", "音声合成"],
        "en": ["Hello", "Test", "Speech synthesis"],
        "zh": ["你好", "测试", "语音合成"],
    }


@pytest.fixture
def sample_mel_spectrogram():
    """Create a sample mel spectrogram"""
    # 80 mel bins, 100 time frames
    mel = torch.randn(1, 80, 100)
    return mel


@pytest.fixture
def sample_audio():
    """Create sample audio data"""
    # 1 second of audio at 22050 Hz
    audio = np.random.randn(22050).astype(np.float32)
    return audio, 22050


@pytest.fixture
def sample_phonemes():
    """Sample phoneme sequence"""
    return "k o n n i ch i w a"


@pytest.fixture
def mock_model_config():
    """Mock model configuration"""
    return {
        "model": {
            "acoustic": {
                "hidden_channels": 192,
                "n_layers": 6,
                "n_heads": 2,
                "kernel_size": 3,
                "dropout_rate": 0.1,
            },
            "speaker": {
                "n_speakers": 10,
                "speaker_embed_dim": 256,
            },
        },
        "audio": {
            "sample_rate": 22050,
            "n_mels": 80,
            "n_fft": 1024,
            "hop_length": 256,
            "win_length": 1024,
        },
    }


@pytest.fixture
def bf16_available():
    """Check if BF16 is available"""
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


# Markers for conditional tests
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu: mark test to run only when GPU is available"
    )
    config.addinivalue_line(
        "markers", "bf16: mark test to run only when BF16 is supported"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests based on markers and available hardware"""
    skip_gpu = pytest.mark.skip(reason="GPU not available")
    skip_bf16 = pytest.mark.skip(reason="BF16 not supported")
    
    for item in items:
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)
        if "bf16" in item.keywords and not (
            torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        ):
            item.add_marker(skip_bf16)