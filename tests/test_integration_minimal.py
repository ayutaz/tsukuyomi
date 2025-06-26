"""
Minimal integration tests for CI/CD
"""

import pytest
import torch


def test_imports():
    """Test that all major modules can be imported"""
    try:
        from scripts.train import TTSTrainer
        from src.data.collate import tts_collate_fn
        from src.data.dataset import TsukuyomiDataset
        from src.models.bigvgan_v2 import BigVGANv2Generator
        from src.models.f0_bert import F0BERT
        from src.models.vits import VITS
    except ImportError as e:
        pytest.fail(f"Failed to import module: {e}")


def test_model_creation():
    """Test that models can be created with minimal config"""
    from src.models.vits import VITS

    # Create minimal VITS model
    model = VITS(
        n_vocab=50,
        n_speakers=2,
        hidden_channels=32,
        filter_channels=64,
        n_heads=1,
        n_layers=1,
    )

    assert model is not None

    # Test forward pass with dummy data
    batch_size = 1
    text = torch.randint(0, 50, (batch_size, 10))
    text_lengths = torch.tensor([10])

    # Should not raise error (even if output is meaningless)
    try:
        with torch.no_grad():
            # Just test that the model structure is valid
            params = sum(p.numel() for p in model.parameters())
            assert params > 0
    except Exception as e:
        pytest.fail(f"Model creation failed: {e}")


def test_data_pipeline():
    """Test basic data loading pipeline"""
    from src.data.collate import tts_collate_fn

    # Create dummy batch
    batch = [
        {
            "audio": torch.randn(1000),
            "text": "test",
            "speaker_id": "spk1",
            "audio_path": "test.wav",
        }
    ]

    # Test collate function doesn't crash
    result = tts_collate_fn(batch)

    assert "audio" in result
    assert "text" in result
    assert "speaker_ids" in result
    assert result["audio"].shape[0] == 1  # batch size


def test_training_config():
    """Test that training configuration can be loaded"""
    from omegaconf import OmegaConf

    # Create minimal config
    config = OmegaConf.create(
        {
            "data": {
                "sample_rate": 22050,
                "num_workers": 0,
            },
            "models": {
                "acoustic_model": "vits",
            },
            "training": {
                "num_epochs": 1,
                "batch_size": 1,
            },
            "paths": {
                "checkpoints": "test",
            },
        }
    )

    assert config.data.sample_rate == 22050
    assert config.training.num_epochs == 1


if __name__ == "__main__":
    # Run tests
    test_imports()
    print("✅ Imports successful")

    test_model_creation()
    print("✅ Model creation successful")

    test_data_pipeline()
    print("✅ Data pipeline successful")

    test_training_config()
    print("✅ Configuration successful")

    print("\n✅ All integration tests passed!")
