"""Minimal test for collate function without heavy dependencies."""

import torch
from src.data.collate import tts_collate_fn


def test_collate_function():
    """Test the custom collate function"""
    # Create dummy batch data
    batch = [
        {
            "audio": torch.randn(10000),
            "text": "Hello world",
            "speaker_id": "spk1",
            "audio_path": "/path/to/audio1.wav",
        },
        {
            "audio": torch.randn(15000),
            "text": "How are you",
            "speaker_id": "spk2",
            "audio_path": "/path/to/audio2.wav",
        },
    ]

    # Test collate function
    collated = tts_collate_fn(batch)

    assert "audio" in collated
    assert "audio_lengths" in collated
    assert "text" in collated
    assert "speaker_ids" in collated

    # Check shapes
    assert collated["audio"].shape[0] == 2  # batch size
    assert collated["audio"].shape[1] == 15000  # max audio length
    assert len(collated["text"]) == 2
    assert collated["speaker_ids"].shape[0] == 2