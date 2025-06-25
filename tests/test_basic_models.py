"""Basic tests for TTS models"""

import pytest
import torch

from src.models.bigvgan_v2 import BigVGANv2
from src.models.emotion_controller import EmotionController
from src.models.f0_bert import F0BERT
from src.models.matcha_tts import MatchaTTS
from src.models.style_transfer import StyleTransferModule
from src.models.vits import VITS
from src.models.voice_morphing import VoiceMorphing
from src.models.xphonebert import XPhoneBERTEncoder


class TestXPhoneBERTEncoder:
    """Test XPhoneBERT encoder model"""

    def test_initialization(self):
        """Test model initialization"""
        model = XPhoneBERTEncoder(
            model_name="vinai/xphonebert-base",
            hidden_size=768,
            num_layers=12,
            num_heads=12,
        )
        assert model is not None
        assert model.hidden_size == 768

    def test_forward_pass(self):
        """Test forward pass"""
        model = XPhoneBERTEncoder(
            model_name="vinai/xphonebert-base",
            hidden_size=768,
            num_layers=12,
            num_heads=12,
        )

        # Create dummy input
        batch_size = 2
        seq_length = 10
        phoneme_ids = torch.randint(0, 100, (batch_size, seq_length))
        language_ids = torch.randint(0, 10, (batch_size, seq_length))

        # Forward pass
        outputs = model(phoneme_ids, language_ids)

        assert "hidden_states" in outputs
        assert outputs["hidden_states"].shape == (batch_size, seq_length, 768)


class TestF0BERT:
    """Test F0-BERT model"""

    def test_initialization(self):
        """Test model initialization"""
        model = F0BERT(
            hidden_size=256,
            num_layers=6,
            num_heads=8,
            pitch_bins=256,
        )
        assert model is not None
        assert model.hidden_size == 256

    def test_forward_pass(self):
        """Test forward pass"""
        model = F0BERT(
            hidden_size=256,
            num_layers=6,
            num_heads=8,
            pitch_bins=256,
        )

        # Create dummy input
        batch_size = 2
        seq_length = 100
        f0_values = torch.randn(batch_size, seq_length)
        f0_mask = torch.ones(batch_size, seq_length)

        # Forward pass
        outputs = model(f0_values, f0_mask)

        assert "hidden_states" in outputs
        assert outputs["hidden_states"].shape == (batch_size, seq_length, 256)


class TestVITS:
    """Test VITS model"""

    def test_initialization(self):
        """Test model initialization"""
        model = VITS(
            n_vocab=256,
            n_speakers=10,
            hidden_channels=192,
        )
        assert model is not None
        assert model.n_vocab == 256
        assert model.n_speakers == 10

    def test_inference(self):
        """Test inference mode"""
        model = VITS(
            n_vocab=256,
            n_speakers=1,
            hidden_channels=192,
        )
        model.eval()

        # Create dummy input
        text = torch.randint(0, 256, (1, 20))
        text_lengths = torch.tensor([20])
        speaker_ids = torch.tensor([0])

        # Inference
        with torch.no_grad():
            outputs = model.infer(text, text_lengths, speaker_ids)

        assert "audio" in outputs
        assert outputs["audio"].dim() == 3  # [B, 1, T]


class TestMatchaTTS:
    """Test Matcha-TTS model"""

    def test_initialization(self):
        """Test model initialization"""
        model = MatchaTTS(
            n_vocab=256,
            n_speakers=10,
            hidden_channels=192,
        )
        assert model is not None
        assert model.n_vocab == 256
        assert model.n_speakers == 10

    def test_forward_pass(self):
        """Test forward pass for training"""
        model = MatchaTTS(
            n_vocab=256,
            n_speakers=1,
            hidden_channels=192,
            n_layers_enc=3,
            n_layers_dec=3,
            n_layers_flow=2,
        )

        # Create dummy input
        batch_size = 2
        text_len = 20
        mel_len = 100
        mel_dim = 80

        text = torch.randint(0, 256, (batch_size, text_len))
        text_lengths = torch.tensor([20, 18])
        mel = torch.randn(batch_size, mel_dim, mel_len)
        mel_lengths = torch.tensor([100, 90])

        # Forward pass
        losses = model(text, text_lengths, mel, mel_lengths)

        assert "loss" in losses
        assert losses["loss"].requires_grad


class TestBigVGANv2:
    """Test BigVGAN v2 vocoder"""

    def test_initialization(self):
        """Test model initialization"""
        model = BigVGANv2(
            num_mels=80,
            upsample_initial_channel=512,
        )
        assert model is not None

    def test_forward_pass(self):
        """Test forward pass"""
        model = BigVGANv2(
            num_mels=80,
            upsample_initial_channel=512,
        )
        model.eval()

        # Create dummy mel-spectrogram
        batch_size = 2
        mel_dim = 80
        mel_len = 100
        mel = torch.randn(batch_size, mel_dim, mel_len)

        # Forward pass
        with torch.no_grad():
            audio = model(mel)

        # Check output shape
        # Default hop_size is 256 with upsample_rates [8, 8, 2, 2]
        expected_audio_len = mel_len * 256
        assert audio.shape == (batch_size, 1, expected_audio_len)


class TestEmotionController:
    """Test emotion controller module"""

    def test_initialization(self):
        """Test model initialization"""
        model = EmotionController(
            feature_dim=768,
            num_emotions=10,
            emotion_embedding_dim=256,
        )
        assert model is not None

    def test_forward_with_emotion_id(self):
        """Test forward pass with emotion ID"""
        model = EmotionController(
            feature_dim=768,
            num_emotions=10,
            emotion_embedding_dim=256,
        )

        # Create dummy input
        batch_size = 2
        seq_length = 100
        features = torch.randn(batch_size, seq_length, 768)
        emotion_id = torch.tensor([3, 7])

        # Forward pass
        outputs = model(features, emotion_id=emotion_id)

        assert "features" in outputs
        assert outputs["features"].shape == features.shape


class TestStyleTransfer:
    """Test style transfer module"""

    def test_initialization(self):
        """Test model initialization"""
        model = StyleTransferModule(
            feature_dim=768,
            style_dim=256,
            num_adapter_layers=4,
        )
        assert model is not None

    def test_forward_with_reference(self):
        """Test forward pass with reference mel"""
        model = StyleTransferModule(
            feature_dim=768,
            style_dim=256,
        )

        # Create dummy input
        batch_size = 2
        seq_length = 100
        features = torch.randn(batch_size, seq_length, 768)
        reference_mel = torch.randn(batch_size, 80, 200)

        # Forward pass
        outputs = model(features, reference_mel=reference_mel)

        assert "features" in outputs
        assert outputs["features"].shape == features.shape


class TestVoiceMorphing:
    """Test voice morphing module"""

    def test_initialization(self):
        """Test model initialization"""
        model = VoiceMorphing(
            input_dim=80,
            content_dim=256,
            speaker_dim=256,
        )
        assert model is not None

    def test_morph_two_voices(self):
        """Test morphing between two voices"""
        model = VoiceMorphing(
            input_dim=80,
            content_dim=256,
            speaker_dim=256,
        )
        model.eval()

        # Create dummy mel-spectrograms
        batch_size = 1
        mel_dim = 80
        mel_len = 100

        mel1 = torch.randn(batch_size, mel_dim, mel_len)
        mel2 = torch.randn(batch_size, mel_dim, mel_len)

        # Morph with equal weights
        weights = torch.tensor([[0.5, 0.5]])

        with torch.no_grad():
            morphed = model.morph([mel1, mel2], weights)

        assert morphed.shape == mel1.shape
