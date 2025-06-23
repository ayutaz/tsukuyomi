"""
Tests for model modules (XPhoneBERT, Acoustic Model)
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock

from src.models.xphonebert import XPhoneBERTWrapper
from src.models.acoustic_model import (
    ConformerBlock,
    TsukuyomiAcousticModel,
    DurationPredictor,
    SpeakerEncoder,
)


class TestXPhoneBERTWrapper:
    """Test XPhoneBERT wrapper"""
    
    @pytest.fixture
    def mock_transformers(self):
        """Mock transformers imports"""
        with patch('src.models.xphonebert.AutoModel') as mock_model, \
             patch('src.models.xphonebert.AutoTokenizer') as mock_tokenizer, \
             patch('src.models.xphonebert.Text2PhonemeSequence') as mock_phonemizer:
            
            # Mock model
            mock_model_instance = MagicMock()
            mock_model_instance.eval.return_value = mock_model_instance
            mock_model_instance.to.return_value = mock_model_instance
            mock_model.from_pretrained.return_value = mock_model_instance
            
            # Mock tokenizer
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            # Mock phonemizer
            mock_phonemizer_instance = MagicMock()
            mock_phonemizer.return_value = mock_phonemizer_instance
            
            yield {
                'model': mock_model,
                'tokenizer': mock_tokenizer,
                'phonemizer': mock_phonemizer,
                'model_instance': mock_model_instance,
                'tokenizer_instance': mock_tokenizer_instance,
                'phonemizer_instance': mock_phonemizer_instance,
            }
    
    def test_initialization(self, mock_transformers):
        """Test XPhoneBERT initialization"""
        wrapper = XPhoneBERTWrapper(device='cpu')
        
        assert wrapper.device == torch.device('cpu')
        mock_transformers['model'].from_pretrained.assert_called_once()
        mock_transformers['tokenizer'].from_pretrained.assert_called_once()
    
    def test_encode_text(self, mock_transformers):
        """Test text encoding"""
        wrapper = XPhoneBERTWrapper(device='cpu')
        
        # Mock phonemizer output
        mock_transformers['phonemizer_instance'].infer_sentence.return_value = "k o n n i ch i w a"
        
        # Mock tokenizer output
        mock_inputs = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]]),
        }
        mock_transformers['tokenizer_instance'].return_value = mock_inputs
        
        # Mock model output
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 5, 768)
        mock_transformers['model_instance'].return_value = mock_output
        
        # Test encoding
        text = "こんにちは"
        embeddings = wrapper.encode(text, language='ja')
        
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (1, 5, 768)
    
    def test_batch_encode(self, mock_transformers):
        """Test batch encoding"""
        wrapper = XPhoneBERTWrapper(device='cpu')
        
        # Mock outputs
        mock_transformers['phonemizer_instance'].infer_sentence.return_value = "t e s t"
        mock_inputs = {
            'input_ids': torch.tensor([[1, 2, 3, 4]] * 3),
            'attention_mask': torch.tensor([[1, 1, 1, 1]] * 3),
        }
        mock_transformers['tokenizer_instance'].return_value = mock_inputs
        
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(3, 4, 768)
        mock_transformers['model_instance'].return_value = mock_output
        
        # Test batch encoding
        texts = ["test1", "test2", "test3"]
        embeddings = wrapper.batch_encode(texts, language='ja')
        
        assert embeddings.shape == (3, 4, 768)
    
    @pytest.mark.gpu
    def test_gpu_encoding(self, mock_transformers):
        """Test GPU encoding"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        wrapper = XPhoneBERTWrapper(device='cuda')
        assert wrapper.device.type == 'cuda'
    
    @pytest.mark.bf16
    def test_bf16_encoding(self, mock_transformers):
        """Test BF16 encoding"""
        wrapper = XPhoneBERTWrapper(device='cuda', use_bf16=True)
        
        # Mock BF16 output
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 5, 768, dtype=torch.bfloat16)
        mock_transformers['model_instance'].return_value = mock_output
        
        embeddings = wrapper.encode("test")
        assert embeddings.dtype == torch.bfloat16


class TestAcousticModel:
    """Test Acoustic Model components"""
    
    @pytest.fixture
    def model_config(self):
        """Small model config for testing"""
        return {
            'hidden_channels': 128,
            'n_layers': 2,
            'n_heads': 4,
            'kernel_size': 3,
            'dropout_rate': 0.1,
            'n_speakers': 10,
        }
    
    def test_conformer_block(self, model_config):
        """Test Conformer block"""
        block = ConformerBlock(
            channels=model_config['hidden_channels'],
            kernel_size=model_config['kernel_size'],
            dropout_rate=model_config['dropout_rate'],
            n_heads=model_config['n_heads'],
        )
        
        # Test forward pass
        x = torch.randn(2, 128, 50)  # batch, channels, time
        output = block(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_duration_predictor(self):
        """Test duration predictor"""
        predictor = DurationPredictor(
            channels=128,
            kernel_size=3,
            dropout_rate=0.1,
        )
        
        x = torch.randn(2, 128, 50)
        x_mask = torch.ones(2, 1, 50)
        
        log_durations = predictor(x, x_mask)
        
        assert log_durations.shape == (2, 1, 50)
        assert not torch.isnan(log_durations).any()
    
    def test_speaker_encoder(self, model_config):
        """Test speaker encoder"""
        encoder = SpeakerEncoder(
            n_speakers=model_config['n_speakers'],
            speaker_embed_dim=256,
            output_dim=128,
        )
        
        speaker_ids = torch.tensor([0, 5, 9])
        embeddings = encoder(speaker_ids)
        
        assert embeddings.shape == (3, 128)
        assert not torch.isnan(embeddings).any()
    
    def test_acoustic_model_init(self, model_config):
        """Test acoustic model initialization"""
        model = TsukuyomiAcousticModel(
            hidden_channels=model_config['hidden_channels'],
            n_layers=model_config['n_layers'],
            n_heads=model_config['n_heads'],
            kernel_size=model_config['kernel_size'],
            dropout_rate=model_config['dropout_rate'],
            n_speakers=model_config['n_speakers'],
        )
        
        # Check components
        assert len(model.encoder_blocks) == model_config['n_layers']
        assert len(model.decoder_blocks) == model_config['n_layers']
        assert model.speaker_encoder is not None
        assert model.duration_predictor is not None
    
    def test_acoustic_model_forward(self, model_config):
        """Test acoustic model forward pass"""
        model = TsukuyomiAcousticModel(
            hidden_channels=model_config['hidden_channels'],
            n_layers=model_config['n_layers'],
            n_heads=model_config['n_heads'],
            kernel_size=model_config['kernel_size'],
            dropout_rate=model_config['dropout_rate'],
            n_speakers=model_config['n_speakers'],
        )
        model.eval()
        
        # Prepare inputs
        phoneme_embeddings = torch.randn(2, 50, 768)  # From XPhoneBERT
        speaker_ids = torch.tensor([0, 1])
        
        with torch.no_grad():
            mel_output, log_durations = model(phoneme_embeddings, speaker_ids)
        
        assert mel_output.shape[0] == 2  # Batch size
        assert mel_output.shape[1] == 80  # Mel bins
        assert not torch.isnan(mel_output).any()
        assert not torch.isnan(log_durations).any()
    
    @pytest.mark.gpu
    @pytest.mark.bf16
    def test_bf16_forward(self, model_config):
        """Test BF16 forward pass"""
        if not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
            pytest.skip("BF16 not supported")
        
        model = TsukuyomiAcousticModel(
            hidden_channels=model_config['hidden_channels'],
            n_layers=model_config['n_layers'],
            n_heads=model_config['n_heads'],
            kernel_size=model_config['kernel_size'],
            dropout_rate=model_config['dropout_rate'],
            n_speakers=model_config['n_speakers'],
            use_bf16=True,
        ).cuda()
        model.eval()
        
        # BF16 inputs
        phoneme_embeddings = torch.randn(2, 50, 768, dtype=torch.bfloat16).cuda()
        speaker_ids = torch.tensor([0, 1]).cuda()
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            mel_output, log_durations = model(phoneme_embeddings, speaker_ids)
        
        assert mel_output.dtype == torch.bfloat16
        assert not torch.isnan(mel_output).any()
    
    def test_gradient_checkpointing(self, model_config):
        """Test gradient checkpointing"""
        model = TsukuyomiAcousticModel(
            hidden_channels=model_config['hidden_channels'],
            n_layers=model_config['n_layers'],
            n_heads=model_config['n_heads'],
            kernel_size=model_config['kernel_size'],
            dropout_rate=model_config['dropout_rate'],
            n_speakers=model_config['n_speakers'],
            use_gradient_checkpointing=True,
        )
        
        # Should work without errors
        phoneme_embeddings = torch.randn(1, 10, 768, requires_grad=True)
        speaker_ids = torch.tensor([0])
        
        mel_output, log_durations = model(phoneme_embeddings, speaker_ids)
        loss = mel_output.mean()
        loss.backward()
        
        assert phoneme_embeddings.grad is not None
    
    @pytest.mark.parametrize("batch_size,seq_len", [
        (1, 10),
        (4, 50),
        (8, 100),
    ])
    def test_different_input_sizes(self, model_config, batch_size, seq_len):
        """Test model with different input sizes"""
        model = TsukuyomiAcousticModel(
            hidden_channels=model_config['hidden_channels'],
            n_layers=model_config['n_layers'],
            n_heads=model_config['n_heads'],
            kernel_size=model_config['kernel_size'],
            dropout_rate=model_config['dropout_rate'],
            n_speakers=model_config['n_speakers'],
        )
        model.eval()
        
        phoneme_embeddings = torch.randn(batch_size, seq_len, 768)
        speaker_ids = torch.randint(0, model_config['n_speakers'], (batch_size,))
        
        with torch.no_grad():
            mel_output, log_durations = model(phoneme_embeddings, speaker_ids)
        
        assert mel_output.shape[0] == batch_size
        assert log_durations.shape[0] == batch_size