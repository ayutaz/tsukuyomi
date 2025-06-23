"""
Mock implementation of BigVGAN for testing
This is a placeholder until the actual package is available
"""

import torch
import torch.nn as nn
import numpy as np


class BigVGAN(nn.Module):
    """Mock BigVGAN vocoder for testing"""
    
    def __init__(self, model_name="nvidia/bigvgan_22khz_80band"):
        super().__init__()
        self.model_name = model_name
        self.sample_rate = 22050
        
        # Extract sample rate from model name
        if "22khz" in model_name.lower():
            self.sample_rate = 22050
        elif "24khz" in model_name.lower():
            self.sample_rate = 24000
        elif "48khz" in model_name.lower():
            self.sample_rate = 48000
        
        # Mock layers
        self.conv1 = nn.Conv1d(80, 512, 7, padding=3)
        self.conv2 = nn.Conv1d(512, 256, 7, padding=3)
        self.conv3 = nn.Conv1d(256, 1, 7, padding=3)
    
    @classmethod
    def from_pretrained(cls, model_name, use_cuda_kernel=False):
        """Load pretrained model (mock)"""
        return cls(model_name)
    
    def remove_weight_norm(self):
        """Remove weight normalization (mock)"""
        pass
    
    def forward(self, mel):
        """
        Generate waveform from mel spectrogram (mock)
        
        Args:
            mel: Mel spectrogram tensor [B, 80, T]
            
        Returns:
            Waveform tensor [B, T * hop_size]
        """
        # Simple mock generation
        batch_size = mel.shape[0]
        mel_frames = mel.shape[2]
        hop_size = 256
        
        # Generate mock waveform
        waveform_length = mel_frames * hop_size
        waveform = torch.randn(batch_size, waveform_length)
        
        return waveform