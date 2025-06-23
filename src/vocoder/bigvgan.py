"""
BigVGAN vocoder wrapper for high-quality waveform generation

This module provides a wrapper around the BigVGAN vocoder for converting
mel-spectrograms to high-quality audio waveforms.
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
import numpy as np

# Import bigvgan from src directory
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import bigvgan
import warnings


class BigVGANVocoder(nn.Module):
    """
    BigVGAN vocoder wrapper for Tsukuyomi TTS.
    
    Converts mel-spectrograms to high-quality 22.05kHz audio waveforms
    using the NVIDIA BigVGAN model.
    """
    
    def __init__(
        self,
        model_name: str = "nvidia/bigvgan_22khz_80band",
        device: Optional[str] = None,
        use_bf16: bool = True,
        use_cuda_kernel: bool = False
    ):
        """
        Initialize BigVGAN vocoder.
        
        Args:
            model_name: Model identifier for BigVGAN
            device: Device to run the model on (None for auto-detect)
            use_bf16: Whether to use BF16 precision
            use_cuda_kernel: Whether to use CUDA kernels (requires compilation)
        """
        super().__init__()
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_bf16 = use_bf16 and self.device == "cuda"
        self.model_name = model_name
        
        # Try to import BigVGAN
        try:
            import bigvgan
            self.bigvgan_available = True
        except ImportError:
            warnings.warn(
                "BigVGAN not installed. Please install it with: "
                "pip install git+https://github.com/NVIDIA/BigVGAN.git"
            )
            self.bigvgan_available = False
            self.model = None
            return
        
        # Load BigVGAN model
        try:
            self.model = bigvgan.BigVGAN.from_pretrained(
                model_name,
                use_cuda_kernel=use_cuda_kernel
            )
            
            # Move to device and set precision
            self.model = self.model.to(self.device)
            if self.use_bf16:
                # Note: BigVGAN may not fully support BF16, so we keep it in FP32
                # but can still benefit from BF16 inputs
                pass
            
            # Set to eval mode
            self.model.eval()
            
            # Disable gradients
            for param in self.model.parameters():
                param.requires_grad = False
                
        except Exception as e:
            warnings.warn(f"Failed to load BigVGAN model: {e}")
            self.model = None
            
    def forward(
        self,
        mel_spectrogram: torch.Tensor,
        return_sample_rate: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
        """
        Convert mel-spectrogram to waveform.
        
        Args:
            mel_spectrogram: Mel-spectrogram tensor of shape (B, n_mels, T) or (n_mels, T)
            return_sample_rate: Whether to return the sample rate along with waveform
            
        Returns:
            If return_sample_rate=False: Waveform tensor of shape (B, T_audio) or (T_audio,)
            If return_sample_rate=True: Tuple of (waveform, sample_rate)
        """
        if self.model is None:
            raise RuntimeError(
                "BigVGAN model not loaded. Please check installation and initialization."
            )
        
        # Handle 2D input (single spectrogram)
        input_ndim = mel_spectrogram.ndim
        if input_ndim == 2:
            mel_spectrogram = mel_spectrogram.unsqueeze(0)
        
        # Ensure input is on correct device
        mel_spectrogram = mel_spectrogram.to(self.device)
        
        # Generate waveform
        with torch.no_grad():
            # BigVGAN expects mel-spectrogram in shape (B, n_mels, T)
            if self.use_bf16 and mel_spectrogram.dtype != torch.bfloat16:
                mel_spectrogram = mel_spectrogram.to(torch.bfloat16)
            
            # Run vocoder
            if hasattr(self.model, 'inference'):
                waveform = self.model.inference(mel_spectrogram)
            else:
                waveform = self.model(mel_spectrogram)
        
        # Remove batch dimension if input was 2D
        if input_ndim == 2:
            waveform = waveform.squeeze(0)
        
        # Return with sample rate if requested
        if return_sample_rate:
            return waveform, 22050  # BigVGAN 22kHz model
        else:
            return waveform
    
    def mel_spectrogram_to_waveform(
        self,
        mel_spectrogram: np.ndarray,
        normalize: bool = True,
        pad_mode: str = "reflect"
    ) -> np.ndarray:
        """
        Convert mel-spectrogram to waveform with preprocessing.
        
        Args:
            mel_spectrogram: Numpy array of mel-spectrogram
            normalize: Whether to normalize the output waveform
            pad_mode: Padding mode for mel-spectrogram
            
        Returns:
            Numpy array of audio waveform
        """
        # Convert to tensor
        mel_tensor = torch.from_numpy(mel_spectrogram).float()
        
        # Pad if needed (BigVGAN benefits from padding for better quality at boundaries)
        if pad_mode != "none":
            pad_length = 10  # Pad 10 frames on each side
            if mel_tensor.ndim == 2:
                mel_tensor = torch.nn.functional.pad(
                    mel_tensor, (pad_length, pad_length), mode=pad_mode
                )
            else:
                mel_tensor = torch.nn.functional.pad(
                    mel_tensor, (pad_length, pad_length), mode=pad_mode
                )
        
        # Generate waveform
        waveform = self.forward(mel_tensor)
        
        # Remove padding from waveform if applied
        if pad_mode != "none":
            # Calculate samples to remove (hop_size * pad_length)
            hop_size = 256  # Standard hop size for 22kHz
            trim_length = hop_size * pad_length
            if waveform.ndim == 1:
                waveform = waveform[trim_length:-trim_length]
            else:
                waveform = waveform[..., trim_length:-trim_length]
        
        # Convert to numpy
        waveform_np = waveform.cpu().numpy()
        
        # Normalize if requested
        if normalize:
            max_val = np.abs(waveform_np).max()
            if max_val > 0:
                waveform_np = waveform_np / max_val * 0.95  # Leave some headroom
        
        return waveform_np
    
    def get_sample_rate(self) -> int:
        """Get the sample rate of the vocoder output."""
        return 22050
    
    def get_hop_size(self) -> int:
        """Get the hop size used by the vocoder."""
        return 256  # Standard for 22kHz BigVGAN
    
    def supports_batch_inference(self) -> bool:
        """Check if the vocoder supports batch inference."""
        return True
    
    @torch.no_grad()
    def inference_onnx(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Inference method compatible with ONNX export.
        
        Args:
            mel_spectrogram: Input mel-spectrogram
            
        Returns:
            Generated waveform
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
            
        # Ensure the model is in eval mode
        self.model.eval()
        
        # Run inference without autocast for ONNX compatibility
        return self.model(mel_spectrogram)