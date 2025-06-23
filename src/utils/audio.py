"""
Audio processing utilities for Tsukuyomi TTS

Contains functions for audio I/O, mel-spectrogram computation,
and other audio processing tasks.
"""

import torch
import torchaudio
import numpy as np
from typing import Union, Optional, Tuple
import soundfile as sf
import librosa
import warnings


def load_audio(
    path: str,
    sample_rate: int = 22050,
    mono: bool = True,
    normalize: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load audio file.
    
    Args:
        path: Path to audio file
        sample_rate: Target sample rate (None to keep original)
        mono: Convert to mono
        normalize: Normalize to [-1, 1]
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    # Load with soundfile for better format support
    audio, sr = sf.read(path, dtype='float32')
    
    # Convert to mono if needed
    if mono and audio.ndim > 1:
        audio = audio.mean(axis=1)
    
    # Resample if needed
    if sample_rate is not None and sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate
    
    # Normalize
    if normalize:
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.95
    
    return audio, sr


def save_audio(
    path: str,
    audio: Union[np.ndarray, torch.Tensor],
    sample_rate: int = 22050,
    normalize: bool = True
):
    """
    Save audio to file.
    
    Args:
        path: Output file path
        audio: Audio data
        sample_rate: Sample rate
        normalize: Whether to normalize before saving
    """
    # Convert to numpy if needed
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    
    # Ensure 1D
    if audio.ndim > 1:
        audio = audio.squeeze()
    
    # Normalize
    if normalize:
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.95
    
    # Save
    sf.write(path, audio, sample_rate, subtype='PCM_16')


def mel_spectrogram(
    audio: Union[np.ndarray, torch.Tensor],
    sample_rate: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: float = 8000.0,
    center: bool = False,
    normalized: bool = False
) -> torch.Tensor:
    """
    Compute mel-spectrogram from audio.
    
    Args:
        audio: Audio waveform
        sample_rate: Sample rate
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
        n_mels: Number of mel bins
        fmin: Minimum frequency
        fmax: Maximum frequency
        center: Whether to center pad
        normalized: Whether to normalize mel-spectrogram
        
    Returns:
        Mel-spectrogram tensor of shape (n_mels, time)
    """
    # Convert to tensor if needed
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio).float()
    
    # Ensure audio is 2D (1, time) for torchaudio
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    
    # Create mel spectrogram transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        f_min=fmin,
        f_max=fmax,
        center=center,
        normalized=normalized,
        power=1.0  # Use magnitude spectrogram
    )
    
    # Compute mel spectrogram
    mel_spec = mel_transform(audio)
    
    # Convert to log scale
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
    
    # Remove batch dimension
    mel_spec = mel_spec.squeeze(0)
    
    return mel_spec


def denormalize_mel(
    mel_spec: torch.Tensor,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Denormalize mel-spectrogram.
    
    Args:
        mel_spec: Normalized mel-spectrogram
        mean: Mean values for denormalization
        std: Standard deviation values for denormalization
        
    Returns:
        Denormalized mel-spectrogram
    """
    if mean is not None and std is not None:
        # Ensure proper shapes
        if mean.ndim == 1:
            mean = mean.unsqueeze(-1)
        if std.ndim == 1:
            std = std.unsqueeze(-1)
        
        mel_spec = mel_spec * std + mean
    
    return mel_spec


def trim_silence(
    audio: Union[np.ndarray, torch.Tensor],
    sample_rate: int = 22050,
    threshold_db: float = 30.0,
    frame_length: int = 2048,
    hop_length: int = 512,
    margin: float = 0.1
) -> Union[np.ndarray, torch.Tensor]:
    """
    Trim silence from audio.
    
    Args:
        audio: Audio waveform
        sample_rate: Sample rate
        threshold_db: Silence threshold in dB
        frame_length: Frame length for energy calculation
        hop_length: Hop length for energy calculation
        margin: Margin to keep (in seconds)
        
    Returns:
        Trimmed audio
    """
    is_tensor = isinstance(audio, torch.Tensor)
    
    # Convert to numpy for librosa
    if is_tensor:
        audio_np = audio.cpu().numpy()
    else:
        audio_np = audio
    
    # Trim silence
    trimmed, _ = librosa.effects.trim(
        audio_np,
        top_db=threshold_db,
        frame_length=frame_length,
        hop_length=hop_length
    )
    
    # Add margin
    margin_samples = int(margin * sample_rate)
    if margin_samples > 0:
        trimmed = np.pad(trimmed, (margin_samples, margin_samples), mode='constant')
    
    # Convert back to tensor if needed
    if is_tensor:
        trimmed = torch.from_numpy(trimmed).to(audio.device)
    
    return trimmed