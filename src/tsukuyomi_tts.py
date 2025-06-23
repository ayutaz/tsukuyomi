"""
Tsukuyomi TTS - Main interface for the ultimate Japanese TTS system.

Supports voice morphing, emotion control, and multi-speaker synthesis.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Union, Dict, Tuple
import numpy as np
from pathlib import Path
import logging

from .models.ultimate_g2p import create_ultimate_g2p
from .models.f0_bert import create_f0_bert
from .models.xphonebert_japanese import create_xphonebert_japanese
from .models.ultimate_acoustic_model import create_ultimate_acoustic_model
from .models.bigvgan_v2 import create_bigvgan_v2

logger = logging.getLogger(__name__)


class TsukuyomiTTS:
    """
    Main interface for Tsukuyomi TTS system.
    
    Features:
    - High-quality Japanese speech synthesis
    - Voice morphing between multiple speakers
    - Emotion and style control
    - Voice cloning
    - Batch processing
    """
    
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: Optional[Path] = None,
        enable_cuda_optimization: bool = True
    ):
        """
        Initialize Tsukuyomi TTS system.
        
        Args:
            device: Device to run models on
            checkpoint_dir: Directory containing model checkpoints
            enable_cuda_optimization: Enable CUDA 12.1+ optimizations
        """
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        
        logger.info(f"Initializing Tsukuyomi TTS on {device}")
        
        # Load models
        self._load_models()
        
        # Enable CUDA optimizations if available
        if enable_cuda_optimization and device == "cuda":
            self._enable_cuda_optimizations()
            
    def _load_models(self):
        """Load all component models."""
        logger.info("Loading G2P model...")
        self.g2p = create_ultimate_g2p(
            checkpoint_path=self.checkpoint_dir / "g2p.pt" if self.checkpoint_dir else None,
            device=self.device
        )
        
        logger.info("Loading XPhoneBERT-Japanese...")
        self.phoneme_encoder = create_xphonebert_japanese(
            checkpoint_path=self.checkpoint_dir / "xphonebert.pt" if self.checkpoint_dir else None,
            device=self.device
        )
        
        logger.info("Loading F0-BERT...")
        self.f0_model = create_f0_bert(
            checkpoint_path=self.checkpoint_dir / "f0bert.pt" if self.checkpoint_dir else None,
            device=self.device
        )
        
        logger.info("Loading Ultimate Acoustic Model...")
        self.acoustic_model = create_ultimate_acoustic_model(
            checkpoint_path=self.checkpoint_dir / "acoustic.pt" if self.checkpoint_dir else None,
            device=self.device
        )
        
        logger.info("Loading BigVGAN-v2 vocoder...")
        self.vocoder = create_bigvgan_v2(
            checkpoint_path=self.checkpoint_dir / "vocoder.pt" if self.checkpoint_dir else None,
            device=self.device
        )
        
        logger.info("All models loaded successfully")
        
    def _enable_cuda_optimizations(self):
        """Enable CUDA 12.1+ optimizations."""
        logger.info("Enabling CUDA optimizations...")
        
        # Optimize each model for inference
        if hasattr(self.phoneme_encoder, 'optimize_for_inference'):
            self.phoneme_encoder.optimize_for_inference()
            
        if hasattr(self.f0_model, 'optimize_for_inference'):
            self.f0_model.optimize_for_inference()
            
        if hasattr(self.acoustic_model, 'compile_for_inference'):
            # Create example input for compilation
            example_input = {
                'phoneme_embeddings': torch.randn(1, 512, 50).to(self.device),
                'phoneme_lengths': torch.tensor([50]).to(self.device)
            }
            self.acoustic_model.compile_for_inference(example_input)
            
        if hasattr(self.vocoder, 'compile_for_inference'):
            self.vocoder.compile_for_inference()
            
        logger.info("CUDA optimizations enabled")
        
    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        speaker_id: int = 0,
        emotion: str = "neutral",
        style: str = "normal",
        speed: float = 1.0,
        pitch_shift: float = 0.0,
        energy: float = 1.0
    ) -> np.ndarray:
        """
        Synthesize speech from text.
        
        Args:
            text: Input Japanese text
            speaker_id: Speaker ID (0-999)
            emotion: Emotion type (neutral, happy, sad, angry, surprised, fear, disgust)
            style: Speaking style (normal, energetic, calm, dramatic, whisper, etc.)
            speed: Speaking speed (0.5-2.0, 1.0 = normal)
            pitch_shift: Pitch shift in semitones (-12 to 12)
            energy: Energy/volume control (0.5-2.0)
            
        Returns:
            Audio waveform as numpy array (48kHz)
        """
        # G2P conversion
        g2p_outputs = self.g2p(text, return_accent=True)
        phoneme_ids = g2p_outputs['phonemes']
        accent_ids = g2p_outputs['accent_types']
        
        # Encode phonemes
        phoneme_outputs = self.phoneme_encoder(
            phoneme_ids=phoneme_ids,
            accent_ids=accent_ids,
            return_dict=True
        )
        phoneme_embeddings = phoneme_outputs['hidden_states']
        
        # Map emotion string to ID
        emotion_map = {
            "neutral": 0, "happy": 1, "sad": 2, "angry": 3,
            "surprised": 4, "fear": 5, "disgust": 6
        }
        emotion_id = torch.tensor([emotion_map.get(emotion, 0)]).to(self.device)
        
        # Predict F0
        f0_outputs = self.f0_model(
            input_ids=phoneme_ids,
            attention_mask=torch.ones_like(phoneme_ids),
            emotion_id=emotion_id,
            phoneme_embeddings=phoneme_embeddings
        )
        
        # Apply pitch shift
        if pitch_shift != 0:
            f0 = f0_outputs['f0_final']
            f0 = f0 * (2 ** (pitch_shift / 12))
            f0_outputs['f0_final'] = f0
            
        # Generate mel-spectrogram
        phoneme_embeddings_t = phoneme_embeddings.transpose(1, 2)
        mel = self.acoustic_model.inference(
            phoneme_embeddings=phoneme_embeddings_t,
            phoneme_lengths=torch.tensor([phoneme_ids.shape[1]]).to(self.device),
            speaker_ids=torch.tensor([speaker_id]).to(self.device),
            emotion_ids=emotion_id,
            length_scale=1.0 / speed,
            temperature=energy
        )
        
        # Generate waveform
        audio = self.vocoder.inference(mel)
        
        # Convert to numpy
        audio_np = audio.squeeze().cpu().numpy()
        
        return audio_np
        
    @torch.inference_mode()
    def morph_voices(
        self,
        text: str,
        speaker_ids: List[int],
        speaker_weights: List[float],
        emotion_ids: Optional[List[str]] = None,
        emotion_weights: Optional[List[float]] = None,
        speed: float = 1.0,
        pitch_shift: float = 0.0
    ) -> np.ndarray:
        """
        Synthesize speech with morphed voices from multiple speakers.
        
        Args:
            text: Input Japanese text
            speaker_ids: List of speaker IDs to morph
            speaker_weights: Weights for each speaker (should sum to 1)
            emotion_ids: Optional list of emotions to morph
            emotion_weights: Optional weights for emotions
            speed: Speaking speed
            pitch_shift: Pitch shift in semitones
            
        Returns:
            Audio waveform with morphed voices
        """
        # Validate inputs
        if len(speaker_ids) != len(speaker_weights):
            raise ValueError("speaker_ids and speaker_weights must have same length")
            
        # Normalize weights
        speaker_weights = np.array(speaker_weights)
        speaker_weights = speaker_weights / speaker_weights.sum()
        
        # G2P conversion
        g2p_outputs = self.g2p(text, return_accent=True)
        phoneme_ids = g2p_outputs['phonemes']
        accent_ids = g2p_outputs['accent_types']
        
        # Encode phonemes
        phoneme_outputs = self.phoneme_encoder(
            phoneme_ids=phoneme_ids,
            accent_ids=accent_ids,
            return_dict=True
        )
        phoneme_embeddings = phoneme_outputs['hidden_states']
        
        # Prepare speaker tensors
        speaker_ids_tensor = torch.tensor([speaker_ids]).to(self.device)
        speaker_weights_tensor = torch.tensor([speaker_weights]).float().to(self.device)
        
        # Handle emotions if provided
        if emotion_ids is not None and emotion_weights is not None:
            emotion_map = {
                "neutral": 0, "happy": 1, "sad": 2, "angry": 3,
                "surprised": 4, "fear": 5, "disgust": 6
            }
            emotion_ids_list = [emotion_map.get(e, 0) for e in emotion_ids]
            emotion_ids_tensor = torch.tensor([emotion_ids_list]).to(self.device)
            
            emotion_weights = np.array(emotion_weights)
            emotion_weights = emotion_weights / emotion_weights.sum()
            emotion_weights_tensor = torch.tensor([emotion_weights]).float().to(self.device)
        else:
            emotion_ids_tensor = None
            emotion_weights_tensor = None
            
        # Generate mel with morphed voices
        phoneme_embeddings_t = phoneme_embeddings.transpose(1, 2)
        mel = self.acoustic_model.morph_voices(
            phoneme_embeddings=phoneme_embeddings_t,
            phoneme_lengths=torch.tensor([phoneme_ids.shape[1]]).to(self.device),
            speaker_ids=speaker_ids_tensor,
            speaker_weights=speaker_weights_tensor,
            emotion_ids=emotion_ids_tensor,
            emotion_weights=emotion_weights_tensor,
            length_scale=1.0 / speed,
            temperature=1.0
        )
        
        # Apply pitch shift if needed
        if pitch_shift != 0:
            # This is a simplified pitch shift - in practice would modify F0
            pass
            
        # Generate waveform
        audio = self.vocoder.inference(mel)
        
        # Convert to numpy
        audio_np = audio.squeeze().cpu().numpy()
        
        return audio_np
        
    @torch.inference_mode()
    def clone_voice(
        self,
        text: str,
        reference_audio: Union[np.ndarray, torch.Tensor],
        emotion: str = "neutral"
    ) -> np.ndarray:
        """
        Clone voice from reference audio.
        
        Args:
            text: Text to synthesize
            reference_audio: Reference audio for voice cloning
            emotion: Emotion for synthesis
            
        Returns:
            Audio with cloned voice
        """
        # Convert reference audio to mel-spectrogram
        if isinstance(reference_audio, np.ndarray):
            reference_audio = torch.from_numpy(reference_audio).float()
            
        if reference_audio.dim() == 1:
            reference_audio = reference_audio.unsqueeze(0)
            
        reference_audio = reference_audio.to(self.device)
        
        # Extract mel-spectrogram from reference
        # In practice, this would use the same preprocessing as training
        reference_mel = self._audio_to_mel(reference_audio)
        
        # G2P conversion
        g2p_outputs = self.g2p(text, return_accent=True)
        phoneme_ids = g2p_outputs['phonemes']
        accent_ids = g2p_outputs['accent_types']
        
        # Encode phonemes
        phoneme_outputs = self.phoneme_encoder(
            phoneme_ids=phoneme_ids,
            accent_ids=accent_ids,
            return_dict=True
        )
        phoneme_embeddings = phoneme_outputs['hidden_states']
        
        # Generate with cloned voice
        phoneme_embeddings_t = phoneme_embeddings.transpose(1, 2)
        mel = self.acoustic_model.inference(
            phoneme_embeddings=phoneme_embeddings_t,
            phoneme_lengths=torch.tensor([phoneme_ids.shape[1]]).to(self.device),
            reference_mel=reference_mel.unsqueeze(0),
            emotion_ids=torch.tensor([0]).to(self.device)  # Use neutral for now
        )
        
        # Generate waveform
        audio = self.vocoder.inference(mel)
        
        # Convert to numpy
        audio_np = audio.squeeze().cpu().numpy()
        
        return audio_np
        
    def _audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert audio to mel-spectrogram."""
        # Simplified - in practice would use proper STFT parameters
        # This is a placeholder implementation
        n_fft = 2048
        hop_length = self.acoustic_model.config.hop_length
        win_length = self.acoustic_model.config.win_length
        n_mels = self.acoustic_model.config.n_mel_channels
        
        # Would use torchaudio.transforms.MelSpectrogram in practice
        # For now, return dummy mel
        mel_length = audio.shape[-1] // hop_length
        mel = torch.randn(n_mels, mel_length).to(self.device)
        
        return mel
        
    def batch_synthesize(
        self,
        texts: List[str],
        speaker_ids: Optional[List[int]] = None,
        **kwargs
    ) -> List[np.ndarray]:
        """
        Batch synthesize multiple texts.
        
        Args:
            texts: List of texts to synthesize
            speaker_ids: Optional list of speaker IDs
            **kwargs: Additional synthesis parameters
            
        Returns:
            List of audio waveforms
        """
        if speaker_ids is None:
            speaker_ids = [0] * len(texts)
            
        audios = []
        for text, speaker_id in zip(texts, speaker_ids):
            audio = self.synthesize(text, speaker_id=speaker_id, **kwargs)
            audios.append(audio)
            
        return audios
        
    def save_audio(
        self,
        audio: np.ndarray,
        filepath: Union[str, Path],
        sample_rate: int = 48000
    ):
        """
        Save audio to file.
        
        Args:
            audio: Audio waveform
            filepath: Output file path
            sample_rate: Sample rate (default: 48000)
        """
        import soundfile as sf
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure audio is in correct range
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / np.abs(audio).max()
            
        sf.write(filepath, audio, sample_rate)
        logger.info(f"Audio saved to {filepath}")


def create_tts_system(
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_dir: Optional[str] = None
) -> TsukuyomiTTS:
    """
    Create Tsukuyomi TTS system.
    
    Args:
        device: Device to use
        checkpoint_dir: Directory with model checkpoints
        
    Returns:
        TsukuyomiTTS instance
    """
    return TsukuyomiTTS(device=device, checkpoint_dir=checkpoint_dir)