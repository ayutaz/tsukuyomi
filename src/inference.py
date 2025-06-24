"""
Main inference pipeline for Tsukuyomi TTS

This module provides the main TTS inference interface that combines
all components for end-to-end text-to-speech synthesis.
"""

import json
import os
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from .frontend import JapaneseTextNormalizer
from .models import AcousticModel, AcousticModelConfig, XPhoneBERTEncoder
from .utils import save_audio, trim_silence
from .vocoder import BigVGANVocoder


class TsukuyomiTTS:
    """
    Main TTS inference class for Tsukuyomi.

    Provides end-to-end text-to-speech synthesis with Japanese language support,
    multi-speaker capabilities, and H100-optimized inference.
    """

    def __init__(
        self,
        config_path: str | None = None,
        checkpoint_path: str | None = None,
        device: str | None = None,
        use_bf16: bool = True,
        enable_h100_optimizations: bool = True,
    ):
        """
        Initialize Tsukuyomi TTS system.

        Args:
            config_path: Path to configuration file
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on (None for auto-detect)
            use_bf16: Whether to use BF16 precision
            enable_h100_optimizations: Enable H100-specific optimizations
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_bf16 = use_bf16 and self.device == "cuda"
        self.enable_h100_optimizations = enable_h100_optimizations

        # Load configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            self.config = AcousticModelConfig(**config_dict)
        else:
            # Use default configuration
            self.config = AcousticModelConfig(use_bf16=self.use_bf16)

        # Initialize components
        self._init_frontend()
        self._init_models()
        self._init_vocoder()

        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)

        # Enable H100 optimizations if requested
        if self.enable_h100_optimizations and self.device == "cuda":
            self._enable_h100_optimizations()

    def _init_frontend(self):
        """Initialize text frontend."""
        self.text_normalizer = JapaneseTextNormalizer(use_gpu=self.device == "cuda")

    def _init_models(self):
        """Initialize acoustic models."""
        # XPhoneBERT encoder
        self.phoneme_encoder = XPhoneBERTEncoder(
            device=self.device, use_bf16=self.use_bf16
        )

        # Acoustic model
        self.acoustic_model = AcousticModel(self.config).to(self.device)
        if self.use_bf16:
            self.acoustic_model = self.acoustic_model.to(dtype=torch.bfloat16)

    def _init_vocoder(self):
        """Initialize vocoder."""
        self.vocoder = BigVGANVocoder(device=self.device, use_bf16=self.use_bf16)

    def _enable_h100_optimizations(self):
        """Enable H100-specific optimizations."""
        # Check if we're on H100
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            if "H100" in device_name:
                print(f"Detected {device_name}, enabling H100 optimizations")

                # Enable TF32 for matmul operations
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

                # Try to compile model with torch.compile if available
                if hasattr(torch, "compile"):
                    print("Compiling acoustic model with torch.compile...")
                    self.acoustic_model = torch.compile(
                        self.acoustic_model, mode="reduce-overhead"
                    )
            else:
                warnings.warn(
                    f"H100 optimizations requested but device is {device_name}"
                )

    @torch.no_grad()
    def synthesize(
        self,
        text: Union[str, List[str]],
        speaker_id: Optional[int] = None,
        speed: float = 1.0,
        pitch_shift: float = 0.0,
        energy_scale: float = 1.0,
        return_mel: bool = False,
        normalize_audio: bool = True,
        trim_silence_audio: bool = True,
    ) -> Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]:
        """
        Synthesize speech from text.

        Args:
            text: Input text or list of texts
            speaker_id: Speaker ID for multi-speaker models
            speed: Speed factor (1.0 = normal speed)
            pitch_shift: Pitch shift in semitones
            energy_scale: Energy/volume scale factor
            return_mel: Whether to return mel-spectrogram along with audio
            normalize_audio: Whether to normalize output audio
            trim_silence_audio: Whether to trim silence from output

        Returns:
            If single text:
                - If return_mel=False: Audio array
                - If return_mel=True: Dict with 'audio' and 'mel' keys
            If list of texts:
                - List of audio arrays or dicts
        """
        # Handle batch vs single input
        is_batch = isinstance(text, list)
        if not is_batch:
            text = [text]

        # Process text
        processed_texts = self.text_normalizer.batch_process(text)

        # Extract phoneme sequences
        phoneme_sequences = [p["phoneme_sequence"] for p in processed_texts]

        # Encode phonemes
        phoneme_embeddings = self.phoneme_encoder(
            phoneme_sequences, return_pooled=False
        )[0]

        # Prepare speaker IDs
        if speaker_id is not None:
            speaker_ids = torch.tensor([speaker_id] * len(text), device=self.device)
        else:
            speaker_ids = None

        # Run acoustic model
        acoustic_output = self.acoustic_model(
            phoneme_embeddings=phoneme_embeddings, speaker_ids=speaker_ids
        )

        mel_outputs = acoustic_output["mel_output"]

        # Apply speed modification
        if speed != 1.0:
            mel_outputs = self._modify_speed(mel_outputs, speed)

        # Apply pitch shift if requested
        if pitch_shift != 0.0:
            mel_outputs = self._apply_pitch_shift(mel_outputs, pitch_shift)

        # Apply energy scaling
        if energy_scale != 1.0:
            mel_outputs = mel_outputs * energy_scale

        # Generate waveforms
        results = []
        for i in range(len(text)):
            # Extract mel for this utterance
            mel = mel_outputs[i].transpose(0, 1)  # (time, mel) -> (mel, time)

            # Generate waveform
            waveform = self.vocoder(mel)
            waveform_np = waveform.cpu().numpy()

            # Post-process audio
            if normalize_audio:
                max_val = np.abs(waveform_np).max()
                if max_val > 0:
                    waveform_np = waveform_np / max_val * 0.95

            if trim_silence_audio:
                waveform_np = trim_silence(
                    waveform_np, sample_rate=self.vocoder.get_sample_rate()
                )

            # Prepare result
            if return_mel:
                result = {"audio": waveform_np, "mel": mel.cpu().numpy()}
            else:
                result = waveform_np

            results.append(result)

        # Return single result if not batch
        if not is_batch:
            return results[0]
        else:
            return results

    def _modify_speed(self, mel: torch.Tensor, speed: float) -> torch.Tensor:
        """Modify speed by interpolating mel-spectrogram."""
        if speed == 1.0:
            return mel

        # Interpolate along time dimension
        original_length = mel.shape[1]
        new_length = int(original_length / speed)

        # Use linear interpolation
        mel = torch.nn.functional.interpolate(
            mel.transpose(1, 2),  # (B, T, C) -> (B, C, T)
            size=new_length,
            mode="linear",
            align_corners=False,
        ).transpose(
            1, 2
        )  # Back to (B, T, C)

        return mel

    def _apply_pitch_shift(self, mel: torch.Tensor, shift: float) -> torch.Tensor:
        """Apply pitch shift to mel-spectrogram."""
        if shift == 0.0:
            return mel

        # Simple frequency shift (this is a approximation)
        # For better quality, implement proper pitch shifting
        shift_factor = 2 ** (shift / 12.0)  # Convert semitones to frequency ratio

        # This is a placeholder - proper implementation would require
        # more sophisticated pitch shifting algorithms
        warnings.warn("Pitch shifting is currently a simple approximation")

        return mel

    def save_audio(
        self, audio: np.ndarray, path: str, sample_rate: Optional[int] = None
    ):
        """
        Save audio to file.

        Args:
            audio: Audio array
            path: Output file path
            sample_rate: Sample rate (None to use vocoder's rate)
        """
        if sample_rate is None:
            sample_rate = self.vocoder.get_sample_rate()

        save_audio(path, audio, sample_rate)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load acoustic model state
        if "acoustic_model" in checkpoint:
            self.acoustic_model.load_state_dict(checkpoint["acoustic_model"])

        # Load configuration if available
        if "config" in checkpoint:
            self.config = AcousticModelConfig(**checkpoint["config"])

        print("Checkpoint loaded successfully")

    def export_onnx(
        self, output_dir: str, opset_version: int = 15, example_text: str = "こんにちは"
    ):
        """
        Export models to ONNX format for deployment.

        Args:
            output_dir: Directory to save ONNX models
            opset_version: ONNX opset version
            example_text: Example text for tracing
        """
        os.makedirs(output_dir, exist_ok=True)

        # Prepare example inputs
        processed = self.text_normalizer.process_for_tts(example_text)
        phoneme_seq = processed["phoneme_sequence"]

        # Export phoneme encoder
        print("Exporting phoneme encoder...")
        phoneme_input = phoneme_seq
        torch.onnx.export(
            self.phoneme_encoder,
            (phoneme_input,),
            os.path.join(output_dir, "phoneme_encoder.onnx"),
            opset_version=opset_version,
            input_names=["phoneme_sequence"],
            output_names=["phoneme_embeddings"],
            dynamic_axes={
                "phoneme_sequence": {0: "batch", 1: "sequence"},
                "phoneme_embeddings": {0: "batch", 1: "sequence"},
            },
        )

        # Export acoustic model
        print("Exporting acoustic model...")
        # Note: This is simplified - full export would need more careful handling
        warnings.warn(
            "ONNX export is simplified and may need additional work for production"
        )

        print(f"Models exported to {output_dir}")

    def benchmark(
        self, text: str = "これは音声合成のベンチマークテストです。", n_runs: int = 10
    ):
        """
        Benchmark inference speed.

        Args:
            text: Text to synthesize
            n_runs: Number of runs for averaging

        Returns:
            Dict with timing information
        """
        import time

        # Warmup
        _ = self.synthesize(text)

        # Benchmark
        times = []
        for _ in range(n_runs):
            start = time.time()
            audio = self.synthesize(text)
            end = time.time()
            times.append(end - start)

        times = np.array(times)
        audio_duration = len(audio) / self.vocoder.get_sample_rate()

        return {
            "mean_time": times.mean(),
            "std_time": times.std(),
            "min_time": times.min(),
            "max_time": times.max(),
            "audio_duration": audio_duration,
            "real_time_factor": audio_duration / times.mean(),
        }
