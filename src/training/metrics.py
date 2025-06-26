"""Metrics for TTS evaluation"""

from typing import Dict, Optional

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pesq import pesq
except ImportError:
    pesq = None
    import warnings
    warnings.warn("PESQ not available. PESQ metric calculation will be disabled.")
from pystoi import stoi
from scipy.stats import pearsonr


class TrainingMetrics:
    """Metrics tracker for training"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.metrics = {
            "loss": [],
            "mel_loss": [],
            "kl_loss": [],
            "duration_loss": [],
            "pitch_loss": [],
            "gen_loss": [],
            "disc_loss": [],
        }

    def update(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]):
        """Update metrics with new values

        Args:
            outputs: Model outputs
            batch: Input batch containing targets
        """
        # Extract losses from outputs
        for key in [
            "loss",
            "mel_loss",
            "kl_loss",
            "duration_loss",
            "pitch_loss",
            "gen_loss",
            "disc_loss",
        ]:
            if key in outputs:
                value = outputs[key]
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().item()
                if key in self.metrics:
                    self.metrics[key].append(value)

        # Also support old interface with metrics_dict
        if isinstance(outputs, dict) and all(
            key in self.metrics for key in outputs
        ):
            for key, value in outputs.items():
                if key in self.metrics:
                    if isinstance(value, torch.Tensor):
                        value = value.detach().cpu().item()
                    self.metrics[key].append(value)

    def compute_average(self) -> Dict[str, float]:
        """Compute average of all metrics"""
        averages = {}
        for key, values in self.metrics.items():
            if len(values) > 0:
                averages[key] = np.mean(values)
            else:
                averages[key] = 0.0
        return averages

    def log_metrics(self, step: int, prefix: str = "train") -> Dict[str, float]:
        """Get metrics for logging"""
        averages = self.compute_average()
        logged_metrics = {}
        for key, value in averages.items():
            logged_metrics[f"{prefix}/{key}"] = value
        return logged_metrics

    def compute(self) -> Dict[str, float]:
        """Compute and return average metrics (alias for compute_average)"""
        return self.compute_average()


class MelCepstralDistortion:
    """Mel-Cepstral Distortion (MCD) metric"""

    def __init__(self, n_mfcc: int = 13):
        self.n_mfcc = n_mfcc

    def __call__(
        self,
        pred_mel: torch.Tensor,
        target_mel: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Calculate MCD between predicted and target mel-spectrograms

        Args:
            pred_mel: Predicted mel-spectrogram [B, n_mels, T]
            target_mel: Target mel-spectrogram [B, n_mels, T]
            lengths: Valid lengths for each sample [B]

        Returns:
            Average MCD in dB
        """
        pred_mel = pred_mel.detach().cpu().numpy()
        target_mel = target_mel.detach().cpu().numpy()

        if lengths is not None:
            lengths = lengths.detach().cpu().numpy()

        mcd_values = []

        for i in range(pred_mel.shape[0]):
            # Extract valid frames
            if lengths is not None:
                valid_len = int(lengths[i])
                pred = pred_mel[i, :, :valid_len]
                target = target_mel[i, :, :valid_len]
            else:
                pred = pred_mel[i]
                target = target_mel[i]

            # Convert to MFCC
            pred_mfcc = librosa.feature.mfcc(S=pred, n_mfcc=self.n_mfcc)
            target_mfcc = librosa.feature.mfcc(S=target, n_mfcc=self.n_mfcc)

            # Calculate MCD
            diff = pred_mfcc - target_mfcc
            mcd = (
                np.mean(np.sqrt(np.sum(diff**2, axis=0))) * 10 / np.log(10) * np.sqrt(2)
            )
            mcd_values.append(mcd)

        return np.mean(mcd_values)


class PitchCorrelation:
    """Pitch correlation metric"""

    def __call__(
        self,
        pred_pitch: torch.Tensor,
        target_pitch: torch.Tensor,
        voiced_mask: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Calculate correlation between predicted and target pitch

        Args:
            pred_pitch: Predicted pitch values [B, T]
            target_pitch: Target pitch values [B, T]
            voiced_mask: Mask for voiced regions [B, T]

        Returns:
            Average pitch correlation
        """
        pred_pitch = pred_pitch.detach().cpu().numpy()
        target_pitch = target_pitch.detach().cpu().numpy()

        if voiced_mask is not None:
            voiced_mask = voiced_mask.detach().cpu().numpy()

        correlations = []

        for i in range(pred_pitch.shape[0]):
            pred = pred_pitch[i]
            target = target_pitch[i]

            # Apply voiced mask
            if voiced_mask is not None:
                mask = voiced_mask[i].astype(bool)
                pred = pred[mask]
                target = target[mask]
            else:
                # Only consider voiced regions (pitch > 0)
                mask = (target > 0) & (pred > 0)
                pred = pred[mask]
                target = target[mask]

            if len(pred) > 1:
                corr, _ = pearsonr(pred, target)
                if not np.isnan(corr):
                    correlations.append(corr)

        return np.mean(correlations) if correlations else 0.0


class VoicedUnvoicedError:
    """Voiced/Unvoiced decision error rate"""

    def __call__(
        self,
        pred_pitch: torch.Tensor,
        target_pitch: torch.Tensor,
        threshold: float = 50.0,
    ) -> float:
        """
        Calculate V/UV error rate

        Args:
            pred_pitch: Predicted pitch values [B, T]
            target_pitch: Target pitch values [B, T]
            threshold: Threshold for voicing decision

        Returns:
            V/UV error rate
        """
        pred_pitch = pred_pitch.detach().cpu().numpy()
        target_pitch = target_pitch.detach().cpu().numpy()

        # Binary voicing decisions
        pred_voiced = pred_pitch > threshold
        target_voiced = target_pitch > threshold

        # Calculate error rate
        errors = pred_voiced != target_voiced
        error_rate = np.mean(errors)

        return error_rate


class SpeakerSimilarity:
    """Speaker similarity metric using cosine distance"""

    def __init__(self, speaker_encoder: Optional[nn.Module] = None):
        self.speaker_encoder = speaker_encoder

    def __call__(
        self,
        pred_audio: torch.Tensor,
        target_audio: torch.Tensor,
        speaker_embeddings: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Calculate speaker similarity

        Args:
            pred_audio: Predicted audio [B, 1, T]
            target_audio: Target audio [B, 1, T]
            speaker_embeddings: Pre-computed speaker embeddings [B, D]

        Returns:
            Average cosine similarity
        """
        if speaker_embeddings is not None:
            # Use provided embeddings
            pred_emb = speaker_embeddings[: pred_audio.size(0)]
            target_emb = speaker_embeddings[pred_audio.size(0) :]
        elif self.speaker_encoder is not None:
            # Extract embeddings using encoder
            with torch.no_grad():
                pred_emb = self.speaker_encoder(pred_audio)
                target_emb = self.speaker_encoder(target_audio)
        else:
            # Simple spectral similarity
            pred_spec = torch.stft(
                pred_audio.squeeze(1), n_fft=1024, hop_length=256, return_complex=True
            ).abs()
            target_spec = torch.stft(
                target_audio.squeeze(1), n_fft=1024, hop_length=256, return_complex=True
            ).abs()

            # Global average pooling
            pred_emb = pred_spec.mean(dim=[1, 2])
            target_emb = target_spec.mean(dim=[1, 2])

        # Normalize embeddings
        pred_emb = F.normalize(pred_emb, dim=1)
        target_emb = F.normalize(target_emb, dim=1)

        # Calculate cosine similarity
        similarities = (pred_emb * target_emb).sum(dim=1)

        return similarities.mean().item()


class AudioQualityMetrics:
    """Comprehensive audio quality metrics"""

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def calculate_pesq(self, pred_audio: np.ndarray, target_audio: np.ndarray) -> float:
        """Calculate PESQ score"""
        try:
            # PESQ expects 16kHz or 8kHz
            if self.sample_rate != 16000:
                pred_audio = librosa.resample(
                    pred_audio, orig_sr=self.sample_rate, target_sr=16000
                )
                target_audio = librosa.resample(
                    target_audio, orig_sr=self.sample_rate, target_sr=16000
                )

            if pesq is not None:
                score = pesq(16000, target_audio, pred_audio, "wb")
                return score
            else:
                return 0.0
        except Exception:
            return 0.0

    def calculate_stoi(self, pred_audio: np.ndarray, target_audio: np.ndarray) -> float:
        """Calculate STOI score"""
        try:
            score = stoi(target_audio, pred_audio, self.sample_rate, extended=False)
            return score
        except Exception:
            return 0.0

    def __call__(
        self, pred_audio: torch.Tensor, target_audio: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate all audio quality metrics

        Args:
            pred_audio: Predicted audio [B, 1, T]
            target_audio: Target audio [B, 1, T]

        Returns:
            Dictionary of metrics
        """
        pred_audio = pred_audio.detach().cpu().numpy()
        target_audio = target_audio.detach().cpu().numpy()

        metrics = {
            "pesq": [],
            "stoi": [],
            "snr": [],
        }

        for i in range(pred_audio.shape[0]):
            pred = pred_audio[i, 0]
            target = target_audio[i, 0]

            # PESQ
            pesq_score = self.calculate_pesq(pred, target)
            metrics["pesq"].append(pesq_score)

            # STOI
            stoi_score = self.calculate_stoi(pred, target)
            metrics["stoi"].append(stoi_score)

            # SNR
            signal_power = np.mean(target**2)
            noise_power = np.mean((pred - target) ** 2)
            if noise_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
            else:
                snr = float("inf")
            metrics["snr"].append(snr)

        # Average metrics
        return {key: np.mean(values) for key, values in metrics.items()}


class EmotionAccuracy:
    """Emotion classification accuracy"""

    def __call__(self, pred_logits: torch.Tensor, target_labels: torch.Tensor) -> float:
        """
        Calculate emotion classification accuracy

        Args:
            pred_logits: Predicted emotion logits [B, num_emotions]
            target_labels: Target emotion labels [B]

        Returns:
            Accuracy
        """
        pred_labels = torch.argmax(pred_logits, dim=1)
        correct = (pred_labels == target_labels).float()
        accuracy = correct.mean().item()

        return accuracy


class DurationError:
    """Duration prediction error metrics"""

    def __call__(
        self,
        pred_durations: torch.Tensor,
        target_durations: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Calculate duration prediction errors

        Args:
            pred_durations: Predicted durations [B, T]
            target_durations: Target durations [B, T]
            mask: Valid mask [B, T]

        Returns:
            Dictionary of duration metrics
        """
        if mask is not None:
            pred_durations = pred_durations * mask
            target_durations = target_durations * mask

        # Absolute error
        abs_error = torch.abs(pred_durations - target_durations)

        # Relative error
        rel_error = abs_error / (target_durations + 1e-8)

        # Total duration error
        pred_total = pred_durations.sum(dim=1)
        target_total = target_durations.sum(dim=1)
        total_error = torch.abs(pred_total - target_total) / (target_total + 1e-8)

        metrics = {
            "duration_mae": abs_error.mean().item(),
            "duration_relative_error": rel_error.mean().item(),
            "duration_total_error": total_error.mean().item(),
        }

        return metrics


def get_evaluation_metrics(config: Dict) -> Dict:
    """Factory function to create evaluation metrics based on config"""
    metrics = {}

    if config.get("use_mcd", True):
        metrics["mcd"] = MelCepstralDistortion()

    if config.get("use_pitch_correlation", True):
        metrics["pitch_correlation"] = PitchCorrelation()

    if config.get("use_vuv_error", True):
        metrics["vuv_error"] = VoicedUnvoicedError()

    if config.get("use_speaker_similarity", True):
        metrics["speaker_similarity"] = SpeakerSimilarity()

    if config.get("use_audio_quality", False):
        metrics["audio_quality"] = AudioQualityMetrics(
            sample_rate=config.get("sample_rate", 22050)
        )

    if config.get("use_emotion_accuracy", True):
        metrics["emotion_accuracy"] = EmotionAccuracy()

    if config.get("use_duration_error", True):
        metrics["duration_error"] = DurationError()

    return metrics
