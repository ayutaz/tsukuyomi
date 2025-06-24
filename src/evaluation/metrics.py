"""
Evaluation Metrics for Tsukuyomi TTS

Comprehensive evaluation metrics including:
- MOS (Mean Opinion Score) prediction
- Speaker similarity
- Pronunciation accuracy
- Prosody evaluation
- Real-time factor (RTF)
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pesq import pesq
from pystoi import stoi
from scipy.stats import pearsonr
from transformers import Wav2Vec2Model, Wav2Vec2Processor

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""

    mel_cepstral_distortion: float
    pitch_correlation: float
    voice_similarity: float
    pronunciation_accuracy: float
    pesq_score: Optional[float] = None
    stoi_score: Optional[float] = None
    mos_prediction: Optional[float] = None
    rtf: Optional[float] = None
    speaker_similarity: Optional[float] = None
    energy_correlation: Optional[float] = None
    duration_error: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @property
    def overall_score(self) -> float:
        """Calculate overall quality score."""
        scores = []
        weights = {
            "mel_cepstral_distortion": -0.2,  # Negative because lower is better
            "pitch_correlation": 0.15,
            "voice_similarity": 0.25,
            "pronunciation_accuracy": 0.2,
            "speaker_similarity": 0.2,
        }

        for metric, weight in weights.items():
            value = getattr(self, metric, None)
            if value is not None:
                scores.append(value * weight)

        # Normalize to 0-5 scale (MOS-like)
        raw_score = sum(scores)
        return max(1.0, min(5.0, 3.0 + raw_score * 2.0))


class MelCepstralDistortion:
    """Calculate Mel-Cepstral Distortion (MCD)."""

    def __init__(self, n_mfcc: int = 13, n_fft: int = 1024, hop_length: int = 256):
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

    def calculate(
        self, reference: np.ndarray, synthesized: np.ndarray, sr: int = 48000
    ) -> float:
        """
        Calculate MCD between reference and synthesized audio.

        Args:
            reference: Reference audio waveform
            synthesized: Synthesized audio waveform
            sr: Sample rate

        Returns:
            MCD value in dB
        """
        # Extract MFCCs
        mfcc_ref = librosa.feature.mfcc(
            y=reference,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        mfcc_syn = librosa.feature.mfcc(
            y=synthesized,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        # Align sequences
        min_len = min(mfcc_ref.shape[1], mfcc_syn.shape[1])
        mfcc_ref = mfcc_ref[:, :min_len]
        mfcc_syn = mfcc_syn[:, :min_len]

        # Calculate MCD
        diff = mfcc_ref - mfcc_syn
        mcd = (
            np.mean(np.sqrt(np.sum(diff**2, axis=0)))
            * (10.0 / np.log(10.0))
            * np.sqrt(2.0)
        )

        return float(mcd)


class PitchEvaluator:
    """Evaluate pitch accuracy and correlation."""

    def __init__(
        self, f0_min: float = 50.0, f0_max: float = 800.0, frame_shift: float = 0.005
    ):
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.frame_shift = frame_shift

    def evaluate(
        self, reference: np.ndarray, synthesized: np.ndarray, sr: int = 48000
    ) -> Dict[str, float]:
        """
        Evaluate pitch accuracy.

        Args:
            reference: Reference audio
            synthesized: Synthesized audio
            sr: Sample rate

        Returns:
            Dictionary with pitch metrics
        """
        # Extract F0 using CREPE or similar
        import crepe

        _, f0_ref, confidence_ref, _ = crepe.predict(
            reference, sr, viterbi=True, step_size=int(self.frame_shift * 1000)
        )

        _, f0_syn, confidence_syn, _ = crepe.predict(
            synthesized, sr, viterbi=True, step_size=int(self.frame_shift * 1000)
        )

        # Filter by confidence
        valid_ref = confidence_ref > 0.5
        valid_syn = confidence_syn > 0.5
        valid_both = valid_ref & valid_syn

        if np.sum(valid_both) < 10:
            return {
                "pitch_correlation": 0.0,
                "pitch_rmse": float("inf"),
                "vuv_error": 1.0,
            }

        # Calculate correlation
        correlation, _ = pearsonr(f0_ref[valid_both], f0_syn[valid_both])

        # Calculate RMSE in cents
        cents_diff = 1200 * np.log2(f0_syn[valid_both] / f0_ref[valid_both])
        rmse_cents = np.sqrt(np.mean(cents_diff**2))

        # Calculate V/UV error
        vuv_error = np.mean(valid_ref != valid_syn)

        return {
            "pitch_correlation": float(correlation),
            "pitch_rmse": float(rmse_cents),
            "vuv_error": float(vuv_error),
        }


class SpeakerSimilarity:
    """Evaluate speaker similarity using speaker embeddings."""

    def __init__(self, model_name: str = "microsoft/wavlm-base-plus-sv"):
        """
        Initialize speaker similarity evaluator.

        Args:
            model_name: Pretrained speaker verification model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            from transformers import AutoModel, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
        except Exception as e:
            logger.warning(f"Failed to load speaker model: {e}")
            self.processor = None
            self.model = None

    def calculate_similarity(
        self, audio1: np.ndarray, audio2: np.ndarray, sr: int = 16000
    ) -> float:
        """
        Calculate cosine similarity between speaker embeddings.

        Args:
            audio1: First audio waveform
            audio2: Second audio waveform
            sr: Sample rate

        Returns:
            Similarity score (0-1)
        """
        if self.model is None:
            return 0.5  # Default similarity

        # Preprocess audio
        inputs1 = self.processor(audio1, sampling_rate=sr, return_tensors="pt")
        inputs2 = self.processor(audio2, sampling_rate=sr, return_tensors="pt")

        # Move to device
        inputs1 = {k: v.to(self.device) for k, v in inputs1.items()}
        inputs2 = {k: v.to(self.device) for k, v in inputs2.items()}

        # Extract embeddings
        with torch.no_grad():
            embeddings1 = self.model(**inputs1).embeddings
            embeddings2 = self.model(**inputs2).embeddings

            # Pool embeddings
            embeddings1 = embeddings1.mean(dim=1)
            embeddings2 = embeddings2.mean(dim=1)

            # Normalize
            embeddings1 = F.normalize(embeddings1, p=2, dim=1)
            embeddings2 = F.normalize(embeddings2, p=2, dim=1)

            # Calculate cosine similarity
            similarity = F.cosine_similarity(embeddings1, embeddings2)

        return float(similarity.cpu().item())


class PronunciationAccuracy:
    """Evaluate pronunciation accuracy using ASR."""

    def __init__(self, model_name: str = "openai/whisper-base"):
        """
        Initialize pronunciation evaluator.

        Args:
            model_name: ASR model for evaluation
        """
        try:
            import whisper

            self.model = whisper.load_model("base")
        except Exception as e:
            logger.warning(f"Failed to load ASR model: {e}")
            self.model = None

    def evaluate(
        self,
        audio: np.ndarray,
        reference_text: str,
        sr: int = 16000,
        language: str = "ja",
    ) -> Dict[str, float]:
        """
        Evaluate pronunciation accuracy.

        Args:
            audio: Audio waveform
            reference_text: Expected text
            sr: Sample rate
            language: Language code

        Returns:
            Dictionary with accuracy metrics
        """
        if self.model is None:
            return {"pronunciation_accuracy": 0.5, "wer": 0.5}

        # Transcribe audio
        result = self.model.transcribe(audio, language=language, task="transcribe")

        transcribed_text = result["text"]

        # Calculate character error rate
        from jiwer import cer, wer

        char_error_rate = cer(reference_text, transcribed_text)
        word_error_rate = wer(reference_text, transcribed_text)

        # Calculate accuracy (1 - error rate)
        accuracy = 1.0 - char_error_rate

        return {
            "pronunciation_accuracy": float(accuracy),
            "cer": float(char_error_rate),
            "wer": float(word_error_rate),
            "transcribed_text": transcribed_text,
        }


class MOSPredictor(nn.Module):
    """Neural MOS predictor based on wav2vec2."""

    def __init__(self, model_name: str = "facebook/wav2vec2-base"):
        super().__init__()

        # Load pretrained model
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)

        # MOS prediction head
        self.mos_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Predict MOS score.

        Args:
            audio: Audio waveform [batch, time]

        Returns:
            MOS scores [batch, 1] (scaled 1-5)
        """
        # Extract features
        features = self.wav2vec2(audio).last_hidden_state

        # Global pooling
        pooled = features.mean(dim=1)

        # Predict MOS
        mos = self.mos_head(pooled)

        # Scale to 1-5 range
        mos = 1.0 + mos * 4.0

        return mos


class ComprehensiveEvaluator:
    """Comprehensive evaluation suite for TTS."""

    def __init__(self, use_gpu: bool = True, cache_models: bool = True):
        """
        Initialize comprehensive evaluator.

        Args:
            use_gpu: Use GPU acceleration
            cache_models: Cache evaluation models
        """
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )

        # Initialize evaluators
        self.mcd_evaluator = MelCepstralDistortion()
        self.pitch_evaluator = PitchEvaluator()
        self.speaker_evaluator = SpeakerSimilarity()
        self.pronunciation_evaluator = PronunciationAccuracy()

        # Load MOS predictor if available
        self.mos_predictor = None
        mos_checkpoint = Path("models/mos_predictor.pt")
        if mos_checkpoint.exists():
            self.mos_predictor = MOSPredictor()
            self.mos_predictor.load_state_dict(torch.load(mos_checkpoint))
            self.mos_predictor.to(self.device)
            self.mos_predictor.eval()

    def evaluate(
        self,
        synthesized_audio: np.ndarray,
        reference_audio: Optional[np.ndarray] = None,
        reference_text: Optional[str] = None,
        sr: int = 48000,
        compute_all: bool = True,
    ) -> EvaluationResult:
        """
        Perform comprehensive evaluation.

        Args:
            synthesized_audio: Synthesized audio waveform
            reference_audio: Reference audio for comparison
            reference_text: Reference text for pronunciation
            sr: Sample rate
            compute_all: Compute all metrics (slower)

        Returns:
            Evaluation results
        """
        results = {}

        # Basic quality metrics
        if reference_audio is not None:
            # MCD
            results["mel_cepstral_distortion"] = self.mcd_evaluator.calculate(
                reference_audio, synthesized_audio, sr
            )

            # Pitch evaluation
            pitch_results = self.pitch_evaluator.evaluate(
                reference_audio, synthesized_audio, sr
            )
            results["pitch_correlation"] = pitch_results["pitch_correlation"]

            # Speaker similarity
            results["speaker_similarity"] = self.speaker_evaluator.calculate_similarity(
                reference_audio, synthesized_audio, sr
            )

            if compute_all:
                # PESQ (requires 16kHz)
                try:
                    ref_16k = librosa.resample(
                        reference_audio, orig_sr=sr, target_sr=16000
                    )
                    syn_16k = librosa.resample(
                        synthesized_audio, orig_sr=sr, target_sr=16000
                    )
                    results["pesq_score"] = pesq(16000, ref_16k, syn_16k, "wb")
                except Exception as e:
                    logger.warning(f"PESQ calculation failed: {e}")

                # STOI
                try:
                    results["stoi_score"] = stoi(
                        reference_audio, synthesized_audio, sr, extended=False
                    )
                except Exception as e:
                    logger.warning(f"STOI calculation failed: {e}")

        # Pronunciation accuracy
        if reference_text:
            pron_results = self.pronunciation_evaluator.evaluate(
                synthesized_audio, reference_text, sr
            )
            results["pronunciation_accuracy"] = pron_results["pronunciation_accuracy"]

        # MOS prediction
        if self.mos_predictor is not None:
            with torch.no_grad():
                audio_tensor = (
                    torch.from_numpy(synthesized_audio)
                    .float()
                    .unsqueeze(0)
                    .to(self.device)
                )
                mos = self.mos_predictor(audio_tensor)
                results["mos_prediction"] = float(mos.cpu().item())

        # Fill in missing required metrics
        if "voice_similarity" not in results:
            results["voice_similarity"] = results.get("speaker_similarity", 0.5)

        return EvaluationResult(**results)

    def evaluate_batch(
        self,
        audio_pairs: List[Tuple[np.ndarray, np.ndarray]],
        texts: Optional[List[str]] = None,
        sr: int = 48000,
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple audio pairs.

        Args:
            audio_pairs: List of (synthesized, reference) audio pairs
            texts: Optional reference texts
            sr: Sample rate

        Returns:
            List of evaluation results
        """
        results = []

        for i, (syn, ref) in enumerate(audio_pairs):
            text = texts[i] if texts else None
            result = self.evaluate(syn, ref, text, sr)
            results.append(result)

        return results

    def save_results(self, results: List[EvaluationResult], output_path: Path):
        """Save evaluation results to file."""
        data = {
            "results": [r.to_dict() for r in results],
            "summary": {
                "mean_mos": np.mean([r.overall_score for r in results]),
                "mean_mcd": np.mean([r.mel_cepstral_distortion for r in results]),
                "mean_speaker_sim": np.mean(
                    [r.speaker_similarity for r in results if r.speaker_similarity]
                ),
                "mean_pronunciation": np.mean(
                    [
                        r.pronunciation_accuracy
                        for r in results
                        if r.pronunciation_accuracy
                    ]
                ),
            },
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)


def calculate_rtf(model_inference_time: float, audio_duration: float) -> float:
    """
    Calculate Real-Time Factor (RTF).

    Args:
        model_inference_time: Time taken for inference in seconds
        audio_duration: Duration of generated audio in seconds

    Returns:
        RTF value (lower is better, <1 means faster than real-time)
    """
    return model_inference_time / audio_duration
