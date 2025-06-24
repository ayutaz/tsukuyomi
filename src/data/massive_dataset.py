"""
Massive dataset processing for 10,000 hours of game character voices

Handles large-scale data with distributed processing, quality assurance,
and efficient data loading for training the ultimate TTS model.
"""

import os
import json
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchaudio
import soundfile as sf
import librosa
import pandas as pd
from tqdm import tqdm
import hashlib

# Quality metrics
from pesq import pesq
from pystoi import stoi

logger = logging.getLogger(__name__)


@dataclass
class AudioMetadata:
    """Metadata for each audio file."""

    file_path: str
    speaker_id: str
    character_name: str
    text: str
    duration: float
    sample_rate: int
    num_channels: int
    bit_depth: int
    snr: float
    energy: float
    f0_mean: float
    f0_std: float
    speaking_rate: float
    quality_score: float
    emotion: Optional[str] = None
    scene_context: Optional[str] = None
    recording_date: Optional[str] = None
    game_title: Optional[str] = None
    file_hash: Optional[str] = None


@dataclass
class DatasetStats:
    """Statistics for the dataset."""

    total_hours: float
    total_files: int
    num_speakers: int
    num_characters: int
    avg_duration: float
    avg_snr: float
    avg_quality: float
    speaker_distribution: Dict[str, int]
    quality_distribution: Dict[str, int]


class QualityChecker:
    """Audio quality assessment for dataset curation."""

    def __init__(
        self,
        min_snr: float = 30.0,
        min_duration: float = 0.5,
        max_duration: float = 30.0,
        target_sr: int = 48000,
        check_clipping: bool = True,
        check_silence: bool = True,
    ):
        self.min_snr = min_snr
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.target_sr = target_sr
        self.check_clipping = check_clipping
        self.check_silence = check_silence

    def check_audio(self, audio_path: Path) -> Tuple[bool, Dict[str, Any]]:
        """
        Check audio quality and return pass/fail with metrics.

        Returns:
            (passed, metrics_dict)
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None, mono=False)

            # Basic checks
            duration = len(audio) / sr
            if duration < self.min_duration or duration > self.max_duration:
                return False, {"reason": "duration_out_of_range", "duration": duration}

            # Convert to mono for analysis
            if audio.ndim > 1:
                audio_mono = audio.mean(axis=0)
            else:
                audio_mono = audio

            # SNR estimation
            noise_floor = np.percentile(np.abs(audio_mono), 10)
            signal_peak = np.percentile(np.abs(audio_mono), 90)
            snr = 20 * np.log10(signal_peak / (noise_floor + 1e-10))

            if snr < self.min_snr:
                return False, {"reason": "low_snr", "snr": snr}

            # Clipping check
            if self.check_clipping:
                clipping_ratio = np.sum(np.abs(audio_mono) > 0.99) / len(audio_mono)
                if clipping_ratio > 0.001:  # More than 0.1% samples clipped
                    return False, {
                        "reason": "clipping",
                        "clipping_ratio": clipping_ratio,
                    }

            # Silence check
            if self.check_silence:
                silence_ratio = np.sum(np.abs(audio_mono) < 0.001) / len(audio_mono)
                if silence_ratio > 0.9:  # More than 90% silence
                    return False, {
                        "reason": "too_silent",
                        "silence_ratio": silence_ratio,
                    }

            # Extract features
            f0, voiced_flag, _ = librosa.pyin(audio_mono, fmin=50, fmax=800, sr=sr)
            f0_valid = f0[voiced_flag == 1]

            metrics = {
                "passed": True,
                "duration": duration,
                "sample_rate": sr,
                "snr": snr,
                "energy": np.mean(audio_mono**2),
                "f0_mean": np.mean(f0_valid) if len(f0_valid) > 0 else 0,
                "f0_std": np.std(f0_valid) if len(f0_valid) > 0 else 0,
                "num_channels": audio.shape[0] if audio.ndim > 1 else 1,
            }

            return True, metrics

        except Exception as e:
            logger.error(f"Error checking {audio_path}: {e}")
            return False, {"reason": "processing_error", "error": str(e)}


class TextProcessor:
    """Process and validate text transcriptions."""

    def __init__(self):
        # Patterns for cleaning
        self.noise_patterns = [
            r"\(.*?\)",  # Remove parentheses
            r"\[.*?\]",  # Remove brackets
            r"<.*?>",  # Remove tags
        ]

    def process_text(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Clean and validate text.

        Returns:
            (cleaned_text, metadata)
        """
        import re

        original_length = len(text)

        # Remove noise patterns
        cleaned = text
        for pattern in self.noise_patterns:
            cleaned = re.sub(pattern, "", cleaned)

        # Normalize whitespace
        cleaned = " ".join(cleaned.split())

        # Extract metadata
        metadata = {
            "original_length": original_length,
            "cleaned_length": len(cleaned),
            "removed_ratio": 1 - len(cleaned) / max(original_length, 1),
        }

        return cleaned, metadata


class MassiveGameVoiceDataset(Dataset):
    """
    Dataset for 10,000 hours of game character voices.

    Features:
    - Efficient loading with caching
    - Multi-speaker support (500+ characters)
    - Quality filtering
    - Distributed training support
    """

    def __init__(
        self,
        data_root: Path,
        metadata_file: Path,
        cache_dir: Optional[Path] = None,
        quality_threshold: float = 0.8,
        load_audio: bool = True,
        transform: Optional[Any] = None,
        target_sample_rate: int = 48000,
        segment_length: Optional[float] = None,
    ):
        self.data_root = Path(data_root)
        self.metadata_file = Path(metadata_file)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_root / ".cache"
        self.quality_threshold = quality_threshold
        self.load_audio = load_audio
        self.transform = transform
        self.target_sample_rate = target_sample_rate
        self.segment_length = segment_length

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load metadata
        self._load_metadata()

        # Build speaker mappings
        self._build_speaker_mappings()

        logger.info(
            f"Loaded dataset with {len(self.metadata)} samples, "
            f"{self.num_speakers} speakers, "
            f"{self.total_hours:.1f} hours"
        )

    def _load_metadata(self):
        """Load and filter metadata."""
        # Check cache first
        cache_file = self.cache_dir / "filtered_metadata.json"

        if cache_file.exists():
            logger.info(f"Loading cached metadata from {cache_file}")
            with open(cache_file, "r") as f:
                self.metadata = json.load(f)
        else:
            logger.info(f"Loading metadata from {self.metadata_file}")

            # Load full metadata
            with open(self.metadata_file, "r") as f:
                full_metadata = json.load(f)

            # Filter by quality
            self.metadata = [
                item
                for item in full_metadata
                if item.get("quality_score", 0) >= self.quality_threshold
            ]

            # Save filtered cache
            with open(cache_file, "w") as f:
                json.dump(self.metadata, f)

        # Calculate stats
        self.total_hours = sum(item["duration"] for item in self.metadata) / 3600

    def _build_speaker_mappings(self):
        """Build speaker ID mappings."""
        # Get unique speakers
        speakers = sorted(list(set(item["speaker_id"] for item in self.metadata)))
        self.speaker_to_id = {spk: idx for idx, spk in enumerate(speakers)}
        self.id_to_speaker = {idx: spk for spk, idx in self.speaker_to_id.items()}
        self.num_speakers = len(speakers)

        # Character mappings
        characters = sorted(list(set(item["character_name"] for item in self.metadata)))
        self.character_to_id = {char: idx for idx, char in enumerate(characters)}
        self.num_characters = len(characters)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """Get a sample."""
        meta = self.metadata[idx]

        sample = {
            "text": meta["text"],
            "speaker_id": self.speaker_to_id[meta["speaker_id"]],
            "character_id": self.character_to_id[meta["character_name"]],
            "duration": meta["duration"],
            "metadata": meta,
        }

        if self.load_audio:
            audio_path = self.data_root / meta["file_path"]
            audio, sr = torchaudio.load(audio_path)

            # Resample if needed
            if sr != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
                audio = resampler(audio)

            # Segment if specified
            if self.segment_length is not None:
                segment_samples = int(self.segment_length * self.target_sample_rate)
                if audio.shape[1] > segment_samples:
                    start = torch.randint(
                        0, audio.shape[1] - segment_samples, (1,)
                    ).item()
                    audio = audio[:, start : start + segment_samples]

            sample["audio"] = audio
            sample["sample_rate"] = self.target_sample_rate

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_speaker_stats(self) -> pd.DataFrame:
        """Get speaker statistics."""
        speaker_data = []

        for speaker_id in self.speaker_to_id:
            speaker_items = [m for m in self.metadata if m["speaker_id"] == speaker_id]
            total_duration = sum(item["duration"] for item in speaker_items) / 3600

            speaker_data.append(
                {
                    "speaker_id": speaker_id,
                    "num_utterances": len(speaker_items),
                    "total_hours": total_duration,
                    "avg_duration": np.mean(
                        [item["duration"] for item in speaker_items]
                    ),
                    "avg_quality": np.mean(
                        [item["quality_score"] for item in speaker_items]
                    ),
                }
            )

        return pd.DataFrame(speaker_data)


class DistributedDataProcessor:
    """
    Process 10,000 hours of audio data efficiently using distributed processing.
    """

    def __init__(
        self, num_workers: int = 32, use_gpu: bool = True, batch_size: int = 100
    ):
        self.num_workers = num_workers
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.batch_size = batch_size
        self.quality_checker = QualityChecker()
        self.text_processor = TextProcessor()

    def process_dataset(
        self, input_dir: Path, output_dir: Path, file_pattern: str = "**/*.wav"
    ) -> DatasetStats:
        """
        Process entire dataset with quality checks and feature extraction.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all audio files
        audio_files = list(Path(input_dir).glob(file_pattern))
        logger.info(f"Found {len(audio_files)} audio files")

        # Process in batches
        all_metadata = []

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Process files in batches
            for i in tqdm(range(0, len(audio_files), self.batch_size)):
                batch_files = audio_files[i : i + self.batch_size]

                # Submit batch for processing
                futures = [
                    executor.submit(self._process_single_file, f, output_dir)
                    for f in batch_files
                ]

                # Collect results
                for future in futures:
                    result = future.result()
                    if result is not None:
                        all_metadata.append(result)

        # Save metadata
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(all_metadata, f, indent=2)

        # Calculate statistics
        stats = self._calculate_stats(all_metadata)

        # Save statistics
        stats_file = output_dir / "dataset_stats.json"
        with open(stats_file, "w") as f:
            json.dump(asdict(stats), f, indent=2)

        return stats

    def _process_single_file(
        self, audio_path: Path, output_dir: Path
    ) -> Optional[Dict]:
        """Process a single audio file."""
        try:
            # Quality check
            passed, metrics = self.quality_checker.check_audio(audio_path)
            if not passed:
                logger.warning(f"Quality check failed for {audio_path}: {metrics}")
                return None

            # Load transcription
            text_path = audio_path.with_suffix(".txt")
            if not text_path.exists():
                logger.warning(f"No transcription found for {audio_path}")
                return None

            with open(text_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            # Process text
            cleaned_text, text_meta = self.text_processor.process_text(text)

            if len(cleaned_text) < 1:
                logger.warning(f"Empty transcription for {audio_path}")
                return None

            # Extract speaker and character info from path
            # Assuming structure: data/game_name/character_name/speaker_id/file.wav
            parts = audio_path.parts
            game_name = parts[-4] if len(parts) > 4 else "unknown"
            character_name = parts[-3] if len(parts) > 3 else "unknown"
            speaker_id = parts[-2] if len(parts) > 2 else "unknown"

            # Calculate file hash
            file_hash = hashlib.md5(open(audio_path, "rb").read()).hexdigest()

            # Create metadata
            metadata = AudioMetadata(
                file_path=str(audio_path.relative_to(audio_path.parents[4])),
                speaker_id=speaker_id,
                character_name=character_name,
                text=cleaned_text,
                duration=metrics["duration"],
                sample_rate=metrics["sample_rate"],
                num_channels=metrics["num_channels"],
                bit_depth=16,  # Assume 16-bit
                snr=metrics["snr"],
                energy=metrics["energy"],
                f0_mean=metrics["f0_mean"],
                f0_std=metrics["f0_std"],
                speaking_rate=len(cleaned_text.split()) / metrics["duration"],
                quality_score=min(metrics["snr"] / 40, 1.0),  # Normalize SNR to 0-1
                game_title=game_name,
                file_hash=file_hash,
            )

            return asdict(metadata)

        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            return None

    def _calculate_stats(self, metadata: List[Dict]) -> DatasetStats:
        """Calculate dataset statistics."""
        total_duration = sum(m["duration"] for m in metadata)
        speakers = set(m["speaker_id"] for m in metadata)
        characters = set(m["character_name"] for m in metadata)

        # Speaker distribution
        speaker_dist = {}
        for m in metadata:
            speaker = m["speaker_id"]
            speaker_dist[speaker] = speaker_dist.get(speaker, 0) + 1

        # Quality distribution
        quality_bins = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        for m in metadata:
            q = m["quality_score"]
            if q >= 0.9:
                quality_bins["excellent"] += 1
            elif q >= 0.7:
                quality_bins["good"] += 1
            elif q >= 0.5:
                quality_bins["fair"] += 1
            else:
                quality_bins["poor"] += 1

        return DatasetStats(
            total_hours=total_duration / 3600,
            total_files=len(metadata),
            num_speakers=len(speakers),
            num_characters=len(characters),
            avg_duration=total_duration / len(metadata) if metadata else 0,
            avg_snr=np.mean([m["snr"] for m in metadata]) if metadata else 0,
            avg_quality=(
                np.mean([m["quality_score"] for m in metadata]) if metadata else 0
            ),
            speaker_distribution=speaker_dist,
            quality_distribution=quality_bins,
        )


def create_massive_dataloader(
    data_root: Path,
    metadata_file: Path,
    batch_size: int = 32,
    num_workers: int = 8,
    distributed: bool = False,
    **kwargs,
) -> DataLoader:
    """Create dataloader for massive dataset."""
    dataset = MassiveGameVoiceDataset(
        data_root=data_root, metadata_file=metadata_file, **kwargs
    )

    if distributed:
        sampler = DistributedSampler(dataset)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )


if __name__ == "__main__":
    # Example usage
    processor = DistributedDataProcessor(num_workers=32)

    stats = processor.process_dataset(
        input_dir=Path("/path/to/game/voices"),
        output_dir=Path("/path/to/processed/data"),
    )

    print(f"Processed {stats.total_hours:.1f} hours of audio")
    print(f"Found {stats.num_speakers} speakers and {stats.num_characters} characters")
