"""
Tsukuyomi Dataset - Flexible dataset loader for TTS training

Supports various metadata formats and handles missing information gracefully.
"""

import csv
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class SpeakerMetadata:
    """Speaker metadata with optional fields."""

    speaker_id: str
    name: Optional[str] = None
    gender: Optional[str] = None
    age_range: Optional[str] = None
    dialect: Optional[str] = None
    voice_characteristics: Optional[Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpeakerMetadata":
        """Create from dictionary, handling missing fields."""
        return cls(
            speaker_id=data.get("speaker_id", ""),
            name=data.get("name"),
            gender=data.get("gender"),
            age_range=data.get("age_range"),
            dialect=data.get("dialect"),
            voice_characteristics=data.get("voice_characteristics", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "speaker_id": self.speaker_id,
            "name": self.name,
            "gender": self.gender,
            "age_range": self.age_range,
            "dialect": self.dialect,
            "voice_characteristics": self.voice_characteristics,
        }


@dataclass
class AudioSample:
    """Single audio sample with metadata."""

    audio_path: Path
    text: str
    speaker_id: str
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


class TsukuyomiDataset(Dataset):
    """
    Flexible dataset for Tsukuyomi TTS training.

    Supports:
    - Multiple transcript formats (txt, csv, json)
    - Optional speaker metadata
    - Flexible directory structures
    - Missing metadata handling
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        transcript_file: Optional[str] = "transcripts.txt",
        metadata_file: Optional[str] = "metadata.json",
        sample_rate: int = 48000,
        max_duration: Optional[float] = 15.0,
        min_duration: Optional[float] = 0.5,
        cache_audio: bool = False,
        validation_split: Optional[float] = None,
        is_validation: bool = False,
        random_seed: int = 42,
    ):
        """
        Initialize dataset.

        Args:
            data_root: Root directory containing audio and transcripts
            transcript_file: Name of transcript file (supports .txt, .csv, .json)
            metadata_file: Optional speaker metadata file
            sample_rate: Target sample rate
            max_duration: Maximum audio duration in seconds
            min_duration: Minimum audio duration in seconds
            cache_audio: Whether to cache audio in memory
            validation_split: Fraction for validation split
            is_validation: Whether this is validation set
            random_seed: Random seed for splitting
        """
        self.data_root = Path(data_root)
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.cache_audio = cache_audio
        self.audio_cache = {} if cache_audio else None

        # Load transcripts
        self.samples = self._load_transcripts(transcript_file)

        # Load speaker metadata if available
        self.speaker_metadata = self._load_metadata(metadata_file)

        # Filter by duration if specified
        if max_duration or min_duration:
            self.samples = self._filter_by_duration()

        # Handle train/validation split
        if validation_split:
            self._split_dataset(validation_split, is_validation, random_seed)

        logger.info(f"Loaded {len(self.samples)} samples from {data_root}")

    def _load_transcripts(self, transcript_file: str) -> List[AudioSample]:
        """Load transcripts from various formats."""
        transcript_path = self.data_root / transcript_file

        if not transcript_path.exists():
            # Try to find transcript file
            transcript_paths = list(self.data_root.glob("transcript*"))
            if transcript_paths:
                transcript_path = transcript_paths[0]
                logger.info(f"Found transcript file: {transcript_path}")
            else:
                raise FileNotFoundError(f"No transcript file found in {self.data_root}")

        samples = []

        # Detect format by extension
        if transcript_path.suffix == ".txt":
            samples = self._load_txt_transcripts(transcript_path)
        elif transcript_path.suffix == ".csv":
            samples = self._load_csv_transcripts(transcript_path)
        elif transcript_path.suffix == ".json":
            samples = self._load_json_transcripts(transcript_path)
        else:
            # Try to detect format
            samples = self._load_auto_format(transcript_path)

        return samples

    def _load_txt_transcripts(self, path: Path) -> List[AudioSample]:
        """Load pipe-delimited text transcripts."""
        samples = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split("|")
                if len(parts) >= 2:
                    audio_file = parts[0]
                    text = parts[1]

                    # Optional speaker ID as third field
                    speaker_id = parts[2] if len(parts) > 2 else "default"

                    # Handle various audio path formats
                    audio_path = self._resolve_audio_path(audio_file)

                    if audio_path and audio_path.exists():
                        samples.append(
                            AudioSample(
                                audio_path=audio_path, text=text, speaker_id=speaker_id
                            )
                        )
                    else:
                        logger.warning(f"Audio file not found: {audio_file}")

        return samples

    def _load_csv_transcripts(self, path: Path) -> List[AudioSample]:
        """Load CSV transcripts."""
        samples = []

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Try common column names
                audio_file = row.get("audio") or row.get("wav") or row.get("file")
                text = row.get("text") or row.get("transcript") or row.get("sentence")
                speaker_id = row.get("speaker") or row.get("speaker_id") or "default"

                if audio_file and text:
                    audio_path = self._resolve_audio_path(audio_file)
                    if audio_path and audio_path.exists():
                        samples.append(
                            AudioSample(
                                audio_path=audio_path,
                                text=text,
                                speaker_id=speaker_id,
                                metadata=row,  # Store all fields as metadata
                            )
                        )

        return samples

    def _load_json_transcripts(self, path: Path) -> List[AudioSample]:
        """Load JSON transcripts."""
        samples = []

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle list format
        if isinstance(data, list):
            for item in data:
                audio_file = item.get("audio") or item.get("wav") or item.get("file")
                text = item.get("text") or item.get("transcript")
                speaker_id = item.get("speaker", "default")

                if audio_file and text:
                    audio_path = self._resolve_audio_path(audio_file)
                    if audio_path and audio_path.exists():
                        samples.append(
                            AudioSample(
                                audio_path=audio_path,
                                text=text,
                                speaker_id=speaker_id,
                                metadata=item,
                            )
                        )

        # Handle dict format (audio_file: transcript)
        elif isinstance(data, dict):
            for audio_file, info in data.items():
                if isinstance(info, str):
                    # Simple format: {"audio.wav": "text"}
                    text = info
                    speaker_id = "default"
                    metadata = {}
                else:
                    # Complex format: {"audio.wav": {"text": "...", "speaker": "..."}}
                    text = info.get("text", "")
                    speaker_id = info.get("speaker", "default")
                    metadata = info

                audio_path = self._resolve_audio_path(audio_file)
                if audio_path and audio_path.exists():
                    samples.append(
                        AudioSample(
                            audio_path=audio_path,
                            text=text,
                            speaker_id=speaker_id,
                            metadata=metadata,
                        )
                    )

        return samples

    def _load_auto_format(self, path: Path) -> List[AudioSample]:
        """Try to automatically detect format."""
        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()

        # Check if it looks like JSON
        if first_line.startswith("{") or first_line.startswith("["):
            return self._load_json_transcripts(path)
        # Check if it has pipes (txt format)
        elif "|" in first_line:
            return self._load_txt_transcripts(path)
        # Otherwise try CSV
        else:
            return self._load_csv_transcripts(path)

    def _resolve_audio_path(self, audio_file: str) -> Optional[Path]:
        """Resolve audio file path with various strategies."""
        # Try as absolute path
        audio_path = Path(audio_file)
        if audio_path.is_absolute() and audio_path.exists():
            return audio_path

        # Try relative to data root
        audio_path = self.data_root / audio_file
        if audio_path.exists():
            return audio_path

        # Try common subdirectories
        for subdir in ["audio", "wav", "wavs", "data"]:
            audio_path = self.data_root / subdir / audio_file
            if audio_path.exists():
                return audio_path

        # Try speaker subdirectories
        parts = Path(audio_file).parts
        if len(parts) > 1:
            # Format: speaker_001/audio.wav
            audio_path = self.data_root / audio_file
            if audio_path.exists():
                return audio_path

        return None

    def _load_metadata(self, metadata_file: str) -> Dict[str, SpeakerMetadata]:
        """Load speaker metadata if available."""
        metadata = {}

        if not metadata_file:
            return metadata

        metadata_path = self.data_root / metadata_file

        # Try alternative paths
        if not metadata_path.exists():
            for alt_name in ["metadata.json", "speakers.json", "speaker_info.json"]:
                alt_path = self.data_root / alt_name
                if alt_path.exists():
                    metadata_path = alt_path
                    break

        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for speaker_id, info in data.items():
                    if not isinstance(info, dict):
                        info = {"name": str(info)}
                    info["speaker_id"] = speaker_id
                    metadata[speaker_id] = SpeakerMetadata.from_dict(info)

                logger.info(f"Loaded metadata for {len(metadata)} speakers")
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")

        return metadata

    def _filter_by_duration(self) -> List[AudioSample]:
        """Filter samples by duration."""
        filtered = []

        for sample in self.samples:
            try:
                info = sf.info(sample.audio_path)
                duration = info.duration
                sample.duration = duration
                sample.sample_rate = info.samplerate

                if self.min_duration and duration < self.min_duration:
                    continue
                if self.max_duration and duration > self.max_duration:
                    continue

                filtered.append(sample)
            except Exception as e:
                logger.warning(
                    f"Failed to read audio info for {sample.audio_path}: {e}"
                )

        logger.info(
            f"Filtered from {len(self.samples)} to {len(filtered)} samples by duration"
        )
        return filtered

    def _split_dataset(self, validation_split: float, is_validation: bool, seed: int):
        """Split dataset for train/validation."""
        random.seed(seed)

        # Group by speaker for speaker-wise split
        speaker_samples = {}
        for sample in self.samples:
            if sample.speaker_id not in speaker_samples:
                speaker_samples[sample.speaker_id] = []
            speaker_samples[sample.speaker_id].append(sample)

        # Shuffle speakers
        speakers = list(speaker_samples.keys())
        random.shuffle(speakers)

        # Split speakers
        n_val_speakers = int(len(speakers) * validation_split)
        val_speakers = set(speakers[:n_val_speakers])

        # Filter samples
        if is_validation:
            self.samples = [s for s in self.samples if s.speaker_id in val_speakers]
        else:
            self.samples = [s for s in self.samples if s.speaker_id not in val_speakers]

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get single sample."""
        sample = self.samples[idx]

        # Load audio
        if self.cache_audio and sample.audio_path in self.audio_cache:
            audio = self.audio_cache[sample.audio_path]
        else:
            audio, sr = librosa.load(sample.audio_path, sr=self.sample_rate)
            if self.cache_audio:
                self.audio_cache[sample.audio_path] = audio

        # Get speaker info
        speaker_info = self.speaker_metadata.get(sample.speaker_id)

        return {
            "audio": torch.from_numpy(audio).float(),
            "text": sample.text,
            "speaker_id": sample.speaker_id,
            "speaker_metadata": speaker_info.to_dict() if speaker_info else None,
            "audio_path": str(sample.audio_path),
            "sample_rate": self.sample_rate,
            "metadata": sample.metadata,
        }

    def get_speaker_ids(self) -> List[str]:
        """Get list of unique speaker IDs."""
        return sorted(list(set(s.speaker_id for s in self.samples)))

    def get_speaker_info(self, speaker_id: str) -> Optional[SpeakerMetadata]:
        """Get metadata for specific speaker."""
        return self.speaker_metadata.get(speaker_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            "total_samples": len(self.samples),
            "total_speakers": len(self.get_speaker_ids()),
            "speakers": {},
        }

        # Per-speaker statistics
        for speaker_id in self.get_speaker_ids():
            speaker_samples = [s for s in self.samples if s.speaker_id == speaker_id]
            durations = [s.duration for s in speaker_samples if s.duration]

            stats["speakers"][speaker_id] = {
                "num_samples": len(speaker_samples),
                "total_duration": sum(durations) if durations else None,
                "metadata": (
                    self.speaker_metadata.get(speaker_id).to_dict()
                    if speaker_id in self.speaker_metadata
                    else None
                ),
            }

        return stats


def create_dataloaders(
    train_path: str,
    val_path: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    **dataset_kwargs,
) -> tuple:
    """Create train and validation dataloaders."""
    from torch.utils.data import DataLoader

    # Create datasets
    train_dataset = TsukuyomiDataset(train_path, **dataset_kwargs)

    if val_path:
        val_dataset = TsukuyomiDataset(val_path, **dataset_kwargs)
    else:
        # Use validation split
        val_dataset = TsukuyomiDataset(
            train_path, validation_split=0.05, is_validation=True, **dataset_kwargs
        )
        train_dataset = TsukuyomiDataset(
            train_path, validation_split=0.05, is_validation=False, **dataset_kwargs
        )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for batching."""
    # This is a simplified version - in practice would handle padding
    return {
        "audio": torch.stack([item["audio"] for item in batch]),
        "text": [item["text"] for item in batch],
        "speaker_ids": [item["speaker_id"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }
