"""Tests for flexible dataset loader."""

import pytest
import json
import csv
from pathlib import Path
import numpy as np
import soundfile as sf
import tempfile
import shutil
import torch

from src.data.dataset import TsukuyomiDataset, SpeakerMetadata, AudioSample


class TestDataset:
    """Test suite for dataset loading."""
    
    @pytest.fixture
    def temp_dataset_dir(self):
        """Create temporary dataset directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_audio_files(self, temp_dataset_dir):
        """Create sample audio files."""
        audio_dir = temp_dataset_dir / "audio"
        audio_dir.mkdir()
        
        # Create dummy audio files
        sample_rate = 48000
        duration = 1.0  # 1 second
        
        files = []
        for i in range(5):
            audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)
            audio_path = audio_dir / f"{i:03d}.wav"
            sf.write(audio_path, audio, sample_rate)
            files.append(audio_path)
            
        return files
    
    def test_load_txt_transcripts(self, temp_dataset_dir, sample_audio_files):
        """Test loading pipe-delimited text transcripts."""
        # Create transcript file
        transcript_path = temp_dataset_dir / "transcripts.txt"
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write("# Comment line\n")
            f.write("audio/000.wav|こんにちは、月読です。\n")
            f.write("audio/001.wav|今日はいい天気ですね。|speaker_001\n")
            f.write("audio/002.wav|音声合成の実験を行います。|speaker_002\n")
            
        # Load dataset
        dataset = TsukuyomiDataset(temp_dataset_dir)
        
        assert len(dataset) == 3
        
        # Check first sample
        sample = dataset[0]
        assert sample["text"] == "こんにちは、月読です。"
        assert sample["speaker_id"] == "default"
        assert sample["audio"].shape[0] == 48000  # 1 second at 48kHz
        
        # Check sample with speaker
        sample = dataset[1]
        assert sample["speaker_id"] == "speaker_001"
    
    def test_load_csv_transcripts(self, temp_dataset_dir, sample_audio_files):
        """Test loading CSV transcripts."""
        transcript_path = temp_dataset_dir / "transcripts.csv"
        with open(transcript_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['audio', 'text', 'speaker'])
            writer.writeheader()
            writer.writerow({
                'audio': 'audio/000.wav',
                'text': 'CSVテスト1',
                'speaker': 'speaker_001'
            })
            writer.writerow({
                'audio': 'audio/001.wav',
                'text': 'CSVテスト2',
                'speaker': 'speaker_002'
            })
            
        dataset = TsukuyomiDataset(temp_dataset_dir, transcript_file="transcripts.csv")
        
        assert len(dataset) == 2
        assert dataset[0]["text"] == "CSVテスト1"
        assert dataset[0]["speaker_id"] == "speaker_001"
    
    def test_load_json_transcripts_list(self, temp_dataset_dir, sample_audio_files):
        """Test loading JSON transcripts in list format."""
        transcript_path = temp_dataset_dir / "transcripts.json"
        data = [
            {
                "audio": "audio/000.wav",
                "text": "JSONテスト1",
                "speaker": "speaker_001",
                "emotion": "happy"
            },
            {
                "audio": "audio/001.wav",
                "text": "JSONテスト2",
                "speaker": "speaker_002"
            }
        ]
        
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            
        dataset = TsukuyomiDataset(temp_dataset_dir, transcript_file="transcripts.json")
        
        assert len(dataset) == 2
        assert dataset[0]["text"] == "JSONテスト1"
        assert dataset[0]["metadata"]["emotion"] == "happy"
    
    def test_load_json_transcripts_dict(self, temp_dataset_dir, sample_audio_files):
        """Test loading JSON transcripts in dict format."""
        transcript_path = temp_dataset_dir / "transcripts.json"
        data = {
            "audio/000.wav": "簡単なテキスト",
            "audio/001.wav": {
                "text": "詳細な情報付き",
                "speaker": "speaker_001",
                "style": "formal"
            }
        }
        
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            
        dataset = TsukuyomiDataset(temp_dataset_dir, transcript_file="transcripts.json")
        
        assert len(dataset) == 2
        assert dataset[0]["text"] == "簡単なテキスト"
        assert dataset[0]["speaker_id"] == "default"
        assert dataset[1]["text"] == "詳細な情報付き"
        assert dataset[1]["metadata"]["style"] == "formal"
    
    def test_speaker_metadata_full(self, temp_dataset_dir, sample_audio_files):
        """Test loading full speaker metadata."""
        # Create transcript
        transcript_path = temp_dataset_dir / "transcripts.txt"
        with open(transcript_path, 'w') as f:
            f.write("audio/000.wav|テスト|speaker_001\n")
            
        # Create metadata
        metadata_path = temp_dataset_dir / "metadata.json"
        metadata = {
            "speaker_001": {
                "name": "キャラクターA",
                "gender": "female",
                "age_range": "20-30",
                "dialect": "tokyo",
                "voice_characteristics": {
                    "pitch": "中高音",
                    "speaking_rate": "標準",
                    "emotion_range": "表現豊か"
                }
            }
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False)
            
        dataset = TsukuyomiDataset(temp_dataset_dir)
        
        sample = dataset[0]
        assert sample["speaker_metadata"] is not None
        assert sample["speaker_metadata"]["name"] == "キャラクターA"
        assert sample["speaker_metadata"]["gender"] == "female"
        assert sample["speaker_metadata"]["voice_characteristics"]["pitch"] == "中高音"
    
    def test_speaker_metadata_partial(self, temp_dataset_dir, sample_audio_files):
        """Test loading partial speaker metadata with nulls."""
        transcript_path = temp_dataset_dir / "transcripts.txt"
        with open(transcript_path, 'w') as f:
            f.write("audio/000.wav|テスト1|speaker_001\n")
            f.write("audio/001.wav|テスト2|speaker_002\n")
            
        # Partial metadata
        metadata_path = temp_dataset_dir / "metadata.json"
        metadata = {
            "speaker_001": {
                "name": "話者1",
                "gender": None,  # Explicit null
                "dialect": "osaka"
                # Missing other fields
            },
            "speaker_002": {
                # Only name
                "name": "話者2"
            }
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False)
            
        dataset = TsukuyomiDataset(temp_dataset_dir)
        
        # Check speaker 1
        sample1 = dataset[0]
        meta1 = sample1["speaker_metadata"]
        assert meta1["name"] == "話者1"
        assert meta1["gender"] is None
        assert meta1["dialect"] == "osaka"
        assert meta1["age_range"] is None  # Missing field
        
        # Check speaker 2
        sample2 = dataset[1]
        meta2 = sample2["speaker_metadata"]
        assert meta2["name"] == "話者2"
        assert meta2["gender"] is None
        assert meta2["dialect"] is None
    
    def test_no_metadata(self, temp_dataset_dir, sample_audio_files):
        """Test dataset without metadata file."""
        transcript_path = temp_dataset_dir / "transcripts.txt"
        with open(transcript_path, 'w') as f:
            f.write("audio/000.wav|テスト|speaker_001\n")
            
        # No metadata file
        dataset = TsukuyomiDataset(temp_dataset_dir)
        
        sample = dataset[0]
        assert sample["speaker_metadata"] is None
    
    def test_duration_filtering(self, temp_dataset_dir):
        """Test filtering by duration."""
        audio_dir = temp_dataset_dir / "audio"
        audio_dir.mkdir()
        
        # Create audio files with different durations
        sample_rate = 48000
        durations = [0.3, 0.8, 1.5, 3.0, 20.0]  # seconds
        
        transcript_path = temp_dataset_dir / "transcripts.txt"
        with open(transcript_path, 'w') as f:
            for i, duration in enumerate(durations):
                audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)
                audio_path = audio_dir / f"{i:03d}.wav"
                sf.write(audio_path, audio, sample_rate)
                f.write(f"audio/{i:03d}.wav|テスト{i}\n")
                
        # Test with duration limits
        dataset = TsukuyomiDataset(
            temp_dataset_dir,
            min_duration=0.5,
            max_duration=15.0
        )
        
        # Should filter out 0.3s and 20.0s files
        assert len(dataset) == 3
    
    def test_validation_split(self, temp_dataset_dir, sample_audio_files):
        """Test train/validation split."""
        transcript_path = temp_dataset_dir / "transcripts.txt"
        with open(transcript_path, 'w') as f:
            # 10 samples, 2 speakers
            for i in range(10):
                speaker = f"speaker_{i % 2}"
                f.write(f"audio/{i%5:03d}.wav|テスト{i}|{speaker}\n")
                
        # Create train dataset
        train_dataset = TsukuyomiDataset(
            temp_dataset_dir,
            validation_split=0.5,
            is_validation=False,
            random_seed=42
        )
        
        # Create validation dataset
        val_dataset = TsukuyomiDataset(
            temp_dataset_dir,
            validation_split=0.5,
            is_validation=True,
            random_seed=42
        )
        
        # Check split
        assert len(train_dataset) + len(val_dataset) == 10
        
        # Check no overlap in speakers
        train_speakers = set(train_dataset.get_speaker_ids())
        val_speakers = set(val_dataset.get_speaker_ids())
        assert len(train_speakers & val_speakers) == 0
    
    def test_auto_format_detection(self, temp_dataset_dir, sample_audio_files):
        """Test automatic format detection."""
        # Test with pipe-delimited file without .txt extension
        transcript_path = temp_dataset_dir / "transcripts"
        with open(transcript_path, 'w') as f:
            f.write("audio/000.wav|自動検出テスト\n")
            
        dataset = TsukuyomiDataset(temp_dataset_dir, transcript_file="transcripts")
        assert len(dataset) == 1
        assert dataset[0]["text"] == "自動検出テスト"
    
    def test_speaker_statistics(self, temp_dataset_dir, sample_audio_files):
        """Test dataset statistics."""
        transcript_path = temp_dataset_dir / "transcripts.txt"
        with open(transcript_path, 'w') as f:
            f.write("audio/000.wav|テスト1|speaker_001\n")
            f.write("audio/001.wav|テスト2|speaker_001\n")
            f.write("audio/002.wav|テスト3|speaker_002\n")
            
        dataset = TsukuyomiDataset(temp_dataset_dir)
        stats = dataset.get_statistics()
        
        assert stats["total_samples"] == 3
        assert stats["total_speakers"] == 2
        assert stats["speakers"]["speaker_001"]["num_samples"] == 2
        assert stats["speakers"]["speaker_002"]["num_samples"] == 1
    
    def test_cache_audio(self, temp_dataset_dir, sample_audio_files):
        """Test audio caching."""
        transcript_path = temp_dataset_dir / "transcripts.txt"
        with open(transcript_path, 'w') as f:
            f.write("audio/000.wav|キャッシュテスト\n")
            
        dataset = TsukuyomiDataset(temp_dataset_dir, cache_audio=True)
        
        # First access
        sample1 = dataset[0]
        audio1 = sample1["audio"]
        
        # Second access (should be cached)
        sample2 = dataset[0]
        audio2 = sample2["audio"]
        
        # Should be the same object
        assert torch.equal(audio1, audio2)
        assert len(dataset.audio_cache) == 1


class TestSpeakerMetadata:
    """Test speaker metadata handling."""
    
    def test_from_dict_full(self):
        """Test creating from full dictionary."""
        data = {
            "speaker_id": "001",
            "name": "テスト話者",
            "gender": "female",
            "age_range": "20-30",
            "dialect": "kansai",
            "voice_characteristics": {
                "pitch": "high",
                "rate": "fast"
            }
        }
        
        metadata = SpeakerMetadata.from_dict(data)
        
        assert metadata.speaker_id == "001"
        assert metadata.name == "テスト話者"
        assert metadata.gender == "female"
        assert metadata.voice_characteristics["pitch"] == "high"
    
    def test_from_dict_partial(self):
        """Test creating from partial dictionary."""
        data = {
            "speaker_id": "002",
            "name": "部分データ"
            # Missing other fields
        }
        
        metadata = SpeakerMetadata.from_dict(data)
        
        assert metadata.speaker_id == "002"
        assert metadata.name == "部分データ"
        assert metadata.gender is None
        assert metadata.age_range is None
        assert metadata.dialect is None
        assert metadata.voice_characteristics == {}
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        metadata = SpeakerMetadata(
            speaker_id="003",
            name="変換テスト",
            gender="male",
            dialect="tokyo"
        )
        
        data = metadata.to_dict()
        
        assert data["speaker_id"] == "003"
        assert data["name"] == "変換テスト"
        assert data["gender"] == "male"
        assert data["age_range"] is None
        assert data["dialect"] == "tokyo"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])