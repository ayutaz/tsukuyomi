#!/usr/bin/env python3
"""Automatic audio preprocessing pipeline for TTS dataset creation

This script handles:
- Audio format conversion
- Resampling
- Silence trimming
- Loudness normalization
- Noise reduction
- Speaker diarization
- Quality filtering
- Dataset splitting
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import webrtcvad
from pydub import AudioSegment
from pydub.silence import split_on_silence
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import noisereduce as nr
import pyloudnorm as pyln

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.frontend.text_normalizer import TextNormalizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Comprehensive audio preprocessing pipeline"""
    
    def __init__(
        self,
        target_sr: int = 24000,
        target_db: float = -20.0,
        silence_threshold: float = -40.0,
        min_silence_len: int = 500,
        max_duration: float = 15.0,
        min_duration: float = 0.5,
        noise_reduction: bool = True,
        normalize_loudness: bool = True,
        trim_silence: bool = True,
        apply_filters: bool = True,
    ):
        self.target_sr = target_sr
        self.target_db = target_db
        self.silence_threshold = silence_threshold
        self.min_silence_len = min_silence_len
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.noise_reduction = noise_reduction
        self.normalize_loudness = normalize_loudness
        self.trim_silence = trim_silence
        self.apply_filters = apply_filters
        
        # Initialize VAD
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(1)  # Aggressiveness mode: 0-3
        
        # Initialize loudness meter
        self.meter = pyln.Meter(target_sr)
        
        # Text normalizer
        self.text_normalizer = TextNormalizer()
        
    def process_audio_file(
        self,
        input_path: Path,
        output_path: Path,
        transcript: Optional[str] = None
    ) -> Dict:
        """Process a single audio file"""
        try:
            # Load audio
            audio, sr = librosa.load(str(input_path), sr=None, mono=True)
            
            # Resample if needed
            if sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
                sr = self.target_sr
                
            # Apply preprocessing steps
            if self.trim_silence:
                audio = self._trim_silence(audio, sr)
                
            if self.noise_reduction:
                audio = self._reduce_noise(audio, sr)
                
            if self.apply_filters:
                audio = self._apply_filters(audio, sr)
                
            if self.normalize_loudness:
                audio = self._normalize_loudness(audio, sr)
                
            # Check duration
            duration = len(audio) / sr
            if duration < self.min_duration or duration > self.max_duration:
                return {
                    'status': 'rejected',
                    'reason': f'Duration {duration:.2f}s outside range [{self.min_duration}, {self.max_duration}]'
                }
                
            # Quality checks
            quality_check = self._check_audio_quality(audio, sr)
            if not quality_check['passed']:
                return {
                    'status': 'rejected',
                    'reason': quality_check['reason']
                }
                
            # Save processed audio
            sf.write(str(output_path), audio, sr)
            
            # Process transcript if provided
            if transcript:
                normalized_transcript = self.text_normalizer.normalize(transcript)
            else:
                normalized_transcript = None
                
            return {
                'status': 'success',
                'duration': duration,
                'sample_rate': sr,
                'transcript': transcript,
                'normalized_transcript': normalized_transcript,
                'snr': quality_check['snr'],
                'energy': quality_check['energy'],
            }
            
        except Exception as e:
            logger.error(f"Error processing {input_path}: {e}")
            return {
                'status': 'error',
                'reason': str(e)
            }
            
    def _trim_silence(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Trim silence from audio"""
        # Convert to pydub AudioSegment
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=1
        )
        
        # Split on silence
        chunks = split_on_silence(
            audio_segment,
            min_silence_len=self.min_silence_len,
            silence_thresh=self.silence_threshold,
            keep_silence=100  # Keep 100ms of silence at boundaries
        )
        
        if not chunks:
            return audio
            
        # Combine chunks
        combined = chunks[0]
        for chunk in chunks[1:]:
            combined += chunk
            
        # Convert back to numpy
        samples = np.array(combined.get_array_of_samples()).astype(np.float32) / 32767.0
        
        return samples
        
    def _reduce_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Reduce noise using spectral gating"""
        # Estimate noise from the first 0.5 seconds
        noise_sample_length = int(0.5 * sr)
        if len(audio) > noise_sample_length:
            noise_sample = audio[:noise_sample_length]
        else:
            noise_sample = audio
            
        # Reduce noise
        reduced_audio = nr.reduce_noise(
            y=audio,
            sr=sr,
            y_noise=noise_sample,
            prop_decrease=0.8
        )
        
        return reduced_audio
        
    def _apply_filters(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply bandpass filter to remove extreme frequencies"""
        # High-pass filter at 80 Hz
        b_high, a_high = butter(4, 80 / (sr / 2), btype='high')
        audio = filtfilt(b_high, a_high, audio)
        
        # Low-pass filter at 8000 Hz
        b_low, a_low = butter(4, 8000 / (sr / 2), btype='low')
        audio = filtfilt(b_low, a_low, audio)
        
        return audio
        
    def _normalize_loudness(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Normalize audio loudness to target dB LUFS"""
        # Measure loudness
        loudness = self.meter.integrated_loudness(audio)
        
        # Normalize
        audio_normalized = pyln.normalize.loudness(
            audio,
            loudness,
            self.target_db
        )
        
        # Prevent clipping
        if np.abs(audio_normalized).max() > 0.99:
            audio_normalized = audio_normalized / (np.abs(audio_normalized).max() + 0.01)
            
        return audio_normalized
        
    def _check_audio_quality(self, audio: np.ndarray, sr: int) -> Dict:
        """Check audio quality metrics"""
        # Calculate SNR
        signal_power = np.mean(audio ** 2)
        noise_floor = np.percentile(np.abs(audio), 5) ** 2
        
        if noise_floor > 0:
            snr = 10 * np.log10(signal_power / noise_floor)
        else:
            snr = float('inf')
            
        # Calculate energy
        energy = np.sqrt(np.mean(audio ** 2))
        
        # VAD check - ensure sufficient voiced segments
        frame_duration = 30  # ms
        frame_length = int(sr * frame_duration / 1000)
        num_frames = len(audio) // frame_length
        
        voiced_frames = 0
        for i in range(num_frames):
            frame = audio[i * frame_length:(i + 1) * frame_length]
            frame_int16 = (frame * 32767).astype(np.int16).tobytes()
            
            if self.vad.is_speech(frame_int16, sr):
                voiced_frames += 1
                
        voiced_ratio = voiced_frames / num_frames if num_frames > 0 else 0
        
        # Quality criteria
        passed = True
        reason = []
        
        if snr < 10:
            passed = False
            reason.append(f"Low SNR: {snr:.1f} dB")
            
        if energy < 0.001:
            passed = False
            reason.append(f"Low energy: {energy:.4f}")
            
        if voiced_ratio < 0.3:
            passed = False
            reason.append(f"Low voiced ratio: {voiced_ratio:.2f}")
            
        return {
            'passed': passed,
            'reason': ', '.join(reason) if reason else 'OK',
            'snr': snr,
            'energy': energy,
            'voiced_ratio': voiced_ratio
        }


class DatasetBuilder:
    """Build TTS dataset from raw audio files"""
    
    def __init__(
        self,
        preprocessor: AudioPreprocessor,
        output_format: str = "ljspeech"
    ):
        self.preprocessor = preprocessor
        self.output_format = output_format
        
    def build_from_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        transcript_file: Optional[Path] = None,
        speaker_id: Optional[str] = None,
        num_workers: int = 4
    ) -> Dict:
        """Build dataset from directory of audio files"""
        # Create output structure
        output_dir.mkdir(parents=True, exist_ok=True)
        wavs_dir = output_dir / "wavs"
        wavs_dir.mkdir(exist_ok=True)
        
        # Load transcripts if provided
        transcripts = {}
        if transcript_file and transcript_file.exists():
            if transcript_file.suffix == '.json':
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    transcripts = json.load(f)
            elif transcript_file.suffix == '.txt':
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if '\t' in line:
                            filename, text = line.strip().split('\t', 1)
                            transcripts[filename] = text
                            
        # Find audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(input_dir.glob(f"**/*{ext}"))
            
        logger.info(f"Found {len(audio_files)} audio files")
        
        # Process files
        if num_workers > 1:
            results = self._process_parallel(
                audio_files,
                wavs_dir,
                transcripts,
                speaker_id,
                num_workers
            )
        else:
            results = self._process_sequential(
                audio_files,
                wavs_dir,
                transcripts,
                speaker_id
            )
            
        # Create metadata
        metadata = self._create_metadata(results, output_dir)
        
        # Split dataset
        self._split_dataset(metadata, output_dir)
        
        # Generate statistics
        stats = self._generate_statistics(results)
        
        # Save statistics
        with open(output_dir / "preprocessing_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
            
        logger.info(f"Dataset built successfully!")
        logger.info(f"Total files: {stats['total_files']}")
        logger.info(f"Successful: {stats['successful']}")
        logger.info(f"Rejected: {stats['rejected']}")
        logger.info(f"Errors: {stats['errors']}")
        
        return stats
        
    def _process_sequential(
        self,
        audio_files: List[Path],
        wavs_dir: Path,
        transcripts: Dict,
        speaker_id: Optional[str]
    ) -> List[Dict]:
        """Process files sequentially"""
        results = []
        
        for idx, audio_file in enumerate(tqdm(audio_files, desc="Processing audio")):
            # Generate output filename
            if speaker_id:
                output_name = f"{speaker_id}_{idx:05d}.wav"
            else:
                output_name = f"{idx:05d}.wav"
                
            output_path = wavs_dir / output_name
            
            # Get transcript
            transcript = transcripts.get(audio_file.stem, None)
            
            # Process audio
            result = self.preprocessor.process_audio_file(
                audio_file,
                output_path,
                transcript
            )
            
            result['input_file'] = str(audio_file)
            result['output_file'] = output_name
            
            results.append(result)
            
        return results
        
    def _process_parallel(
        self,
        audio_files: List[Path],
        wavs_dir: Path,
        transcripts: Dict,
        speaker_id: Optional[str],
        num_workers: int
    ) -> List[Dict]:
        """Process files in parallel"""
        # Prepare tasks
        tasks = []
        for idx, audio_file in enumerate(audio_files):
            if speaker_id:
                output_name = f"{speaker_id}_{idx:05d}.wav"
            else:
                output_name = f"{idx:05d}.wav"
                
            output_path = wavs_dir / output_name
            transcript = transcripts.get(audio_file.stem, None)
            
            tasks.append((
                audio_file,
                output_path,
                transcript,
                output_name
            ))
            
        # Process in parallel
        with mp.Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(self._process_worker, tasks),
                total=len(tasks),
                desc="Processing audio"
            ))
            
        return results
        
    def _process_worker(self, task: Tuple) -> Dict:
        """Worker function for parallel processing"""
        audio_file, output_path, transcript, output_name = task
        
        result = self.preprocessor.process_audio_file(
            audio_file,
            output_path,
            transcript
        )
        
        result['input_file'] = str(audio_file)
        result['output_file'] = output_name
        
        return result
        
    def _create_metadata(self, results: List[Dict], output_dir: Path) -> List[Dict]:
        """Create metadata file"""
        metadata = []
        
        if self.output_format == "ljspeech":
            # Create LJSpeech format metadata.csv
            csv_lines = []
            
            for result in results:
                if result['status'] == 'success':
                    filename = result['output_file'].replace('.wav', '')
                    transcript = result.get('transcript', '')
                    normalized = result.get('normalized_transcript', transcript)
                    
                    csv_lines.append(f"{filename}|{transcript}|{normalized}")
                    
                    metadata.append({
                        'filename': filename,
                        'transcript': transcript,
                        'normalized': normalized,
                        'duration': result['duration'],
                        'speaker_id': 0  # Single speaker for LJSpeech
                    })
                    
            # Save CSV
            with open(output_dir / "metadata.csv", 'w', encoding='utf-8') as f:
                f.write('\n'.join(csv_lines))
                
        else:
            # Custom JSON format
            for result in results:
                if result['status'] == 'success':
                    metadata.append({
                        'audio_file': result['output_file'],
                        'transcript': result.get('transcript', ''),
                        'normalized_transcript': result.get('normalized_transcript', ''),
                        'duration': result['duration'],
                        'sample_rate': result['sample_rate'],
                        'snr': result.get('snr', 0),
                        'energy': result.get('energy', 0),
                    })
                    
            # Save JSON
            with open(output_dir / "metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
                
        return metadata
        
    def _split_dataset(
        self,
        metadata: List[Dict],
        output_dir: Path,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1
    ):
        """Split dataset into train/val/test"""
        np.random.shuffle(metadata)
        
        n_total = len(metadata)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_files = [m['filename'] for m in metadata[:n_train]]
        val_files = [m['filename'] for m in metadata[n_train:n_train + n_val]]
        test_files = [m['filename'] for m in metadata[n_train + n_val:]]
        
        # Save file lists
        with open(output_dir / "train_files.txt", 'w') as f:
            f.write('\n'.join(train_files))
            
        with open(output_dir / "val_files.txt", 'w') as f:
            f.write('\n'.join(val_files))
            
        with open(output_dir / "test_files.txt", 'w') as f:
            f.write('\n'.join(test_files))
            
        logger.info(f"Dataset split: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")
        
    def _generate_statistics(self, results: List[Dict]) -> Dict:
        """Generate preprocessing statistics"""
        stats = {
            'total_files': len(results),
            'successful': 0,
            'rejected': 0,
            'errors': 0,
            'total_duration': 0,
            'rejection_reasons': {},
            'quality_metrics': {
                'snr': [],
                'energy': [],
                'duration': []
            }
        }
        
        for result in results:
            if result['status'] == 'success':
                stats['successful'] += 1
                stats['total_duration'] += result['duration']
                stats['quality_metrics']['duration'].append(result['duration'])
                
                if 'snr' in result:
                    stats['quality_metrics']['snr'].append(result['snr'])
                if 'energy' in result:
                    stats['quality_metrics']['energy'].append(result['energy'])
                    
            elif result['status'] == 'rejected':
                stats['rejected'] += 1
                reason = result.get('reason', 'Unknown')
                stats['rejection_reasons'][reason] = stats['rejection_reasons'].get(reason, 0) + 1
                
            else:
                stats['errors'] += 1
                
        # Calculate averages
        if stats['quality_metrics']['snr']:
            stats['avg_snr'] = np.mean(stats['quality_metrics']['snr'])
        if stats['quality_metrics']['energy']:
            stats['avg_energy'] = np.mean(stats['quality_metrics']['energy'])
        if stats['quality_metrics']['duration']:
            stats['avg_duration'] = np.mean(stats['quality_metrics']['duration'])
            
        # Remove raw lists for cleaner output
        del stats['quality_metrics']
        
        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Automatic audio preprocessing for TTS dataset creation"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory containing audio files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for processed dataset",
    )
    parser.add_argument(
        "--transcript-file",
        type=str,
        help="Path to transcript file (JSON or TSV)",
    )
    parser.add_argument(
        "--speaker-id",
        type=str,
        help="Speaker ID for multi-speaker datasets",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=24000,
        help="Target sample rate",
    )
    parser.add_argument(
        "--target-db",
        type=float,
        default=-20.0,
        help="Target loudness in dB LUFS",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=15.0,
        help="Maximum audio duration in seconds",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.5,
        help="Minimum audio duration in seconds",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--no-noise-reduction",
        action="store_true",
        help="Disable noise reduction",
    )
    parser.add_argument(
        "--no-normalization",
        action="store_true",
        help="Disable loudness normalization",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="ljspeech",
        choices=["ljspeech", "json"],
        help="Output dataset format",
    )
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = AudioPreprocessor(
        target_sr=args.sample_rate,
        target_db=args.target_db,
        max_duration=args.max_duration,
        min_duration=args.min_duration,
        noise_reduction=not args.no_noise_reduction,
        normalize_loudness=not args.no_normalization,
    )
    
    # Create dataset builder
    builder = DatasetBuilder(preprocessor, args.output_format)
    
    # Build dataset
    stats = builder.build_from_directory(
        Path(args.input_dir),
        Path(args.output_dir),
        Path(args.transcript_file) if args.transcript_file else None,
        args.speaker_id,
        args.num_workers
    )
    
    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    main()