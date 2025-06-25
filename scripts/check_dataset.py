#!/usr/bin/env python3
"""JVS LJSpeech形式のデータセットを確認するスクリプト"""

import sys
from pathlib import Path

def check_dataset(data_dir: str):
    """データセットの構造を確認"""
    data_path = Path(data_dir)
    
    print(f"Checking dataset at: {data_path}")
    print(f"Directory exists: {data_path.exists()}")
    
    if not data_path.exists():
        print("ERROR: Directory does not exist!")
        return
    
    # Check for metadata file
    metadata_files = list(data_path.glob("metadata.*"))
    print(f"\nMetadata files found: {len(metadata_files)}")
    for f in metadata_files:
        print(f"  - {f.name}")
        
    # Check specific metadata.csv
    metadata_csv = data_path / "metadata.csv"
    if metadata_csv.exists():
        print(f"\nmetadata.csv exists: {metadata_csv}")
        # Read first few lines
        with open(metadata_csv, 'r', encoding='utf-8') as f:
            print("\nFirst 5 lines of metadata.csv:")
            for i, line in enumerate(f):
                if i >= 5:
                    break
                print(f"  {i+1}: {line.strip()}")
        
        # Count total lines
        with open(metadata_csv, 'r') as f:
            total_lines = sum(1 for line in f)
        print(f"\nTotal lines in metadata.csv: {total_lines}")
    
    # Check for wavs directory
    wavs_dir = data_path / "wavs"
    if wavs_dir.exists():
        wav_files = list(wavs_dir.glob("*.wav"))
        print(f"\nwavs/ directory exists with {len(wav_files)} .wav files")
        if wav_files:
            print("Sample wav files:")
            for f in wav_files[:5]:
                print(f"  - {f.name}")
    else:
        print("\nERROR: wavs/ directory not found!")
        
    # Check if there are audio files in root
    audio_files = list(data_path.glob("*.wav")) + list(data_path.glob("*.mp3"))
    if audio_files:
        print(f"\nFound {len(audio_files)} audio files in root directory")
        for f in audio_files[:5]:
            print(f"  - {f.name}")

if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/jvs_ljspeech"
    check_dataset(data_dir)