#\!/usr/bin/env python3
"""データセットの確認スクリプト"""

import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.dataset import TsukuyomiDataset


def main():
    """データセットの内容を確認"""
    data_root = Path("data/jvs_ljspeech")
    
    print(f"データディレクトリ: {data_root}")
    print(f"存在確認: {data_root.exists()}")
    
    if data_root.exists():
        # metadata.csvの確認
        metadata_files = list(data_root.glob("metadata*.csv"))
        print(f"\nメタデータファイル: {metadata_files}")
        
        if metadata_files:
            with open(metadata_files[0], "r", encoding="utf-8") as f:
                lines = f.readlines()
                print(f"総行数: {len(lines)}")
                print("最初の5行:")
                for i, line in enumerate(lines[:5]):
                    print(f"  {i+1}: {line.strip()}")
        
        # wavファイルの確認
        wav_dir = data_root / "wavs"
        if wav_dir.exists():
            wav_files = list(wav_dir.glob("*.wav"))
            print(f"\n音声ファイル数: {len(wav_files)}")
            if wav_files:
                print("最初の5ファイル:")
                for f in sorted(wav_files)[:5]:
                    print(f"  {f.name}")
    
    # データセットの読み込みテスト
    try:
        print("\n=== データセット読み込みテスト ===")
        dataset = TsukuyomiDataset(
            data_root=data_root,
            transcript_file="metadata.csv",
            sample_rate=22050,
            max_duration=10.0,
            min_duration=0.5,
            cache_audio=False,
            validation_split=0.1,
            is_validation=False,
        )
        print(f"学習サンプル数: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nサンプル例:")
            print(f"  テキスト: {sample['text']}")
            print(f"  話者ID: {sample['speaker_id']}")
            print(f"  音声shape: {sample['audio'].shape}")
            
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()