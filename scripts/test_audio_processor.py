#!/usr/bin/env python3
"""音声処理のテストスクリプト"""

import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from src.data.audio_processor import AudioProcessor
from src.data.dataset import TsukuyomiDataset
from src.data.collate import tts_collate_fn


def main():
    """音声処理のテスト"""
    # AudioProcessorの初期化
    audio_processor = AudioProcessor(
        sample_rate=22050,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        n_mels=80,
    )
    
    print(f"AudioProcessor initialized:")
    print(f"  Sample rate: {audio_processor.sample_rate}")
    print(f"  Hop length: {audio_processor.hop_length}")
    print(f"  Mel bins: {audio_processor.n_mels}")
    
    # データセットの読み込み
    dataset = TsukuyomiDataset(
        data_root="data/jvs_ljspeech",
        transcript_file="metadata.csv",
        sample_rate=22050,
        cache_audio=False,
    )
    
    if len(dataset) > 0:
        # 単一サンプルのテスト
        sample = dataset[0]
        print(f"\n=== 単一サンプルのテスト ===")
        print(f"テキスト: {sample['text']}")
        print(f"音声shape: {sample['audio'].shape}")
        
        # メルスペクトログラムに変換
        mel = audio_processor.wav_to_mel(sample['audio'])
        print(f"メルスペクトログラムshape: {mel.shape}")
        
        # バッチテスト
        print(f"\n=== バッチ処理のテスト ===")
        batch_size = min(2, len(dataset))
        batch = [dataset[i] for i in range(batch_size)]
        
        # collate関数でバッチ化
        collated = tts_collate_fn(batch, audio_processor=audio_processor)
        
        print(f"バッチサイズ: {batch_size}")
        print(f"音声バッチshape: {collated['audio'].shape}")
        print(f"音声長: {collated['audio_lengths']}")
        print(f"メルバッチshape: {collated['mel_targets'].shape}")
        print(f"メル長: {collated['mel_lengths']}")
        
        # 正規化の確認
        print(f"\n=== 正規化の確認 ===")
        mel_min = collated['mel_targets'].min().item()
        mel_max = collated['mel_targets'].max().item()
        mel_mean = collated['mel_targets'].mean().item()
        print(f"メル値の範囲: [{mel_min:.3f}, {mel_max:.3f}]")
        print(f"メル値の平均: {mel_mean:.3f}")
        
    else:
        print("データセットが空です")


if __name__ == "__main__":
    main()