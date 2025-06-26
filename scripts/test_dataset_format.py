#!/usr/bin/env python3
"""データセットフォーマットのテストスクリプト"""

import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.dataset import TsukuyomiDataset


def main():
    """データセットのロードをテスト"""
    data_root = Path("data/jvs_ljspeech")
    
    print("JVSデータセットのロードテスト")
    print(f"データディレクトリ: {data_root}")
    
    # データセットを作成
    dataset = TsukuyomiDataset(
        data_root=data_root,
        transcript_file="metadata.csv",
        sample_rate=22050,
        cache_audio=False,
        max_duration=10.0,
        min_duration=0.5,
    )
    
    print(f"\n総サンプル数: {len(dataset)}")
    
    # スピーカーIDの分布を確認
    speaker_ids = dataset.get_speaker_ids()
    print(f"ユニークスピーカー数: {len(speaker_ids)}")
    print(f"スピーカーID例: {speaker_ids[:5]}")
    
    # 最初の5サンプルを確認
    print("\n=== 最初の5サンプル ===")
    for i in range(min(5, len(dataset))):
        sample = dataset.samples[i]
        print(f"\nサンプル {i+1}:")
        print(f"  音声パス: {sample.audio_path}")
        print(f"  スピーカーID: {sample.speaker_id}")
        print(f"  テキスト: {sample.text}")
        print(f"  テキスト長: {len(sample.text)} 文字")
        
        # 日本語文字の種類をチェック
        has_hiragana = any('\u3040' <= c <= '\u309f' for c in sample.text)
        has_katakana = any('\u30a0' <= c <= '\u30ff' for c in sample.text)
        has_kanji = any('\u4e00' <= c <= '\u9fff' for c in sample.text)
        
        char_types = []
        if has_hiragana:
            char_types.append("ひらがな")
        if has_katakana:
            char_types.append("カタカナ")
        if has_kanji:
            char_types.append("漢字")
        
        print(f"  文字種: {', '.join(char_types) if char_types else 'なし'}")
    
    # 実際のデータをロード（1サンプルのみ）
    if len(dataset) > 0:
        print("\n=== 実データのロードテスト ===")
        data = dataset[0]
        print(f"データキー: {list(data.keys())}")
        print(f"音声shape: {data['audio'].shape}")
        print(f"テキスト: {data['text']}")
        print(f"スピーカーID: {data['speaker_id']}")

if __name__ == "__main__":
    main()
