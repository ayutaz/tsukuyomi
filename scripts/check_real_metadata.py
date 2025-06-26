#!/usr/bin/env python3
"""実際のJVSメタデータファイルを確認"""

from pathlib import Path


def check_metadata():
    """メタデータファイルの内容を詳しく確認"""
    data_root = Path("data/jvs_ljspeech")
    
    # 可能性のあるメタデータファイル
    metadata_files = [
        "metadata.csv",
        "metadata_multispeaker.csv",
        "metadata_multispeaker_pyopenjtalk_plus_processed.csv",
    ]
    
    for filename in metadata_files:
        file_path = data_root / filename
        if file_path.exists():
            print(f"\n=== {filename} ===")
            print(f"ファイルサイズ: {file_path.stat().st_size:,} bytes")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                # 最初の10行を読む
                lines = []
                for i, line in enumerate(f):
                    if i >= 10:
                        break
                    lines.append(line.strip())
                
                print("\n最初の10行:")
                for i, line in enumerate(lines):
                    print(f"{i+1}: {line}")
                
                # 各行のカラム数を確認
                print("\nカラム分析:")
                for i, line in enumerate(lines[:5]):
                    parts = line.split('|')
                    print(f"行{i+1}: {len(parts)}カラム")
                    for j, part in enumerate(parts):
                        # 日本語を含むかチェック
                        has_japanese = any(
                            '\u3040' <= c <= '\u309f' or  # ひらがな
                            '\u30a0' <= c <= '\u30ff' or  # カタカナ
                            '\u4e00' <= c <= '\u9fff'     # 漢字
                            for c in part
                        )
                        if len(part) > 50:
                            part = part[:50] + "..."
                        japanese_flag = " [日本語]" if has_japanese else ""
                        print(f"  カラム{j+1}: {part}{japanese_flag}")
    
    # wavsディレクトリの内容も確認
    wavs_dir = data_root / "wavs"
    if wavs_dir.exists():
        wav_files = list(wavs_dir.glob("*.wav"))
        print("\n\n=== WAVファイル ===")
        print(f"総数: {len(wav_files)}")
        print("最初の10ファイル:")
        for wav in sorted(wav_files)[:10]:
            print(f"  {wav.name}")

if __name__ == "__main__":
    check_metadata()
