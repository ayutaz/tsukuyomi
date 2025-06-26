#!/usr/bin/env python3
"""データディレクトリ構造を確認するスクリプト"""

import os
from pathlib import Path


def check_directory_structure():
    """ディレクトリ構造を詳しく確認"""
    base_path = Path("data/jvs_ljspeech")

    print(f"Base path: {base_path.absolute()}")
    print(f"Exists: {base_path.exists()}")

    if base_path.exists():
        print("\n=== Directory contents ===")
        for item in sorted(base_path.iterdir()):
            if item.is_file():
                print(f"FILE: {item.name} ({item.stat().st_size:,} bytes)")
            else:
                print(f"DIR:  {item.name}/")
                # サブディレクトリの内容も表示
                if item.name == "wavs" or item.name == "audio":
                    sub_items = list(item.iterdir())
                    print(f"      Contains {len(sub_items)} files")
                    for sub_item in sorted(sub_items)[:5]:  # 最初の5つ
                        print(f"      - {sub_item.name}")
                    if len(sub_items) > 5:
                        print(f"      ... and {len(sub_items) - 5} more files")

    # 他の可能性のあるパスも確認
    other_paths = [
        "data/jvs",
        "data/JVS",
        "data/jvs_dataset",
        "../data/jvs_ljspeech",
        "~/data/jvs_ljspeech",
    ]

    print("\n=== Checking other possible paths ===")
    for path_str in other_paths:
        path = Path(path_str).expanduser()
        if path.exists():
            print(f"✓ Found: {path_str} -> {path.absolute()}")
            # 音声ファイルを探す
            wav_files = list(path.rglob("*.wav"))[:5]
            if wav_files:
                print("  Contains WAV files:")
                for wav in wav_files:
                    print(f"    - {wav.relative_to(path)}")
        else:
            print(f"✗ Not found: {path_str}")

    # 環境変数も確認
    print("\n=== Environment variables ===")
    for key in ["DATA_DIR", "JVS_DATA_DIR", "TSUKUYOMI_DATA_DIR"]:
        value = os.environ.get(key)
        if value:
            print(f"{key}={value}")


if __name__ == "__main__":
    check_directory_structure()
