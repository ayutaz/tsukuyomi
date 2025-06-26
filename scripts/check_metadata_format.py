#!/usr/bin/env python3
"""メタデータファイルのフォーマットを確認"""

from pathlib import Path


def check_metadata_format():
    """各メタデータファイルのフォーマットを確認"""
    data_root = Path("data/jvs_ljspeech")

    # 確認するメタデータファイル
    metadata_files = [
        "metadata.csv",
        "metadata_multispeaker.csv",
        "metadata_multispeaker_pyopenjtalk_plus_processed.csv",
        "metadata_pyopenjtalk_plus_processed.csv",
    ]

    # WAVファイルのサンプルも取得
    wavs_dir = data_root / "wavs"
    wav_files = sorted(list(wavs_dir.glob("*.wav")))[:5] if wavs_dir.exists() else []

    print("=== WAVファイルのサンプル ===")
    for wav in wav_files:
        print(f"  {wav.name}")

    # 各メタデータファイルを確認
    for filename in metadata_files:
        file_path = data_root / filename
        if not file_path.exists():
            continue

        print(f"\n=== {filename} ===")
        print(f"ファイルサイズ: {file_path.stat().st_size:,} bytes")

        with open(file_path, "r", encoding="utf-8") as f:
            # 最初の5行を読む
            print("\n最初の5行:")
            for i in range(5):
                line = f.readline().strip()
                if not line:
                    break

                parts = line.split("|")
                print(f"\n行{i+1}: {len(parts)}カラム")

                # 各カラムを表示
                for j, part in enumerate(parts):
                    if len(part) > 60:
                        part = part[:60] + "..."

                    # WAVファイルが存在するかチェック
                    if j == 0:  # 最初のカラムがファイル名の場合
                        wav_path = data_root / "wavs" / f"{part}.wav"
                        exists = "✓" if wav_path.exists() else "✗"
                        print(f"  カラム{j+1} [{exists}]: {part}")
                    else:
                        # 日本語を含むかチェック
                        has_japanese = any(
                            "\u3040" <= c <= "\u309f"  # ひらがな
                            or "\u30a0" <= c <= "\u30ff"  # カタカナ
                            or "\u4e00" <= c <= "\u9fff"  # 漢字
                            for c in part
                        )
                        japanese_flag = " [日本語]" if has_japanese else ""
                        print(f"  カラム{j+1}: {part}{japanese_flag}")


if __name__ == "__main__":
    check_metadata_format()
