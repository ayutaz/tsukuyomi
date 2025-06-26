#!/usr/bin/env python3
"""メタデータファイルのフォーマットを調査するスクリプト"""

import csv
from pathlib import Path


def inspect_metadata_file(file_path):
    """メタデータファイルの内容を調査"""
    print(f"\n=== {file_path.name} ===")

    with open(file_path, "r", encoding="utf-8") as f:
        # 最初の5行を表示
        print("\n最初の5行:")
        for i, line in enumerate(f):
            if i >= 5:
                break
            print(f"  {i+1}: {line.strip()}")

        # ファイルの先頭に戻る
        f.seek(0)

        # パイプ区切りの場合
        if "|" in line:
            print("\nパイプ区切りとして解析:")
            f.seek(0)
            for i, line in enumerate(f):
                if i >= 3:
                    break
                parts = line.strip().split("|")
                print(f"  行{i+1}: {len(parts)}カラム")
                for j, part in enumerate(parts):
                    # 長いテキストは省略
                    if len(part) > 50:
                        part = part[:50] + "..."
                    print(f"    カラム{j+1}: {part}")


def main():
    """メイン処理"""
    data_root = Path("data/jvs_ljspeech")

    # 調査するメタデータファイル
    metadata_files = [
        "metadata_multispeaker.csv",
        "metadata_multispeaker_pyopenjtalk_plus_processed.csv",
        "metadata.csv",
        "transcripts.txt",
    ]

    print("JVSデータセットのメタデータファイルを調査します")
    print(f"データディレクトリ: {data_root}")

    # 存在するファイルをリスト
    print("\n存在するメタデータファイル:")
    for file_name in metadata_files:
        file_path = data_root / file_name
        if file_path.exists():
            print(f"  ✓ {file_name}")
            inspect_metadata_file(file_path)
        else:
            print(f"  ✗ {file_name}")

    # CSVファイルの詳細分析
    print("\n\n=== 詳細分析 ===")
    target_file = data_root / "metadata_multispeaker.csv"
    if target_file.exists():
        with open(target_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")
            rows = list(reader)

            print(f"\n総行数: {len(rows)}")

            # カラム数の統計
            col_counts = {}
            for row in rows:
                col_count = len(row)
                col_counts[col_count] = col_counts.get(col_count, 0) + 1

            print("\nカラム数の分布:")
            for count, freq in sorted(col_counts.items()):
                print(f"  {count}カラム: {freq}行")

            # テキストらしいカラムを探す
            print("\n各カラムのサンプル（日本語を含むかチェック）:")
            for i in range(min(5, len(rows))):
                row = rows[i]
                print(f"\n行{i+1}:")
                for j, col in enumerate(row):
                    # 日本語文字を含むかチェック
                    has_japanese = any(
                        "\u3040" <= c <= "\u309f"  # ひらがな
                        or "\u30a0" <= c <= "\u30ff"  # カタカナ
                        or "\u4e00" <= c <= "\u9fff"  # 漢字
                        for c in col
                    )

                    preview = col[:50] + "..." if len(col) > 50 else col
                    japanese_flag = " [日本語]" if has_japanese else ""
                    print(f"  カラム{j+1}: {preview}{japanese_flag}")


if __name__ == "__main__":
    main()
