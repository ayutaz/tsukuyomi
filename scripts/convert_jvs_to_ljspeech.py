#!/usr/bin/env python3
"""JVSデータセットをLJSpeech形式に変換するスクリプト

LJSpeech形式:
- wavs/: 音声ファイルディレクトリ（WAVファイル）
- metadata.csv: ファイル名|転写テキスト|正規化テキスト
"""

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import soundfile as sf
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))


def normalize_text(text: str) -> str:
    """テキストの正規化

    LJSpeechと同様の正規化処理を適用
    """
    # 基本的な正規化
    text = text.strip()

    # 句読点の統一
    text = text.replace("。", ".")
    text = text.replace("、", ",")
    text = text.replace("！", "!")
    text = text.replace("？", "?")
    text = text.replace("・", " ")

    # 括弧の統一
    text = text.replace("（", "(")
    text = text.replace("）", ")")
    text = text.replace("「", '"')
    text = text.replace("」", '"')
    text = text.replace("『", "'")
    text = text.replace("』", "'")

    # 連続する空白を1つに
    text = re.sub(r"\s+", " ", text)

    return text


def load_jvs_transcripts(speaker_dir: Path) -> Dict[str, str]:
    """JVSの転写テキストを読み込む"""
    transcripts = {}

    # parallel100の転写ファイル
    transcript_file = speaker_dir / "parallel100" / "transcripts_utf8.txt"

    if transcript_file.exists():
        with open(transcript_file, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    filename, text = line.strip().split(":", 1)
                    transcripts[filename] = text.strip()

    # nonpara30の転写ファイル
    nonpara_file = speaker_dir / "nonpara30" / "transcripts_utf8.txt"

    if nonpara_file.exists():
        with open(nonpara_file, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    filename, text = line.strip().split(":", 1)
                    transcripts[filename] = text.strip()

    return transcripts


def convert_speaker_to_ljspeech(
    speaker_dir: Path, output_dir: Path, speaker_id: str, start_index: int = 0
) -> Tuple[List[Tuple[str, str, str]], int]:
    """1人の話者をLJSpeech形式に変換

    Returns:
        metadata_entries: (filename, transcript, normalized_transcript)のリスト
        num_files: 処理したファイル数
    """
    metadata_entries = []
    transcripts = load_jvs_transcripts(speaker_dir)

    # 出力ディレクトリの作成
    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    file_index = start_index

    # parallel100の処理
    parallel_dir = speaker_dir / "parallel100" / "wav24kHz16bit"
    if parallel_dir.exists():
        wav_files = sorted(parallel_dir.glob("*.wav"))

        for wav_file in wav_files:
            basename = wav_file.stem

            # 転写テキストの取得
            if basename not in transcripts:
                print(f"警告: {basename} の転写テキストが見つかりません")
                continue

            transcript = transcripts[basename]
            normalized = normalize_text(transcript)

            # LJSpeech形式のファイル名
            # LJ001-0001.wav のような形式
            ljspeech_filename = f"JVS-{speaker_id}-{file_index:04d}"

            # 音声ファイルのコピー
            dst_path = wavs_dir / f"{ljspeech_filename}.wav"
            shutil.copy(wav_file, dst_path)

            # メタデータエントリの追加
            metadata_entries.append((ljspeech_filename, transcript, normalized))

            file_index += 1

    # nonpara30の処理（オプション）
    nonpara_dir = speaker_dir / "nonpara30" / "wav24kHz16bit"
    if nonpara_dir.exists():
        wav_files = sorted(nonpara_dir.glob("*.wav"))

        for wav_file in wav_files:
            basename = wav_file.stem

            if basename not in transcripts:
                continue

            transcript = transcripts[basename]
            normalized = normalize_text(transcript)

            ljspeech_filename = f"JVS-{speaker_id}-{file_index:04d}"

            dst_path = wavs_dir / f"{ljspeech_filename}.wav"
            shutil.copy(wav_file, dst_path)

            metadata_entries.append((ljspeech_filename, transcript, normalized))

            file_index += 1

    return metadata_entries, file_index - start_index


def create_metadata_csv(
    metadata_entries: List[Tuple[str, str, str]], output_path: Path
):
    """metadata.csvの作成"""
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|", quoting=csv.QUOTE_MINIMAL)

        for entry in metadata_entries:
            writer.writerow(entry)

    print(f"メタデータを保存: {output_path}")


def create_file_lists(
    metadata_entries: List[Tuple[str, str, str]],
    output_dir: Path,
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
):
    """train/val/testのファイルリストを作成"""
    n_total = len(metadata_entries)
    n_train = int(n_total * split_ratio[0])
    n_val = int(n_total * split_ratio[1])

    train_entries = metadata_entries[:n_train]
    val_entries = metadata_entries[n_train : n_train + n_val]
    test_entries = metadata_entries[n_train + n_val :]

    # trainリスト
    with open(output_dir / "train_files.txt", "w", encoding="utf-8") as f:
        for entry in train_entries:
            f.write(f"{entry[0]}\n")

    # valリスト
    with open(output_dir / "val_files.txt", "w", encoding="utf-8") as f:
        for entry in val_entries:
            f.write(f"{entry[0]}\n")

    # testリスト
    with open(output_dir / "test_files.txt", "w", encoding="utf-8") as f:
        for entry in test_entries:
            f.write(f"{entry[0]}\n")

    print("\nデータ分割:")
    print(f"  - Train: {len(train_entries)} files")
    print(f"  - Val: {len(val_entries)} files")
    print(f"  - Test: {len(test_entries)} files")


def convert_jvs_to_ljspeech(
    jvs_dir: Path, output_dir: Path, speakers: List[str], include_nonpara: bool = False
):
    """JVSデータセット全体をLJSpeech形式に変換"""
    print("JVSデータセットをLJSpeech形式に変換中...")
    print(f"入力: {jvs_dir}")
    print(f"出力: {output_dir}")
    print(f"話者: {speakers}")

    output_dir.mkdir(parents=True, exist_ok=True)

    all_metadata_entries = []
    file_index = 0

    # 各話者の処理
    for speaker in tqdm(speakers, desc="話者を処理中"):
        speaker_dir = jvs_dir / speaker

        if not speaker_dir.exists():
            print(f"警告: {speaker} が見つかりません")
            continue

        # 話者の変換
        metadata_entries, num_files = convert_speaker_to_ljspeech(
            speaker_dir, output_dir, speaker, start_index=file_index
        )

        all_metadata_entries.extend(metadata_entries)
        file_index += num_files

        print(f"{speaker}: {num_files} ファイルを変換")

    # metadata.csvの作成
    metadata_path = output_dir / "metadata.csv"
    create_metadata_csv(all_metadata_entries, metadata_path)

    # ファイルリストの作成
    create_file_lists(all_metadata_entries, output_dir)

    # 統計情報
    print("\n変換完了!")
    print(f"総ファイル数: {len(all_metadata_entries)}")

    # サンプル表示
    print("\nメタデータのサンプル:")
    for i, entry in enumerate(all_metadata_entries[:3]):
        print(f"{i+1}: {entry[0]}|{entry[1][:50]}...")


def validate_ljspeech_format(output_dir: Path):
    """LJSpeech形式の検証"""
    print("\n=== 形式検証 ===")

    # 必須ファイル/ディレクトリのチェック
    wavs_dir = output_dir / "wavs"
    metadata_csv = output_dir / "metadata.csv"

    if not wavs_dir.exists():
        print("❌ wavs/ ディレクトリが見つかりません")
        return False

    if not metadata_csv.exists():
        print("❌ metadata.csv が見つかりません")
        return False

    # メタデータの検証
    with open(metadata_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        rows = list(reader)

    print(f"✅ metadata.csv: {len(rows)} エントリ")

    # WAVファイルの検証
    wav_files = list(wavs_dir.glob("*.wav"))
    print(f"✅ wavs/: {len(wav_files)} ファイル")

    # サンプリングレートのチェック（最初の5ファイル）
    for wav_file in wav_files[:5]:
        info = sf.info(str(wav_file))
        print(f"  - {wav_file.name}: {info.samplerate}Hz, {info.duration:.2f}秒")

    # ファイル名の一致チェック
    metadata_filenames = {row[0] for row in rows}
    wav_filenames = {f.stem for f in wav_files}

    missing_wavs = metadata_filenames - wav_filenames
    if missing_wavs:
        print(
            f"⚠️  メタデータに存在するが音声ファイルがない: {len(missing_wavs)} ファイル"
        )

    extra_wavs = wav_filenames - metadata_filenames
    if extra_wavs:
        print(
            f"⚠️  音声ファイルは存在するがメタデータにない: {len(extra_wavs)} ファイル"
        )

    print("\n✅ LJSpeech形式への変換が完了しました!")
    return True


def main():
    parser = argparse.ArgumentParser(description="JVSデータセットをLJSpeech形式に変換")
    parser.add_argument(
        "--jvs-dir",
        type=str,
        required=True,
        help="JVSデータセットのディレクトリ（jvs_ver1/）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="出力ディレクトリ（LJSpeech形式）",
    )
    parser.add_argument(
        "--speakers",
        type=str,
        nargs="+",
        default=[
            "jvs001",
            "jvs002",
            "jvs003",
            "jvs004",
            "jvs005",
            "jvs006",
            "jvs007",
            "jvs008",
            "jvs009",
            "jvs010",
        ],
        help="変換する話者ID",
    )
    parser.add_argument(
        "--include-nonpara",
        action="store_true",
        help="nonpara30（自由発話）も含める",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="変換後の形式を検証",
    )

    args = parser.parse_args()

    jvs_dir = Path(args.jvs_dir)
    output_dir = Path(args.output_dir)

    if not jvs_dir.exists():
        print(f"エラー: {jvs_dir} が見つかりません")
        return

    # 変換実行
    convert_jvs_to_ljspeech(
        jvs_dir, output_dir, args.speakers, include_nonpara=args.include_nonpara
    )

    # 検証
    if args.validate:
        validate_ljspeech_format(output_dir)

    # 使用例の表示
    print("\n使用例:")
    print("python scripts/train_jvs_experiment.py \\")
    print("    --config configs/experiment_jvs_4070ti.yaml \\")
    print(f"    --data-dir {output_dir}")


if __name__ == "__main__":
    main()
