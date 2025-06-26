#!/usr/bin/env python3
"""JVSデータセット準備スクリプト

JVS (Japanese versatile speech) コーパスをダウンロードして準備
"""

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import soundfile as sf
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))


def download_jvs(output_dir: Path):
    """JVSデータセットをダウンロード"""
    jvs_url = "https://drive.google.com/uc?id=19oAw8wWn3Y7z6CKChRdAyGOB9yupL_Xt"

    print("JVSデータセットのダウンロード情報:")
    print("- サイズ: 約2.7GB")
    print("- 話者数: 100人")
    print("- 各話者: parallel100（読み上げ音声）+ nonpara30（自由発話）")
    print("\n手動でダウンロードしてください:")
    print(f"1. {jvs_url} にアクセス")
    print("2. jvs_ver1.zip をダウンロード")
    print(f"3. {output_dir} に配置")

    zip_path = output_dir / "jvs_ver1.zip"
    if not zip_path.exists():
        print(f"\n{zip_path} が見つかりません。")
        print("上記の手順でダウンロードしてください。")
        return False

    return True


def extract_jvs(zip_path: Path, output_dir: Path):
    """JVSデータセットを展開"""
    print(f"\n{zip_path} を展開中...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    print("展開完了！")
    return output_dir / "jvs_ver1"


def prepare_jvs_metadata(jvs_dir: Path, speakers: List[str]) -> Dict:
    """JVSのメタデータを準備"""
    metadata = {
        "dataset": "JVS",
        "version": "1.0",
        "speakers": {},
        "total_duration": 0.0,
        "total_files": 0,
    }

    for speaker in speakers:
        speaker_dir = jvs_dir / speaker
        if not speaker_dir.exists():
            print(f"警告: {speaker} が見つかりません")
            continue

        # 話者情報
        speaker_info = {
            "id": speaker,
            "gender": (
                "F" if int(speaker[3:]) <= 50 else "M"
            ),  # JVS001-050: 女性, JVS051-100: 男性
            "parallel100": [],
            "nonpara30": [],
            "total_duration": 0.0,
        }

        # parallel100の処理
        parallel_dir = speaker_dir / "parallel100" / "wav24kHz16bit"
        if parallel_dir.exists():
            transcript_file = speaker_dir / "parallel100" / "transcripts_utf8.txt"
            transcripts = {}

            if transcript_file.exists():
                with open(transcript_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if ":" in line:
                            filename, text = line.strip().split(":", 1)
                            transcripts[filename] = text

            for wav_file in sorted(parallel_dir.glob("*.wav")):
                basename = wav_file.stem

                # 音声情報
                info = sf.info(str(wav_file))
                duration = info.duration

                file_info = {
                    "filename": wav_file.name,
                    "path": str(wav_file.relative_to(jvs_dir)),
                    "text": transcripts.get(basename, ""),
                    "duration": duration,
                    "sample_rate": info.samplerate,
                }

                speaker_info["parallel100"].append(file_info)
                speaker_info["total_duration"] += duration
                metadata["total_duration"] += duration
                metadata["total_files"] += 1

        metadata["speakers"][speaker] = speaker_info

    return metadata


def create_file_lists(
    jvs_dir: Path,
    metadata: Dict,
    output_dir: Path,
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
):
    """学習/検証/テスト用のファイルリストを作成"""
    train_files = []
    val_files = []
    test_files = []

    for speaker_id, speaker_info in metadata["speakers"].items():
        files = speaker_info["parallel100"]

        # ファイルを分割
        n_files = len(files)
        n_train = int(n_files * split_ratio[0])
        n_val = int(n_files * split_ratio[1])

        train_files.extend(files[:n_train])
        val_files.extend(files[n_train : n_train + n_val])
        test_files.extend(files[n_train + n_val :])

    # ファイルリストを保存
    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }

    for split_name, file_list in splits.items():
        split_file = output_dir / f"jvs_{split_name}.json"
        with open(split_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "split": split_name,
                    "files": file_list,
                    "total_files": len(file_list),
                    "total_duration": sum(f["duration"] for f in file_list),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(
            f"{split_name}: {len(file_list)} ファイル, {sum(f['duration'] for f in file_list):.1f}秒"
        )


def create_experiment_subset(
    jvs_dir: Path, output_dir: Path, speakers: List[str], files_per_speaker: int = 50
):
    """実験用の小規模サブセットを作成"""
    subset_dir = output_dir / "jvs_subset"
    subset_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n実験用サブセットを作成中 ({len(speakers)}話者 × {files_per_speaker}ファイル)..."
    )

    for speaker in tqdm(speakers):
        speaker_src = jvs_dir / speaker / "parallel100" / "wav24kHz16bit"
        speaker_dst = subset_dir / speaker / "wav"
        speaker_dst.mkdir(parents=True, exist_ok=True)

        # トランスクリプトのコピー
        transcript_src = jvs_dir / speaker / "parallel100" / "transcripts_utf8.txt"
        transcript_dst = subset_dir / speaker / "transcripts.txt"

        if transcript_src.exists():
            shutil.copy(transcript_src, transcript_dst)

        # 音声ファイルのコピー（最初のN個）
        wav_files = sorted(speaker_src.glob("*.wav"))[:files_per_speaker]

        for wav_file in wav_files:
            shutil.copy(wav_file, speaker_dst / wav_file.name)

    print(f"サブセットを作成しました: {subset_dir}")
    return subset_dir


def main():
    parser = argparse.ArgumentParser(description="JVSデータセット準備")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="データディレクトリ",
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
        help="使用する話者ID",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=50,
        help="実験用サブセットのファイル数（話者あたり）",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="ダウンロードをスキップ",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # JVSダウンロード
    if not args.skip_download:
        if not download_jvs(data_dir):
            return

    # 展開
    zip_path = data_dir / "jvs_ver1.zip"
    jvs_dir = data_dir / "jvs_ver1"

    if not jvs_dir.exists() and zip_path.exists():
        jvs_dir = extract_jvs(zip_path, data_dir)
    elif not jvs_dir.exists():
        print(f"エラー: {jvs_dir} が見つかりません")
        return

    # メタデータ準備
    print("\nメタデータを準備中...")
    metadata = prepare_jvs_metadata(jvs_dir, args.speakers)

    # メタデータ保存
    metadata_path = data_dir / "jvs_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\nデータセット統計:")
    print(f"- 話者数: {len(metadata['speakers'])}")
    print(f"- 総ファイル数: {metadata['total_files']}")
    print(f"- 総時間: {metadata['total_duration'] / 3600:.1f}時間")

    # ファイルリスト作成
    print("\nファイルリストを作成中...")
    create_file_lists(jvs_dir, metadata, data_dir)

    # 実験用サブセット作成
    subset_dir = create_experiment_subset(
        jvs_dir, data_dir, args.speakers, args.subset_size
    )

    print("\n✅ JVSデータセットの準備が完了しました！")
    print("\n実験を開始するには:")
    print(
        f"python scripts/train_jvs_experiment.py --config configs/experiment_jvs_4070ti.yaml --data-dir {subset_dir}"
    )


if __name__ == "__main__":
    main()
