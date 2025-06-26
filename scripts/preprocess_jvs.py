#!/usr/bin/env python3
"""JVSデータセットの前処理スクリプト

JVSコーパスをTsukuyomi TTSの学習形式に変換します。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import soundfile as sf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.text_processing import normalize_text
from src.utils.audio import AudioProcessor


def parse_transcript(transcript_path: Path) -> Dict[str, str]:
    """JVSのtranscriptファイルを解析

    Args:
        transcript_path: prompt-lab/parallel100/transcripts_utf8.txt のパス

    Returns:
        Dict[utterance_id, text]
    """
    transcripts = {}

    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # JVSのフォーマット: BASIC5000_0001:こんにちは。
            parts = line.split(":", 1)
            if len(parts) == 2:
                utt_id = parts[0]
                text = parts[1].strip()
                transcripts[utt_id] = text

    return transcripts


def process_speaker(
    speaker_dir: Path,
    output_dir: Path,
    transcripts: Dict[str, str],
    audio_processor: AudioProcessor,
    subset: str = "parallel100",
) -> List[Dict]:
    """1話者のデータを処理

    Args:
        speaker_dir: 話者ディレクトリ (例: jvs001)
        output_dir: 出力ディレクトリ
        transcripts: 転写テキスト辞書
        audio_processor: 音声処理器
        subset: 使用するサブセット

    Returns:
        メタデータのリスト
    """
    speaker_id = speaker_dir.name  # jvs001 など
    metadata = []

    # 音声ファイルディレクトリ
    wav_dir = speaker_dir / subset / "wav24kHz16bit"
    if not wav_dir.exists():
        print(f"Warning: {wav_dir} does not exist")
        return metadata

    # 出力ディレクトリの作成
    speaker_output_dir = output_dir / speaker_id
    speaker_output_dir.mkdir(parents=True, exist_ok=True)

    # 音声ファイルの処理
    wav_files = sorted(wav_dir.glob("*.wav"))

    for wav_path in tqdm(wav_files, desc=f"Processing {speaker_id}"):
        utt_id = wav_path.stem

        # テキストの取得
        if utt_id not in transcripts:
            print(f"Warning: No transcript for {utt_id}")
            continue

        text = transcripts[utt_id]
        normalized_text = normalize_text(text)

        # 音声の読み込みと処理
        try:
            audio, sr = librosa.load(wav_path, sr=None)

            # サンプリングレートの変換（22050Hzに統一）
            if sr != 22050:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)

            # 音声の正規化
            audio = audio / np.max(np.abs(audio)) * 0.95

            # 無音区間のトリミング
            audio, _ = librosa.effects.trim(audio, top_db=30)

            # メルスペクトログラムの計算
            mel = audio_processor.wav_to_mel(audio)

            # ファイルの保存
            audio_filename = f"{speaker_id}_{utt_id}.wav"
            mel_filename = f"{speaker_id}_{utt_id}.npy"

            audio_path = speaker_output_dir / audio_filename
            mel_path = speaker_output_dir / mel_filename

            sf.write(audio_path, audio, 22050, subtype="PCM_16")
            np.save(mel_path, mel)

            # メタデータの追加
            metadata.append(
                {
                    "audio_path": str(audio_path.relative_to(output_dir)),
                    "mel_path": str(mel_path.relative_to(output_dir)),
                    "text": text,
                    "normalized_text": normalized_text,
                    "speaker_id": speaker_id,
                    "utterance_id": utt_id,
                    "duration": len(audio) / 22050,
                    "mel_frames": mel.shape[1],
                }
            )

        except Exception as e:
            print(f"Error processing {wav_path}: {e}")
            continue

    return metadata


def split_metadata(
    metadata: List[Dict],
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    random_state: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """メタデータを学習/検証/テストに分割

    話者ごとにバランスよく分割します。
    """
    # 話者ごとにグループ化
    speaker_data = {}
    for item in metadata:
        speaker_id = item["speaker_id"]
        if speaker_id not in speaker_data:
            speaker_data[speaker_id] = []
        speaker_data[speaker_id].append(item)

    train_data = []
    val_data = []
    test_data = []

    # 各話者のデータを分割
    for speaker_id, items in speaker_data.items():
        # まず学習とそれ以外に分割
        train_items, temp_items = train_test_split(
            items, test_size=(1 - train_ratio), random_state=random_state
        )

        # 検証とテストに分割
        val_size = val_ratio / (val_ratio + test_ratio)
        val_items, test_items = train_test_split(
            temp_items, test_size=(1 - val_size), random_state=random_state
        )

        train_data.extend(train_items)
        val_data.extend(val_items)
        test_data.extend(test_items)

    return train_data, val_data, test_data


def main():
    parser = argparse.ArgumentParser(description="JVSデータセットの前処理")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="JVSデータセットのルートディレクトリ",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="前処理済みデータの出力ディレクトリ",
    )
    parser.add_argument(
        "--num_speakers", type=int, default=10, help="処理する話者数（デフォルト: 10）"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="parallel100",
        choices=["parallel100", "nonpara30", "whisper10"],
        help="使用するサブセット",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="並列処理のワーカー数"
    )

    args = parser.parse_args()

    # パスの設定
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # 出力ディレクトリの作成
    output_dir.mkdir(parents=True, exist_ok=True)

    # 音声処理器の初期化
    audio_processor = AudioProcessor(
        sample_rate=22050,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=80,
        fmin=0,
        fmax=8000,
    )

    # 転写テキストの読み込み
    transcript_path = (
        input_dir / "jvs_ver1" / "prompt-lab" / args.subset / "transcripts_utf8.txt"
    )
    if not transcript_path.exists():
        print(f"Error: Transcript file not found: {transcript_path}")
        return

    transcripts = parse_transcript(transcript_path)
    print(f"Loaded {len(transcripts)} transcripts")

    # 話者ディレクトリの取得
    speaker_dirs = sorted(
        [
            d
            for d in (input_dir / "jvs_ver1").iterdir()
            if d.is_dir() and d.name.startswith("jvs")
        ]
    )[: args.num_speakers]

    print(f"Processing {len(speaker_dirs)} speakers")

    # 全メタデータの収集
    all_metadata = []

    for speaker_dir in speaker_dirs:
        metadata = process_speaker(
            speaker_dir, output_dir / "audio", transcripts, audio_processor, args.subset
        )
        all_metadata.extend(metadata)

    print(f"Processed {len(all_metadata)} utterances")

    # データの分割
    train_data, val_data, test_data = split_metadata(all_metadata)

    print(f"Train: {len(train_data)} utterances")
    print(f"Val: {len(val_data)} utterances")
    print(f"Test: {len(test_data)} utterances")

    # メタデータの保存
    for split_name, split_data in [
        ("train", train_data),
        ("val", val_data),
        ("test", test_data),
    ]:
        split_dir = output_dir / split_name
        split_dir.mkdir(exist_ok=True)

        # JSONファイルとして保存
        metadata_path = split_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)

        # テキストファイルとしても保存（簡易確認用）
        filelist_path = split_dir / "filelist.txt"
        with open(filelist_path, "w", encoding="utf-8") as f:
            for item in split_data:
                f.write(
                    f"{item['audio_path']}|{item['normalized_text']}|{item['speaker_id']}\n"
                )

    # 統計情報の出力
    stats = {
        "num_speakers": len(set(item["speaker_id"] for item in all_metadata)),
        "num_utterances": len(all_metadata),
        "total_duration": sum(item["duration"] for item in all_metadata),
        "avg_duration": np.mean([item["duration"] for item in all_metadata]),
        "splits": {
            "train": len(train_data),
            "val": len(val_data),
            "test": len(test_data),
        },
    }

    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n=== 処理完了 ===")
    print(f"総話者数: {stats['num_speakers']}")
    print(f"総発話数: {stats['num_utterances']}")
    print(f"総時間: {stats['total_duration'] / 3600:.2f} 時間")
    print(f"平均発話長: {stats['avg_duration']:.2f} 秒")
    print(f"\n出力ディレクトリ: {output_dir}")


if __name__ == "__main__":
    main()
