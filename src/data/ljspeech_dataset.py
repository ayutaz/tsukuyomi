"""LJSpeech形式のデータセットローダー

標準的なLJSpeech形式:
- wavs/: 音声ファイルディレクトリ
- metadata.csv: ファイル名|転写テキスト|正規化テキスト
"""

import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset


class LJSpeechDataset(Dataset):
    """LJSpeech形式のデータセット"""

    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        n_mels: int = 80,
        hop_length: int = 256,
        win_length: int = 1024,
        f_min: int = 0,
        f_max: int = 8000,
        max_audio_len: Optional[int] = None,
        min_audio_len: int = 1024,
        file_list: Optional[str] = None,
        text_cleaners: Optional[List[str]] = None,
        speaker_id: int = 0,  # シングルスピーカーの場合のデフォルト
    ):
        """
        Args:
            data_dir: LJSpeech形式のデータディレクトリ
            sample_rate: サンプリングレート
            n_fft: FFTサイズ
            n_mels: メル次元数
            hop_length: ホップ長
            win_length: 窓長
            f_min: 最小周波数
            f_max: 最大周波数
            max_audio_len: 最大音声長（サンプル数）
            min_audio_len: 最小音声長（サンプル数）
            file_list: 使用するファイルリスト（train_files.txt等）
            text_cleaners: テキストクリーナー
            speaker_id: 話者ID（マルチスピーカーの場合は上書き）
        """
        self.data_dir = Path(data_dir)
        self.wavs_dir = self.data_dir / "wavs"
        self.sample_rate = sample_rate
        self.max_audio_len = max_audio_len
        self.min_audio_len = min_audio_len
        self.speaker_id = speaker_id

        # メルスペクトログラム変換
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=hop_length,
            win_length=win_length,
            f_min=f_min,
            f_max=f_max,
            power=1.0,
        )

        # メタデータの読み込み
        self.metadata = self._load_metadata(file_list)

        # テキスト処理（簡易版）
        self.text_cleaners = text_cleaners or []

        print(f"LJSpeechDataset: {len(self.metadata)} files loaded from {data_dir}")

    def _load_metadata(self, file_list: Optional[str]) -> List[Dict]:
        """メタデータの読み込み"""
        metadata = []

        # メタデータCSVの読み込み
        metadata_path = self.data_dir / "metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.csv not found in {self.data_dir}")

        # ファイルリストの読み込み（指定された場合）
        allowed_files = None
        if file_list:
            file_list_path = self.data_dir / file_list
            if file_list_path.exists():
                with open(file_list_path, "r", encoding="utf-8") as f:
                    allowed_files = set(line.strip() for line in f)

        # メタデータの解析
        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")
            for row in reader:
                if len(row) >= 2:
                    filename = row[0]
                    transcript = row[1]
                    normalized = row[2] if len(row) > 2 else transcript

                    # ファイルリストでフィルタリング
                    if allowed_files and filename not in allowed_files:
                        continue

                    # 音声ファイルの存在確認
                    wav_path = self.wavs_dir / f"{filename}.wav"
                    if not wav_path.exists():
                        print(f"Warning: {wav_path} not found")
                        continue

                    metadata.append(
                        {
                            "filename": filename,
                            "wav_path": str(wav_path),
                            "transcript": transcript,
                            "normalized": normalized,
                        }
                    )

        return metadata

    def _load_audio(self, wav_path: str) -> torch.Tensor:
        """音声ファイルの読み込み"""
        # torchaudioで読み込み
        waveform, sr = torchaudio.load(wav_path)

        # モノラルに変換
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # リサンプリング
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # 長さの調整
        waveform = waveform.squeeze(0)
        audio_len = waveform.shape[0]

        if audio_len < self.min_audio_len:
            # パディング
            pad_len = self.min_audio_len - audio_len
            waveform = F.pad(waveform, (0, pad_len), mode="constant", value=0)
        elif self.max_audio_len and audio_len > self.max_audio_len:
            # トリミング（ランダムな位置から）
            start = random.randint(0, audio_len - self.max_audio_len)
            waveform = waveform[start : start + self.max_audio_len]

        return waveform

    def _text_to_tokens(self, text: str) -> torch.Tensor:
        """テキストをトークンに変換（簡易版）"""
        # 文字レベルのトークン化（実際にはG2P処理が必要）
        tokens = []
        for char in text:
            # 簡易的な文字->数値変換
            tokens.append(ord(char) % 256)

        return torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]

        # 音声の読み込み
        audio = self._load_audio(item["wav_path"])

        # メルスペクトログラムの計算
        mel = self.mel_transform(audio.unsqueeze(0)).squeeze(0)
        mel = torch.log(torch.clamp(mel, min=1e-5))

        # テキストのトークン化
        text_tokens = self._text_to_tokens(item["normalized"])

        # ピッチ抽出（ダミー）
        pitch = torch.zeros(mel.shape[1])

        return {
            "audio": audio,
            "mel": mel,
            "text": text_tokens,
            "pitch": pitch,
            "speaker_id": torch.tensor(self.speaker_id, dtype=torch.long),
            "filename": item["filename"],
        }


class LJSpeechCollator:
    """LJSpeechデータセット用のコレーター"""

    def __init__(self, return_ids: bool = True):
        self.return_ids = return_ids

    def __call__(self, batch):
        # バッチ内の最大長を取得
        max_audio_len = max(x["audio"].shape[0] for x in batch)
        max_mel_len = max(x["mel"].shape[1] for x in batch)
        max_text_len = max(x["text"].shape[0] for x in batch)

        # バッチテンソルの初期化
        batch_size = len(batch)

        audios = torch.zeros(batch_size, max_audio_len)
        mels = torch.zeros(batch_size, batch[0]["mel"].shape[0], max_mel_len)
        texts = torch.zeros(batch_size, max_text_len, dtype=torch.long)
        pitches = torch.zeros(batch_size, max_mel_len)
        speaker_ids = torch.zeros(batch_size, dtype=torch.long)

        audio_lengths = torch.zeros(batch_size, dtype=torch.long)
        mel_lengths = torch.zeros(batch_size, dtype=torch.long)
        text_lengths = torch.zeros(batch_size, dtype=torch.long)

        filenames = []

        # バッチにデータを詰める
        for i, item in enumerate(batch):
            audio_len = item["audio"].shape[0]
            mel_len = item["mel"].shape[1]
            text_len = item["text"].shape[0]

            audios[i, :audio_len] = item["audio"]
            mels[i, :, :mel_len] = item["mel"]
            texts[i, :text_len] = item["text"]
            pitches[i, :mel_len] = item["pitch"][:mel_len]
            speaker_ids[i] = item["speaker_id"]

            audio_lengths[i] = audio_len
            mel_lengths[i] = mel_len
            text_lengths[i] = text_len

            filenames.append(item["filename"])

        output = {
            "audio": audios,
            "mel": mels,
            "text": texts,
            "pitch": pitches,
            "speaker_id": speaker_ids,
            "audio_lengths": audio_lengths,
            "mel_lengths": mel_lengths,
            "text_lengths": text_lengths,
        }

        if self.return_ids:
            output["filenames"] = filenames

        return output


def create_ljspeech_datasets(
    data_dir: str,
    sample_rate: int = 22050,
    n_mels: int = 80,
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs,
):
    """LJSpeechデータセットとデータローダーの作成"""
    from torch.utils.data import DataLoader

    # データセットの作成
    train_dataset = LJSpeechDataset(
        data_dir=data_dir,
        sample_rate=sample_rate,
        n_mels=n_mels,
        file_list="train_files.txt",
        **kwargs,
    )

    val_dataset = LJSpeechDataset(
        data_dir=data_dir,
        sample_rate=sample_rate,
        n_mels=n_mels,
        file_list="val_files.txt",
        **kwargs,
    )

    # コレーターの作成
    collator = LJSpeechCollator()

    # データローダーの作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # テスト
    import sys

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "data/ljspeech"

    print(f"Testing LJSpeech dataset from {data_dir}")

    # データセットの作成
    dataset = LJSpeechDataset(
        data_dir=data_dir,
        sample_rate=22050,
        n_mels=80,
    )

    print(f"Dataset size: {len(dataset)}")

    # サンプルの取得
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Audio shape: {sample['audio'].shape}")
    print(f"  Mel shape: {sample['mel'].shape}")
    print(f"  Text shape: {sample['text'].shape}")
    print(f"  Filename: {sample['filename']}")

    # データローダーのテスト
    train_loader, val_loader = create_ljspeech_datasets(
        data_dir=data_dir,
        batch_size=4,
        num_workers=0,
    )

    print(f"\nDataLoader test:")
    batch = next(iter(train_loader))
    print(f"  Batch audio shape: {batch['audio'].shape}")
    print(f"  Batch mel shape: {batch['mel'].shape}")
    print(f"  Batch text shape: {batch['text'].shape}")
    print(f"  Audio lengths: {batch['audio_lengths']}")
