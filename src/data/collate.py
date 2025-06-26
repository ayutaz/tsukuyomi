"""
Collate functions for TTS data loading.
"""

from typing import Dict, List, Optional

import torch

from .audio_processor import AudioProcessor


def tts_collate_fn(
    batch: List[Dict],
    audio_processor: Optional[AudioProcessor] = None,
    speaker_to_id: Optional[Dict[str, int]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Collate function for TTS training.

    Args:
        batch: List of dictionaries from TsukuyomiDataset
        audio_processor: AudioProcessor instance for mel-spectrogram extraction
        speaker_to_id: Optional global speaker ID mapping

    Returns:
        Dictionary with batched tensors
    """
    # Separate different fields
    texts = [item["text"] for item in batch]
    speaker_ids = [item["speaker_id"] for item in batch]
    audio_paths = [item["audio_path"] for item in batch]

    # Handle audio - find max length and pad
    audios = [item["audio"] for item in batch]
    max_audio_len = max(audio.shape[0] for audio in audios)

    # Pad audios to same length
    padded_audios = []
    audio_lengths = []
    for audio in audios:
        audio_len = audio.shape[0]
        audio_lengths.append(audio_len)
        if audio_len < max_audio_len:
            padding = max_audio_len - audio_len
            padded_audio = torch.nn.functional.pad(audio, (0, padding), value=0.0)
            padded_audios.append(padded_audio)
        else:
            padded_audios.append(audio)

    # Stack into batch tensors
    audio_batch = torch.stack(padded_audios, dim=0)
    audio_lengths = torch.tensor(audio_lengths, dtype=torch.long)

    # Map speaker IDs to indices
    if speaker_to_id is not None:
        # Use global speaker mapping
        speaker_indices = torch.tensor(
            [speaker_to_id[sid] for sid in speaker_ids], dtype=torch.long
        )
    else:
        # Fallback to local mapping (for compatibility)
        unique_speakers = sorted(list(set(speaker_ids)))
        speaker_id_to_idx = {sid: idx for idx, sid in enumerate(unique_speakers)}
        speaker_indices = torch.tensor(
            [speaker_id_to_idx[sid] for sid in speaker_ids], dtype=torch.long
        )

    result = {
        "audio": audio_batch,
        "audio_lengths": audio_lengths,
        "text": texts,
        "speaker_ids": speaker_indices,
        "speaker_names": speaker_ids,
        "audio_paths": audio_paths,
    }

    # メルスペクトログラムの生成（audio_processorが提供された場合）
    if audio_processor is not None:
        mel_spectrograms = []
        mel_lengths = []

        for audio, audio_len in zip(audio_batch, audio_lengths):
            # 実際の長さまでトリミング
            audio_trimmed = audio[:audio_len]

            # メルスペクトログラムに変換
            mel = audio_processor.wav_to_mel(audio_trimmed)
            mel_spectrograms.append(mel)

            # メル長を計算
            mel_len = audio_processor.get_mel_lengths(audio_len.unsqueeze(0)).squeeze()
            mel_lengths.append(mel_len)

        # パディング
        max_mel_len = max(mel.shape[-1] for mel in mel_spectrograms)
        padded_mels = []

        for mel in mel_spectrograms:
            mel_len = mel.shape[-1]
            if mel_len < max_mel_len:
                padding = max_mel_len - mel_len
                mel = torch.nn.functional.pad(mel, (0, padding), value=0.0)
            padded_mels.append(mel)

        # バッチ化
        mel_batch = torch.stack(padded_mels, dim=0)
        mel_lengths = torch.tensor(mel_lengths, dtype=torch.long)

        result["mel_targets"] = mel_batch
        result["mel_lengths"] = mel_lengths

    return result
