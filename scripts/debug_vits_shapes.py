#!/usr/bin/env python3
"""VITSモデルの形状問題を詳細に調査するスクリプト"""

import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch

from src.data.text_tokenizer import JapaneseTextTokenizer
from src.models.vits import VITS
from src.utils.text_preprocessor import simple_text_to_katakana


def debug_vits_forward():
    """VITSのforward処理をステップバイステップでデバッグ"""

    # パラメータ
    batch_size = 16
    n_vocab = 93
    n_speakers = 100
    seq_len = 50
    mel_len = 200

    print("=== VITS Forward Debug ===")
    print(f"Batch size: {batch_size}")
    print(f"Vocabulary size: {n_vocab}")
    print(f"Number of speakers: {n_speakers}")
    print(f"Text sequence length: {seq_len}")
    print(f"Mel sequence length: {mel_len}")

    # VITSモデルを作成
    vits = VITS(
        n_vocab=n_vocab,
        n_speakers=n_speakers,
        speaker_embedding_dim=256,
        hidden_channels=192,
        filter_channels=768,
        n_heads=2,  # マルチヘッド
        n_layers=6,
        kernel_size=3,
        p_dropout=0.0,  # デバッグ用にドロップアウトなし
        n_flows=4,
    )
    vits.eval()  # 評価モードに

    # ダミー入力を作成
    text = torch.randint(0, n_vocab, (batch_size, seq_len))
    text_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
    mel = torch.randn(batch_size, 80, mel_len)
    mel_lengths = torch.full((batch_size,), mel_len, dtype=torch.long)
    speaker_ids = torch.randint(0, n_speakers, (batch_size,))

    print("\nInput shapes:")
    print(f"  text: {text.shape}")
    print(f"  text_lengths: {text_lengths.shape}")
    print(f"  mel: {mel.shape}")
    print(f"  mel_lengths: {mel_lengths.shape}")
    print(f"  speaker_ids: {speaker_ids.shape}")

    # フォワードパスを実行（トレーニングモード）
    print("\n=== Running forward pass (training mode) ===")
    try:
        with torch.no_grad():
            outputs = vits(
                text=text,
                text_lengths=text_lengths,
                mel=mel,
                mel_lengths=mel_lengths,
                speaker_ids=speaker_ids,
            )
        print("✓ Forward pass successful!")
        print(f"Output keys: {list(outputs.keys())}")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            elif isinstance(value, dict):
                print(f"  {key}: {type(value)} with {len(value)} items")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback

        traceback.print_exc()

    # 推論モードもテスト
    print("\n=== Running forward pass (inference mode) ===")
    try:
        with torch.no_grad():
            outputs = vits(
                text=text,
                text_lengths=text_lengths,
                mel=None,  # 推論時はmelなし
                mel_lengths=None,
                speaker_ids=speaker_ids,
            )
        print("✓ Inference pass successful!")
        print(f"Output keys: {list(outputs.keys())}")
    except Exception as e:
        print(f"✗ Inference pass failed: {e}")


def test_specific_case():
    """実際のエラーケースを再現"""
    print("\n=== Testing specific error case ===")

    # エラーが起きているケースのパラメータ
    batch_size = 16
    n_speakers = 100

    # 実際のテキスト（最初の2つ）
    texts = [
        "しかし、後悔することがふたつある。",
        "祭神は、ヴィシュヌ派の聖人、スワーミーナーラーヤン。",
    ]

    # カタカナに変換
    katakana_texts = [simple_text_to_katakana(text) for text in texts]
    print(f"Katakana texts: {katakana_texts}")

    # トークナイザー
    tokenizer = JapaneseTextTokenizer()

    # バッチ全体を作成（同じテキストを繰り返し）
    full_texts = katakana_texts * (batch_size // 2)
    if len(full_texts) < batch_size:
        full_texts.extend(katakana_texts[: batch_size - len(full_texts)])

    # エンコード
    encoding = tokenizer.batch_encode(
        full_texts,
        add_special_tokens=True,
        max_length=None,
        padding=True,
        return_tensors=True,
    )

    text_tokens = encoding["input_ids"]
    text_lengths = encoding["lengths"]

    print("\nToken shapes:")
    print(f"  text_tokens: {text_tokens.shape}")
    print(f"  text_lengths: {text_lengths}")

    # スピーカーIDを作成（エラーログと同じ）
    speaker_ids = torch.tensor([12, 11] * (batch_size // 2))
    if len(speaker_ids) < batch_size:
        speaker_ids = torch.cat(
            [speaker_ids, speaker_ids[: batch_size - len(speaker_ids)]]
        )

    print(f"  speaker_ids: {speaker_ids.shape}, first 2: {speaker_ids[:2].tolist()}")

    # VITSモデルを作成
    vits = VITS(
        n_vocab=93,
        n_speakers=n_speakers,
        speaker_embedding_dim=256,
        hidden_channels=192,
        filter_channels=768,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=0.1,
        n_flows=4,
    )
    vits.eval()

    # ダミーのメルスペクトログラム
    mel = torch.randn(batch_size, 80, 500)
    mel_lengths = torch.randint(300, 500, (batch_size,))

    print("\n=== Running specific case ===")
    try:
        with torch.no_grad():
            outputs = vits(
                text=text_tokens,
                text_lengths=text_lengths,
                mel=mel,
                mel_lengths=mel_lengths,
                speaker_ids=speaker_ids,
            )
        print("✓ Specific case successful!")
    except Exception as e:
        print(f"✗ Specific case failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_vits_forward()
    test_specific_case()
