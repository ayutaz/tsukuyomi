# LJSpeech形式のデータセットでの学習ガイド

## 概要

このガイドでは、LJSpeech形式のデータセットを使用してTsukuyomi TTSシステムを学習する方法を説明します。

## LJSpeech形式とは

LJSpeech形式は、音声合成の標準的なデータセット形式です：

```
dataset_root/
├── wavs/               # 音声ファイルディレクトリ
│   ├── file1.wav
│   ├── file2.wav
│   └── ...
├── metadata.csv        # メタデータファイル
├── train_files.txt     # 学習用ファイルリスト（オプション）
├── val_files.txt       # 検証用ファイルリスト（オプション）
└── test_files.txt      # テスト用ファイルリスト（オプション）
```

### metadata.csv の形式

```csv
filename1|transcript1|normalized_transcript1
filename2|transcript2|normalized_transcript2
...
```

- `filename`: 音声ファイル名（拡張子なし）
- `transcript`: 元の転写テキスト
- `normalized_transcript`: 正規化された転写テキスト

## 学習の実行

### 1. 基本的な学習

```bash
python scripts/train_ljspeech.py \
    --config configs/ljspeech_base.yaml \
    --data-dir /path/to/your/ljspeech_dataset
```

### 2. カスタム設定での学習

```bash
python scripts/train_ljspeech.py \
    --config configs/ljspeech_base.yaml \
    --data-dir /path/to/your/dataset \
    --batch-size 16 \
    --learning-rate 1e-4 \
    --num-epochs 300 \
    --output-dir checkpoints/custom_model
```

### 3. 分散学習（マルチGPU）

```bash
# 4GPUでの分散学習
torchrun --nproc_per_node=4 scripts/train_ljspeech.py \
    --config configs/ljspeech_base.yaml \
    --data-dir /path/to/your/dataset \
    --distributed
```

### 4. チェックポイントからの再開

```bash
python scripts/train_ljspeech.py \
    --config configs/ljspeech_base.yaml \
    --data-dir /path/to/your/dataset \
    --checkpoint checkpoints/ljspeech/checkpoint_epoch_100.pt
```

## 設定ファイルのカスタマイズ

### 日本語データセット用の設定例

```yaml
# configs/japanese_ljspeech.yaml

data:
  dataset: "japanese_ljspeech"
  data_dir: "data/japanese_dataset"
  sample_rate: 24000  # 日本語は24kHzを推奨
  hop_length: 240
  n_mels: 80

models:
  # 日本語用にXPhoneBERTを有効化
  xphonebert:
    enabled: true
    model_name: "vinai/xphonebert-base"
    
  # マルチスピーカー対応
  vits:
    n_vocab: 512  # 日本語の音素数に合わせて増加
    n_speakers: 10  # 話者数
```

### メモリ制限がある場合の設定

```yaml
# configs/ljspeech_small.yaml

data:
  # バッチサイズを削減
  batch_size: 8
  # 最大音声長を制限
  max_audio_len: 110250  # 5秒

models:
  # モデルサイズを削減
  vits:
    hidden_channels: 96
    filter_channels: 384
    n_layers: 3
    
training:
  # 勾配累積を使用
  gradient_accumulation_steps: 4
  # 混合精度を使用
  mixed_precision: "fp16"
```

## データセットの準備

### 既存の音声データをLJSpeech形式に変換

```python
import csv
import shutil
from pathlib import Path

def convert_to_ljspeech(source_dir, output_dir, transcript_file):
    """音声データをLJSpeech形式に変換"""
    
    output_dir = Path(output_dir)
    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    
    # トランスクリプトの読み込み
    with open(transcript_file, 'r', encoding='utf-8') as f:
        transcripts = json.load(f)
    
    # メタデータの作成
    metadata = []
    for audio_file, text in transcripts.items():
        # 音声ファイルのコピー
        src = Path(source_dir) / audio_file
        dst = wavs_dir / audio_file
        shutil.copy(src, dst)
        
        # メタデータエントリ
        filename = Path(audio_file).stem
        normalized = normalize_text(text)
        metadata.append([filename, text, normalized])
    
    # metadata.csvの保存
    with open(output_dir / "metadata.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerows(metadata)
```

## トラブルシューティング

### メモリ不足エラー

```bash
# バッチサイズを削減
--batch-size 4

# 勾配累積を増やす
--gradient-accumulation-steps 8

# 音声長を制限（設定ファイルで）
max_audio_len: 55125  # 2.5秒
```

### 学習が不安定

```bash
# 学習率を下げる
--learning-rate 5e-5

# 勾配クリッピングを調整（設定ファイルで）
gradient_clip: 1.0
```

### データローダーが遅い

```bash
# ワーカー数を増やす（設定ファイルで）
num_workers: 8

# キャッシュを有効化
use_cache: true
cache_dir: "data/cache"
```

## モニタリングとログ

### TensorBoard

```bash
# TensorBoardの起動
tensorboard --logdir logs/ljspeech/tensorboard

# ブラウザで http://localhost:6006 にアクセス
```

### Weights & Biases

```bash
# W&Bの設定
wandb login

# 設定ファイルでW&Bを有効化
logging:
  trackers: ["tensorboard", "wandb"]
  wandb_project: "tsukuyomi-tts"
```

## 評価とテスト

### 学習済みモデルでの推論

```python
from tsukuyomi import TsukuyomiTTS

# モデルの読み込み
tts = TsukuyomiTTS.from_checkpoint("checkpoints/ljspeech/best_model.pt")

# 音声生成
audio = tts.synthesize("Hello, this is a test of the TTS system.")
tts.save_audio(audio, "output.wav")
```

### 客観評価

```bash
# 評価スクリプトの実行
python scripts/evaluate.py \
    --checkpoint checkpoints/ljspeech/best_model.pt \
    --test-data data/ljspeech/test_files.txt \
    --metrics mcd pitch_correlation voice_quality
```

## ベストプラクティス

1. **データ品質**: 高品質な録音（16kHz以上、低ノイズ）を使用
2. **テキスト正規化**: 一貫した正規化ルールを適用
3. **学習スケジュール**: 最初は高い学習率、後半は低い学習率
4. **定期的な評価**: 10エポックごとに音声サンプルを生成して確認
5. **早期終了**: 検証損失が改善しなくなったら学習を停止