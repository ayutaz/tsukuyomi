# JVSデータセットを使用したTsukuyomi学習ガイド

## 概要

このガイドでは、JVS（Japanese Versatile Speech）コーパスを使用してTsukuyomi TTSモデルを学習する手順を説明します。JVSは100人の日本語話者による音声データセットで、小規模な実験に最適です。

## 前提条件

- Python 3.11以上
- CUDA対応GPU（推奨: RTX 3090以上）
- 16GB以上のGPUメモリ
- 50GB以上の空きディスク容量

## 1. 環境構築

### 1.1 リポジトリのクローン

```bash
git clone https://github.com/ayutaz/tsukuyomi.git
cd tsukuyomi
```

### 1.2 仮想環境の作成（UV使用）

```bash
# UVのインストール（未インストールの場合）
pip install uv

# 仮想環境の作成とアクティベート
uv venv
source .venv/bin/activate  # Linux/macOS
# または
.venv\Scripts\activate  # Windows
```

### 1.3 依存関係のインストール

```bash
# キャッシュをクリア（必要に応じて）
uv cache clean

# librosaの依存関係の問題を回避するため、先に必要なパッケージをインストール
uv pip install "librosa>=0.10.1" "numba>=0.57.0"

# 基本パッケージ
uv pip install -e .

# 学習用追加パッケージ
uv pip install accelerate wandb tensorboard
```

**注意**: Python 3.11でlibrosaの古いバージョンがインストールされる問題がある場合は、上記のように先にlibrosaとnumbaを明示的にインストールしてください。

## 2. JVSデータセットの準備

### 2.1 データセットのダウンロード

```bash
# データディレクトリの作成
mkdir -p data/jvs

# JVSコーパスのダウンロード（公式サイトから）
# https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus
cd data/jvs
wget https://drive.google.com/uc?id=19oAw8wWn3Y7z6CKChRdAyGOB9yupL_Xt -O jvs_ver1.zip
unzip jvs_ver1.zip
cd ../..
```

### 2.2 データの前処理

```bash
# 前処理スクリプトの実行
python scripts/preprocess_jvs.py \
    --input_dir data/jvs/jvs_ver1 \
    --output_dir data/processed/jvs \
    --num_workers 4
```

## 3. 設定ファイルの作成

### 3.1 小規模学習用設定

`configs/jvs_small.yaml`を作成：

```yaml
# JVS小規模学習設定
experiment_name: "tsukuyomi_jvs_small"

# データ設定
data:
  train_dir: "data/processed/jvs/train"
  val_dir: "data/processed/jvs/val"
  test_dir: "data/processed/jvs/test"
  
  # 音声パラメータ
  sample_rate: 22050
  hop_length: 256
  win_length: 1024
  n_fft: 1024
  n_mels: 80
  
  # データローダー
  batch_size: 16  # GPUメモリに応じて調整
  num_workers: 4
  use_cache: true
  cache_dir: "data/cache/jvs"

# モデル設定
models:
  # XPhoneBERTは軽量版を使用
  xphonebert:
    enabled: true
    model_name: "vinai/xphonebert-base"
    hidden_size: 256  # 小規模版
    num_layers: 6
    num_heads: 8
    freeze_base: true  # 事前学習済み部分は固定
  
  # F0-BERT
  f0_bert:
    enabled: true
    hidden_size: 128
    num_layers: 4
    num_heads: 4
    pitch_bins: 256
  
  # 音響モデル（VITSを使用）
  acoustic_model: "vits"
  
  vits:
    n_vocab: 256
    n_speakers: 100  # JVSの話者数
    hidden_channels: 192
    inter_channels: 192
    filter_channels: 768
    n_heads: 2
    n_layers: 6
    kernel_size: 3
    p_dropout: 0.1
    n_flows: 4
  
  # ボコーダー
  vocoder: "bigvgan"
  
  bigvgan:
    num_mels: 80
    upsample_initial_channel: 512
    resblock_kernel_sizes: [3, 7, 11]
    upsample_rates: [8, 8, 2, 2]  # hop_length = 256
    upsample_kernel_sizes: [16, 16, 4, 4]

# 学習設定
training:
  num_epochs: 100  # 小規模なので少なめ
  gradient_accumulation_steps: 1
  gradient_clip: 1.0
  
  # オプティマイザー
  optimizers:
    default:
      lr: 1e-4
      beta1: 0.8
      beta2: 0.99
      eps: 1e-9
      weight_decay: 0.01
    
    # BERTモデルは学習率を下げる
    xphonebert:
      lr: 1e-5
    
    f0_bert:
      lr: 5e-5
  
  # スケジューラー
  schedulers:
    default:
      type: "cosine_annealing_warm_restarts"
      T_0: 10
      T_mult: 2
      eta_min: 1e-6
  
  # チェックポイント
  save_interval: 10
  eval_interval: 5
  
  # ログ設定
  logging:
    log_interval: 100
    trackers: ["tensorboard"]  # wandbも使用可能
  
  # 混合精度学習
  mixed_precision: "fp16"  # または "bf16" (A100/H100の場合)
  
  # 早期終了
  early_stopping:
    patience: 20
    min_delta: 0.001

# パス設定
paths:
  checkpoints: "checkpoints/jvs_small"
  tensorboard: "logs/tensorboard/jvs_small"
  model_registry: "models/registry"

# 評価設定
evaluation:
  metrics:
    - mcd  # Mel Cepstral Distortion
    - pitch_correlation
    - speaker_similarity
  
  # 推論設定
  inference:
    batch_size: 1
    temperature: 1.0
    length_scale: 1.0
```

## 4. 学習の実行

### 4.1 シングルGPU学習

```bash
python scripts/train.py --config configs/jvs_small.yaml
```

### 4.2 マルチGPU学習（2GPU以上の場合）

```bash
# 2GPUの例
torchrun --nproc_per_node=2 scripts/train.py --config configs/jvs_small.yaml
```

### 4.3 学習の監視

```bash
# 別ターミナルでTensorBoardを起動
tensorboard --logdir logs/tensorboard/jvs_small
```

ブラウザで `http://localhost:6006` にアクセスして進捗を確認。

## 5. 推論とテスト

### 5.1 チェックポイントから推論

```python
# scripts/inference_jvs.py
import torch
from src.models.vits import VITS
from src.models.bigvgan_v2 import BigVGANv2
from src.data.text_processing import text_to_sequence

# モデルのロード
checkpoint = torch.load("checkpoints/jvs_small/best_model.pt")
model = VITS(**checkpoint['config']['models']['vits'])
model.load_state_dict(checkpoint['model_acoustic'])
model.eval()

vocoder = BigVGANv2(**checkpoint['config']['models']['bigvgan'])
vocoder.load_state_dict(checkpoint['model_vocoder'])
vocoder.eval()

# テキストから音声生成
text = "こんにちは、私はつくよみです。"
phoneme_ids = text_to_sequence(text)
speaker_id = 0  # JVS001の話者

with torch.no_grad():
    mel = model.infer(
        torch.tensor([phoneme_ids]),
        torch.tensor([len(phoneme_ids)]),
        torch.tensor([speaker_id])
    )
    audio = vocoder(mel)

# 音声を保存
import soundfile as sf
sf.write("output.wav", audio.squeeze().cpu().numpy(), 22050)
```

## 6. トラブルシューティング

### メモリ不足エラー

```yaml
# configs/jvs_small.yaml のバッチサイズを減らす
data:
  batch_size: 8  # 16から減らす
  
training:
  gradient_accumulation_steps: 2  # 実効的なバッチサイズを維持
```

### 学習が収束しない

```yaml
# 学習率を調整
optimizers:
  default:
    lr: 5e-5  # 1e-4から下げる
```

### 音質が悪い

- エポック数を増やす（100→200）
- より大きなモデルを使用（hidden_channels: 192→256）
- データ拡張を追加

## 7. 次のステップ

1. **モデルのスケールアップ**
   - より大きなモデルサイズ
   - Matcha-TTSの使用
   - 高度な機能（感情制御など）の有効化

2. **データセットの拡張**
   - JVS全話者（100人）の使用
   - 他のデータセットとの組み合わせ
   - カスタムデータの追加

3. **ファインチューニング**
   - 特定の話者やスタイルに特化
   - ゲームキャラクター音声への適応

## 参考リンク

- [JVSコーパス公式サイト](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)
- [Tsukuyomi GitHub](https://github.com/ayutaz/tsukuyomi)
- [学習済みモデル](https://huggingface.co/tsukuyomi-tts)（公開予定）