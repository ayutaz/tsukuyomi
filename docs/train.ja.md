# Tsukuyomi TTS 学習ガイド

## 概要

このガイドでは、Tsukuyomi TTSモデルの学習方法について詳しく説明します。小規模な実験から大規模な本番学習まで、段階的なアプローチを提供します。

## 目次

1. [環境構築](#環境構築)
2. [データセット準備](#データセット準備)
3. [設定ファイル](#設定ファイル)
4. [学習の実行](#学習の実行)
5. [分散学習](#分散学習)
6. [チェックポイント管理](#チェックポイント管理)
7. [評価とモニタリング](#評価とモニタリング)
8. [トラブルシューティング](#トラブルシューティング)

## 環境構築

### 推奨環境

- **OS**: Ubuntu 20.04/22.04 または Windows 11 (WSL2)
- **Python**: 3.11以上
- **GPU**: NVIDIA RTX 3090以上（開発）、A100/H100（本番）
- **CUDA**: 12.1以上
- **メモリ**: 32GB以上（システム）、24GB以上（GPU）

### 依存関係のインストール

```bash
# 基本パッケージ
uv pip install -e .

# 学習用追加パッケージ
uv pip install accelerate wandb tensorboard

# Flash Attention 2（オプション、A100/H100推奨）
uv pip install flash-attn --no-build-isolation
```

## データセット準備

### サポートされるデータセット形式

Tsukuyomiは以下の形式をサポートします：

1. **標準形式** (推奨)
   ```
   dataset/
   ├── train/
   │   ├── metadata.json
   │   └── audio/
   │       ├── speaker001/
   │       │   ├── utterance001.wav
   │       │   └── utterance001.txt
   │       └── speaker002/
   └── val/
   ```

2. **CSV形式**
   ```csv
   audio_path,text,speaker_id,duration
   audio/001.wav,こんにちは,0,2.5
   ```

### メタデータ形式

`metadata.json`の例：

```json
[
  {
    "audio_path": "audio/speaker001/utterance001.wav",
    "text": "こんにちは、今日はいい天気ですね。",
    "speaker_id": "speaker001",
    "duration": 3.2,
    "emotion": "happy",
    "style": "casual"
  }
]
```

### 前処理パイプライン

```bash
# 一般的なデータセットの前処理
python scripts/preprocess_dataset.py \
    --input_dir /path/to/raw/data \
    --output_dir data/processed/my_dataset \
    --sample_rate 22050 \
    --trim_silence \
    --normalize
```

## 設定ファイル

### 基本設定テンプレート

```yaml
# configs/base_config.yaml
experiment_name: "tsukuyomi_base"

# データ設定
data:
  train_dir: "data/processed/my_dataset/train"
  val_dir: "data/processed/my_dataset/val"
  sample_rate: 22050
  hop_length: 256
  win_length: 1024
  n_mels: 80
  
  # バッチ設定
  batch_size: 32
  num_workers: 8
  pin_memory: true
  
  # データ拡張
  augmentation:
    speed_perturb: [0.9, 1.1]
    pitch_shift: [-2, 2]
    add_noise: true

# モデル設定
models:
  # 音素エンコーダー
  xphonebert:
    enabled: true
    model_name: "vinai/xphonebert-base"
    hidden_size: 768
    num_layers: 12
    freeze_layers: 6  # 下位6層は固定
  
  # ピッチモデル
  f0_bert:
    enabled: true
    hidden_size: 256
    num_layers: 6
    pitch_bins: 256
  
  # 音響モデル
  acoustic_model: "vits"  # または "matcha"
  
  vits:
    n_vocab: 256
    n_speakers: 100
    hidden_channels: 192
    filter_channels: 768
    n_layers: 6
    n_flows: 4
    use_sdp: true  # 確率的持続時間予測
  
  # ボコーダー
  vocoder: "bigvgan"
  
  bigvgan:
    num_mels: 80
    upsample_initial_channel: 1536
    resblock_kernel_sizes: [3, 7, 11]
    resblock_dilation_sizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

# 学習設定
training:
  num_epochs: 1000
  gradient_accumulation_steps: 2
  gradient_clip: 1.0
  
  # 混合精度学習
  mixed_precision: "bf16"  # "fp16", "bf16", or "no"
  
  # オプティマイザー
  optimizer:
    type: "AdamW"
    lr: 2e-4
    betas: [0.8, 0.99]
    eps: 1e-9
    weight_decay: 0.01
  
  # スケジューラー
  scheduler:
    type: "cosine_annealing_warm_restarts"
    T_0: 50
    T_mult: 2
    eta_min: 1e-6
    warmup_steps: 4000
  
  # 損失重み
  loss_weights:
    mel: 45.0
    kl: 1.0
    duration: 1.0
    pitch: 10.0
    gen: 1.0
    fm: 2.0

# チェックポイント設定
checkpointing:
  save_interval: 10  # エポック
  keep_last: 5
  save_best: true
  resume: null  # チェックポイントパス

# ログ設定
logging:
  log_interval: 100  # ステップ
  eval_interval: 5  # エポック
  trackers: ["tensorboard", "wandb"]
  wandb_project: "tsukuyomi-tts"
```

### 高度な設定

#### 感情制御を有効化

```yaml
models:
  emotion_control:
    enabled: true
    num_emotions: 10
    emotion_embedding_dim: 256
    use_vad: true  # Valence-Arousal-Dominance
```

#### スタイル転送を有効化

```yaml
models:
  style_transfer:
    enabled: true
    style_dim: 256
    num_style_tokens: 10
    reference_encoder: "ecapa-tdnn"
```

## 学習の実行

### シングルGPU学習

```bash
python scripts/train.py --config configs/base_config.yaml
```

### 学習の再開

```bash
python scripts/train.py \
    --config configs/base_config.yaml \
    --resume checkpoints/tsukuyomi_base/checkpoint_epoch_100.pt
```

### 学習のモニタリング

```bash
# TensorBoard
tensorboard --logdir logs/tensorboard

# WandB（設定済みの場合）
# https://wandb.ai/your-username/tsukuyomi-tts
```

## 分散学習

### マルチGPU学習（単一ノード）

```bash
# 4 GPU
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/base_config.yaml

# または専用スクリプト
./scripts/train_multi_gpu.sh configs/base_config.yaml
```

### マルチノード学習

```bash
# ノード0（マスター）
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=192.168.1.1 \
    --master_port=29500 \
    scripts/train.py --config configs/base_config.yaml

# ノード1
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=192.168.1.1 \
    --master_port=29500 \
    scripts/train.py --config configs/base_config.yaml
```

### DeepSpeed統合（大規模学習）

```yaml
# configs/deepspeed_config.json
{
  "train_batch_size": 256,
  "gradient_accumulation_steps": 4,
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu"
    }
  }
}
```

```bash
deepspeed scripts/train.py \
    --config configs/base_config.yaml \
    --deepspeed configs/deepspeed_config.json
```

## チェックポイント管理

### チェックポイントの構造

```
checkpoints/
└── tsukuyomi_base/
    ├── checkpoint_epoch_100.pt
    ├── best_model.pt
    ├── optimizer_state.pt
    └── training_state.json
```

### チェックポイントの変換

```bash
# ONNXへの変換
python scripts/export_onnx.py \
    --checkpoint checkpoints/tsukuyomi_base/best_model.pt \
    --output models/tsukuyomi_base.onnx

# TorchScriptへの変換
python scripts/export_torchscript.py \
    --checkpoint checkpoints/tsukuyomi_base/best_model.pt \
    --output models/tsukuyomi_base.pt
```

## 評価とモニタリング

### 自動評価メトリクス

- **MCD (Mel Cepstral Distortion)**: 音響的類似度
- **F0 RMSE**: ピッチ精度
- **V/UV Error**: 有声/無声判定精度
- **Speaker Similarity**: 話者類似度（コサイン類似度）
- **PESQ/STOI**: 音声品質（オプション）

### 評価の実行

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/tsukuyomi_base/best_model.pt \
    --test_dir data/processed/my_dataset/test \
    --output_dir evaluation/tsukuyomi_base
```

### カスタムメトリクスの追加

```python
# src/evaluation/custom_metrics.py
from src.evaluation.metrics import BaseMetric

class MyCustomMetric(BaseMetric):
    def __init__(self):
        super().__init__()
        
    def compute(self, predictions, targets):
        # カスタムメトリクスの計算
        return metric_value
```

## トラブルシューティング

### よくある問題と解決策

#### 1. CUDA Out of Memory

```yaml
# バッチサイズを減らす
data:
  batch_size: 16  # 32から減少

# 勾配累積を増やす
training:
  gradient_accumulation_steps: 4  # 2から増加
```

#### 2. 学習が不安定

```yaml
# 学習率を下げる
optimizer:
  lr: 1e-4  # 2e-4から減少

# 勾配クリッピングを強化
training:
  gradient_clip: 0.5  # 1.0から減少
```

#### 3. 過学習

```yaml
# ドロップアウトを増やす
models:
  vits:
    p_dropout: 0.2  # 0.1から増加

# データ拡張を強化
data:
  augmentation:
    spec_augment: true
    time_masking: 0.2
```

#### 4. 収束が遅い

```yaml
# 学習率スケジュールを調整
scheduler:
  type: "one_cycle"
  max_lr: 5e-4
  pct_start: 0.3
```

### デバッグモード

```bash
# 小規模データでのデバッグ
python scripts/train.py \
    --config configs/base_config.yaml \
    --debug \
    --max_steps 100
```

## ベストプラクティス

1. **段階的な学習**
   - まず小規模データで動作確認
   - 徐々にデータとモデルサイズを拡大

2. **定期的な評価**
   - 5-10エポックごとに評価を実行
   - 主観評価も定期的に実施

3. **チェックポイント管理**
   - ベストモデルを常に保存
   - 定期的にバックアップを作成

4. **ハイパーパラメータチューニング**
   - Optuna等を使用した自動チューニング
   - 学習率が最も重要なパラメータ

5. **データ品質**
   - 高品質な音声データが最重要
   - ノイズ除去と正規化を徹底

## 次のステップ

- [推論最適化ガイド](inference_optimization.ja.md)
- [モデルデプロイメントガイド](deployment.ja.md)
- [APIサーバー構築ガイド](api_server.ja.md)