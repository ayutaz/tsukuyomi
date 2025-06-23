# 月読（Tsukuyomi）学習ガイド

このガイドでは、月読究極TTSシステムの完全な学習パイプラインについて、データの前処理からモデルの評価まで詳しく説明します。

## 目次

1. [概要](#概要)
2. [前提条件](#前提条件)
3. [データ準備](#データ準備)
4. [前処理パイプライン](#前処理パイプライン)
5. [学習ステージ](#学習ステージ)
6. [評価方法](#評価方法)
7. [トラブルシューティング](#トラブルシューティング)
8. [高度なテクニック](#高度なテクニック)

## 概要

月読の学習プロセスは、段階的アプローチによりMOS 4.7以上（人間レベル）の品質を達成するよう設計されています：

- **ステージ1**: 基礎（100時間、10話者）
- **ステージ2**: スケールアップ（1,000時間、100話者）
- **ステージ3**: フルスケール（10,000時間、500以上の話者）

### 学習タイムライン

| ステージ | データサイズ | GPU | 学習時間 | 期待品質 |
|---------|-------------|-----|----------|----------|
| 1 | 100時間 | 2x H100 | 3-5日 | MOS 4.0-4.2 |
| 2 | 1,000時間 | 4x H100 | 2週間 | MOS 4.3-4.5 |
| 3 | 10,000時間 | 8x H100 | 6-8週間 | MOS 4.7+ |

## 前提条件

### ハードウェア要件

- **最小**: 2x NVIDIA A100/H100 GPU（80GB）
- **推奨**: 8x NVIDIA H100 GPU
- **RAM**: 512GB以上のシステムメモリ
- **ストレージ**: 50TB以上のNVMe SSD（フルデータセット用）

### ソフトウェアセットアップ

```bash
# UVパッケージマネージャーのインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# リポジトリのクローンと環境セットアップ
git clone https://github.com/ayutaz/tsukuyomi.git
cd tsukuyomi
uv venv
source .venv/bin/activate
uv pip install -e .
uv pip install -r requirements.txt
```

### インストールの確認

```bash
# GPU利用可能性の確認
python -c "import torch; print(f'GPU数: {torch.cuda.device_count()}')"

# コンポーネントテストの実行
pytest tests/test_ultimate_g2p.py -v
pytest tests/test_ultimate_acoustic_model.py -v
```

## データ準備

### 音声要件

- **フォーマット**: WAVファイル、48kHz、16ビットPCM
- **長さ**: 発話あたり3-15秒
- **品質**: スタジオ録音品質が望ましい
- **無音**: 先頭/末尾の無音は0.5秒未満

### ディレクトリ構造

```
data/
├── raw/
│   ├── speaker_001/
│   │   ├── audio/
│   │   │   ├── 001.wav
│   │   │   └── ...
│   │   └── transcripts.txt
│   └── speaker_002/
│       └── ...
├── processed/
│   ├── train/
│   ├── val/
│   └── test/
└── metadata/
    ├── speaker_info.json
    └── data_stats.json
```

### 書き起こしフォーマット

```text
# transcripts.txt
001.wav|こんにちは、月読です。
002.wav|今日はいい天気ですね。
003.wav|音声合成の実験を行います。
```

### 話者メタデータ

```json
{
  "speaker_001": {
    "name": "キャラクターA",
    "gender": "female",
    "age_range": "20-30",
    "dialect": "tokyo",
    "voice_characteristics": {
      "pitch": "中高音",
      "speaking_rate": "標準",
      "emotion_range": "表現豊か"
    }
  }
}
```

## 前処理パイプライン

### ステップ1: 音声検証と正規化

```bash
python scripts/preprocess/validate_audio.py \
    --input_dir data/raw \
    --output_dir data/validated \
    --sample_rate 48000 \
    --check_quality \
    --normalize_loudness -23
```

このスクリプトは以下を実行します：
- 音声フォーマットと品質の検証
- ラウドネスを-23 LUFSに正規化
- 破損ファイルの除去
- 検証レポートの生成

### ステップ2: 強制アライメント

```bash
python scripts/preprocess/forced_alignment.py \
    --audio_dir data/validated \
    --transcript_dir data/raw \
    --output_dir data/aligned \
    --model_name "japanese-wav2vec2" \
    --device cuda \
    --batch_size 32
```

出力内容：
- 音素レベルのアライメント
- 単語境界
- ポーズ位置

### ステップ3: 特徴抽出

```bash
python scripts/preprocess/extract_features.py \
    --input_dir data/aligned \
    --output_dir data/features \
    --features mel,f0,energy,duration \
    --mel_bins 128 \
    --hop_length 480 \
    --win_length 1920 \
    --f0_method dio \
    --num_workers 16
```

抽出される特徴：
- 128ビンメルスペクトログラム
- F0輪郭（DIOアルゴリズム）
- エネルギーエンベロープ
- 音素持続時間

### ステップ4: G2P変換

```bash
python scripts/preprocess/run_g2p.py \
    --transcript_dir data/aligned \
    --output_dir data/phonemes \
    --g2p_model ultimate \
    --add_accent \
    --add_word_boundary \
    --quality_check
```

生成内容：
- 音素シーケンス
- アクセントパターン
- 単語境界
- 信頼度スコア

### ステップ5: データ分割

```bash
python scripts/preprocess/split_data.py \
    --input_dir data/features \
    --output_dir data/processed \
    --train_ratio 0.95 \
    --val_ratio 0.03 \
    --test_ratio 0.02 \
    --speaker_balanced \
    --min_utterances_per_speaker 100
```

### ステップ6: 統計情報と検証

```bash
python scripts/preprocess/compute_stats.py \
    --data_dir data/processed \
    --output_file data/metadata/data_stats.json \
    --compute_speaker_embeddings \
    --plot_distributions
```

## 学習ステージ

### ステージ1: 基礎モデル（100時間）

#### 設定

```yaml
# configs/stage1_foundation.yaml
model:
  type: ultimate_tts
  g2p:
    pretrained: false
    accuracy_target: 0.90
  acoustic:
    n_speakers: 10
    n_flows: 8
    hidden_channels: 384
  vocoder:
    type: bigvgan_v2_small
    
data:
  train_path: data/processed/stage1/train
  val_path: data/processed/stage1/val
  batch_size: 32
  num_workers: 8
  
training:
  max_steps: 100000
  learning_rate: 2e-4
  warmup_steps: 5000
  gradient_accumulation: 4
  mixed_precision: bf16
  
logging:
  log_interval: 100
  eval_interval: 1000
  checkpoint_interval: 5000
```

#### 学習の開始

```bash
# シングルGPU
python train.py --config configs/stage1_foundation.yaml

# マルチGPU（2x H100）
torchrun --nproc_per_node=2 train.py \
    --config configs/stage1_foundation.yaml \
    --distributed
```

#### 進捗モニタリング

```bash
# TensorBoard
tensorboard --logdir logs/stage1

# カスタムモニタリング
python scripts/monitor_training.py \
    --experiment stage1 \
    --metrics loss,accuracy,mel_loss,f0_rmse
```

### ステージ2: スケールアップ（1,000時間）

#### ステージ2データの準備

```bash
python scripts/prepare_stage2.py \
    --stage1_checkpoint checkpoints/stage1/best.pt \
    --new_speakers 90 \
    --augmentation_config configs/augmentation.yaml
```

#### 設定の更新

```yaml
# configs/stage2_scaleup.yaml
model:
  checkpoint: checkpoints/stage1/best.pt
  acoustic:
    n_speakers: 100
    n_flows: 10
    hidden_channels: 512
    speaker_encoder:
      use_reference: true
      
training:
  max_steps: 500000
  learning_rate: 1e-4
  batch_size: 64
  gradient_checkpointing: true
  
# 4GPU用のFSDP有効化
distributed:
  backend: nccl
  strategy: fsdp
  sharding_strategy: full_shard
```

#### 分散学習の開始

```bash
# 4x H100 GPU
torchrun --nproc_per_node=4 \
    --master_port=29500 \
    train.py \
    --config configs/stage2_scaleup.yaml \
    --resume_from checkpoints/stage1/best.pt
```

### ステージ3: フルスケール（10,000時間）

#### 高度な設定

```yaml
# configs/stage3_fullscale.yaml
model:
  checkpoint: checkpoints/stage2/best.pt
  g2p:
    accuracy_target: 0.97
    use_neural_correction: true
  acoustic:
    n_speakers: 1000
    n_flows: 12
    hidden_channels: 512
    use_residual_adapters: true
  vocoder:
    type: bigvgan_v2_full
    
data:
  train_path: data/processed/stage3/train
  streaming: true  # 大規模データセット用
  cache_size: 10000
  prefetch_factor: 4
  
training:
  max_steps: 2000000
  learning_rate: 5e-5
  batch_size: 256  # 全GPU合計
  gradient_accumulation: 8
  
optimization:
  optimizer: adamw
  weight_decay: 0.01
  beta1: 0.9
  beta2: 0.95
  eps: 1e-8
  
distributed:
  strategy: fsdp
  sharding_strategy: hybrid_shard
  cpu_offload: true
  activation_checkpointing: true
```

#### フル学習の開始

```bash
# 8x H100 GPUで最適化
OMP_NUM_THREADS=8 torchrun \
    --nproc_per_node=8 \
    --master_addr=$MASTER_ADDR \
    --master_port=29500 \
    --nnodes=1 \
    train.py \
    --config configs/stage3_fullscale.yaml \
    --resume_from checkpoints/stage2/best.pt \
    --compile  # PyTorch 2.0コンパイル
```

## 評価方法

### 客観的指標

#### 1. メルケプストラル歪み（MCD）

```bash
python evaluate/compute_mcd.py \
    --synthesized_dir outputs/test \
    --reference_dir data/test/audio \
    --output_file results/mcd_scores.json
```

目標: MCD < 4.5 dB

#### 2. F0フレーム誤差（FFE）

```bash
python evaluate/compute_f0_metrics.py \
    --synthesized_dir outputs/test \
    --reference_dir data/test/audio \
    --metrics rmse,corr,vuv \
    --output_file results/f0_metrics.json
```

目標:
- F0 RMSE < 25 Hz
- F0相関 > 0.9
- 有声/無声誤差 < 5%

#### 3. 文字誤り率（CER）

```bash
python evaluate/compute_cer.py \
    --audio_dir outputs/test \
    --transcript_dir data/test/transcripts \
    --asr_model "japanese-whisper-large" \
    --output_file results/cer_scores.json
```

目標: CER < 2%

### 主観的評価

#### 1. 平均意見スコア（MOS）

```bash
# 評価セットの生成
python evaluate/prepare_mos_test.py \
    --num_samples 200 \
    --speakers 20 \
    --include_ground_truth \
    --output_dir evaluation/mos

# Webインターフェースの起動
python evaluate/mos_server.py \
    --test_dir evaluation/mos \
    --port 8080
```

#### 2. 話者類似度

```bash
python evaluate/speaker_similarity.py \
    --synthesized_dir outputs/test \
    --reference_dir data/test/audio \
    --model "japanese-speaker-encoder" \
    --output_file results/speaker_sim.json
```

目標: コサイン類似度 > 0.85

### 自動テスト

```bash
# 全評価の実行
python evaluate/run_all_tests.py \
    --checkpoint checkpoints/stage3/best.pt \
    --test_data data/test \
    --output_dir results/full_evaluation \
    --metrics all
```

## トラブルシューティング

### よくある問題

#### 1. OOM（メモリ不足）

```bash
# バッチサイズの削減
python train.py --config config.yaml \
    --override training.batch_size=16 \
    --override training.gradient_accumulation=8

# CPUオフロードの有効化
python train.py --config config.yaml \
    --override distributed.cpu_offload=true
```

#### 2. 学習が遅い

```bash
# 学習のプロファイリング
python train.py --config config.yaml \
    --profile \
    --profile_steps 100

# データローディングの最適化
python train.py --config config.yaml \
    --override data.num_workers=16 \
    --override data.pin_memory=true \
    --override data.prefetch_factor=4
```

#### 3. 勾配爆発

```yaml
# 設定に追加
training:
  gradient_clip_norm: 1.0
  gradient_clip_value: 5.0
  detect_anomaly: true
```

### デバッグツール

```bash
# アテンションマップの可視化
python debug/visualize_attention.py \
    --checkpoint checkpoints/model.pt \
    --text "テストです" \
    --output_dir debug/attention

# 勾配フローのチェック
python debug/check_gradients.py \
    --checkpoint checkpoints/model.pt \
    --num_steps 10

# 損失曲線の分析
python debug/analyze_losses.py \
    --log_file logs/train.log \
    --smooth_window 100
```

## 高度なテクニック

### 1. マルチ解像度学習

```python
# 複数解像度でのカスタムデータセット
class MultiResolutionDataset(Dataset):
    def __init__(self, data_dir, resolutions=[8, 16, 24]):
        self.resolutions = resolutions
        # 複数のホップサイズでデータをロード
        
    def __getitem__(self, idx):
        # ランダムな解像度を返す
        resolution = random.choice(self.resolutions)
        return self.load_item(idx, resolution)
```

### 2. プログレッシブ学習

```python
# 段階的にモデル容量を増加
def progressive_training(config):
    # 小さなモデルから開始
    model = create_model(hidden_size=256, n_layers=6)
    train(model, epochs=10)
    
    # モデルを拡張
    model = expand_model(model, hidden_size=512, n_layers=12)
    train(model, epochs=20)
```

### 3. データ拡張

```yaml
# augmentation.yaml
audio:
  - type: pitch_shift
    range: [-2, 2]  # 半音
    prob: 0.3
  - type: time_stretch
    range: [0.9, 1.1]
    prob: 0.3
  - type: add_noise
    snr_range: [20, 40]
    prob: 0.2
    
text:
  - type: synonym_replacement
    prob: 0.1
  - type: accent_variation
    dialects: ["osaka", "kyoto"]
    prob: 0.2
```

### 4. 効率的な推論

```python
# デプロイ用の最適化
python scripts/optimize_model.py \
    --checkpoint checkpoints/best.pt \
    --output_dir deployed \
    --quantize int8 \
    --optimize_onnx \
    --batch_size 1
```

### 5. 継続学習

```bash
# 新しい話者でのファインチューニング
python finetune.py \
    --base_checkpoint checkpoints/stage3/best.pt \
    --new_data data/new_speakers \
    --freeze_encoder \
    --adapter_rank 64 \
    --learning_rate 1e-5 \
    --max_steps 10000
```

## 学習のベストプラクティス

### 1. データ品質管理

- **音声チェック**: SNR < 20dBのクリップを除去
- **書き起こし検証**: ASRを使用して書き起こしを検証
- **話者の一貫性**: 一貫した録音条件を確保
- **バランスの取れたデータセット**: 音素と韻律のカバレッジを維持

### 2. ハイパーパラメータチューニング

```bash
# グリッドサーチ
python scripts/hyperparameter_search.py \
    --config configs/search_space.yaml \
    --metric mos \
    --trials 50 \
    --gpus_per_trial 2
```

### 3. 実験トラッキング

```python
# Weights & Biasesの使用
import wandb

wandb.init(project="tsukuyomi", config=config)
wandb.log({
    "loss": loss,
    "mel_loss": mel_loss,
    "f0_rmse": f0_rmse,
    "mos_estimate": mos
})
```

### 4. モデルアンサンブル

```python
# 複数チェックポイントのアンサンブル
models = [
    load_checkpoint(f"checkpoint_{i}.pt")
    for i in range(5)
]

def ensemble_inference(text):
    outputs = [model(text) for model in models]
    return torch.mean(torch.stack(outputs), dim=0)
```

## 音声モーフィング機能

### 基本的な音声モーフィング

```python
# 2人の話者間でモーフィング
audio = tts.morph_voices(
    text="こんにちは、月読です",
    speaker_ids=[0, 5],
    speaker_weights=[0.7, 0.3]  # 70% 話者0、30% 話者5
)
```

### 複数話者のモーフィング

```python
# 4人の話者をブレンド
audio = tts.morph_voices(
    text="複数の声をミックスします",
    speaker_ids=[0, 5, 10, 15],
    speaker_weights=[0.4, 0.3, 0.2, 0.1]
)
```

### 感情のモーフィング

```python
# 話者と感情の同時モーフィング
audio = tts.morph_voices(
    text="感情豊かな音声です",
    speaker_ids=[0, 5],
    speaker_weights=[0.6, 0.4],
    emotion_ids=["happy", "excited"],
    emotion_weights=[0.7, 0.3]
)
```

### グラデーションモーフィング

```python
# 話者間の段階的な変化
for i in range(11):
    weight = i / 10.0
    audio = tts.morph_voices(
        text="声が徐々に変化します",
        speaker_ids=[0, 5],
        speaker_weights=[1-weight, weight]
    )
    save_audio(audio, f"morph_{i}.wav")
```

## まとめ

月読をMOS 4.7以上に学習するには：

1. **高品質データ**: 10,000時間のクリーンで多様な音声
2. **段階的アプローチ**: 100時間から10,000時間への段階的スケーリング
3. **強力なハードウェア**: フル学習には8x H100 GPU
4. **慎重なモニタリング**: 客観的・主観的指標の両方を追跡
5. **忍耐**: 完全な学習には6-8週間

質問や問題がある場合：
- GitHub Issues: https://github.com/ayutaz/tsukuyomi/issues
- ドキュメント: https://github.com/ayutaz/tsukuyomi/wiki

学習頑張ってください！ 🚀