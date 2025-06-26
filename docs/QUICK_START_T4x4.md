# 🚀 T4×4 GPU環境 クイックスタートガイド

このガイドは、T4 GPU 4台を使用してTsukuyomi TTSモデルを学習するための最速手順です。

## 前提条件

- **Python**: 3.11以上（プロジェクト要件）
- **GPU**: NVIDIA T4 × 4台
- **CUDA**: 12.4.1（PyTorchはCUDA 12.1版を使用 - 互換性あり）
- **メモリ**: 各GPU 16GB VRAM
- **ストレージ**: 100GB以上の空き容量

## 1分で始める方法

```bash
# 1. リポジトリをクローン
git clone https://github.com/ayutaz/tsukuyomi.git
cd tsukuyomi

# 2. uvをインストール（高速パッケージマネージャー）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. データを配置
# data/jvs_ljspeech/metadata_multispeaker.csv
# data/jvs_ljspeech/wavs/*.wav

# 4. クイックスタートスクリプトを実行
./scripts/quick_start_4gpu_uv.sh
```

スクリプトが自動的に：
- ✅ Python 3.11環境を作成
- ✅ 必要なパッケージをインストール
- ✅ GPU環境を検証
- ✅ データを確認
- ✅ 学習を開始

## 学習オプション

スクリプト実行時に選択できます：

1. **バックグラウンド実行**（推奨）
   - nohupで実行、ログファイルに出力
   - ターミナルを閉じても継続

2. **フォアグラウンド実行**
   - 直接出力を確認
   - Ctrl+Cで中断可能

3. **テスト実行**
   - 1エポックのみ実行
   - 環境の動作確認用

## モニタリング

```bash
# 別ターミナルで実行

# TensorBoard（学習曲線の確認）
source .venv/bin/activate
tensorboard --logdir logs/vits_jvs_4gpu/tensorboard

# GPU使用状況
watch -n 1 nvidia-smi

# ログ確認
tail -f logs/vits_jvs_4gpu/training_*.log
```

## トラブルシューティング

### メモリ不足エラー
```yaml
# configs/experiment_vits_jvs_4gpu.yaml
training:
  batch_size: 12  # 16から削減
```

### Python バージョンエラー
```bash
# Python 3.11以上が必要
python --version  # 確認
uv venv --python 3.11  # 3.11を指定
```

### NCCL通信エラー
```bash
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
```

## 期待される結果

| エポック数 | 所要時間 | 品質レベル |
|-----------|---------|-----------|
| 1 | 1.3-1.5時間 | 動作確認 |
| 10 | 13-15時間 | 基本的な音声 |
| 20-30 | 1-2日 | 実用レベル |
| 50 | 3-4日 | 高品質 |
| 100 | 5-7日 | 最高品質 |

## 推論テスト

学習中でもテスト可能：

```bash
source .venv/bin/activate
python scripts/inference.py \
  --checkpoint checkpoints/vits_jvs_4gpu/checkpoint_latest.pt \
  --text "こんにちは、音声合成のテストです。" \
  --speaker jvs001 \
  --output test.wav
```

## サポート

問題が発生した場合：
1. エラーログを確認: `grep -i error logs/vits_jvs_4gpu/training_*.log`
2. GitHub Issues: https://github.com/ayutaz/tsukuyomi/issues
3. ドキュメント: `docs/T4x4_training_guide_uv.md`（詳細版）

---

**準備ができたら `./scripts/quick_start_4gpu_uv.sh` を実行！**