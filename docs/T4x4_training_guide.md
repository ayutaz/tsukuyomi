# T4×4 GPU環境でのVITS学習ガイド（完全版）

## 1. 環境セットアップ

### 1.1 リポジトリのクローン
```bash
git clone https://github.com/ayutaz/tsukuyomi.git
cd tsukuyomi
```

### 1.2 Python環境の作成（推奨: Python 3.10）
```bash
# Conda環境の作成
conda create -n tsukuyomi python=3.10
conda activate tsukuyomi

# または venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
```

### 1.3 依存関係のインストール
```bash
# PyTorchのインストール（CUDA 11.8対応）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# その他の依存関係
pip install -r requirements.txt

# 開発モードでインストール
pip install -e .
```

### 1.4 GPUとNCCLの確認
```bash
# GPUの確認
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# NCCLの動作確認
python -c "import torch.distributed as dist; print('NCCL available:', dist.is_nccl_available())"
```

## 2. データの準備

### 2.1 JVSデータセット（LJSpeech形式）の配置
```bash
# データディレクトリの作成
mkdir -p data/jvs_ljspeech

# データの配置（既に変換済みのデータがある場合）
# data/jvs_ljspeech/
#   ├── metadata_multispeaker.csv  # 全話者のメタデータ
#   └── wavs/                       # 音声ファイル
#       ├── jvs001_001.wav
#       ├── jvs001_002.wav
#       └── ...
```

### 2.2 データの確認
```bash
# データ数の確認
wc -l data/jvs_ljspeech/metadata_multispeaker.csv

# 話者数の確認
cut -d'|' -f2 data/jvs_ljspeech/metadata_multispeaker.csv | sort -u | wc -l

# サンプルの確認
head -5 data/jvs_ljspeech/metadata_multispeaker.csv
```

## 3. 4GPU分散学習の設定

### 3.1 設定ファイルの確認
```bash
# 4GPU用設定ファイルを確認
cat configs/experiment_vits_jvs_4gpu.yaml
```

### 3.2 必要に応じて設定を調整
```yaml
# configs/experiment_vits_jvs_4gpu.yaml の主要パラメータ

training:
  batch_size: 16        # GPU当たりのバッチサイズ（総バッチサイズ64）
  num_epochs: 100       # エポック数
  
  optimizers:
    default:
      lr: 8e-4          # 4GPU用に調整済み（線形スケーリング）
```

## 4. 学習の実行

### 4.1 分散学習スクリプトの実行権限付与
```bash
chmod +x scripts/train_multi_gpu_distributed.sh
```

### 4.2 学習の開始
```bash
# 4GPU分散学習を開始
./scripts/train_multi_gpu_distributed.sh configs/experiment_vits_jvs_4gpu.yaml

# または直接torchrunを使用
torchrun --nproc_per_node=4 \
         --master_port=29500 \
         scripts/train.py \
         --config configs/experiment_vits_jvs_4gpu.yaml
```

### 4.3 バックグラウンドで実行（推奨）
```bash
# nohupを使用
nohup ./scripts/train_multi_gpu_distributed.sh configs/experiment_vits_jvs_4gpu.yaml > training.log 2>&1 &

# またはtmux/screenを使用
tmux new -s vits_training
./scripts/train_multi_gpu_distributed.sh configs/experiment_vits_jvs_4gpu.yaml
# Ctrl+B, D でデタッチ
# tmux attach -t vits_training で再接続
```

## 5. 学習のモニタリング

### 5.1 TensorBoardの起動
```bash
# 別ターミナルで実行
tensorboard --logdir logs/vits_jvs_4gpu/tensorboard --port 6006

# リモートサーバーの場合
# ローカルマシンから: ssh -L 6006:localhost:6006 user@server
# ブラウザで: http://localhost:6006
```

### 5.2 GPU使用状況の監視
```bash
# リアルタイムモニタリング
watch -n 1 nvidia-smi

# または詳細な情報
nvidia-smi dmon -s um -d 1
```

### 5.3 ログの確認
```bash
# 学習ログの確認
tail -f training.log

# エラーの確認
grep -i error training.log
```

## 6. トラブルシューティング

### 6.1 NCCL通信エラー
```bash
# デバッグ情報を有効化
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# タイムアウトの延長
export NCCL_TIMEOUT=3600

# ネットワークインターフェースの指定
export NCCL_SOCKET_IFNAME=eth0
```

### 6.2 メモリ不足（OOM）
```yaml
# configs/experiment_vits_jvs_4gpu.yaml を編集
training:
  batch_size: 12  # 16から削減
  memory_optimization:
    gradient_checkpointing: true  # 有効化
```

### 6.3 学習が遅い場合
```bash
# データローディングのボトルネック確認
# configs/experiment_vits_jvs_4gpu.yaml
data:
  num_workers: 4  # 調整（CPU数の半分程度が目安）
  use_cache: true  # キャッシュを有効化
```

## 7. チェックポイントの管理

### 7.1 チェックポイントの確認
```bash
# 保存されたチェックポイント
ls -lah checkpoints/vits_jvs_4gpu/

# 最新のチェックポイント
ls -t checkpoints/vits_jvs_4gpu/checkpoint_*.pt | head -1
```

### 7.2 学習の再開
```bash
# 中断した学習を再開
torchrun --nproc_per_node=4 \
         scripts/train.py \
         --config configs/experiment_vits_jvs_4gpu.yaml \
         --resume checkpoints/vits_jvs_4gpu/checkpoint_latest.pt
```

## 8. 期待される学習時間

- **1エポック**: 約1.3-1.5時間
- **100エポック**: 約5-7日
- **実用レベル（20-30エポック）**: 1-2日

## 9. 学習完了後の評価

### 9.1 推論テスト
```bash
# 学習済みモデルでテスト
python scripts/inference.py \
  --checkpoint checkpoints/vits_jvs_4gpu/best_model.pt \
  --text "こんにちは、音声合成のテストです。" \
  --speaker jvs001 \
  --output output.wav
```

### 9.2 音質評価
```bash
# 評価スクリプトの実行
python scripts/evaluate.py \
  --checkpoint checkpoints/vits_jvs_4gpu/best_model.pt \
  --test-data data/jvs_ljspeech/test_set.csv
```

## 10. 推奨される実行順序

1. **まず10話者で動作確認（1-2時間）**
   ```bash
   # 少数話者でテスト
   python scripts/create_subset.py --num-speakers 10
   ./scripts/train_multi_gpu_distributed.sh configs/experiment_vits_jvs_10speakers.yaml
   ```

2. **問題なければ全話者で本番学習**
   ```bash
   ./scripts/train_multi_gpu_distributed.sh configs/experiment_vits_jvs_4gpu.yaml
   ```

## 注意事項

- 初回実行時はデータの前処理でメモリを大量に使用する可能性があります
- 学習中は定期的にチェックポイントが保存されるので、中断しても問題ありません
- T4のメモリ（16GB）を考慮してバッチサイズを調整してください