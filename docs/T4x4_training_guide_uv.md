# T4×4 GPU環境でのVITS学習ガイド（uv版）

## 1. 環境セットアップ（uvを使用）

### 1.1 リポジトリのクローン
```bash
git clone https://github.com/ayutaz/tsukuyomi.git
cd tsukuyomi
```

### 1.2 uvのインストール（まだの場合）
```bash
# uvのインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# または pip でインストール
pip install uv
```

### 1.3 uv環境の作成とセットアップ
```bash
# Python 3.11環境を作成（プロジェクト要件: Python >= 3.11）
uv venv --python 3.11

# 環境をアクティベート
source .venv/bin/activate  # Linux/Mac
# または
.venv\Scripts\activate  # Windows
```

### 1.4 依存関係のインストール（uv使用）
```bash
# PyTorchのインストール（CUDA 12.1対応 - CUDA 12.4と互換性あり）
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# プロジェクトの依存関係をインストール
uv pip install -r requirements.txt

# 開発モードでインストール
uv pip install -e .

# 追加で必要なパッケージ
uv pip install tensorboard accelerate
```

### 1.5 GPUとNCCLの確認
```bash
# GPUの確認
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# NCCLの動作確認
python -c "import torch.distributed as dist; print('NCCL available:', dist.is_nccl_available())"
```

## 2. データの準備

### 2.1 JVSデータセット（LJSpeech形式）の配置
```bash
# データディレクトリの作成
mkdir -p data/jvs_ljspeech

# データの配置構造
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

## 3. クイックスタートスクリプトの実行

### 3.1 uv対応版のクイックスタートスクリプト
```bash
# 実行権限を付与
chmod +x scripts/quick_start_4gpu_uv.sh

# スクリプトを実行
./scripts/quick_start_4gpu_uv.sh
```

## 4. 手動での学習実行

### 4.1 環境変数の設定
```bash
# GPU設定
export CUDA_VISIBLE_DEVICES=0,1,2,3

# NCCL設定（通信最適化）
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=3600
export OMP_NUM_THREADS=1  # DataLoaderの最適化
```

### 4.2 4GPU分散学習の開始
```bash
# torchrunを使用（PyTorch 1.9+）
torchrun --nproc_per_node=4 \
         --master_port=29500 \
         scripts/train.py \
         --config configs/experiment_vits_jvs_4gpu.yaml
```

### 4.3 バックグラウンド実行（推奨）
```bash
# tmuxを使用
tmux new -s vits_training

# tmux内で実行
source .venv/bin/activate  # uv環境の再アクティベート
torchrun --nproc_per_node=4 scripts/train.py --config configs/experiment_vits_jvs_4gpu.yaml

# Ctrl+B, D でデタッチ
# tmux attach -t vits_training で再接続
```

## 5. 学習のモニタリング

### 5.1 TensorBoardの起動
```bash
# 別ターミナルで、uv環境をアクティベート
source .venv/bin/activate
tensorboard --logdir logs/vits_jvs_4gpu/tensorboard --port 6006

# リモートサーバーの場合
# ローカルマシンから: ssh -L 6006:localhost:6006 user@server
# ブラウザで: http://localhost:6006
```

### 5.2 リアルタイムモニタリング
```bash
# GPU使用状況
watch -n 1 nvidia-smi

# 学習ログ
tail -f logs/vits_jvs_4gpu/training_*.log

# プロセス確認
ps aux | grep train.py
```

## 6. トラブルシューティング

### 6.1 uv関連の問題
```bash
# uv環境が見つからない場合
which python  # .venv/bin/python を指しているか確認

# パッケージの再インストール
uv pip install --upgrade -r requirements.txt
```

### 6.2 CUDA/GPUエラー
```bash
# CUDAバージョンの確認
nvidia-smi  # ドライバーバージョン確認（CUDA 12.4.1）
python -c "import torch; print(torch.version.cuda)"  # PyTorchのCUDAバージョン

# 不一致の場合は適切なPyTorchを再インストール
uv pip uninstall torch torchvision torchaudio
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4は12.1と互換性があるため、cu121版が使用可能
```

### 6.3 メモリ不足（OOM）
```yaml
# configs/experiment_vits_jvs_4gpu.yaml を編集
training:
  batch_size: 12  # 16から削減
  memory_optimization:
    gradient_checkpointing: true
    clear_cache_interval: 50  # より頻繁にキャッシュクリア
```

## 7. 学習の再開

### 7.1 中断した学習の再開
```bash
# uv環境をアクティベート
source .venv/bin/activate

# 最新のチェックポイントから再開
LATEST_CHECKPOINT=$(ls -t checkpoints/vits_jvs_4gpu/checkpoint_*.pt | head -1)
echo "Resuming from: $LATEST_CHECKPOINT"

torchrun --nproc_per_node=4 \
         scripts/train.py \
         --config configs/experiment_vits_jvs_4gpu.yaml \
         --resume $LATEST_CHECKPOINT
```

## 8. 推論テスト

### 8.1 学習中のモデルでテスト
```bash
# uv環境内で実行
python scripts/inference.py \
  --checkpoint checkpoints/vits_jvs_4gpu/checkpoint_latest.pt \
  --text "こんにちは、音声合成のテストです。" \
  --speaker jvs001 \
  --output test_output.wav

# 音声ファイルの再生（Linux）
aplay test_output.wav
```

## 9. ベストプラクティス

### 9.1 uv環境の管理
```bash
# 環境の情報確認
uv pip list

# Pythonバージョン確認
python --version  # 3.11以上であることを確認

# 環境のエクスポート
uv pip freeze > requirements_frozen.txt

# 別マシンでの再現
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements_frozen.txt
```

### 9.2 効率的な学習
1. **最初は少ないデータで確認**
   ```bash
   # 10話者のサブセットで動作確認
   python scripts/create_subset.py --num-speakers 10
   ```

2. **段階的な学習**
   - 10話者 × 10エポック（動作確認）
   - 50話者 × 30エポック（中間評価）
   - 100話者 × 100エポック（最終学習）

## 10. 期待される結果

- **1エポック**: 約1.3-1.5時間（4GPU）
- **実用レベル（20-30エポック）**: 1-2日
- **高品質（50エポック）**: 3-4日
- **最高品質（100エポック）**: 5-7日

## 注意事項

- uvは高速なパッケージ管理を提供しますが、一部のパッケージで互換性問題が発生する可能性があります
- 問題が発生した場合は、通常のpipにフォールバックしてください
- T4のVRAM（16GB）制限に注意してバッチサイズを調整してください