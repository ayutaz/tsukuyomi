#!/bin/bash
# T4×4 クイックスタートスクリプト（uv版）

echo "🚀 Tsukuyomi TTS - T4×4 GPU Training Quick Start (uv version)"
echo "============================================================="

# 色付き出力の設定
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 0. uv環境の確認
echo -e "\n${GREEN}[0/6] uv環境確認${NC}"
if [ -d ".venv" ]; then
    echo "✅ .venv ディレクトリを検出"
    # 環境をアクティベート
    source .venv/bin/activate
    echo "✅ uv環境をアクティベートしました"
    python --version
else
    echo -e "${YELLOW}⚠️  .venv が見つかりません。uv環境を作成します...${NC}"
    
    # uvがインストールされているか確認
    if ! command -v uv &> /dev/null; then
        echo -e "${RED}❌ uvがインストールされていません${NC}"
        echo "以下のコマンドでuvをインストールしてください:"
        echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    
    # uv環境を作成
    echo "Python 3.11環境を作成中（プロジェクト要件: Python >= 3.11）..."
    uv venv --python 3.11
    source .venv/bin/activate
    
    # 基本パッケージのインストール
    echo "PyTorchをインストール中（CUDA 12.1版 - CUDA 12.4と互換）..."
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    echo "依存関係をインストール中..."
    uv pip install -r requirements.txt
    uv pip install -e .
    uv pip install tensorboard accelerate
    
    echo -e "${GREEN}✅ uv環境のセットアップが完了しました${NC}"
fi

# 1. GPU確認
echo -e "\n${GREEN}[1/6] GPU確認${NC}"
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
if [ "$GPU_COUNT" -eq "4" ]; then
    echo "✅ 4つのGPUを検出しました"
    python -c "import torch; [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(4)]"
else
    echo -e "${RED}⚠️  GPUが4つ検出されませんでした（検出数: $GPU_COUNT）${NC}"
    echo "続行しますか？ (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
fi

# Python、PyTorch、CUDAバージョンの確認
echo -e "\n${BLUE}Python/PyTorch/CUDA情報:${NC}"
python -c "import sys; print(f'Python: {sys.version.split()[0]}')"
if python -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"; then
    echo "✅ Python 3.11以上を確認"
else
    echo -e "${RED}❌ Python 3.11以上が必要です${NC}"
    exit 1
fi
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.version.cuda}')"
python -c "import torch.distributed as dist; print(f'NCCL available: {dist.is_nccl_available()}')"

# 2. データ確認
echo -e "\n${GREEN}[2/6] データ確認${NC}"
if [ -f "data/jvs_ljspeech/metadata_multispeaker.csv" ]; then
    SAMPLE_COUNT=$(wc -l < data/jvs_ljspeech/metadata_multispeaker.csv)
    SPEAKER_COUNT=$(cut -d'|' -f2 data/jvs_ljspeech/metadata_multispeaker.csv | sort -u | wc -l)
    echo "✅ データセット検出: ${SAMPLE_COUNT} サンプル, ${SPEAKER_COUNT} 話者"
    
    # WAVファイルの存在確認
    WAV_COUNT=$(find data/jvs_ljspeech/wavs -name "*.wav" 2>/dev/null | wc -l)
    echo "   WAVファイル数: ${WAV_COUNT}"
else
    echo -e "${RED}⚠️  データが見つかりません${NC}"
    echo "data/jvs_ljspeech/metadata_multispeaker.csv が存在することを確認してください"
    exit 1
fi

# 3. 設定ファイル確認
echo -e "\n${GREEN}[3/6] 設定ファイル確認${NC}"
CONFIG_FILE="configs/experiment_vits_jvs_4gpu.yaml"
if [ -f "$CONFIG_FILE" ]; then
    echo "✅ 設定ファイル: $CONFIG_FILE"
    # 主要パラメータを表示
    BATCH_SIZE=$(grep -A1 "training:" "$CONFIG_FILE" | grep "batch_size:" | awk '{print $2}')
    NUM_EPOCHS=$(grep "num_epochs:" "$CONFIG_FILE" | awk '{print $2}')
    LR=$(grep -A3 "optimizers:" "$CONFIG_FILE" | grep "lr:" | awk '{print $2}')
    echo "   - バッチサイズ: ${BATCH_SIZE} per GPU (総バッチサイズ: $((BATCH_SIZE * 4)))"
    echo "   - エポック数: ${NUM_EPOCHS}"
    echo "   - 学習率: ${LR}"
else
    echo -e "${RED}⚠️  設定ファイルが見つかりません${NC}"
    exit 1
fi

# 4. ディレクトリ作成
echo -e "\n${GREEN}[4/6] ディレクトリ準備${NC}"
mkdir -p logs/vits_jvs_4gpu
mkdir -p checkpoints/vits_jvs_4gpu
mkdir -p outputs/vits_jvs_4gpu
echo "✅ 必要なディレクトリを作成しました"

# 5. 環境変数の設定
echo -e "\n${GREEN}[5/6] 環境変数設定${NC}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=WARN  # INFOは冗長なのでWARNに
export NCCL_TIMEOUT=3600
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
echo "✅ CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "✅ NCCL設定完了"
echo "✅ その他の最適化設定完了"

# 6. 学習開始の確認
echo -e "\n${GREEN}[6/6] 学習開始${NC}"
echo -e "${YELLOW}======================================${NC}"
echo -e "${YELLOW}学習設定サマリー:${NC}"
echo -e "${YELLOW}======================================${NC}"
echo "- 環境: uv (.venv)"
echo "- GPU数: 4 (T4)"
echo "- データ: ${SAMPLE_COUNT} サンプル, ${SPEAKER_COUNT} 話者"
echo "- バッチサイズ: ${BATCH_SIZE} per GPU (総: $((BATCH_SIZE * 4)))"
echo "- エポック数: ${NUM_EPOCHS}"
echo "- 推定学習時間: 約5-7日"
echo -e "${YELLOW}======================================${NC}"
echo ""
echo -e "${BLUE}実行オプション:${NC}"
echo "1) バックグラウンドで実行（推奨）"
echo "2) フォアグラウンドで実行"
echo "3) テスト実行（1エポックのみ）"
echo "4) キャンセル"
echo ""
echo -n "選択してください (1-4): "
read -r choice

case $choice in
    1)
        # バックグラウンド実行
        LOG_FILE="logs/vits_jvs_4gpu/training_$(date +%Y%m%d_%H%M%S).log"
        echo -e "\n${GREEN}🎯 バックグラウンドで学習を開始します...${NC}"
        echo "ログファイル: $LOG_FILE"
        
        nohup torchrun --nproc_per_node=4 \
                       --master_port=29500 \
                       scripts/train.py \
                       --config configs/experiment_vits_jvs_4gpu.yaml \
                       > "$LOG_FILE" 2>&1 &
        
        TRAIN_PID=$!
        echo -e "\n${GREEN}✅ 学習がバックグラウンドで開始されました (PID: $TRAIN_PID)${NC}"
        echo ""
        echo "📊 モニタリングコマンド:"
        echo "  - ログ確認: tail -f $LOG_FILE"
        echo "  - GPU監視: watch -n 1 nvidia-smi"
        echo "  - TensorBoard: tensorboard --logdir logs/vits_jvs_4gpu/tensorboard"
        echo "  - プロセス確認: ps aux | grep $TRAIN_PID"
        echo "  - 学習停止: kill $TRAIN_PID"
        ;;
    2)
        # フォアグラウンド実行
        echo -e "\n${GREEN}🎯 フォアグラウンドで学習を開始します...${NC}"
        echo "Ctrl+Cで中断できます"
        sleep 2
        
        torchrun --nproc_per_node=4 \
                 --master_port=29500 \
                 scripts/train.py \
                 --config configs/experiment_vits_jvs_4gpu.yaml
        ;;
    3)
        # テスト実行
        echo -e "\n${GREEN}🎯 テスト実行を開始します（1エポック）...${NC}"
        
        # 一時的な設定ファイルを作成
        TEST_CONFIG="configs/test_1epoch.yaml"
        cp $CONFIG_FILE $TEST_CONFIG
        sed -i 's/num_epochs: .*/num_epochs: 1/' $TEST_CONFIG
        
        torchrun --nproc_per_node=4 \
                 --master_port=29500 \
                 scripts/train.py \
                 --config $TEST_CONFIG
        
        rm $TEST_CONFIG
        ;;
    4)
        echo "学習をキャンセルしました"
        exit 0
        ;;
    *)
        echo "無効な選択です"
        exit 1
        ;;
esac