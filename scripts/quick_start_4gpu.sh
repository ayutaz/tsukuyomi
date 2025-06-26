#!/bin/bash
# T4×4 クイックスタートスクリプト

echo "🚀 Tsukuyomi TTS - T4×4 GPU Training Quick Start"
echo "================================================"

# 色付き出力の設定
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 1. GPU確認
echo -e "\n${GREEN}[1/5] GPU確認${NC}"
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
if [ "$GPU_COUNT" -eq "4" ]; then
    echo "✅ 4つのGPUを検出しました"
    nvidia-smi --query-gpu=name,memory.total --format=csv
else
    echo -e "${RED}⚠️  GPUが4つ検出されませんでした（検出数: $GPU_COUNT）${NC}"
    echo "続行しますか？ (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
fi

# 2. データ確認
echo -e "\n${GREEN}[2/5] データ確認${NC}"
if [ -f "data/jvs_ljspeech/metadata_multispeaker.csv" ]; then
    SAMPLE_COUNT=$(wc -l < data/jvs_ljspeech/metadata_multispeaker.csv)
    SPEAKER_COUNT=$(cut -d'|' -f2 data/jvs_ljspeech/metadata_multispeaker.csv | sort -u | wc -l)
    echo "✅ データセット検出: ${SAMPLE_COUNT} サンプル, ${SPEAKER_COUNT} 話者"
else
    echo -e "${RED}⚠️  データが見つかりません${NC}"
    echo "data/jvs_ljspeech/metadata_multispeaker.csv が存在することを確認してください"
    exit 1
fi

# 3. 設定ファイル確認
echo -e "\n${GREEN}[3/5] 設定ファイル確認${NC}"
CONFIG_FILE="configs/experiment_vits_jvs_4gpu.yaml"
if [ -f "$CONFIG_FILE" ]; then
    echo "✅ 設定ファイル: $CONFIG_FILE"
    # バッチサイズとエポック数を表示
    BATCH_SIZE=$(grep -A1 "training:" "$CONFIG_FILE" | grep "batch_size:" | awk '{print $2}')
    NUM_EPOCHS=$(grep "num_epochs:" "$CONFIG_FILE" | awk '{print $2}')
    echo "   - バッチサイズ: ${BATCH_SIZE} per GPU (総バッチサイズ: $((BATCH_SIZE * 4)))"
    echo "   - エポック数: ${NUM_EPOCHS}"
else
    echo -e "${RED}⚠️  設定ファイルが見つかりません${NC}"
    exit 1
fi

# 4. 環境変数の設定
echo -e "\n${GREEN}[4/5] 環境変数設定${NC}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=3600
echo "✅ CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "✅ NCCL設定完了"

# 5. 学習開始の確認
echo -e "\n${GREEN}[5/5] 学習開始${NC}"
echo -e "${YELLOW}以下の設定で学習を開始します:${NC}"
echo "- GPU数: 4 (T4)"
echo "- データ: ${SAMPLE_COUNT} サンプル, ${SPEAKER_COUNT} 話者"
echo "- バッチサイズ: ${BATCH_SIZE} per GPU"
echo "- エポック数: ${NUM_EPOCHS}"
echo "- 推定学習時間: 約5-7日"
echo ""
echo -e "${YELLOW}学習を開始しますか？ (y/n)${NC}"
read -r response

if [ "$response" = "y" ]; then
    # ログディレクトリの作成
    mkdir -p logs/vits_jvs_4gpu
    mkdir -p checkpoints/vits_jvs_4gpu
    
    # 学習開始
    echo -e "\n${GREEN}🎯 学習を開始します...${NC}"
    echo "ログファイル: logs/vits_jvs_4gpu/training_$(date +%Y%m%d_%H%M%S).log"
    
    # nohupで実行（バックグラウンド）
    nohup torchrun --nproc_per_node=4 \
                   --master_port=29500 \
                   scripts/train.py \
                   --config configs/experiment_vits_jvs_4gpu.yaml \
                   > "logs/vits_jvs_4gpu/training_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
    
    TRAIN_PID=$!
    echo -e "\n${GREEN}✅ 学習がバックグラウンドで開始されました (PID: $TRAIN_PID)${NC}"
    echo ""
    echo "📊 進捗確認コマンド:"
    echo "  - ログ確認: tail -f logs/vits_jvs_4gpu/training_*.log"
    echo "  - GPU監視: watch -n 1 nvidia-smi"
    echo "  - TensorBoard: tensorboard --logdir logs/vits_jvs_4gpu/tensorboard"
    echo "  - プロセス確認: ps aux | grep $TRAIN_PID"
    echo ""
    echo "💡 ヒント: tmuxやscreenを使用することをお勧めします"
else
    echo "学習をキャンセルしました"
fi