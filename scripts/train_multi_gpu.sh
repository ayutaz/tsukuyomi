#!/bin/bash
# マルチGPU学習スクリプト

# エラー時に停止
set -e

# 引数の確認
if [ $# -lt 1 ]; then
    echo "Usage: $0 <config_file> [options]"
    echo "Example: $0 configs/train_config.yaml --resume checkpoints/latest.pt"
    exit 1
fi

CONFIG_FILE=$1
shift  # 残りの引数を保持

# GPUの数を取得
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "検出されたGPU数: $NUM_GPUS"

# 環境変数の設定
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 8台のGPUを使用
export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO

# 学習の実行
echo "マルチGPU学習を開始します..."
echo "設定ファイル: $CONFIG_FILE"

# PyTorchの分散学習
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    scripts/train.py \
    --config $CONFIG_FILE \
    $@

echo "学習が完了しました。"