#!/bin/bash
# Multi-GPU distributed training script for VITS

# 使用するGPU数を設定
NUM_GPUS=4

# 設定ファイルのパス
CONFIG_FILE="${1:-configs/experiment_vits_jvs_full.yaml}"

# マスターポートを設定（他のプロセスと衝突しないように）
MASTER_PORT="${MASTER_PORT:-29500}"

# 分散学習の起動
echo "Starting distributed training with $NUM_GPUS GPUs..."
echo "Config: $CONFIG_FILE"
echo "Master port: $MASTER_PORT"

# torchrunを使用（PyTorch 1.9+推奨）
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    scripts/train.py \
    --config $CONFIG_FILE \
    --distributed \
    "${@:2}"  # 追加の引数を渡す

# 古いバージョンのPyTorchの場合は以下を使用
# python -m torch.distributed.launch \
#     --nproc_per_node=$NUM_GPUS \
#     --master_port=$MASTER_PORT \
#     scripts/train.py \
#     --config $CONFIG_FILE \
#     --distributed \
#     "${@:2}"