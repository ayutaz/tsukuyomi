#!/bin/bash
# Tsukuyomi TTS Training Script for JVS Dataset

# 環境変数の設定
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128

# T4 GPU用の最適化設定
export TORCH_CUDA_ARCH_LIST="7.5"
export CUDA_LAUNCH_BLOCKING=0

# 学習の実行
echo "Starting Tsukuyomi TTS training with JVS dataset..."
echo "Configuration: configs/experiment_jvs_t4.yaml"
echo "GPU: NVIDIA T4 (16GB)"

# データディレクトリの確認
if [ ! -d "data/jvs_ljspeech" ]; then
    echo "Error: data/jvs_ljspeech directory not found!"
    exit 1
fi

# 学習開始
python scripts/train.py \
    --config configs/experiment_jvs_t4.yaml

echo "Training completed!"