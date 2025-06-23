#!/bin/bash
# Tsukuyomi H100 Training Environment Setup Script

echo "=== Tsukuyomi H100 Training Setup ==="
echo "Setting up optimized environment for H100 GPUs with BF16 training..."

# Check CUDA version
echo "Checking CUDA version..."
nvidia-smi
nvcc --version

# Create virtual environment
echo "Creating Python virtual environment..."
python3.11 -m venv venv_h100
source venv_h100/bin/activate

# Upgrade pip and essential tools
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support
echo "Installing PyTorch 2.2.0 with CUDA 12.1..."
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# Install NVIDIA Transformer Engine for H100
echo "Installing NVIDIA Transformer Engine..."
pip install transformer-engine

# Install Flash Attention v2
echo "Installing Flash Attention v2..."
pip install ninja
pip install flash-attn --no-build-isolation

# Install NVIDIA Apex from source (for latest optimizations)
echo "Installing NVIDIA Apex..."
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
cd ..

# Install other dependencies
echo "Installing remaining dependencies..."
pip install -r requirements.txt

# Install additional H100 optimization tools
pip install nvidia-dlprof
pip install nvidia-ml-py
pip install py3nvml

# Setup environment variables for optimal performance
echo "Setting up environment variables..."
cat > setup_env.sh << 'EOF'
# H100 Optimization Environment Variables

# Enable TF32 for matmul and convolutions
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# Flash Attention settings
export FLASH_ATTENTION_SKIP_GEMM=0

# NCCL optimizations for multi-GPU
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_LEVEL=NVL

# CUDA settings
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_LAUNCH_BLOCKING=0

# PyTorch settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_CUDA_ARCH_LIST="9.0"  # H100 architecture

# Distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=8

echo "Environment variables set for H100 optimization"
EOF

# Create distributed training launcher
echo "Creating distributed training launcher..."
cat > launch_training.sh << 'EOF'
#!/bin/bash
# Distributed training launcher for H100 x8

source setup_env.sh

# Launch with torchrun (PyTorch 2.0+ distributed launcher)
torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_tsukuyomi.py \
    --use_bf16 \
    --use_flash_attention \
    --use_transformer_engine \
    --batch_size=32 \
    --gradient_accumulation=8 \
    --learning_rate=5e-4 \
    --num_epochs=100 \
    --checkpoint_dir=./checkpoints \
    --tensorboard_dir=./logs
EOF

chmod +x launch_training.sh

# Create performance monitoring script
echo "Creating performance monitoring script..."
cat > monitor_training.py << 'EOF'
#!/usr/bin/env python3
"""H100 Training Performance Monitor"""

import nvidia_ml_py as nvml
import time
import torch

def monitor_gpus():
    nvml.nvmlInit()
    device_count = nvml.nvmlDeviceGetCount()
    
    print("=== H100 GPU Status ===")
    for i in range(device_count):
        handle = nvml.nvmlDeviceGetHandleByIndex(i)
        
        # GPU utilization
        util = nvml.nvmlDeviceGetUtilizationRates(handle)
        print(f"GPU {i}: {util.gpu}% utilized")
        
        # Memory info
        mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"  Memory: {mem_info.used / 1024**3:.1f}GB / {mem_info.total / 1024**3:.1f}GB")
        
        # Temperature
        temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
        print(f"  Temperature: {temp}°C")
        
        # Power
        power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000
        print(f"  Power: {power:.1f}W")
    
    # PyTorch memory stats
    if torch.cuda.is_available():
        print("\n=== PyTorch Memory Stats ===")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}:")
            print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"  Reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")

if __name__ == "__main__":
    while True:
        monitor_gpus()
        print("\n" + "="*40 + "\n")
        time.sleep(5)
EOF

chmod +x monitor_training.py

# Verify installation
echo "Verifying installation..."
python -c "
import torch
import transformer_engine
import flash_attn

print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('cuDNN version:', torch.backends.cudnn.version())
print('Number of GPUs:', torch.cuda.device_count())

if torch.cuda.is_available():
    print('GPU 0:', torch.cuda.get_device_name(0))
    
# Test BF16 support
if torch.cuda.is_bf16_supported():
    print('BF16 is supported!')
    
    # Quick BF16 test
    x = torch.randn(1024, 1024, dtype=torch.bfloat16, device='cuda')
    y = torch.matmul(x, x)
    print('BF16 matmul test passed!')
else:
    print('WARNING: BF16 not supported on this device')
"

echo "=== Setup Complete ==="
echo "To start training, run: ./launch_training.sh"
echo "To monitor performance, run: python monitor_training.py"