# CUDA互換性ガイド

## CUDA 12.4.1環境でのPyTorchセットアップ

### 互換性情報

CUDA 12.4.1を使用している場合、PyTorchの公式ビルドは以下の通りです：

- **CUDA 12.4.1**: ドライバーバージョン
- **PyTorch CUDA 12.1**: 使用するPyTorchビルド（後方互換性あり）

NVIDIA CUDAは後方互換性があるため、新しいCUDAドライバーで古いCUDAランタイムのアプリケーションを実行できます。

### インストールコマンド

```bash
# uvを使用
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# または通常のpip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 互換性の確認

```bash
# システムのCUDAバージョン確認
nvidia-smi
# Driver Version: 550.xxx.xx   CUDA Version: 12.4

# PyTorchが認識するCUDAバージョン
python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"
# PyTorch CUDA: 12.1

# CUDAが利用可能か確認
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# CUDA available: True

# GPU数の確認
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
# GPU count: 4
```

### トラブルシューティング

#### CUDAが認識されない場合

1. **環境変数の確認**
   ```bash
   echo $CUDA_HOME
   echo $LD_LIBRARY_PATH
   ```

2. **PyTorchの再インストール**
   ```bash
   uv pip uninstall torch torchvision torchaudio
   uv pip cache clear
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **代替オプション（CUDA 11.8版）**
   ```bash
   # CUDA 12.xで問題がある場合
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### パフォーマンス最適化

CUDA 12.4の新機能を活用：

```python
# PyTorchでの設定
import torch

# TF32を有効化（Ampere以降のGPU）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# cuDNN最適化
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
```

### 推奨される構成

T4 GPUでの最適な設定：

- **PyTorch**: 2.2.0以上（CUDA 12.1ビルド）
- **cuDNN**: 8.9.x（自動的に含まれる）
- **NCCL**: 2.19.x（分散学習用）

### 参考リンク

- [PyTorch CUDA互換性マトリックス](https://pytorch.org/get-started/locally/)
- [NVIDIA CUDA互換性ガイド](https://docs.nvidia.com/deploy/cuda-compatibility/)
- [T4 GPU仕様](https://www.nvidia.com/en-us/data-center/tesla-t4/)