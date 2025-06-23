# Tsukuyomi TTS - Installation Guide

## Prerequisites

- Python 3.11 or higher
- CUDA 12.1+ (for GPU support)
- 8GB+ RAM (16GB+ recommended)
- 10GB+ free disk space

## Quick Installation

### Option 1: Using UV (Recommended)

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/ayutaz/tsukuyomi.git
cd tsukuyomi

# Create virtual environment and install
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e .
uv pip install -r requirements.txt
```

### Option 2: Using pip

```bash
# Clone repository
git clone https://github.com/ayutaz/tsukuyomi.git
cd tsukuyomi

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
pip install -r requirements.txt
```

### Option 3: Using Docker

```bash
# Clone repository
git clone https://github.com/ayutaz/tsukuyomi.git
cd tsukuyomi

# Build Docker image
docker build -t tsukuyomi .

# Run container
docker run -it --gpus all -v $(pwd):/workspace tsukuyomi
```

## Verify Installation

Run the test script to verify everything is installed correctly:

```bash
python scripts/test_installation.py
```

You should see:
```
✅ All tests passed! Tsukuyomi TTS is ready to use.
```

## Platform-Specific Instructions

### Windows

1. Install Visual Studio Build Tools for C++ extensions
2. Install MeCab separately:
   ```bash
   # Using conda
   conda install -c conda-forge mecab
   
   # Or download from: https://taku910.github.io/mecab/
   ```

### macOS

1. Install MeCab:
   ```bash
   brew install mecab mecab-ipadic
   ```

2. For Apple Silicon (M1/M2), install PyTorch with MPS support:
   ```bash
   pip install torch torchvision torchaudio
   ```

### Linux

1. Install system dependencies:
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install -y libsndfile1 mecab libmecab-dev mecab-ipadic-utf8
   
   # CentOS/RHEL
   sudo yum install -y libsndfile mecab mecab-devel mecab-ipadic
   ```

## GPU Support

### NVIDIA GPU

1. Install CUDA 12.1+:
   - Download from: https://developer.nvidia.com/cuda-downloads

2. Install PyTorch with CUDA:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. Verify GPU is detected:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should print True
   print(torch.cuda.get_device_name(0))  # Should show your GPU
   ```

### AMD GPU (ROCm)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

## Common Issues

### 1. MeCab Error on Windows

**Error:** `error: Microsoft Visual C++ 14.0 or greater is required`

**Solution:**
1. Install Visual Studio Build Tools
2. Or use pre-built wheel:
   ```bash
   pip install mecab-python3-windows
   ```

### 2. CUDA Out of Memory

**Error:** `CUDA out of memory`

**Solution:**
- Reduce batch size in training config
- Enable gradient checkpointing
- Use mixed precision training (fp16/bf16)

### 3. Import Errors

**Error:** `ModuleNotFoundError`

**Solution:**
```bash
# Ensure you're in the project directory
cd tsukuyomi

# Reinstall in editable mode
pip install -e .
```

### 4. Audio Backend Issues

**Error:** `No audio backend available`

**Solution:**
```bash
# Linux
sudo apt-get install libsndfile1 ffmpeg

# macOS
brew install libsndfile ffmpeg

# Windows
# Install ffmpeg from: https://ffmpeg.org/download.html
```

## Optional Dependencies

### For Development

```bash
pip install -e ".[dev]"
```

This includes:
- pytest (testing)
- black (code formatting)
- ruff (linting)
- mypy (type checking)
- pre-commit (git hooks)

### For Documentation

```bash
pip install -e ".[docs]"
```

### For All Features

```bash
pip install -e ".[all]"
```

## Next Steps

1. **Test the installation:**
   ```bash
   python scripts/test_installation.py
   ```

2. **Try the Web UI:**
   ```bash
   streamlit run app.py
   ```

3. **Prepare your data:**
   ```bash
   python scripts/preprocess_audio.py --input-dir your_audio/ --output-dir processed/
   ```

4. **Start training:**
   ```bash
   python train.py --config configs/stage1_foundation.yaml
   ```

## Support

If you encounter issues:

1. Check the [FAQ](docs/FAQ.md)
2. Search [GitHub Issues](https://github.com/ayutaz/tsukuyomi/issues)
3. Join our [Discord community](https://discord.gg/tsukuyomi)
4. Create a new issue with:
   - Your OS and Python version
   - Full error message
   - Steps to reproduce