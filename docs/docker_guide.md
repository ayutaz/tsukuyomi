# Docker Guide for Tsukuyomi TTS

This guide explains how to run Tsukuyomi TTS using Docker on both Linux/macOS and Windows.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Windows Setup](#windows-setup)
- [Linux/macOS Setup](#linux-macos-setup)
- [Training with Docker](#training-with-docker)
- [Development Environment](#development-environment)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### All Platforms
- Docker installed ([Docker Desktop](https://www.docker.com/products/docker-desktop))
- At least 16GB RAM (32GB+ recommended for training)
- 100GB+ free disk space

### For GPU Support
- NVIDIA GPU with CUDA 12.1+ support
- NVIDIA Docker runtime (Linux) or WSL2 with GPU support (Windows)

### Windows Specific
- Windows 10/11 Pro, Enterprise, or Education (for Hyper-V)
- WSL2 installed (recommended)
- Visual Studio Build Tools (for native Windows containers)

## Quick Start

### Linux/macOS
```bash
# Clone the repository
git clone https://github.com/your-username/tsukuyomi.git
cd tsukuyomi

# Build and run
./scripts/docker_run.sh build
./scripts/docker_run.sh run

# Access the services
# API: http://localhost:8000
# Demo UI: http://localhost:7860
# API Docs: http://localhost:8000/docs
```

### Windows
```powershell
# Clone the repository
git clone https://github.com/your-username/tsukuyomi.git
cd tsukuyomi

# Build and run
.\scripts\docker_run.bat build
.\scripts\docker_run.bat run

# Access the services (same URLs as above)
```

## Windows Setup

### Option 1: WSL2 with GPU Support (Recommended)

1. **Install WSL2**
   ```powershell
   wsl --install
   wsl --set-default-version 2
   ```

2. **Install NVIDIA GPU Support for WSL2**
   - Download and install [NVIDIA GPU Driver for WSL](https://developer.nvidia.com/cuda/wsl)
   - No need to install CUDA toolkit inside WSL2

3. **Use Linux containers**
   ```powershell
   # Docker Desktop will automatically use WSL2 backend
   docker-compose up -d
   ```

### Option 2: Native Windows Containers

1. **Switch to Windows containers**
   - Right-click Docker Desktop tray icon
   - Select "Switch to Windows containers..."

2. **Build and run**
   ```powershell
   docker-compose -f docker-compose.windows.yml build
   docker-compose -f docker-compose.windows.yml up -d
   ```

## Linux/macOS Setup

### Installing NVIDIA Docker Runtime (Linux only)

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Training with Docker

### Prepare Training Data

1. **Create data directories**
   ```bash
   mkdir -p data/raw data/processed
   ```

2. **Copy your audio files and transcripts**
   ```bash
   # Example structure:
   # data/raw/
   #   ├── speaker_001/
   #   │   ├── audio_001.wav
   #   │   └── ...
   #   └── transcripts.txt
   ```

### Start Training

#### Linux/macOS
```bash
# Run preprocessing
docker-compose run --rm tsukuyomi-tts python scripts/preprocess.py

# Start training with 100h config
./scripts/docker_run.sh train
```

#### Windows
```powershell
# Run preprocessing
docker-compose run --rm tsukuyomi-tts python scripts/preprocess.py

# Start training
.\scripts\docker_run.bat train
```

### Monitor Training

```bash
# View logs
docker-compose logs -f tsukuyomi-tts

# Start TensorBoard
docker-compose --profile monitoring up -d tensorboard
# Access at http://localhost:6006
```

## Development Environment

### Jupyter Notebook

```bash
# Start Jupyter environment
docker-compose --profile dev up -d jupyter

# Access at http://localhost:8888
# No password required
```

### Interactive Shell

```bash
# Linux/macOS
./scripts/docker_run.sh shell

# Windows
.\scripts\docker_run.bat shell
```

### Running Tests

```bash
# Run all tests
docker-compose run --rm tsukuyomi-tts python -m pytest tests/ -v

# Run specific test
docker-compose run --rm tsukuyomi-tts python -m pytest tests/test_models.py -v
```

## Docker Compose Profiles

Different profiles are available for various use cases:

```bash
# Default - Just the inference server
docker-compose up -d

# Development - Includes Jupyter
docker-compose --profile dev up -d

# Training - Optimized for training
docker-compose --profile training up -d

# Production - Includes Redis cache
docker-compose --profile production up -d

# Monitoring - Includes TensorBoard
docker-compose --profile monitoring up -d
```

## Volume Mounts

The Docker setup uses several volume mounts:

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./models` | `/app/models` | Model checkpoints |
| `./data` | `/app/data` | Training data |
| `./outputs` | `/app/outputs` | Generated audio |
| `./logs` | `/app/logs` | Training logs |
| `./configs` | `/app/configs` | Configuration files |

## Environment Variables

You can customize the Docker environment using `.env` file:

```bash
# .env
CUDA_VISIBLE_DEVICES=0,1
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
OMP_NUM_THREADS=8
```

## Troubleshooting

### Common Issues

#### 1. **Out of Memory (OOM)**
```bash
# Reduce batch size in config
# Enable gradient checkpointing
# Use gradient accumulation
```

#### 2. **CUDA not available in Docker**
```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

# If fails, reinstall nvidia-container-toolkit
```

#### 3. **Windows: "Docker daemon not running"**
- Ensure Docker Desktop is running
- Check if Hyper-V is enabled
- Try restarting Docker Desktop

#### 4. **Permission denied errors**
```bash
# Linux: Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

#### 5. **Slow performance on Windows**
- Use WSL2 backend instead of Hyper-V
- Ensure files are in WSL2 filesystem, not Windows filesystem
- Allocate more resources in Docker Desktop settings

### Performance Optimization

#### GPU Memory Management
```yaml
# In docker-compose.yml
environment:
  - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
  - CUDA_LAUNCH_BLOCKING=0
```

#### CPU Optimization
```yaml
# Limit CPU usage
deploy:
  resources:
    limits:
      cpus: '8'
      memory: 32G
```

### Debugging

```bash
# Check container logs
docker-compose logs -f tsukuyomi-tts

# Inspect running container
docker exec -it tsukuyomi-tts bash

# Check resource usage
docker stats tsukuyomi-tts

# Clean up everything
docker-compose down -v
docker system prune -af
```

## Security Considerations

1. **Don't run as root in production**
   ```dockerfile
   # In Dockerfile
   USER app
   ```

2. **Use secrets for sensitive data**
   ```yaml
   # docker-compose.yml
   secrets:
     - api_key
   ```

3. **Limit network exposure**
   ```yaml
   # Only expose necessary ports
   ports:
     - "127.0.0.1:8000:8000"
   ```

## Next Steps

- Check the [Training Guide](../train_ja.md) for detailed training instructions
- See [API Documentation](http://localhost:8000/docs) for inference API
- Join our [Discord](https://discord.gg/tsukuyomi) for support