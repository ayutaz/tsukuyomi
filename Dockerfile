# Tsukuyomi TTS Docker Image
# Supports both Linux and Windows containers with CUDA 12.1+

# Base image with CUDA 12.1 and cuDNN 8
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
ENV FORCE_CUDA=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    sox \
    libsox-dev \
    libsox-fmt-all \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock* ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# Install Python dependencies with uv
RUN uv pip install --system -e . --no-cache

# Install additional dependencies for Windows compatibility
RUN uv pip install --system \
    pywin32-ctypes \
    colorama \
    --no-cache || true

# Install Flash Attention 2 (if available)
RUN uv pip install --system \
    flash-attn>=2.0.0 \
    --no-cache || echo "Flash Attention not available for this platform"

# Create directories for models and data
RUN mkdir -p /app/models /app/data /app/outputs /app/logs

# Expose ports
EXPOSE 8000 7860 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden)
CMD ["python", "-m", "src.server.inference_server"]