# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tsukuyomi is a state-of-the-art Japanese Text-to-Speech (TTS) system targeting MOS 4.7+ quality (indistinguishable from human speech). It integrates cutting-edge models like XPhoneBERT, F0-BERT, VITS/Matcha-TTS, and BigVGAN-v2 to generate natural and expressive speech.

## Development Commands

### Environment Setup
```bash
# Create virtual environment and install package (uses UV package manager)
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Fix Python 3.11 dependency issues with librosa
uv cache clean
uv pip install "librosa>=0.10.1" "numba>=0.57.0"

# Install the project
uv pip install -e .

# Development installation with all extras
uv pip install -e .[dev,test]

# H100-optimized installation
uv pip install -e .[h100]
```

### Common Development Tasks
```bash
# Run tests
pytest tests/ -v                    # All tests
pytest -m "not slow and not gpu"    # Fast tests only
pytest -m gpu                       # GPU tests only
pytest -v --cov=src                 # With coverage

# Run single test file or function
pytest tests/test_models.py -v
pytest tests/test_models.py::test_xphonebert -v

# Code quality
make lint      # Run linting (ruff, black, isort, mypy)
make format    # Auto-format code

# Training
python scripts/train.py --config configs/train_config.yaml
./scripts/train_multi_gpu.sh configs/train_config.yaml  # Multi-GPU

# Inference and benchmarking
python scripts/benchmark_inference.py --model checkpoints/best_model.pt
python scripts/test_synthesis.py --text "こんにちは"

# Demo advanced features
python scripts/demo_advanced_features.py --model-path checkpoints/best_model.pt --demo-type all
```

## High-Level Architecture

### Core Model Pipeline
```
Text → G2P++ → XPhoneBERT-JP → F0-BERT → Acoustic Model (VITS/Matcha-TTS) → BigVGAN-v2 → Audio
```

### Key Components Integration

1. **Frontend Processing (src/frontend/)**
   - Japanese text normalization with pyopenjtalk-plus
   - G2P++ system achieving 97%+ accuracy through rule-based + neural correction
   - Context-aware BERT-based accent prediction

2. **Model Architecture (src/models/)**
   - **XPhoneBERT**: Multilingual phoneme encoder optimized for Japanese with LoRA fine-tuning
   - **F0-BERT**: High-precision pitch modeling with emotion/style conditioning
   - **Acoustic Models**: Both VITS (VAE-based) and Matcha-TTS (flow matching) supported
   - **BigVGAN-v2**: 48kHz vocoder with Snake-Beta activation
   - **Advanced Features**: Emotion control, style transfer, voice morphing, real-time streaming

3. **Training Infrastructure (src/training/)**
   - Distributed training support for H100 GPU clusters
   - Mixed precision training with BF16
   - Gradient accumulation and checkpointing
   - Custom loss functions for multi-task learning

4. **Inference Optimization (src/inference/)**
   - ONNX export for Unity integration
   - Triton Inference Server deployment
   - Batch processing with dynamic padding
   - Real-time streaming with <100ms latency via WebSocket

5. **API Server (src/server/)**
   - FastAPI-based REST API with authentication
   - WebSocket support for streaming
   - OpenAPI/Swagger documentation
   - Rate limiting and caching

### Advanced Features

- **Emotion Control**: 10 basic emotions + VAD (Valence-Arousal-Dominance) model
- **Style Transfer**: Reference audio extraction, style mixing, adaptive transfer
- **Voice Morphing**: SLERP interpolation, multi-speaker blending, continuous morphing
- **Real-time Streaming**: Chunk-based processing, CUDA streams, WebSocket server

### Configuration System

The project uses YAML configs (configs/) for experiments. Key config files:
- `train_config.yaml`: Main training configuration
- `model_config.yaml`: Model architecture settings
- `data_config.yaml`: Dataset and preprocessing

### Testing Strategy

- **Unit tests**: Model components, data processing, utilities
- **Integration tests**: End-to-end synthesis, API endpoints
- **Performance tests**: Inference speed, memory usage
- **Markers**: `slow`, `gpu`, `integration`, `unit`, `benchmark`

## Important Notes

- Preprocessing and pre-training should be done by users
- Designed for 8x H100 GPU training
- Requires 100+ hours of audio data for quality results
- Supports incomplete metadata with null tolerance
- Japanese-first design with multilingual expansion capability