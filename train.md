# Tsukuyomi (月読) Training Guide

This comprehensive guide covers the complete training pipeline for the Tsukuyomi Ultimate TTS system, from data preprocessing through model evaluation.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Data Preparation](#data-preparation)
4. [Preprocessing Pipeline](#preprocessing-pipeline)
5. [Training Stages](#training-stages)
6. [Evaluation Methods](#evaluation-methods)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Tips](#advanced-tips)

## Overview

The Tsukuyomi training process is designed to achieve MOS 4.7+ (human-level) quality through a staged approach:

- **Stage 1**: Foundation (100 hours, 10 speakers)
- **Stage 2**: Scale-up (1,000 hours, 100 speakers)
- **Stage 3**: Full-scale (10,000 hours, 500+ speakers)

### Training Timeline

| Stage | Data Size | GPUs | Training Time | Expected Quality |
|-------|-----------|------|---------------|------------------|
| 1 | 100 hours | 2x H100 | 3-5 days | MOS 4.0-4.2 |
| 2 | 1,000 hours | 4x H100 | 2 weeks | MOS 4.3-4.5 |
| 3 | 10,000 hours | 8x H100 | 6-8 weeks | MOS 4.7+ |

## Prerequisites

### Hardware Requirements

- **Minimum**: 2x NVIDIA A100/H100 GPUs (80GB)
- **Recommended**: 8x NVIDIA H100 GPUs
- **RAM**: 512GB+ system memory
- **Storage**: 50TB+ NVMe SSD (for full dataset)

### Software Setup

```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup environment
git clone https://github.com/ayutaz/tsukuyomi.git
cd tsukuyomi
uv venv
source .venv/bin/activate
uv pip install -e .
uv pip install -r requirements.txt
```

### Verify Installation

```bash
# Check GPU availability
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# Run component tests
pytest tests/test_ultimate_g2p.py -v
pytest tests/test_ultimate_acoustic_model.py -v
```

## Data Preparation

### Audio Requirements

- **Format**: WAV files, 48kHz, 16-bit PCM
- **Duration**: 3-15 seconds per utterance
- **Quality**: Studio recording quality preferred
- **Silence**: < 0.5s leading/trailing silence

### Directory Structure

```
data/
├── raw/
│   ├── speaker_001/
│   │   ├── audio/
│   │   │   ├── 001.wav
│   │   │   └── ...
│   │   └── transcripts.txt
│   └── speaker_002/
│       └── ...
├── processed/
│   ├── train/
│   ├── val/
│   └── test/
└── metadata/
    ├── speaker_info.json
    └── data_stats.json
```

### Transcript Format

```text
# transcripts.txt
001.wav|こんにちは、月読です。
002.wav|今日はいい天気ですね。
003.wav|音声合成の実験を行います。
```

### Speaker Metadata

```json
{
  "speaker_001": {
    "name": "Character A",
    "gender": "female",
    "age_range": "20-30",
    "dialect": "tokyo",
    "voice_characteristics": {
      "pitch": "medium-high",
      "speaking_rate": "normal",
      "emotion_range": "expressive"
    }
  }
}
```

## Preprocessing Pipeline

### Step 1: Audio Validation and Normalization

```bash
python scripts/preprocess/validate_audio.py \
    --input_dir data/raw \
    --output_dir data/validated \
    --sample_rate 48000 \
    --check_quality \
    --normalize_loudness -23
```

This script:
- Validates audio format and quality
- Normalizes loudness to -23 LUFS
- Removes corrupted files
- Generates validation report

### Step 2: Forced Alignment

```bash
python scripts/preprocess/forced_alignment.py \
    --audio_dir data/validated \
    --transcript_dir data/raw \
    --output_dir data/aligned \
    --model_name "japanese-wav2vec2" \
    --device cuda \
    --batch_size 32
```

Outputs:
- Phone-level alignments
- Word boundaries
- Pause locations

### Step 3: Feature Extraction

```bash
python scripts/preprocess/extract_features.py \
    --input_dir data/aligned \
    --output_dir data/features \
    --features mel,f0,energy,duration \
    --mel_bins 128 \
    --hop_length 480 \
    --win_length 1920 \
    --f0_method dio \
    --num_workers 16
```

Extracts:
- 128-bin mel-spectrograms
- F0 contours (DIO algorithm)
- Energy envelope
- Phone durations

### Step 4: G2P Conversion

```bash
python scripts/preprocess/run_g2p.py \
    --transcript_dir data/aligned \
    --output_dir data/phonemes \
    --g2p_model ultimate \
    --add_accent \
    --add_word_boundary \
    --quality_check
```

Generates:
- Phoneme sequences
- Accent patterns
- Word boundaries
- Confidence scores

### Step 5: Data Splitting

```bash
python scripts/preprocess/split_data.py \
    --input_dir data/features \
    --output_dir data/processed \
    --train_ratio 0.95 \
    --val_ratio 0.03 \
    --test_ratio 0.02 \
    --speaker_balanced \
    --min_utterances_per_speaker 100
```

### Step 6: Statistics and Verification

```bash
python scripts/preprocess/compute_stats.py \
    --data_dir data/processed \
    --output_file data/metadata/data_stats.json \
    --compute_speaker_embeddings \
    --plot_distributions
```

## Training Stages

### Stage 1: Foundation Model (100 hours)

#### Configuration

```yaml
# configs/stage1_foundation.yaml
model:
  type: ultimate_tts
  g2p:
    pretrained: false
    accuracy_target: 0.90
  acoustic:
    n_speakers: 10
    n_flows: 8
    hidden_channels: 384
  vocoder:
    type: bigvgan_v2_small
    
data:
  train_path: data/processed/stage1/train
  val_path: data/processed/stage1/val
  batch_size: 32
  num_workers: 8
  
training:
  max_steps: 100000
  learning_rate: 2e-4
  warmup_steps: 5000
  gradient_accumulation: 4
  mixed_precision: bf16
  
logging:
  log_interval: 100
  eval_interval: 1000
  checkpoint_interval: 5000
```

#### Launch Training

```bash
# Single GPU
python train.py --config configs/stage1_foundation.yaml

# Multi-GPU (2x H100)
torchrun --nproc_per_node=2 train.py \
    --config configs/stage1_foundation.yaml \
    --distributed
```

#### Monitor Progress

```bash
# TensorBoard
tensorboard --logdir logs/stage1

# Custom monitoring
python scripts/monitor_training.py \
    --experiment stage1 \
    --metrics loss,accuracy,mel_loss,f0_rmse
```

### Stage 2: Scale-up (1,000 hours)

#### Prepare Stage 2 Data

```bash
python scripts/prepare_stage2.py \
    --stage1_checkpoint checkpoints/stage1/best.pt \
    --new_speakers 90 \
    --augmentation_config configs/augmentation.yaml
```

#### Configuration Updates

```yaml
# configs/stage2_scaleup.yaml
model:
  checkpoint: checkpoints/stage1/best.pt
  acoustic:
    n_speakers: 100
    n_flows: 10
    hidden_channels: 512
    speaker_encoder:
      use_reference: true
      
training:
  max_steps: 500000
  learning_rate: 1e-4
  batch_size: 64
  gradient_checkpointing: true
  
# Enable FSDP for 4 GPUs
distributed:
  backend: nccl
  strategy: fsdp
  sharding_strategy: full_shard
```

#### Launch Distributed Training

```bash
# 4x H100 GPUs
torchrun --nproc_per_node=4 \
    --master_port=29500 \
    train.py \
    --config configs/stage2_scaleup.yaml \
    --resume_from checkpoints/stage1/best.pt
```

### Stage 3: Full-scale (10,000 hours)

#### Advanced Configuration

```yaml
# configs/stage3_fullscale.yaml
model:
  checkpoint: checkpoints/stage2/best.pt
  g2p:
    accuracy_target: 0.97
    use_neural_correction: true
  acoustic:
    n_speakers: 1000
    n_flows: 12
    hidden_channels: 512
    use_residual_adapters: true
  vocoder:
    type: bigvgan_v2_full
    
data:
  train_path: data/processed/stage3/train
  streaming: true  # For large dataset
  cache_size: 10000
  prefetch_factor: 4
  
training:
  max_steps: 2000000
  learning_rate: 5e-5
  batch_size: 256  # Total across all GPUs
  gradient_accumulation: 8
  
optimization:
  optimizer: adamw
  weight_decay: 0.01
  beta1: 0.9
  beta2: 0.95
  eps: 1e-8
  
distributed:
  strategy: fsdp
  sharding_strategy: hybrid_shard
  cpu_offload: true
  activation_checkpointing: true
```

#### Launch Full Training

```bash
# 8x H100 GPUs with optimizations
OMP_NUM_THREADS=8 torchrun \
    --nproc_per_node=8 \
    --master_addr=$MASTER_ADDR \
    --master_port=29500 \
    --nnodes=1 \
    train.py \
    --config configs/stage3_fullscale.yaml \
    --resume_from checkpoints/stage2/best.pt \
    --compile  # PyTorch 2.0 compile
```

## Evaluation Methods

### Objective Metrics

#### 1. Mel Cepstral Distortion (MCD)

```bash
python evaluate/compute_mcd.py \
    --synthesized_dir outputs/test \
    --reference_dir data/test/audio \
    --output_file results/mcd_scores.json
```

Target: MCD < 4.5 dB

#### 2. F0 Frame Error (FFE)

```bash
python evaluate/compute_f0_metrics.py \
    --synthesized_dir outputs/test \
    --reference_dir data/test/audio \
    --metrics rmse,corr,vuv \
    --output_file results/f0_metrics.json
```

Targets:
- F0 RMSE < 25 Hz
- F0 Correlation > 0.9
- V/UV Error < 5%

#### 3. Character Error Rate (CER)

```bash
python evaluate/compute_cer.py \
    --audio_dir outputs/test \
    --transcript_dir data/test/transcripts \
    --asr_model "japanese-whisper-large" \
    --output_file results/cer_scores.json
```

Target: CER < 2%

### Subjective Evaluation

#### 1. Mean Opinion Score (MOS)

```bash
# Generate evaluation set
python evaluate/prepare_mos_test.py \
    --num_samples 200 \
    --speakers 20 \
    --include_ground_truth \
    --output_dir evaluation/mos

# Launch web interface
python evaluate/mos_server.py \
    --test_dir evaluation/mos \
    --port 8080
```

#### 2. Speaker Similarity

```bash
python evaluate/speaker_similarity.py \
    --synthesized_dir outputs/test \
    --reference_dir data/test/audio \
    --model "japanese-speaker-encoder" \
    --output_file results/speaker_sim.json
```

Target: Cosine similarity > 0.85

### Automated Testing

```bash
# Run all evaluations
python evaluate/run_all_tests.py \
    --checkpoint checkpoints/stage3/best.pt \
    --test_data data/test \
    --output_dir results/full_evaluation \
    --metrics all
```

## Troubleshooting

### Common Issues

#### 1. OOM (Out of Memory)

```bash
# Reduce batch size
python train.py --config config.yaml \
    --override training.batch_size=16 \
    --override training.gradient_accumulation=8

# Enable CPU offloading
python train.py --config config.yaml \
    --override distributed.cpu_offload=true
```

#### 2. Slow Training

```bash
# Profile training
python train.py --config config.yaml \
    --profile \
    --profile_steps 100

# Optimize data loading
python train.py --config config.yaml \
    --override data.num_workers=16 \
    --override data.pin_memory=true \
    --override data.prefetch_factor=4
```

#### 3. Gradient Explosion

```yaml
# Add to config
training:
  gradient_clip_norm: 1.0
  gradient_clip_value: 5.0
  detect_anomaly: true
```

### Debugging Tools

```bash
# Visualize attention maps
python debug/visualize_attention.py \
    --checkpoint checkpoints/model.pt \
    --text "テストです" \
    --output_dir debug/attention

# Check gradient flow
python debug/check_gradients.py \
    --checkpoint checkpoints/model.pt \
    --num_steps 10

# Analyze loss curves
python debug/analyze_losses.py \
    --log_file logs/train.log \
    --smooth_window 100
```

## Advanced Tips

### 1. Multi-Resolution Training

```python
# Custom dataset with multiple resolutions
class MultiResolutionDataset(Dataset):
    def __init__(self, data_dir, resolutions=[8, 16, 24]):
        self.resolutions = resolutions
        # Load data at multiple hop sizes
        
    def __getitem__(self, idx):
        # Return random resolution
        resolution = random.choice(self.resolutions)
        return self.load_item(idx, resolution)
```

### 2. Progressive Training

```python
# Gradually increase model capacity
def progressive_training(config):
    # Start with small model
    model = create_model(hidden_size=256, n_layers=6)
    train(model, epochs=10)
    
    # Expand model
    model = expand_model(model, hidden_size=512, n_layers=12)
    train(model, epochs=20)
```

### 3. Data Augmentation

```yaml
# augmentation.yaml
audio:
  - type: pitch_shift
    range: [-2, 2]  # semitones
    prob: 0.3
  - type: time_stretch
    range: [0.9, 1.1]
    prob: 0.3
  - type: add_noise
    snr_range: [20, 40]
    prob: 0.2
    
text:
  - type: synonym_replacement
    prob: 0.1
  - type: accent_variation
    dialects: ["osaka", "kyoto"]
    prob: 0.2
```

### 4. Efficient Inference

```python
# Optimize for deployment
python scripts/optimize_model.py \
    --checkpoint checkpoints/best.pt \
    --output_dir deployed \
    --quantize int8 \
    --optimize_onnx \
    --batch_size 1
```

### 5. Continuous Learning

```bash
# Fine-tune on new speakers
python finetune.py \
    --base_checkpoint checkpoints/stage3/best.pt \
    --new_data data/new_speakers \
    --freeze_encoder \
    --adapter_rank 64 \
    --learning_rate 1e-5 \
    --max_steps 10000
```

## Training Best Practices

### 1. Data Quality Control

- **Audio Check**: Remove clips with SNR < 20dB
- **Transcript Verification**: Use ASR to verify transcripts
- **Speaker Consistency**: Ensure consistent recording conditions
- **Balanced Dataset**: Maintain phoneme and prosody coverage

### 2. Hyperparameter Tuning

```bash
# Grid search
python scripts/hyperparameter_search.py \
    --config configs/search_space.yaml \
    --metric mos \
    --trials 50 \
    --gpus_per_trial 2
```

### 3. Experiment Tracking

```python
# Use Weights & Biases
import wandb

wandb.init(project="tsukuyomi", config=config)
wandb.log({
    "loss": loss,
    "mel_loss": mel_loss,
    "f0_rmse": f0_rmse,
    "mos_estimate": mos
})
```

### 4. Model Ensemble

```python
# Ensemble multiple checkpoints
models = [
    load_checkpoint(f"checkpoint_{i}.pt")
    for i in range(5)
]

def ensemble_inference(text):
    outputs = [model(text) for model in models]
    return torch.mean(torch.stack(outputs), dim=0)
```

## Conclusion

Training Tsukuyomi to achieve MOS 4.7+ requires:

1. **High-quality data**: 10,000 hours of clean, diverse audio
2. **Staged approach**: Progressive scaling from 100 to 10,000 hours
3. **Powerful hardware**: 8x H100 GPUs for full training
4. **Careful monitoring**: Track both objective and subjective metrics
5. **Patience**: 6-8 weeks for complete training

For questions or issues:
- GitHub Issues: https://github.com/ayutaz/tsukuyomi/issues
- Documentation: https://github.com/ayutaz/tsukuyomi/wiki

Good luck with your training! 🚀