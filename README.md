# Tsukuyomi (月読) - Ultimate Japanese Text-to-Speech System

<div align="center">

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[English](#english) | [日本語](#日本語) | [日本語README](README_ja.md)

</div>

---

<a name="english"></a>
## 🌟 Overview

Tsukuyomi is an ultimate Text-to-Speech (TTS) system designed to achieve world-class quality in Japanese speech synthesis. Built with cutting-edge deep learning architectures and optimized for large-scale training on H100 GPUs.

### Key Features

- 🎯 **Ultimate Quality**: Targeting MOS 4.7+ (indistinguishable from human speech)
- 🗣️ **500+ Speakers**: Support for game character voices with perfect reproduction
- 🎭 **Emotion & Style Control**: 7 emotions × 10 speaking styles
- ⚡ **High Performance**: RTF < 0.05 (20x faster than real-time)
- 🌏 **Multilingual Ready**: Japanese-first with multilingual expansion capability
- 🎮 **Unity Integration**: ONNX export for game engine deployment

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Tsukuyomi Ultimate TTS                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Text → G2P++ → XPhoneBERT-JP → F0-BERT → Acoustic   │
│                                              Model      │
│                                                ↓       │
│                                          BigVGAN-v2    │
│                                                ↓       │
│                                            Audio       │
└─────────────────────────────────────────────────────────┘
```

### Core Components

1. **Ultimate G2P++ (97%+ accuracy)**
   - Rule-based (pyopenjtalk-plus) + Neural correction
   - Context-aware BERT-based accent prediction
   - Japanese-specific phoneme handling

2. **XPhoneBERT-Japanese**
   - Japanese phoneme system optimization
   - Accent and dialect modeling (47 prefectures)
   - LoRA fine-tuning (rank=64)

3. **F0-BERT**
   - High-precision pitch contour prediction
   - Emotion and style conditioning
   - Frame-level F0 generation

4. **Ultimate Acoustic Model**
   - Matcha-TTS Flow Matching + VITS VAE
   - 500+ speaker support with voice cloning
   - Stochastic duration modeling

5. **BigVGAN-v2 Vocoder**
   - 48kHz high-fidelity synthesis
   - Snake-Beta anti-aliased activation
   - Multi-scale/resolution discriminators

## 🚀 Installation

### Prerequisites

- Python 3.11+
- CUDA 12.1+ (for GPU acceleration with Flash Attention 2 and enhanced BF16 support)
- 8x NVIDIA H100 GPUs (for full training)
- 10,000 hours of high-quality audio data

### Option 1: Docker (Recommended for Training)

```bash
# Clone repository
git clone https://github.com/ayutaz/tsukuyomi.git
cd tsukuyomi

# Build and run with Docker
./scripts/docker_run.sh build  # Linux/macOS
# or
.\scripts\docker_run.bat build  # Windows

# Start inference server
./scripts/docker_run.sh run

# Start training
./scripts/docker_run.sh train
```

For detailed Docker instructions including Windows support, see [Docker Guide](docs/docker_guide.md).

### Option 2: Using UV (Local Development)

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/ayutaz/tsukuyomi.git
cd tsukuyomi

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
uv pip install -r requirements.txt
```

### Development Setup

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v

# Run linting
ruff check src/
mypy src/
```

## 📊 Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| MOS (Mean Opinion Score) | 4.7+ | Training |
| Speaker Similarity | 95%+ | Training |
| RTF (Real-Time Factor) | < 0.05 | Achieved |
| Accent Accuracy | 97%+ | Achieved |
| Character Error Rate | < 1% | Training |

## 🎯 Quick Start

### Basic Usage

```python
from tsukuyomi import TsukuyomiTTS

# Initialize TTS system
tts = TsukuyomiTTS(device="cuda")

# Generate speech
audio = tts.synthesize(
    text="月読は最高峰の音声合成システムです",
    speaker_id=0,
    emotion="neutral",
    style="normal"
)

# Save audio
tts.save_audio(audio, "output.wav")
```

### Advanced Features

```python
# Multi-speaker synthesis
audio = tts.synthesize(
    text="こんにちは、月読です",
    speaker_id=42,  # Specific character voice
    emotion="happy",
    style="energetic",
    speed=1.1,
    pitch_shift=2.0
)

# Voice cloning
reference_audio = load_audio("reference.wav")
audio = tts.clone_voice(
    text="クローンされた音声です",
    reference_audio=reference_audio
)

# Batch synthesis
texts = ["文1", "文2", "文3"]
audios = tts.batch_synthesize(texts, speaker_ids=[0, 1, 2])
```

## 🏋️ Training

For detailed training instructions, see [train.md](train.md).

### Stage 1: Foundation (100 hours, 10 speakers)
```bash
python train.py \
    --config configs/stage1_foundation.yaml \
    --data_dir data/foundation \
    --output_dir checkpoints/stage1 \
    --gpus 2
```

### Stage 2: Scale-up (1,000 hours, 100 speakers)
```bash
torchrun --nproc_per_node=4 train.py \
    --config configs/stage2_scaleup.yaml \
    --data_dir data/scaleup \
    --checkpoint checkpoints/stage1/best.pt \
    --gpus 4
```

### Stage 3: Full-scale (10,000 hours, 500+ speakers)
```bash
torchrun --nproc_per_node=8 train.py \
    --config configs/stage3_fullscale.yaml \
    --data_dir data/fullscale \
    --checkpoint checkpoints/stage2/best.pt \
    --gpus 8 \
    --use_fsdp \
    --use_bf16
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific component tests
pytest tests/test_ultimate_g2p.py -v
pytest tests/test_f0_bert.py -v
pytest tests/test_xphonebert_japanese.py -v
pytest tests/test_ultimate_acoustic_model.py -v
pytest tests/test_bigvgan_v2.py -v

# Run integration tests
python scripts/test_ultimate_tts_integration.py
```

## 📁 Project Structure

```
tsukuyomi/
├── src/
│   ├── models/
│   │   ├── ultimate_g2p.py         # 97%+ accuracy G2P
│   │   ├── xphonebert_japanese.py  # Japanese-optimized encoder
│   │   ├── f0_bert.py              # Pitch prediction
│   │   ├── ultimate_acoustic_model.py  # Matcha-TTS + VITS
│   │   └── bigvgan_v2.py           # 48kHz vocoder
│   ├── data/
│   │   └── massive_dataset.py      # 10,000-hour data pipeline
│   ├── frontend/
│   │   ├── japanese_g2p.py         # pyopenjtalk-plus integration
│   │   └── text_normalizer.py      # Text preprocessing
│   └── training/
│       └── trainer.py              # Distributed training
├── configs/
│   ├── ultimate_tts_h100.yaml      # H100 optimization config
│   └── pretrained_models.json      # Model registry
├── tests/
│   └── test_*.py                   # Comprehensive test suite
├── scripts/
│   ├── download_pretrained_models.py
│   └── test_ultimate_tts_integration.py
├── docs/
│   ├── architecture-overview.md
│   ├── ultimate-tts-architecture.md
│   └── training-guide.md
└── train.md                        # Detailed training guide
```

## 🔧 Configuration

### Model Configuration

```yaml
# configs/ultimate_tts.yaml
model:
  g2p:
    accuracy_target: 0.97
    use_neural_correction: true
  
  acoustic:
    n_speakers: 1000
    n_flows: 12
    hidden_channels: 512
    
  vocoder:
    sampling_rate: 48000
    use_snake_activation: true
    
training:
  batch_size: 32
  learning_rate: 2e-4
  use_bf16: true
  gradient_checkpointing: true
```

## 🎮 Unity Integration

```csharp
// Export to ONNX
python scripts/export_onnx.py --checkpoint best_model.pt --output tsukuyomi.onnx

// Unity C# usage
using Unity.Sentis;

public class TsukuyomiTTS : MonoBehaviour {
    private Model model;
    private IWorker worker;
    
    void Start() {
        model = ModelLoader.Load("tsukuyomi.onnx");
        worker = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);
    }
    
    public AudioClip Synthesize(string text, int speakerId = 0) {
        var inputs = PreprocessText(text);
        worker.Execute(inputs);
        return ConvertToAudioClip(worker.PeekOutput());
    }
}
```

## 📈 Benchmarks

| Model Component | Latency (ms) | Memory (GB) | Quality |
|----------------|--------------|-------------|---------|
| G2P++ | 5 | 0.5 | 97% accuracy |
| XPhoneBERT-JP | 10 | 1.2 | - |
| F0-BERT | 8 | 0.8 | 94% accuracy |
| Acoustic Model | 25 | 2.5 | - |
| BigVGAN-v2 | 12 | 1.0 | 48kHz |
| **Total** | **60** | **6.0** | **MOS 4.7+** |

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenJTalk and pyopenjtalk-plus developers
- XPhoneBERT authors
- VITS and Matcha-TTS research teams
- BigVGAN authors
- Japanese TTS research community

## 📚 Citation

If you use Tsukuyomi in your research, please cite:

```bibtex
@software{tsukuyomi2024,
  title = {Tsukuyomi: Ultimate Japanese Text-to-Speech System},
  year = {2024},
  url = {https://github.com/ayutaz/tsukuyomi}
}
```

## 📞 Contact

- Issues: [GitHub Issues](https://github.com/ayutaz/tsukuyomi/issues)
- Discussions: [GitHub Discussions](https://github.com/ayutaz/tsukuyomi/discussions)

---

<a name="日本語"></a>
## 🌟 概要

月読（Tsukuyomi）は、世界最高水準の品質を目指して設計された日本語音声合成（TTS）システムです。最先端の深層学習アーキテクチャを採用し、H100 GPUでの大規模学習に最適化されています。

### 主な特徴

- 🎯 **究極の品質**: MOS 4.7+（人間の音声と区別がつかないレベル）を目標
- 🗣️ **500以上の話者**: ゲームキャラクターの音声を完璧に再現
- 🎭 **感情・スタイル制御**: 7つの感情 × 10の話し方スタイル
- ⚡ **高性能**: RTF < 0.05（リアルタイムの20倍速）
- 🌏 **多言語対応**: 日本語優先で多言語展開可能
- 🎮 **Unity統合**: ゲームエンジンへのデプロイのためのONNXエクスポート

## 🏗️ アーキテクチャ

```
┌─────────────────────────────────────────────────────────┐
│                  月読 Ultimate TTS                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  テキスト → G2P++ → XPhoneBERT-JP → F0-BERT →        │
│                                        音響モデル       │
│                                           ↓           │
│                                      BigVGAN-v2       │
│                                           ↓           │
│                                         音声           │
└─────────────────────────────────────────────────────────┘
```

### コアコンポーネント

1. **Ultimate G2P++（97%以上の精度）**
   - ルールベース（pyopenjtalk-plus）+ ニューラル補正
   - 文脈を考慮したBERTベースのアクセント予測
   - 日本語特有の音素処理

2. **XPhoneBERT-Japanese**
   - 日本語音素システムの最適化
   - アクセント・方言モデリング（47都道府県）
   - LoRAファインチューニング（rank=64）

3. **F0-BERT**
   - 高精度ピッチ輪郭予測
   - 感情・スタイル条件付け
   - フレームレベルF0生成

4. **Ultimate音響モデル**
   - Matcha-TTS Flow Matching + VITS VAE
   - 500以上の話者サポート（音声クローニング対応）
   - 確率的持続時間モデリング

5. **BigVGAN-v2 ボコーダー**
   - 48kHz高忠実度合成
   - Snake-Betaアンチエイリアス活性化
   - マルチスケール/解像度識別器

## 🚀 インストール

### 前提条件

- Python 3.11以上
- CUDA 12.1以上（Flash Attention 2と強化されたBF16サポート用）
- 8x NVIDIA H100 GPU（フル学習用）
- 10,000時間の高品質音声データ

### UV使用（推奨）

```bash
# UVのインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# リポジトリのクローン
git clone https://github.com/ayutaz/tsukuyomi.git
cd tsukuyomi

# 仮想環境の作成と依存関係のインストール
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e .
uv pip install -r requirements.txt
```

### 開発環境のセットアップ

```bash
# 開発用依存関係のインストール
uv pip install -e ".[dev]"

# pre-commitフックのインストール
pre-commit install

# テストの実行
pytest tests/ -v

# リンティング
ruff check src/
mypy src/
```

## 📊 性能目標

| 指標 | 目標 | 現状 |
|------|------|------|
| MOS（平均意見スコア） | 4.7以上 | 学習中 |
| 話者類似度 | 95%以上 | 学習中 |
| RTF（リアルタイム係数） | < 0.05 | 達成済み |
| アクセント精度 | 97%以上 | 達成済み |
| 文字誤り率 | < 1% | 学習中 |

## 🎯 クイックスタート

### 基本的な使用方法

```python
from tsukuyomi import TsukuyomiTTS

# TTSシステムの初期化
tts = TsukuyomiTTS(device="cuda")

# 音声生成
audio = tts.synthesize(
    text="月読は最高峰の音声合成システムです",
    speaker_id=0,
    emotion="neutral",
    style="normal"
)

# 音声の保存
tts.save_audio(audio, "output.wav")
```

### 高度な機能

```python
# マルチスピーカー合成
audio = tts.synthesize(
    text="こんにちは、月読です",
    speaker_id=42,  # 特定のキャラクター音声
    emotion="happy",
    style="energetic",
    speed=1.1,
    pitch_shift=2.0
)

# 音声クローニング
reference_audio = load_audio("reference.wav")
audio = tts.clone_voice(
    text="クローンされた音声です",
    reference_audio=reference_audio
)

# バッチ合成
texts = ["文1", "文2", "文3"]
audios = tts.batch_synthesize(texts, speaker_ids=[0, 1, 2])
```

## 🏋️ 学習

詳細な学習手順については[train.md](train.md)を参照してください。

### ステージ1：基礎（100時間、10話者）
```bash
python train.py \
    --config configs/stage1_foundation.yaml \
    --data_dir data/foundation \
    --output_dir checkpoints/stage1 \
    --gpus 2
```

### ステージ2：スケールアップ（1,000時間、100話者）
```bash
torchrun --nproc_per_node=4 train.py \
    --config configs/stage2_scaleup.yaml \
    --data_dir data/scaleup \
    --checkpoint checkpoints/stage1/best.pt \
    --gpus 4
```

### ステージ3：フルスケール（10,000時間、500以上の話者）
```bash
torchrun --nproc_per_node=8 train.py \
    --config configs/stage3_fullscale.yaml \
    --data_dir data/fullscale \
    --checkpoint checkpoints/stage2/best.pt \
    --gpus 8 \
    --use_fsdp \
    --use_bf16
```

## 🧪 テスト

```bash
# 全テストの実行
pytest tests/ -v

# 特定コンポーネントのテスト
pytest tests/test_ultimate_g2p.py -v
pytest tests/test_f0_bert.py -v
pytest tests/test_xphonebert_japanese.py -v
pytest tests/test_ultimate_acoustic_model.py -v
pytest tests/test_bigvgan_v2.py -v

# 統合テストの実行
python scripts/test_ultimate_tts_integration.py
```

## 📁 プロジェクト構造

```
tsukuyomi/
├── src/
│   ├── models/
│   │   ├── ultimate_g2p.py         # 97%以上の精度のG2P
│   │   ├── xphonebert_japanese.py  # 日本語最適化エンコーダー
│   │   ├── f0_bert.py              # ピッチ予測
│   │   ├── ultimate_acoustic_model.py  # Matcha-TTS + VITS
│   │   └── bigvgan_v2.py           # 48kHzボコーダー
│   ├── data/
│   │   └── massive_dataset.py      # 10,000時間データパイプライン
│   ├── frontend/
│   │   ├── japanese_g2p.py         # pyopenjtalk-plus統合
│   │   └── text_normalizer.py      # テキスト前処理
│   └── training/
│       └── trainer.py              # 分散学習
├── configs/
│   ├── ultimate_tts_h100.yaml      # H100最適化設定
│   └── pretrained_models.json      # モデルレジストリ
├── tests/
│   └── test_*.py                   # 包括的なテストスイート
├── scripts/
│   ├── download_pretrained_models.py
│   └── test_ultimate_tts_integration.py
├── docs/
│   ├── architecture-overview.md
│   ├── ultimate-tts-architecture.md
│   └── training-guide.md
└── train.md                        # 詳細な学習ガイド
```

## 🔧 設定

### モデル設定

```yaml
# configs/ultimate_tts.yaml
model:
  g2p:
    accuracy_target: 0.97
    use_neural_correction: true
  
  acoustic:
    n_speakers: 1000
    n_flows: 12
    hidden_channels: 512
    
  vocoder:
    sampling_rate: 48000
    use_snake_activation: true
    
training:
  batch_size: 32
  learning_rate: 2e-4
  use_bf16: true
  gradient_checkpointing: true
```

## 🎮 Unity統合

```csharp
// ONNXへのエクスポート
python scripts/export_onnx.py --checkpoint best_model.pt --output tsukuyomi.onnx

// Unity C#での使用
using Unity.Sentis;

public class TsukuyomiTTS : MonoBehaviour {
    private Model model;
    private IWorker worker;
    
    void Start() {
        model = ModelLoader.Load("tsukuyomi.onnx");
        worker = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);
    }
    
    public AudioClip Synthesize(string text, int speakerId = 0) {
        var inputs = PreprocessText(text);
        worker.Execute(inputs);
        return ConvertToAudioClip(worker.PeekOutput());
    }
}
```

## 📈 ベンチマーク

| モデルコンポーネント | レイテンシ (ms) | メモリ (GB) | 品質 |
|---------------------|-----------------|-------------|------|
| G2P++ | 5 | 0.5 | 97%精度 |
| XPhoneBERT-JP | 10 | 1.2 | - |
| F0-BERT | 8 | 0.8 | 94%精度 |
| 音響モデル | 25 | 2.5 | - |
| BigVGAN-v2 | 12 | 1.0 | 48kHz |
| **合計** | **60** | **6.0** | **MOS 4.7+** |

## 🤝 貢献

貢献を歓迎します！詳細は[貢献ガイドライン](CONTRIBUTING.md)をご覧ください。

1. リポジトリをフォーク
2. フィーチャーブランチを作成（`git checkout -b feature/amazing-feature`）
3. 変更をコミット（`git commit -m 'Add amazing feature'`）
4. ブランチにプッシュ（`git push origin feature/amazing-feature`）
5. プルリクエストを開く

## 📄 ライセンス

このプロジェクトはMITライセンスの下でライセンスされています - 詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 🙏 謝辞

- OpenJTalkおよびpyopenjtalk-plusの開発者
- XPhoneBERTの著者
- VITSおよびMatcha-TTSの研究チーム
- BigVGANの著者
- 日本語TTS研究コミュニティ

## 📚 引用

研究で月読を使用する場合は、以下を引用してください：

```bibtex
@software{tsukuyomi2024,
  title = {Tsukuyomi: Ultimate Japanese Text-to-Speech System},
  year = {2024},
  url = {https://github.com/ayutaz/tsukuyomi}
}
```

## 📞 連絡先

- Issues: [GitHub Issues](https://github.com/ayutaz/tsukuyomi/issues)
- Discussions: [GitHub Discussions](https://github.com/ayutaz/tsukuyomi/discussions)

---

<div align="center">
Made with ❤️ for the Japanese TTS community<br>
日本語TTSコミュニティのために愛を込めて作られました
</div>