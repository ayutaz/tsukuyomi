# Tsukuyomi (月読) - 究極の日本語音声合成システム

<div align="center">

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[English](README.md) | [日本語](#日本語) | [クイックスタート](docs/jvs_training_guide.ja.md) | [学習ガイド](docs/train.ja.md) | [APIリファレンス](docs/api/api-reference.md)

</div>

---

## 🌟 概要

Tsukuyomiは、日本語音声合成において世界最高品質を目指して設計された究極のText-to-Speech (TTS)システムです。最先端の深層学習アーキテクチャを採用し、H100 GPUでの大規模学習に最適化されています。

### 主な特徴

- 🎯 **究極の品質**: MOS 4.7以上（人間の音声と区別不可能）を目標
- 🗣️ **500人以上の話者**: ゲームキャラクター音声の完璧な再現をサポート
- 🎭 **感情・スタイル制御**: 7つの感情 × 10の話し方スタイル
- ⚡ **高性能**: RTF < 0.05（リアルタイムの20倍速）
- 🌏 **多言語対応**: 日本語優先で多言語展開可能
- 🎮 **Unity統合**: ゲームエンジンへのデプロイのためのONNXエクスポート

## 🏗️ アーキテクチャ

```
┌─────────────────────────────────────────────────────────┐
│                  Tsukuyomi Ultimate TTS                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  テキスト → G2P++ → XPhoneBERT-JP → F0-BERT → 音響   │
│                                                モデル   │
│                                                  ↓      │
│                                            BigVGAN-v2   │
│                                                  ↓      │
│                                               音声      │
└─────────────────────────────────────────────────────────┘
```

### コアコンポーネント

1. **Ultimate G2P++ (97%以上の精度)**
   - ルールベース（pyopenjtalk-plus）+ ニューラル補正
   - コンテキスト認識BERTベースのアクセント予測
   - 日本語特有の音素処理

2. **XPhoneBERT-Japanese**
   - 日本語音素システムの最適化
   - アクセントと方言モデリング（47都道府県）
   - LoRAファインチューニング（rank=64）

3. **F0-BERT**
   - 高精度ピッチ輪郭予測
   - 感情とスタイルの条件付け
   - フレームレベルのF0生成

4. **Ultimate音響モデル**
   - Matcha-TTS Flow Matching + VITS VAE
   - 500人以上の話者サポートと音声クローニング
   - 確率的持続時間モデリング

5. **BigVGAN-v2ボコーダー**
   - 48kHz高忠実度合成
   - Snake-Betaアンチエイリアス活性化
   - マルチスケール/解像度ディスクリミネーター

## 🚀 インストール

### 前提条件

- Python 3.11以上
- CUDA 12.1以上（Flash Attention 2とBF16サポートのためのGPUアクセラレーション）
- 8x NVIDIA H100 GPU（フル学習用）
- 10,000時間の高品質音声データ

### オプション1: UV（推奨）

```bash
# リポジトリのクローン
git clone https://github.com/ayutaz/tsukuyomi.git
cd tsukuyomi

# UVのインストール（未インストールの場合）
pip install uv

# 仮想環境の作成とパッケージのインストール
uv venv
source .venv/bin/activate  # Linux/macOS
# または
.venv\Scripts\activate  # Windows

# 依存関係のインストール
uv pip install -e .
```

### オプション2: Docker（学習用推奨）

```bash
# Dockerイメージのビルドと実行
./scripts/docker_run.sh build  # Linux/macOS
# または
.\scripts\docker_run.bat build  # Windows

# 推論サーバーの起動
./scripts/docker_run.sh run

# 学習の開始
./scripts/docker_run.sh train
```

## 🎓 学習

### JVSデータセットでの小規模学習（推奨開始点）

```bash
# JVSデータセットの前処理
python scripts/preprocess_jvs.py \
    --input_dir data/jvs/jvs_ver1 \
    --output_dir data/processed/jvs \
    --num_speakers 10

# 学習の実行
python scripts/train.py --config configs/jvs_small.yaml
```

詳細は[JVS学習ガイド](docs/jvs_training_guide.ja.md)を参照してください。

### 大規模学習

```bash
# マルチGPU学習（8x H100）
./scripts/train_multi_gpu.sh configs/train_full.yaml
```

詳細は[学習ガイド](docs/train.ja.md)を参照してください。

## 🔧 推論

### Pythonでの使用

```python
from tsukuyomi import TsukuyomiTTS

# モデルの初期化
tts = TsukuyomiTTS.from_pretrained("tsukuyomi-base")

# 音声生成
audio = tts.synthesize(
    "こんにちは、私は月読です。",
    speaker_id=0,
    emotion="happy",
    style="energetic"
)

# 音声の保存
tts.save_audio(audio, "output.wav")
```

### APIサーバー

```bash
# APIサーバーの起動
python -m tsukuyomi.server --model tsukuyomi-base --port 8000

# 音声生成リクエスト
curl -X POST http://localhost:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "こんにちは、私は月読です。",
    "speaker_id": 0,
    "emotion": "happy"
  }' \
  --output output.wav
```

詳細は[APIリファレンス](docs/api/api-reference.md)を参照してください。

## 📊 パフォーマンス

| メトリクス | 目標 | 現在 |
|----------|------|------|
| MOS (Mean Opinion Score) | 4.7+ | 開発中 |
| 話者類似度 | 95%+ | 開発中 |
| RTF (Real-Time Factor) | < 0.05 | 開発中 |
| レイテンシ（最初の音声まで） | < 100ms | 開発中 |

## 🎮 Unity統合

```csharp
using TsukuyomiTTS;

public class TTSManager : MonoBehaviour
{
    private TsukuyomiEngine tts;
    
    void Start()
    {
        // ONNXモデルのロード
        tts = new TsukuyomiEngine("path/to/model.onnx");
    }
    
    public void Speak(string text, int speakerId = 0)
    {
        // 音声生成と再生
        var audioClip = tts.Synthesize(text, speakerId);
        AudioSource.PlayClipAtPoint(audioClip, transform.position);
    }
}
```

## 🛠️ 高度な機能

### 感情制御

```python
# 10種類の基本感情
emotions = ["neutral", "happy", "sad", "angry", "fearful", 
           "surprised", "disgusted", "excited", "calm", "confident"]

# 感情強度の調整
audio = tts.synthesize(
    text="感情を込めて話します",
    emotion="happy",
    emotion_intensity=1.5  # 0.0-2.0
)
```

### スタイル転送

```python
# 参照音声からスタイルを抽出
reference_audio = "path/to/reference.wav"
audio = tts.synthesize(
    text="このスタイルで話します",
    style_reference=reference_audio
)
```

### 音声モーフィング

```python
# 複数話者のブレンド
audio = tts.morph_voices(
    text="モーフィングされた音声です",
    speaker_weights={0: 0.7, 5: 0.3}  # 話者0と話者5を7:3でブレンド
)
```

## 📁 プロジェクト構造

```
tsukuyomi/
├── src/
│   ├── models/          # コアモデル実装
│   ├── data/            # データ処理
│   ├── training/        # 学習関連
│   ├── inference/       # 推論最適化
│   ├── serving/         # モデルサービング
│   └── utils/           # ユーティリティ
├── scripts/             # 実行スクリプト
├── configs/             # 設定ファイル
├── tests/               # テストコード
├── docker/              # Docker関連
└── docs/                # ドキュメント
```

## 🤝 貢献

プルリクエストを歓迎します！大きな変更の場合は、まずissueを開いて変更内容について議論してください。

1. プロジェクトをフォーク
2. フィーチャーブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを開く

## 📝 ライセンス

このプロジェクトはMITライセンスの下でライセンスされています - 詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 🙏 謝辞

- [XPhoneBERT](https://github.com/VinAIResearch/XPhoneBERT) - 多言語音素表現
- [VITS](https://github.com/jaywalnut310/vits) - エンドツーエンドTTSモデル
- [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) - 高速音声合成
- [BigVGAN](https://github.com/NVIDIA/BigVGAN) - 高品質ボコーダー
- [pyopenjtalk-plus](https://github.com/open-jtalk/pyopenjtalk-plus) - 日本語テキスト処理

## 📧 連絡先

質問や提案がある場合は、[Issues](https://github.com/ayutaz/tsukuyomi/issues)を開くか、[Discussions](https://github.com/ayutaz/tsukuyomi/discussions)で議論してください。

---

<div align="center">
  <sub>日本から世界へ、最高品質の音声合成を。</sub>
</div>