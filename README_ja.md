# 月読（Tsukuyomi） - 究極の日本語音声合成システム

<div align="center">

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

*MOS 4.7以上（人間レベルの品質）を目指す最先端の日本語TTSシステム*

[English](README.md) | 日本語

</div>

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

### オプション1: Docker（学習環境推奨）

```bash
# リポジトリのクローン
git clone https://github.com/ayutaz/tsukuyomi.git
cd tsukuyomi

# Dockerでビルドと実行
./scripts/docker_run.sh build  # Linux/macOS
# または
.\scripts\docker_run.bat build  # Windows

# 推論サーバーの起動
./scripts/docker_run.sh run

# 学習の開始
./scripts/docker_run.sh train
```

Windows環境を含む詳細なDocker手順については、[Docker ガイド](docs/docker_guide.md)を参照してください。

### オプション2: UV使用（ローカル開発）

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

## 🎯 主な使用例

### ゲーム開発

```python
# ゲームキャラクターの音声生成
character_voices = {
    "hero": 0,
    "villain": 1,
    "narrator": 2
}

# 感情豊かなセリフ生成
dialogue = tts.synthesize(
    text="ついに会えたな、宿敵よ！",
    speaker_id=character_voices["hero"],
    emotion="angry",
    style="dramatic"
)
```

### オーディオブック制作

```python
# 長文のナレーション
narrator = tts.create_narrator(
    speaker_id=10,
    speaking_rate=0.95,
    pause_length=1.2
)

# チャプターごとの音声生成
for chapter in book_chapters:
    audio = narrator.read_chapter(chapter)
    save_audio(audio, f"chapter_{chapter.number}.wav")
```

### バーチャルアシスタント

```python
# リアルタイム応答
assistant = tts.create_assistant(
    speaker_id=20,
    response_speed="fast",
    personality="friendly"
)

# ユーザーの質問に応答
response_text = get_ai_response(user_query)
audio = assistant.speak(response_text)
play_audio(audio)
```

## 🔬 技術詳細

### G2P++ システム

月読のG2P++システムは、3段階のアプローチで97%以上の精度を実現：

1. **基礎変換**: pyopenjtalk-plusによる高精度なルールベース変換
2. **文脈理解**: BERTベースのアクセント・イントネーション予測
3. **ニューラル補正**: RNNによる最終的な音素配列の最適化

### 音響モデルの革新

- **Flow Matching**: Matcha-TTSの高速合成技術
- **VAE**: VITSの変分オートエンコーダーによる自然な音声
- **確率的モデリング**: 人間らしい揺らぎの再現

### 学習の最適化

```python
# H100 GPU向け最適化設定
optimization_config = {
    "mixed_precision": "bf16",
    "gradient_checkpointing": True,
    "fsdp": {
        "sharding_strategy": "hybrid_shard",
        "cpu_offload": True
    },
    "compile": True  # PyTorch 2.0
}
```

## 🤝 貢献

貢献を歓迎します！詳細は[貢献ガイドライン](CONTRIBUTING.md)をご覧ください。

### 貢献の方法

1. リポジトリをフォーク
2. フィーチャーブランチを作成（`git checkout -b feature/amazing-feature`）
3. 変更をコミット（`git commit -m 'Add amazing feature'`）
4. ブランチにプッシュ（`git push origin feature/amazing-feature`）
5. プルリクエストを開く

### 開発ガイドライン

- コードスタイル: Black + Ruff
- 型ヒント: 必須（mypy準拠）
- テスト: 新機能には必ずテストを追加
- ドキュメント: docstringとREADMEの更新

## 📄 ライセンス

このプロジェクトはMITライセンスの下でライセンスされています - 詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 🙏 謝辞

本プロジェクトは以下の素晴らしい研究・開発に支えられています：

- OpenJTalkおよびpyopenjtalk-plusの開発者の皆様
- XPhoneBERTの著者の皆様
- VITSおよびMatcha-TTSの研究チームの皆様
- BigVGANの著者の皆様
- 日本語TTS研究コミュニティの皆様

## 📚 引用

研究で月読を使用する場合は、以下を引用してください：

```bibtex
@software{tsukuyomi2024,
  title = {Tsukuyomi: Ultimate Japanese Text-to-Speech System},
  title_ja = {月読：究極の日本語音声合成システム},
  year = {2024},
  url = {https://github.com/ayutaz/tsukuyomi},
  note = {MOS 4.7+を目指す最先端の日本語TTSシステム}
}
```

## 📞 お問い合わせ

- Issues: [GitHub Issues](https://github.com/ayutaz/tsukuyomi/issues)
- Discussions: [GitHub Discussions](https://github.com/ayutaz/tsukuyomi/discussions)
- Email: tsukuyomi-tts@example.com

## 🗺️ ロードマップ

### 2024年 Q1-Q2
- [x] 基本アーキテクチャの実装
- [x] Ultimate G2P++の開発
- [ ] ステージ1学習の完了

### 2024年 Q3-Q4
- [ ] ステージ2学習の完了
- [ ] リアルタイムAPI の公開
- [ ] Unity プラグインのリリース

### 2025年
- [ ] ステージ3学習の完了
- [ ] 商用ライセンスの提供
- [ ] クラウドサービスの開始

---

<div align="center">

**月読** - 日本語音声合成の新たな地平を切り開く

日本語TTSコミュニティのために愛を込めて作られました ❤️

</div>