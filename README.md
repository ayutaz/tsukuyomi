# Tsukuyomi - 日本語音声合成システム

Tsukuyomi（月読）は、最先端の深層学習技術を活用した高品質な日本語音声合成システムです。

## 特徴

- 🎯 **高精度な日本語アクセント**: F0-BERTによる自然な韻律生成
- 🌐 **多言語対応基盤**: XPhoneBERTによる音素表現
- 🎭 **豊かな表現力**: 話者適応と感情制御機能
- ⚡ **高速推論**: ONNXによる最適化されたデプロイメント
- 🔧 **モジュラー設計**: 柔軟な拡張と保守が可能

## システム要件

- Python 3.8以上
- CUDA対応GPU（推奨: 8GB以上のVRAM）
- Ubuntu 20.04 / Windows 10 / macOS（開発環境）

## クイックスタート

### 1. 環境構築

```bash
# リポジトリのクローン
git clone https://github.com/yourusername/tsukuyomi.git
cd tsukuyomi

# 仮想環境の作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存関係のインストール
pip install -r requirements.txt
```

### 2. モデルのダウンロード

```bash
python scripts/download_models.py
```

### 3. 音声合成の実行

```python
from src.inference import TsukuyomiTTS

# TTSシステムの初期化
tts = TsukuyomiTTS(device='cuda')

# 音声合成
text = "こんにちは、月読です。"
audio, sample_rate = tts.synthesize(text)

# 保存
import soundfile as sf
sf.write("output.wav", audio, sample_rate)
```

## ドキュメント

詳細なドキュメントは`docs/`ディレクトリを参照してください：

- [アーキテクチャ概要](docs/architecture-overview.md)
- [技術仕様書](docs/technical-specifications.md)
- [実装ガイド](docs/implementation-guide.md)

## 開発ロードマップ

### フェーズ1（現在）
- [x] 基本的なVITSパイプライン
- [x] XPhoneBERTの統合
- [x] BigVGANボコーダー
- [ ] 基本的な音声合成機能

### フェーズ2
- [ ] F0-BERTによる韻律予測
- [ ] 日本語特化の最適化
- [ ] 品質評価システム

### フェーズ3
- [ ] Residual Adaptersによる話者適応
- [ ] HVAEによる感情制御
- [ ] kNNベースの音声モーフィング
- [ ] ONNX変換と最適化

## コントリビューション

プルリクエストや Issue の報告を歓迎します。コントリビューションの際は以下をご確認ください：

1. Issue を作成して議論
2. フィーチャーブランチでの開発
3. テストの追加
4. ドキュメントの更新

## ライセンス

本プロジェクトは研究・教育目的での使用を想定しています。商用利用については別途ご相談ください。

## 謝辞

本プロジェクトは以下の研究・プロジェクトに基づいています：

- XPhoneBERT (VinAI Research)
- VITS (Conditional Variational Autoencoder with Adversarial Learning)
- BigVGAN (NVIDIA)
- Matcha-TTS (Conditional Flow Matching)

## 連絡先

質問や提案がある場合は、Issue を作成するか、[メールアドレス] までご連絡ください。