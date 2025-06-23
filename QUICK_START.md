# Tsukuyomi TTS クイックスタートガイド

このガイドでは、Tsukuyomi TTSを素早く始める方法を説明します。

## 📋 前提条件

- Python 3.11以上
- CUDA対応GPU（推奨）またはCPU
- 4GB以上のRAM（GPU使用時）、8GB以上（CPU使用時）

## 🚀 インストール

### 方法1: UV（推奨）

```bash
# UVをインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# プロジェクトをクローン
git clone https://github.com/ayutaz/tsukuyomi.git
cd tsukuyomi

# 環境をセットアップ
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e .
```

### 方法2: pip

```bash
# プロジェクトをクローン
git clone https://github.com/ayutaz/tsukuyomi.git
cd tsukuyomi

# 仮想環境を作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# インストール
pip install -e .
```

## 🎯 基本的な使用方法

### 1. コマンドラインから

```bash
# シンプルな音声合成
tsukuyomi "こんにちは、月読です" -o hello.wav

# 話者を指定
tsukuyomi "おはようございます" -s 1 -o morning.wav

# 感情を指定
tsukuyomi "嬉しいです！" -e happy -o happy.wav

# ヘルプを表示
tsukuyomi --help
```

### 2. Pythonスクリプトから

```python
from tsukuyomi import TsukuyomiTTS

# 初期化
tts = TsukuyomiTTS()

# 基本的な合成
audio = tts.synthesize("月読は高品質な日本語音声合成システムです")
tts.save_audio(audio, "output.wav")

# 詳細な制御
audio = tts.synthesize(
    text="感情豊かな音声を生成します",
    speaker_id=0,
    emotion="happy",
    speed=1.1,
    pitch_shift=2.0
)
```

### 3. APIサーバーとして

```bash
# サーバーを起動
tsukuyomi --serve --port 8080

# 別のターミナルから使用
curl -X POST http://localhost:8080/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "APIから音声を生成", "speaker_id": 0}' \
  -o api_output.wav
```

## 🎨 高度な機能

### 感情制御

```python
# 10種類の基本感情
emotions = ["neutral", "happy", "sad", "angry", "fearful", 
           "surprised", "disgusted", "excited", "calm", "confident"]

for emotion in emotions:
    audio = tts.synthesize(
        text=f"{emotion}な声です",
        emotion=emotion,
        emotion_intensity=1.5
    )
    tts.save_audio(audio, f"{emotion}.wav")
```

### 音声モーフィング

```python
# 複数話者をブレンド
audio = tts.morph_voices(
    text="複数の話者をミックスした声",
    speaker_ids=[0, 1, 2],
    weights=[0.5, 0.3, 0.2]
)
```

### ストリーミング合成

```python
import asyncio

async def stream_example():
    async for chunk in tts.stream_synthesis("長い文章をストリーミングで合成します"):
        # チャンクごとに処理
        play_audio_chunk(chunk)

asyncio.run(stream_example())
```

## 📁 サンプルデータ

プロジェクトには以下のサンプルが含まれています：

- `examples/basic_synthesis.py` - 基本的な音声合成
- `examples/emotion_control.py` - 感情制御のデモ
- `examples/voice_morphing.py` - 音声モーフィング
- `examples/streaming.py` - ストリーミング合成

## 🔧 トラブルシューティング

### CUDA関連のエラー

```bash
# CPUモードで実行
tsukuyomi "テスト" -o test.wav --device cpu
```

### メモリ不足

```python
# バッチサイズを減らす
tts = TsukuyomiTTS(device="cuda", max_batch_size=1)
```

### 音質の問題

```python
# 高品質設定を使用
audio = tts.synthesize(
    text="高品質な音声",
    quality="high",  # "fast", "balanced", "high"
    denoise=True
)
```

## 📚 次のステップ

1. [詳細なドキュメント](docs/README.md)を読む
2. [サンプルコード](examples/)を試す
3. [APIリファレンス](docs/api_reference.md)を確認
4. 独自のモデルを[学習](docs/training_guide.md)する

## 💡 Tips

- GPUを使用すると約10倍高速
- 長い文章は自動的に分割されます
- 日本語以外のテキストは自動的に翻訳されます（実験的機能）

## 🆘 ヘルプ

問題が解決しない場合：

1. [FAQ](docs/FAQ.md)を確認
2. [GitHub Issues](https://github.com/ayutaz/tsukuyomi/issues)で検索
3. 新しいIssueを作成

---

Happy speech synthesis! 🎵