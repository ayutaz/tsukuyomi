# Tsukuyomi 実装ガイド

## プロジェクト構造

```
tsukuyomi/
├── docs/                    # ドキュメント
│   ├── architecture-overview.md
│   ├── technical-specifications.md
│   └── implementation-guide.md
├── src/                     # ソースコード
│   ├── frontend/           # テキスト前処理
│   │   ├── __init__.py
│   │   ├── text_normalizer.py
│   │   └── phonemizer.py
│   ├── models/             # モデル実装
│   │   ├── __init__.py
│   │   ├── xphonebert.py
│   │   ├── f0_bert.py
│   │   ├── vits.py
│   │   └── adapters.py
│   ├── vocoder/            # ボコーダー
│   │   ├── __init__.py
│   │   └── bigvgan.py
│   ├── utils/              # ユーティリティ
│   │   ├── __init__.py
│   │   ├── audio.py
│   │   └── visualization.py
│   └── inference.py        # 推論パイプライン
├── configs/                # 設定ファイル
│   ├── model_config.yaml
│   └── training_config.yaml
├── scripts/                # 実行スクリプト
│   ├── setup_environment.sh
│   ├── download_models.py
│   └── test_synthesis.py
├── tests/                  # テストコード
│   ├── test_frontend.py
│   ├── test_models.py
│   └── test_integration.py
├── data/                   # データディレクトリ
│   ├── models/            # 学習済みモデル
│   ├── samples/           # サンプル音声
│   └── cache/             # キャッシュ
├── requirements.txt        # 依存関係
├── setup.py               # パッケージ設定
├── README.md              # プロジェクト説明
└── .gitignore            # Git除外設定
```

## フェーズ1: 基本実装

### 1. 環境構築

```bash
# 仮想環境の作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 基本パッケージのインストール
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers text2phonemesequence bigvgan
pip install mecab-python3 unidic-lite
pip install soundfile librosa matplotlib
```

### 2. XPhoneBERTの実装

```python
# src/models/xphonebert.py
import torch
from transformers import AutoModel, AutoTokenizer
from text2phonemesequence import Text2PhonemeSequence

class XPhoneBERTWrapper:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = AutoModel.from_pretrained("vinai/xphonebert-base")
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/xphonebert-base")
        self.phonemizer = Text2PhonemeSequence(language='jpn', is_cuda=True)
        
        self.model.to(device)
        self.model.eval()
    
    def encode(self, text):
        # テキストを音素列に変換
        phoneme_seq = self.phonemizer.infer_sentence(text)
        
        # トークン化
        inputs = self.tokenizer(phoneme_seq, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 埋め込み取得
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
        
        return embeddings
```

### 3. VITSモデルの統合

```python
# src/models/vits.py
from transformers import VitsForConditionalGeneration, AutoTokenizer

class VITSWrapper:
    def __init__(self, model_name="facebook/mms-tts-jpn", device='cuda'):
        self.device = device
        self.model = VitsForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.model.to(device)
        self.model.eval()
    
    def generate_mel(self, text=None, phoneme_embeddings=None):
        if phoneme_embeddings is not None:
            # XPhoneBERT埋め込みを使用（要実装）
            # ここでは埋め込みをVITSに統合する処理を実装
            pass
        else:
            # 通常のテキスト入力
            inputs = self.tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, output_hidden_states=True)
            # メルスペクトログラムの取得
            mel = outputs.hidden_states
        
        return mel
```

### 4. BigVGANボコーダー

```python
# src/vocoder/bigvgan.py
import bigvgan
import torch

class BigVGANVocoder:
    def __init__(self, model_name="nvidia/bigvgan_22khz_80band", device='cuda'):
        self.device = device
        self.model = bigvgan.BigVGAN.from_pretrained(model_name)
        self.model.remove_weight_norm()
        self.model.eval().to(device)
    
    def generate_waveform(self, mel):
        mel_tensor = torch.tensor(mel).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            audio = self.model(mel_tensor)
        
        return audio.squeeze().cpu().numpy()
```

### 5. 統合パイプライン

```python
# src/inference.py
from src.frontend.text_normalizer import normalize_japanese
from src.models.xphonebert import XPhoneBERTWrapper
from src.models.vits import VITSWrapper
from src.vocoder.bigvgan import BigVGANVocoder

class TsukuyomiTTS:
    def __init__(self, device='cuda'):
        self.device = device
        self.xphonebert = XPhoneBERTWrapper(device)
        self.vits = VITSWrapper(device=device)
        self.vocoder = BigVGANVocoder(device=device)
    
    def synthesize(self, text, use_xphonebert=True):
        # テキスト正規化
        normalized_text = normalize_japanese(text)
        
        if use_xphonebert:
            # XPhoneBERT経由
            embeddings = self.xphonebert.encode(normalized_text)
            mel = self.vits.generate_mel(phoneme_embeddings=embeddings)
        else:
            # 直接VITS
            mel = self.vits.generate_mel(text=normalized_text)
        
        # 波形生成
        waveform = self.vocoder.generate_waveform(mel)
        
        return waveform, 22050  # サンプリングレート
```

## テストスクリプト

```python
# scripts/test_synthesis.py
import soundfile as sf
from src.inference import TsukuyomiTTS

def main():
    # TTSシステムの初期化
    tts = TsukuyomiTTS(device='cuda')
    
    # テストテキスト
    test_texts = [
        "こんにちは、音声合成のテストです。",
        "今日は良い天気ですね。",
        "月読は日本語音声合成システムです。"
    ]
    
    for i, text in enumerate(test_texts):
        print(f"合成中: {text}")
        
        # XPhoneBERT使用
        audio, sr = tts.synthesize(text, use_xphonebert=True)
        sf.write(f"output_xphonebert_{i}.wav", audio, sr)
        
        # 通常のVITS
        audio, sr = tts.synthesize(text, use_xphonebert=False)
        sf.write(f"output_vits_{i}.wav", audio, sr)
        
        print(f"保存完了: output_*_{i}.wav")

if __name__ == "__main__":
    main()
```

## 次のステップ

### フェーズ2の準備
1. F0-BERTの実装準備
2. 日本語音声データセットの収集
3. XPhoneBERTとVITSの統合最適化

### 品質評価
1. MOSテストの準備
2. アクセント評価の実施
3. ベースライン性能の記録

このガイドに従って実装を進めることで、Tsukuyomiの基本機能を構築できます。