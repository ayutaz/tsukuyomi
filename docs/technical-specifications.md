# 技術仕様書

## 1. フロントエンド（前処理・音韻表現層）

### 1.1 テキスト前処理
- **形態素解析**: MeCab + UniDic/IPAdic
- **テキスト正規化**: 
  - 数字→読み変換
  - 記号処理
  - 英語混じり文の処理
- **分かち書き**: 単語・記号単位での空白挿入

### 1.2 XPhoneBERT
- **モデル**: vinai/xphonebert-base
- **入力**: 音素化されたテキスト
- **出力**: 768次元の音素埋め込みベクトル
- **実装**:
  ```python
  from transformers import AutoModel, AutoTokenizer
  from text2phonemesequence import Text2PhonemeSequence
  
  # 日本語用音素変換器
  phonemizer_ja = Text2PhonemeSequence(language='jpn', is_cuda=True)
  ```

### 1.3 F0-BERT（フェーズ2以降）
- **ベースモデル**: 日本語BERT（東北大学版など）
- **入力**: 漢字かな混じり文 + モーラ単位カタカナ
- **出力**: 連続F0系列（対数スケール）
- **学習データ**: 1万時間の日本語音声から抽出したF0

## 2. 音響モデル層

### 2.1 基盤モデル（フェーズ1: VITS、フェーズ3: Matcha-TTS）

#### VITS（初期実装）
- **実装選択肢**:
  - Facebook MMS-TTS（facebook/mms-tts-jpn）
  - ESPnet日本語VITS
  - 独自学習済みモデル
- **入力**: 音素ID列 or XPhoneBERT埋め込み
- **出力**: メルスペクトログラム（80次元）

#### Matcha-TTS（将来実装）
- **技術**: Conditional Flow Matching (CFM)
- **利点**: VITSより高速・高品質
- **アーキテクチャ**: U-Netベースデコーダー + OT-CFM

### 2.2 話者適応（Residual Adapters）
- **パラメータ効率**: 全体の0.1%のみ更新
- **構造**: 各Conformerブロックにボトルネック層を挿入
- **必要データ**: 話者あたり30分〜1時間
- **話者エンコーダ**: ECAPA2（事前学習済み）

### 2.3 感情制御（HVAE）
- **階層構造**: フレーム・音素・単語・文レベル
- **制御方式**:
  - カテゴリカル: 離散感情ラベル（喜び、怒り等）
  - 連続制御: AVD空間（覚醒度・快不快度・優位度）

## 3. ボコーダー層

### BigVGAN
- **モデル**: nvidia/bigvgan_22khz_80band
- **入力**: 80次元メルスペクトログラム
- **出力**: 22.05kHz音声波形
- **実装**:
  ```python
  import bigvgan
  vocoder = bigvgan.BigVGAN.from_pretrained(
      "nvidia/bigvgan_22khz_80band", 
      use_cuda_kernel=False
  )
  ```

## 4. 音声モーフィング（オプション）

### kNNベースSSL特徴量補間
- **SSLモデル**: WavLM-Large
- **特徴量次元**: 1024次元
- **kNN検索**: FAISSライブラリ使用
- **補間パラメータ**: λ ∈ [0, 1]

## 5. デプロイメント仕様

### ONNX変換
- **各モジュール独立エクスポート**
- **量子化**: INT8静的量子化
- **推論フレームワーク**: ONNX Runtime

### 性能目標
- **レイテンシ**: < 100ms（短文）
- **スループット**: リアルタイムファクター > 10
- **メモリ使用量**: < 2GB（全モジュール合計）

## 6. データ要件

### 学習データ
- **規模**: 1万時間の日本語音声
- **品質基準**: MOSスコア > 3.8
- **前処理**:
  - 音源分離（MSS）
  - エコー除去
  - 話者ダイアライゼーション
  - 高精度文字起こし（Whisper-large-v3）

### 話者適応データ
- **必要量**: 話者あたり30分〜1時間
- **品質**: スタジオ録音品質推奨

## 7. 開発環境

### 必須ライブラリ
```bash
# 基本環境
python >= 3.8
pytorch >= 2.0 (CUDA対応)

# 音声処理
transformers
text2phonemesequence
bigvgan
soundfile
librosa

# 日本語処理
mecab-python3
unidic-lite

# 最適化・デプロイ
onnx
onnxruntime
```

### ハードウェア要件
- **開発**: NVIDIA GPU（8GB以上のVRAM）
- **学習**: A100 80GB × 8（推奨）
- **推論**: CPU or GPU（ONNXランタイム）

## 8. 評価指標

### 客観指標
- **MOS（Mean Opinion Score）**: > 4.0目標
- **話者類似度**: > 0.9（コサイン類似度）
- **文字誤り率**: < 5%

### 主観指標
- アクセントの自然さ
- 感情表現の適切さ
- 長時間聴取での疲労感