# Tsukuyomi TTS プロジェクト

## プロジェクト概要

Tsukuyomi TTSは、最高峰の日本語音声合成システムです。XPhoneBERT、F0-BERT、VITS/Matcha-TTS、BigVGAN-v2などの最先端技術を統合し、自然で表現豊かな音声を生成します。

### 主な特徴

1. **最高品質の音声合成**
   - XPhoneBERTによる多言語音素エンコーディング
   - F0-BERTによる高精度なピッチモデリング
   - VITS/Matcha-TTSによる高品質音響モデル
   - BigVGAN-v2による高忠実度ボコーダー

2. **大規模データ対応**
   - 1万時間以上のデータセット処理能力
   - 100人以上の話者サポート
   - 効率的な分散学習

3. **本番環境対応**
   - ONNX変換によるUnity統合
   - Triton Inference Serverサポート
   - バッチ処理とキャッシングによる高速推論
   - 包括的なAPIとドキュメント

## 実装状況

### ✅ 完了した機能

#### コアモデル
- [x] XPhoneBERT統合（日本語最適化）
- [x] F0-BERT実装
- [x] VITS音響モデル
- [x] Matcha-TTS（フローマッチング）
- [x] BigVGAN-v2ボコーダー

#### インフラストラクチャ
- [x] Docker環境（Windows/Linux対応）
- [x] 分散学習スクリプト
- [x] GitHub Actions CI/CD
- [x] UV パッケージマネージャー統合

#### 最適化・統合
- [x] ONNX変換・最適化
- [x] Unity C#ラッパー生成
- [x] 推論最適化（バッチ処理、キャッシング）
- [x] Triton Inference Server統合

#### 評価・品質保証
- [x] 包括的な評価システム（MCD、ピッチ、話者類似度）
- [x] パフォーマンスベンチマーク
- [x] ユニットテスト（モデル、データ、推論）

#### API・ドキュメント
- [x] FastAPI サーバー実装
- [x] 認証・レート制限
- [x] OpenAPI/Swagger ドキュメント
- [x] クライアントSDK生成（Python、TypeScript）

#### 高度な機能
- [x] 感情制御（10種類の感情 + VADモデル）
- [x] スタイル転送（参照音声、スタイルミキシング）
- [x] 音声モーフィング（複数話者ブレンド、連続遷移）
- [x] リアルタイムストリーミング（低レイテンシ、WebSocket）
- [x] 統合AdvancedTTSモデル

### 🔧 開発コマンド

```bash
# 環境セットアップ
uv venv
uv pip install -e .

# 学習の実行
python scripts/train.py --config configs/train_config.yaml

# マルチGPU学習
./scripts/train_multi_gpu.sh configs/train_config.yaml

# 推論ベンチマーク
python scripts/benchmark_inference.py --model checkpoints/best_model.pt

# Tritonデプロイ
python scripts/deploy_triton.py --checkpoint checkpoints/best_model.pt

# APIドキュメント生成
python scripts/generate_api_docs.py --output-dir docs/api

# テストの実行
pytest tests/ -v

# リント
ruff check src
black src tests

# 高度な機能のデモ
python scripts/demo_advanced_features.py --model-path checkpoints/best_model.pt --demo-type all
```

### 📁 プロジェクト構造

```
tsukuyomi/
├── src/
│   ├── models/          # コアモデル実装
│   │   ├── xphonebert.py
│   │   ├── f0_bert.py
│   │   ├── vits.py
│   │   ├── matcha_tts.py
│   │   ├── bigvgan.py
│   │   ├── emotion_controller.py  # 感情制御
│   │   ├── style_transfer.py      # スタイル転送
│   │   ├── voice_morphing.py      # 音声モーフィング
│   │   └── advanced_tts.py        # 統合モデル
│   ├── data/            # データ処理
│   ├── training/        # 学習関連
│   ├── inference/       # 推論最適化
│   │   ├── optimized_inference.py
│   │   └── realtime_streaming.py  # ストリーミング
│   ├── serving/         # モデルサービング
│   ├── server/          # APIサーバー
│   ├── evaluation/      # 評価メトリクス
│   ├── export/          # モデル変換
│   └── utils/           # ユーティリティ
├── scripts/             # 実行スクリプト
│   ├── train.py
│   ├── benchmark_inference.py
│   └── demo_advanced_features.py  # 高度な機能デモ
├── configs/             # 設定ファイル
├── tests/               # テストコード
├── docker/              # Docker関連
└── docs/                # ドキュメント
```

## 高度な機能の詳細

### 🎭 感情制御
- **10種類の基本感情**: neutral, happy, sad, angry, fearful, surprised, disgusted, excited, calm, confident
- **VADモデル**: Valence（感情価）、Arousal（覚醒度）、Dominance（支配性）による連続的な感情表現
- **感情予測**: テキストから自動的に感情を推定
- **感情強度制御**: 0.0〜2.0の範囲で感情の強さを調整

### 🎨 スタイル転送
- **参照音声からのスタイル抽出**: 任意の音声からスタイルを学習
- **スタイルミキシング**: 複数のスタイルをブレンド
- **適応的スタイル転送**: コンテンツに応じて転送強度を自動調整
- **スタイルバンク**: 事前定義された50種類のスタイル

### 🔀 音声モーフィング
- **球面線形補間（SLERP）**: 滑らかな話者間の遷移
- **多点モーフィング**: 3人以上の話者を同時にブレンド
- **連続モーフィング**: 時間軸に沿った動的な話者変化
- **声質変換**: コンテンツを保持したまま声質のみ変更

### ⚡ リアルタイムストリーミング
- **低レイテンシ**: 100ms以下の遅延
- **チャンク処理**: 256文字単位での逐次生成
- **CUDAストリーム**: 並列処理による高速化
- **WebSocketサーバー**: リアルタイム双方向通信

## 今後の拡張案

1. **追加の前処理・拡張**
   - より高度なテキスト正規化
   - ~~感情制御機能~~ ✅ 実装済み
   - ~~声質変換~~ ✅ 実装済み

2. **モデル改良**
   - より大規模な事前学習
   - ファインチューニング戦略
   - マルチタスク学習

3. **デプロイメント**
   - Kubernetes オーケストレーション
   - エッジデバイス対応
   - WebAssembly 統合

## 注意事項

- 前処理と事前学習はユーザー側で実施
- H100 GPU 8台での学習を想定
- 100時間以上のデータでの学習を推奨
- メタデータが不完全な場合はnull許容

## リンク

- [日本語README](README.ja.md)
- [英語README](README.md)
- [学習ガイド](docs/train.ja.md)
- [APIリファレンス](docs/api/api-reference.md)