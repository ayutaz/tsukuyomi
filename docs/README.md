# Tsukuyomi TTS Documentation

このディレクトリには、Tsukuyomi TTSプロジェクトのドキュメントが含まれています。

## ドキュメント構成

### 開発ガイド

- **[アーキテクチャ概要](architecture-overview.md)** - システム全体の設計と構成
- **[実装ガイド](implementation-guide.md)** - 各コンポーネントの実装詳細
- **[技術仕様](technical-specifications.md)** - 詳細な技術仕様書

### 学習ガイド

- **[JVSデータセット学習ガイド](jvs_training_guide.ja.md)** - JVSデータセットを使用した学習方法
- **[LJSpeech学習ガイド](ljspeech_training_guide.md)** - LJSpeech形式でのデータ準備と学習
- **[学習ガイド（日本語）](train.ja.md)** - 日本語版の詳細な学習手順

### APIとデプロイメント

- **[APIリファレンス](api/api-reference.md)** - REST APIの詳細仕様
- **[Dockerガイド](docker_guide.md)** - Dockerを使用したデプロイメント

### 技術比較・評価

- **[G2P戦略](g2p-strategy.md)** - 日本語Grapheme-to-Phoneme変換の戦略
- **[pyopenjtalk比較](pyopenjtalk-comparison.md)** - 各種pyopenjtalkライブラリの比較
- **[pyopenjtalk-plusの利点](pyopenjtalk-plus-benefits.md)** - pyopenjtalk-plusを採用する理由

## クイックスタート

1. **開発を始める方**: [アーキテクチャ概要](architecture-overview.md)から読み始めてください
2. **学習を実行する方**: [JVSデータセット学習ガイド](jvs_training_guide.ja.md)を参照してください
3. **APIを使用する方**: [APIリファレンス](api/api-reference.md)を確認してください

## 言語

- 日本語ドキュメントには `.ja.md` の拡張子が付いています
- 英語ドキュメントは拡張子なしの `.md` ファイルです

## 貢献

ドキュメントの改善や追加は歓迎します。プルリクエストを送る前に、既存のドキュメントスタイルに従ってください。