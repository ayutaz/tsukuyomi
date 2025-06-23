# Tsukuyomi TTS - 重要課題リスト

## 🚨 致命的な問題（即座に対応が必要）

### 1. 外部依存パッケージの欠落
- [ ] `text2phonemesequence`の実装または代替ライブラリの選定
- [ ] `bigvgan`の実装またはHiFi-GAN等の代替実装
- [ ] 実際に動作する音素変換システムの構築

### 2. テストの実行不可能問題
- [ ] 存在しないクラス（DurationPredictor、VarianceAdaptor等）の削除
- [ ] モックとテストの整合性確保
- [ ] 実際に実行可能なテストスイートの構築

### 3. コア機能の未実装
- [ ] F0-BERTによる韻律予測の実装
- [ ] 実際の音響モデル（Matcha-TTS/VITS）の実装
- [ ] 話者エンコーダーの実装

## ⚠️ 重要な改善事項

### アーキテクチャの再設計
```python
# 推奨: 責務の分離
class TTSPipeline:
    def __init__(self, components: TTSComponents):
        self.text_processor = components.text_processor
        self.phoneme_encoder = components.phoneme_encoder
        self.acoustic_model = components.acoustic_model
        self.vocoder = components.vocoder
```

### エラーハンドリングの強化
```python
@dataclass
class TTSError:
    error_type: ErrorType
    message: str
    recovery_action: Optional[Callable]

def synthesize(self, text: str) -> Result[Audio, TTSError]:
    # 適切なエラーハンドリング
```

### 型安全性の向上
```python
from typing import TypedDict, Protocol

class AudioOutput(TypedDict):
    waveform: np.ndarray
    sample_rate: int
    duration_seconds: float

class Synthesizer(Protocol):
    def synthesize(self, text: str) -> AudioOutput:
        ...
```

## 📋 段階的な改善計画

### Phase 1: 基盤の修正（2週間）
1. 外部依存の解決
2. テストの修正
3. 基本的な音声合成の動作確認

### Phase 2: コア機能の実装（4週間）
1. 実際の音素変換システム
2. 基本的な音響モデル
3. 動作するボコーダー

### Phase 3: 品質向上（4週間）
1. BF16最適化の実装
2. マルチGPU対応
3. キャッシング戦略

### Phase 4: 高度な機能（6週間）
1. 話者適応
2. 感情制御
3. リアルタイム合成

## 🎯 成功の指標

- [ ] 基本的な日本語音声合成が動作する
- [ ] MOS 4.0以上の品質
- [ ] RTF < 0.1 (リアルタイム合成)
- [ ] 95%以上のテストカバレッジ
- [ ] ゼロダウンタイムデプロイメント