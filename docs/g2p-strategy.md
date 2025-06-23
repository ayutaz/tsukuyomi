# Tsukuyomi TTS - G2P戦略

## なぜG2Pが必要か

### アーキテクチャ上の要求
```
[日本語テキスト] → [G2P] → [音素列] → [XPhoneBERT] → [音素埋め込み]
                              ↓
                        必須の変換ステップ
```

- **XPhoneBERT**：音素レベルの入力が必須（文字入力は不可）
- **音響モデル**：音素埋め込みを期待
- **品質への影響**：G2Pの精度が最終的な音声品質に直結

## 実装戦略：3段階アプローチ

### 🚀 Phase 1: 安定基盤（現在実装済み）
```python
# 実装済み: src/frontend/japanese_g2p.py
g2p = create_japanese_g2p(backend="openjtalk")
```

**特徴**：
- ✅ pyopenjtalk-plus使用（改良版）
- ✅ アクセント情報付き（fullcontext labels）
- ✅ 商用利用可能（Modified BSDライセンス）
- ✅ 標準版より高い精度と安定性
- ⚠️ 辞書ベースのため新語には限界あり

**精度**：
- アクセント核予測：89-90%（標準版87.48%から向上）
- 実用レベルで高品質

### 🎯 Phase 2: 精度向上（3-4週間）

#### Option A: ESPnet統合
```python
# 高精度バックエンド
g2p = create_japanese_g2p(backend="espnet")
```

**メリット**：
- アクセント予測：94.66%（ニューラルモデル）
- ストリーミング対応
- 活発な開発コミュニティ

#### Option B: カスタムBERTモデル
```python
# F0-BERTとの統合
class JapaneseG2PBERT(nn.Module):
    def __init__(self):
        self.bert = AutoModel.from_pretrained("tohoku-nlp/bert-base-japanese")
        self.phoneme_head = nn.Linear(768, phoneme_vocab_size)
        self.accent_head = nn.Linear(768, 3)  # 0,1,2
```

**学習データ**：
- 10,000時間の音声データから自動アライメント
- アクセント辞書との組み合わせ

### 🚀 Phase 3: 統合最適化（オプション）

#### エンドツーエンド学習
```python
class IntegratedG2P(nn.Module):
    def __init__(self):
        self.char_embedding = nn.Embedding(char_vocab_size, 256)
        self.transformer = nn.TransformerEncoder(...)
        self.phoneme_decoder = nn.Linear(256, phoneme_vocab_size)
        self.xphonebert = XPhoneBERT()
    
    def forward(self, text):
        # 文字→音素→埋め込みを一貫して学習
        char_emb = self.char_embedding(text)
        hidden = self.transformer(char_emb)
        phonemes = self.phoneme_decoder(hidden)
        embeddings = self.xphonebert(phonemes)
        return embeddings
```

## 技術選定の判断基準

### 1. 品質要件
| 要件 | OpenJTalk | ESPnet | カスタム |
|-----|-----------|---------|----------|
| アクセント精度 | 87% | 94% | 95%+ (目標) |
| 処理速度 | ◎ | ○ | △ |
| 安定性 | ◎ | ○ | △ |
| カスタマイズ性 | △ | ○ | ◎ |

### 2. リソース要件
- **OpenJTalk**：最小（CPU only）
- **ESPnet**：中程度（GPU推奨）
- **カスタム**：大（学習に大量のGPU時間）

### 3. 開発期間
- **OpenJTalk**：即座に利用可能 ✅
- **ESPnet**：1-2週間の統合作業
- **カスタム**：2-3ヶ月の開発

## 推奨実装計画

### 短期（〜2週間）
```bash
# 現在の実装を活用
pip install pyopenjtalk
python scripts/test_basic_tts.py
```

### 中期（1-2ヶ月）
```python
# 精度向上
if high_quality_mode:
    g2p = create_japanese_g2p(backend="espnet")
else:
    g2p = create_japanese_g2p(backend="openjtalk")
```

### 長期（3ヶ月〜）
- 10,000時間データでカスタムG2Pを学習
- XPhoneBERTとの統合最適化
- マルチタスク学習（G2P + アクセント + 韻律）

## ベンチマーク目標

### Phase 1（現在）
- [ ] 基本的な音素変換：✅ 動作確認
- [ ] アクセント精度：87%以上
- [ ] 処理速度：1000文字/秒以上

### Phase 2
- [ ] アクセント精度：94%以上
- [ ] 未知語対応：80%以上の精度
- [ ] 多言語対応：日英中

### Phase 3
- [ ] アクセント精度：95%以上
- [ ] エンドツーエンド最適化
- [ ] リアルタイム係数：0.05以下

## 結論

1. **G2Pは必須**：XPhoneBERTが音素入力を要求
2. **段階的アプローチ**：安定→高精度→最適化
3. **現実的な選択**：まずOpenJTalkで動作確認、その後ESPnet統合

この戦略により、即座に動作するシステムを構築しながら、
段階的に世界最高水準の精度を実現できます。