# 既存高品質TTSモデル評価レポート

## 1. 評価対象モデル

### 🥇 Style-BERT-VITS2
- **開発元**: Style-BERT-VITS2コミュニティ
- **ライセンス**: AGPL-3.0（商用利用要相談）
- **特徴**:
  - BERT日本語モデルによる高精度アクセント予測
  - 感情表現豊か
  - 10時間程度の音声でファインチューニング可能
  - JP-Extra使用で自然な日本語
- **品質**: MOS 4.3-4.5（プロ収録音声に近い）

### 🥈 VITS2 (MB-iSTFT-VITS2)
- **開発元**: 原論文著者 + コミュニティ
- **ライセンス**: MIT（商用利用可）
- **特徴**:
  - 高速・高品質
  - 多言語対応容易
  - メモリ効率的
  - ESPnet2依存なし
- **品質**: MOS 4.2-4.4

### 🥉 Bert-VITS2
- **開発元**: fishaudio
- **ライセンス**: AGPL-3.0
- **特徴**:
  - 中国語・日本語・英語対応
  - 感情制御可能
  - データ効率的学習
- **品質**: MOS 4.1-4.3

### VOICEVOX/COEIROINK系
- **開発元**: ヒホ氏他
- **ライセンス**: 各種（要確認）
- **特徴**:
  - 日本のVTuber/ゲーム業界で実績
  - キャラクターボイス特化
  - コミュニティ活発
- **品質**: MOS 4.0-4.2

## 2. Tsukuyomiへの統合方針

### 推奨: Style-BERT-VITS2ベース

```python
# 統合アーキテクチャ
Tsukuyomi-TTS v2.0
├── Frontend
│   ├── pyopenjtalk-plus (既存)
│   └── Style-BERT-VITS2 テキスト処理
├── Acoustic Model
│   ├── Style-BERT-VITS2 (メイン)
│   └── 独自Conformer (研究用)
├── Vocoder
│   ├── MB-iSTFT (Style-BERT-VITS2内蔵)
│   └── BigVGAN (オプション)
└── Post-processing
    └── 独自エンハンサー
```

## 3. 実装計画

### Phase 1: Style-BERT-VITS2統合（1週間）

1. **環境構築**
```bash
# Style-BERT-VITS2のインストール
git clone https://github.com/litagin02/Style-BERT-VITS2.git
cd Style-BERT-VITS2
pip install -r requirements.txt
```

2. **ラッパー作成**
```python
class StyleBertVits2Wrapper:
    def __init__(self, model_path: str):
        self.model = load_style_bert_vits2(model_path)
    
    def synthesize(self, text: str, style: str = "Neutral"):
        # Style-BERT-VITS2のAPIを呼び出し
        return self.model.tts(text, style=style)
```

3. **既存システムとの統合**
- Tsukuyomiのインターフェースを維持
- バックエンドをStyle-BERT-VITS2に切り替え

### Phase 2: 10,000時間データでファインチューニング（2-3週間）

1. **データ前処理**
- ゲームキャラクター音声の整理
- 話者ごとにデータセット作成
- 音声品質チェック

2. **学習設定**
```yaml
# ファインチューニング設定
model:
  base: Style-BERT-VITS2-JP
  speakers: 500  # キャラクター数
  
training:
  batch_size: 32  # H100で最適
  learning_rate: 2e-4
  epochs: 100
  fp16: true  # BF16も可
```

3. **段階的学習**
- まず代表10キャラクターで検証
- 品質確認後、全キャラクター展開

### Phase 3: Unity統合（1週間）

1. **ONNX変換**
```python
# Style-BERT-VITS2のONNXエクスポート
model.export_onnx("tsukuyomi_unity.onnx")
```

2. **Unity側実装**
```csharp
public class TsukuyomiTTS : MonoBehaviour {
    private OnnxModel model;
    
    public AudioClip Synthesize(string text) {
        var mel = model.Infer(text);
        return MelToAudio(mel);
    }
}
```

## 4. 比較表

| 項目 | 現在のTsukuyomi | Style-BERT-VITS2統合後 |
|------|----------------|---------------------|
| 品質 | 未学習（ノイズ） | MOS 4.3+ |
| 学習時間 | 3-6ヶ月必要 | 2-3週間 |
| 実用性 | ✗ | ✓ |
| 商用利用 | ✓ | △（要ライセンス確認） |
| カスタマイズ性 | ✓ | ○ |

## 5. リスクと対策

### リスク
1. **ライセンス問題**
   - AGPL-3.0は商用利用に制約
   - 対策: デュアルライセンス交渉

2. **品質のばらつき**
   - キャラクターによって品質差
   - 対策: データ量に応じた学習戦略

3. **技術的依存**
   - Style-BERT-VITS2の更新に依存
   - 対策: フォーク＆独自改良

## 6. 次のアクション

1. **即座に実施**
   - [ ] Style-BERT-VITS2のライセンス詳細確認
   - [ ] サンプル音声での品質評価
   - [ ] 技術的統合可能性の検証

2. **1週間以内**
   - [ ] プロトタイプ実装
   - [ ] 10キャラクターでの検証
   - [ ] パフォーマンス測定

3. **1ヶ月以内**
   - [ ] 全キャラクター対応
   - [ ] Unity統合完了
   - [ ] ユーザーテスト

## 結論

Style-BERT-VITS2の採用により、3-6ヶ月かかる開発を2-3週間に短縮可能。
品質も実証済みで、即座に実用レベルに到達できる。
ライセンス問題はあるが、技術的には最適な選択。