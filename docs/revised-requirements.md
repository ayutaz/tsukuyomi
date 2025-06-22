# Tsukuyomi 改訂版要件定義と実現戦略

## 明確化された要件

### 1. 日本語を含む多言語のアクセントおよび話者の再現性が追加学習ありで最高峰であること

この要件を分解すると：
- **多言語対応**: 日本語を中心に、英語、中国語、韓国語等
- **アクセントの完璧な再現**: 各言語のネイティブレベルの韻律
- **話者の再現性**: 追加学習により新しい話者の声質を完璧に模倣
- **最高峰の品質**: 現存する全てのTTSシステムを上回る

### 2. 将来的にUnityで動かすためにONNXで動かせる

- **ONNX対応**: 全モジュールをONNX形式でエクスポート可能
- **Unity統合**: Unity Barracuda/Sentisでの実行
- **クロスプラットフォーム**: Windows/Mac/Linux/Mobile/Console対応

### 3. 推論時間よりは精度を優先する

- **品質最優先**: リアルタイム性を犠牲にしても最高品質を追求
- **用途**: カットシーン、ストーリーパート等の事前生成も想定
- **許容レイテンシ**: 1秒以内なら許容（通常のTTSは50-100ms）

## 要件を満たすアーキテクチャ設計

### 1. 多言語アクセント対応

```python
class MultilingualAccentSystem:
    def __init__(self):
        # 言語別専門モデル
        self.language_experts = {
            'ja': JapaneseAccentExpert(),  # F0-BERT + 日本語特化
            'en': EnglishProsodyExpert(),  # 英語韻律モデル
            'zh': ChineseToneExpert(),     # 中国語声調モデル
            'ko': KoreanIntonationExpert() # 韓国語イントネーション
        }
        
        # 統一音素表現
        self.universal_phoneme = XPhoneBERT(
            enhanced_for=['ja', 'en', 'zh', 'ko']
        )
        
        # 言語間知識転移
        self.cross_lingual_adapter = CrossLingualAdapter()
```

### 2. 最高峰の話者再現性

```python
class SpeakerReproductionSystem:
    def __init__(self):
        # ベースモデル: 1万時間で学習
        self.foundation_model = UniversalSpeakerModel(
            n_speakers=5000,  # 基礎話者数
            speaker_dim=1024  # 高次元話者空間
        )
        
        # 高速話者適応メカニズム
        self.adaptation_methods = {
            'few_shot': FewShotAdapter(n_shots=10),      # 10発話で適応
            'fine_tuning': ResidualAdapter(rank=64),     # 高品質適応
            'voice_cloning': NeuralVoiceCloner(),        # 音声クローニング
            'zero_shot': ZeroShotSpeakerEncoder()        # 即座適応
        }
```

## 技術実装の詳細

### フェーズ1: 多言語基盤モデル構築

#### データ戦略
```yaml
training_data:
  japanese: 
    hours: 8000  # 1万時間の80%
    speakers: 400 # ゲームキャラクター
    quality: studio_grade
  
  english:
    hours: 1000
    speakers: 50
    datasets: ["LibriTTS", "VCTK", "Custom"]
  
  chinese:
    hours: 500
    speakers: 30
    datasets: ["AISHELL-3", "Custom"]
  
  korean:
    hours: 500
    speakers: 20
    datasets: ["KSS", "Custom"]
```

#### モデルアーキテクチャ
```python
# 最新技術の統合
class TsukuyomiMultilingualTTS:
    def __init__(self):
        # 1. 言語認識層
        self.language_identifier = LanguageID()
        
        # 2. 統一エンコーダー（XPhoneBERT改良版）
        self.phoneme_encoder = EnhancedXPhoneBERT(
            base_model="xlm-roberta-large",
            phoneme_layers=6,
            language_specific_heads=True
        )
        
        # 3. 言語別エキスパート
        self.prosody_experts = {
            'ja': F0BERT_JP(trained_on="8000h_game_data"),
            'en': ProsodyBERT_EN(),
            'zh': ToneBERT_ZH(),
            'ko': IntonationBERT_KO()
        }
        
        # 4. 音響モデル（最先端）
        self.acoustic_model = FlowMatchingTTS(
            model="F5-TTS",  # 2024年最新
            mel_bins=128,
            sampling_rate=48000  # 高音質
        )
        
        # 5. ニューラルボコーダー
        self.vocoder = BigVGAN_v2(
            universal=True,
            languages=['ja', 'en', 'zh', 'ko']
        )
```

### フェーズ2: 話者再現性の極限追求

#### 追加学習メソッド

```python
class ExtremeSpeakerAdaptation:
    def __init__(self, base_model):
        self.base_model = base_model
        
    def adapt_new_speaker(self, audio_samples, method='best'):
        if method == 'best':
            # 音声量に応じて最適手法を選択
            if len(audio_samples) < 10:
                return self.zero_shot_adaptation(audio_samples)
            elif len(audio_samples) < 100:
                return self.few_shot_adaptation(audio_samples)
            else:
                return self.full_adaptation(audio_samples)
    
    def zero_shot_adaptation(self, samples):
        # 数秒の音声から即座に適応
        speaker_emb = self.extract_speaker_embedding(samples)
        return ZeroShotAdapter(speaker_emb)
    
    def few_shot_adaptation(self, samples):
        # 10-100発話で高品質適応
        adapter = ResidualAdapter(rank=32)
        adapter.train(samples, epochs=100)
        return adapter
    
    def full_adaptation(self, samples):
        # 完全なファインチューニング
        model_copy = self.base_model.copy()
        model_copy.fine_tune(samples, epochs=1000)
        return model_copy
```

## H100活用による学習高速化

```yaml
distributed_training:
  strategy: "Hybrid Parallelism"
  
  data_parallel:
    gpus: 4  # 4GPU でデータ並列
    batch_size_per_gpu: 64
    
  model_parallel:
    gpus: 4  # 4GPU でモデル並列
    pipeline_stages: 4
    
  optimization:
    - FlashAttention-3
    - BF16 Mixed Precision
    - Gradient Accumulation
    - ZeRO-3 Optimization
    
  expected_speedup: 25x  # vs 単一GPU
```

## 期待される成果

### 1. アクセント精度
- **日本語**: ネイティブ話者と区別不可能（MOS 4.7+）
- **英語**: ネイティブレベル（MOS 4.5+）
- **中国語**: 声調完璧再現（声調認識精度 99%+）
- **韓国語**: 自然なイントネーション（MOS 4.4+）

### 2. 話者再現性
- **ゼロショット**: 10秒音声で80%の類似度
- **Few-shot**: 10発話で95%の類似度
- **フルアダプテーション**: 99%+の類似度

### 3. 実用性
- **新規話者追加時間**: 最短1分（ゼロショット）〜最長1時間（フル適応）
- **推論速度**: リアルタイムの50倍速
- **メモリ効率**: 話者あたり1-10MB

## ONNX対応とUnity統合

### ONNX最適化アーキテクチャ

```python
class ONNXOptimizedTTS:
    def __init__(self):
        # モジュラー設計でONNXエクスポート
        self.modules = {
            'text_encoder': 'text_encoder.onnx',      # 300MB
            'prosody_model': 'prosody_model.onnx',    # 500MB
            'acoustic_model': 'acoustic_model.onnx',   # 800MB
            'vocoder': 'vocoder.onnx'                 # 200MB
        }
        
    def export_to_onnx(self, precision='fp32'):
        # 精度優先のため量子化は最小限
        export_config = {
            'opset_version': 17,
            'optimization_level': 'BASIC_OPT',  # 品質維持
            'quantization': None  # 量子化なし（品質優先）
        }
```

### Unity統合設計

```csharp
// Unity/C# インターフェース
public class TsukuyomiTTS : MonoBehaviour 
{
    private SentisModel textEncoder;
    private SentisModel prosodyModel;
    private SentisModel acousticModel;
    private SentisModel vocoder;
    
    public AudioClip GenerateSpeech(
        string text, 
        string language, 
        int speakerId,
        float emotionIntensity = 1.0f)
    {
        // 品質優先: 複数パスで精度向上
        var phonemes = textEncoder.Execute(text, language);
        var prosody = prosodyModel.Execute(phonemes, speakerId);
        var acoustic = acousticModel.Execute(prosody, emotionIntensity);
        var waveform = vocoder.Execute(acoustic);
        
        return CreateAudioClip(waveform);
    }
}
```

## 精度優先の設計判断

### 1. モデルサイズと精度のトレードオフ

```yaml
model_configs:
  # 通常のリアルタイムTTS
  realtime_model:
    size: 500MB
    quality: 4.0/5.0
    latency: 50ms
    
  # Tsukuyomi (精度優先)
  tsukuyomi_model:
    size: 2GB  # 4倍のサイズ
    quality: 4.8/5.0  # 最高品質
    latency: 500-1000ms  # 許容範囲内
    
  optimizations:
    - No quantization (FP32維持)
    - Large hidden dimensions (2048)
    - Deep networks (24+ layers)
    - High-resolution features (256 mel bins)
```

### 2. 精度向上のための技術選択

```python
class QualityFirstAcousticModel:
    def __init__(self):
        # より大きく深いモデル
        self.encoder = TransformerEncoder(
            n_layers=24,  # 通常の2倍
            d_model=2048,  # 通常の2倍
            n_heads=32
        )
        
        # 高解像度特徴量
        self.mel_specs = HighResolutionMelSpec(
            n_mels=256,  # 通常80-128
            n_fft=4096,  # 通常1024-2048
            sample_rate=48000  # 通常22050
        )
        
        # アンサンブル推論
        self.ensemble_models = [
            Model1(),  # Matcha-TTS
            Model2(),  # F5-TTS
            Model3()   # VITS2
        ]
```

### 3. Unity向け最適化戦略

```yaml
deployment_strategy:
  # オフライン生成モード
  offline_generation:
    - カットシーン用音声を事前生成
    - 高品質設定で一括処理
    - Unityアセットとして保存
    
  # ストリーミング生成モード
  streaming_generation:
    - 1秒バッファで逐次生成
    - 品質を維持しつつ体感速度向上
    
  # ハイブリッドモード
  hybrid_mode:
    - 重要な台詞: 最高品質モード (1秒)
    - 汎用台詞: 標準品質モード (200ms)
```

## 実装上の考慮事項

### メモリ管理
```csharp
public class TsukuyomiMemoryManager
{
    // Unity向けメモリ最適化
    private const int MAX_CACHE_SIZE = 4096; // 4GB
    private Dictionary<string, AudioClip> voiceCache;
    
    // 話者モデルの動的ロード
    public void LoadSpeakerModel(int speakerId)
    {
        UnloadUnusedSpeakers();
        LoadONNXModel($"speaker_{speakerId}.onnx");
    }
}
```

### プラットフォーム対応
```yaml
platform_support:
  windows:
    backend: "DirectML"
    performance: "最高"
    
  macos:
    backend: "CoreML" 
    performance: "高"
    
  mobile:
    backend: "NNAPI/Metal"
    mode: "オフライン生成推奨"
    
  console:
    backend: "Custom"
    optimization: "プラットフォーム別最適化"
```

## 他システムとの比較（改訂版）

| システム | 多言語 | 話者適応 | 日本語品質 | ONNX | Unity | 精度優先 |
|---------|--------|----------|-----------|------|-------|----------|
| Google TTS | ◎ | △ | ○ | × | △ | △ |
| Amazon Polly | ◎ | △ | △ | × | △ | △ |
| Azure TTS | ◎ | ○ | ○ | △ | ○ | ○ |
| ElevenLabs | ○ | ◎ | × | × | × | ○ |
| Coqui TTS | ○ | ○ | △ | ○ | △ | △ |
| **Tsukuyomi** | ◎ | ◎ | ◎ | ◎ | ◎ | ◎ |

## 結論

3つの要件を総合的に満たすシステム設計：

1. **多言語・話者再現性**: 1万時間データとH100×8で世界最高水準
2. **ONNX/Unity対応**: モジュラー設計で完全対応、Sentis統合
3. **精度優先**: 2GBモデル、500-1000msレイテンシで最高品質

特に精度優先の要件により：
- 量子化を避けFP32精度維持
- モデルサイズ制限を緩和（2GB）
- アンサンブル推論で品質向上
- 高解像度特徴量（256mel, 48kHz）

これによりゲーム業界で前例のない高品質音声合成が実現可能です。