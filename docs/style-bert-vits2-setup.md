# Style-BERT-VITS2 セットアップガイド

## 概要

Style-BERT-VITS2は、高品質な日本語音声合成を実現する最先端のTTSシステムです。
Tsukuyomiでは、即座に実用的な音声合成を実現するため、Style-BERT-VITS2を統合します。

## インストール手順

### 1. リポジトリのクローン

```bash
# Tsukuyomiプロジェクトルートで実行
cd external
git clone https://github.com/litagin02/Style-BERT-VITS2.git
cd Style-BERT-VITS2
git checkout v2.3  # 安定版を使用
```

### 2. 依存関係のインストール

```bash
# Style-BERT-VITS2の依存関係
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 追加の日本語処理ライブラリ
pip install pyopenjtalk-plus[marine]
pip install jp-extra
```

### 3. 事前学習済みモデルのダウンロード

```bash
# JP-Extra版（推奨）
python download_models.py --model jp-extra

# または標準版
python download_models.py --model jp-base
```

### 4. Tsukuyomi統合の設定

```python
# config/style_bert_vits2_config.yaml
model:
  type: "jp-extra"
  path: "external/Style-BERT-VITS2/models/jp-extra"
  device: "cuda"
  use_fp16: true

synthesis:
  default_speaker: 0
  default_style: "Neutral"
  sample_rate: 24000
  
integration:
  use_tsukuyomi_frontend: true  # Tsukuyomiの前処理を使用
  enable_onnx_export: true       # Unity用ONNX出力
```

## 使用方法

### 基本的な使用例

```python
from src.integrations.style_bert_vits2 import create_style_bert_vits2_tts

# TTSシステムの初期化
tts = create_style_bert_vits2_tts()
tts.setup(model_name="jp-extra")

# 音声合成
audio = tts.tts(
    text="こんにちは、月読です。",
    speaker=0,
    style="Neutral"
)
```

### キャラクターボイスのファインチューニング

```bash
# データ準備
python prepare_dataset.py \
    --input_dir /path/to/game_voices \
    --output_dir datasets/game_characters \
    --speaker_name "キャラクター名"

# ファインチューニング
python train.py \
    --model_base jp-extra \
    --dataset datasets/game_characters \
    --batch_size 32 \
    --epochs 100 \
    --use_bf16  # H100で高速化
```

## 10,000時間データでの学習

### 1. データ前処理パイプライン

```python
# scripts/prepare_massive_dataset.py
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import librosa
import soundfile as sf

def process_character_data(character_dir: Path):
    """キャラクターごとのデータ処理"""
    # 音声ファイルの正規化
    # テキストの抽出と整形
    # データセット形式への変換
    pass

def prepare_massive_dataset(base_dir: Path, num_workers: int = 32):
    """10,000時間データの並列処理"""
    character_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(process_character_data, character_dirs)
```

### 2. 段階的学習戦略

```yaml
# training_stages.yaml
stage1:
  name: "代表キャラクター学習"
  speakers: ["主人公", "ヒロインA", "ヒロインB"]
  epochs: 50
  validation_interval: 5

stage2:
  name: "キャラクター拡張"
  speakers: 50  # 上位50キャラクター
  epochs: 100
  checkpoint: "stage1_best.pth"

stage3:
  name: "全キャラクター対応"
  speakers: all  # 500+キャラクター
  epochs: 200
  use_gradient_checkpointing: true  # メモリ節約
```

## Unity/ONNX統合

### ONNXエクスポート

```python
# scripts/export_onnx.py
from src.integrations.style_bert_vits2 import StyleBertVits2Wrapper

wrapper = StyleBertVits2Wrapper()
wrapper.load_model("finetuned_model")
wrapper.export_onnx(Path("unity/Assets/Models/tsukuyomi_tts.onnx"))
```

### Unity側の実装

```csharp
// Unity/TsukuyomiTTS.cs
using Unity.Sentis;
using UnityEngine;

public class TsukuyomiTTS : MonoBehaviour
{
    [SerializeField] private ModelAsset modelAsset;
    private Model runtimeModel;
    private IWorker worker;
    
    void Start()
    {
        runtimeModel = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(
            BackendType.GPUCompute, 
            runtimeModel
        );
    }
    
    public AudioClip Synthesize(string text, int speakerId = 0)
    {
        // テキストを音素列に変換
        var phonemes = ConvertToPhonemes(text);
        
        // 推論実行
        var inputs = new Dictionary<string, Tensor>
        {
            ["phonemes"] = new TensorInt(phonemes),
            ["speaker_id"] = new TensorInt(new int[] { speakerId })
        };
        
        worker.Execute(inputs);
        var output = worker.PeekOutput("audio");
        
        // AudioClipに変換
        return TensorToAudioClip(output);
    }
}
```

## パフォーマンス最適化

### H100での最適化設定

```python
# BF16混合精度学習
trainer_config = {
    "precision": "bf16-mixed",
    "accumulate_grad_batches": 4,
    "gradient_clip_val": 1.0,
    "use_flash_attention": True,
    "compile_model": True,  # PyTorch 2.0+
}

# データローダー最適化
dataloader_config = {
    "batch_size": 64,  # H100で可能な大バッチ
    "num_workers": 16,
    "pin_memory": True,
    "persistent_workers": True,
}
```

## トラブルシューティング

### よくある問題

1. **CUDA out of memory**
   ```python
   # gradient checkpointingを有効化
   model.gradient_checkpointing_enable()
   ```

2. **音質が低い**
   ```yaml
   # config調整
   audio:
     sample_rate: 48000  # 高品質設定
     mel_channels: 128
     hop_length: 256
   ```

3. **推論が遅い**
   ```python
   # バッチ推論の実装
   texts = ["文1", "文2", "文3"]
   audios = tts.batch_synthesize(texts)
   ```

## 評価とベンチマーク

### 自動評価スクリプト

```python
# scripts/evaluate_quality.py
from src.evaluation import calculate_mos, calculate_similarity

def evaluate_model(model_path: str, test_set: str):
    # MOS推定
    mos_scores = calculate_mos(model_path, test_set)
    
    # 話者類似度
    similarity_scores = calculate_similarity(
        model_path, 
        test_set,
        reference_dir="datasets/original_voices"
    )
    
    print(f"平均MOS: {mos_scores.mean():.2f}")
    print(f"話者類似度: {similarity_scores.mean():.2f}")
```

## 次のステップ

1. [ ] Style-BERT-VITS2のインストール確認
2. [ ] サンプルモデルでの動作確認
3. [ ] 10キャラクターでのファインチューニング
4. [ ] 品質評価とフィードバック
5. [ ] 全キャラクター展開