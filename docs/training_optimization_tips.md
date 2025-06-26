# T4 GPUでの学習最適化ガイド

## 現状の問題
- T4 1台での100エポック学習: 約**21.7日**
- 1エポック: 約5.2時間
- 1イテレーション: 約45秒

## 推奨する最適化方法

### 1. モデルの軽量化
```yaml
# 変更前（フルモデル）
hidden_channels: 192
filter_channels: 768
n_layers: 6
n_flows: 4

# 変更後（T4最適化）
hidden_channels: 128  # -33%
filter_channels: 512  # -33%
n_layers: 4          # -33%
n_flows: 3           # -25%
```

### 2. バッチサイズとGradient Accumulation
```yaml
batch_size: 16  # メモリ節約
gradient_accumulation_steps: 2  # 実効バッチサイズ32を維持
```

### 3. Mixed Precision Training（必須）
```yaml
mixed_precision: "fp16"  # 約2倍高速化
```

### 4. Gradient Checkpointing
```yaml
gradient_checkpointing: true  # メモリ使用量を削減
```

### 5. データセットのサブセット化
```bash
# 10話者のみで試す場合
python scripts/create_subset.py \
  --input-dir data/jvs_ljspeech \
  --output-dir data/jvs_ljspeech_10speakers \
  --num-speakers 10 \
  --samples-per-speaker 100
```

### 6. 段階的な学習
1. **第1段階**: 10話者、20エポック（約1日）
2. **第2段階**: 50話者、30エポック（約3日）  
3. **第3段階**: 100話者、50エポック（約10日）

### 7. 複数GPU利用（可能であれば）
```bash
# 2 GPU利用
torchrun --nproc_per_node=2 scripts/train.py \
  --config configs/experiment_vits_jvs_t4_optimized.yaml
```

## 期待される結果
- **学習時間**: 21.7日 → 約10日（50エポック、軽量モデル）
- **メモリ使用量**: 約40%削減
- **音質**: わずかに低下するが、実用レベルは維持

## Cloud GPU利用の検討
### コスト効率の良い選択肢：
1. **Google Colab Pro+**: T4/P100 (月額$49.99)
2. **Paperspace Gradient**: P4000/P5000 ($0.51-0.78/時)
3. **Lambda Labs**: A10 ($0.60/時)
4. **Vast.ai**: 3090/4090 ($0.20-0.50/時)

### H100利用（高速だが高価）：
- AWS: 約$36/時
- Lambda Labs: 約$2.49/時
- 100エポック学習: 約4-6時間で完了（$150-200）

## モニタリングとデバッグ

### TensorBoard確認
```bash
tensorboard --logdir logs/vits_jvs_t4/tensorboard
```

### 学習曲線の確認ポイント
1. **Loss収束**: 10エポック程度で大きく下がるはず
2. **Duration Loss**: 特に重要、早期に収束すべき
3. **KL Divergence**: 徐々に減少

### 早期終了の活用
```yaml
early_stopping:
  enabled: true
  patience: 10
  metric: "val_loss"
```

## トラブルシューティング

### OOMエラーの場合
```yaml
batch_size: 8  # さらに削減
gradient_accumulation_steps: 4
clear_cache_interval: 25
```

### 学習が遅い場合
```yaml
num_workers: 0  # データローダーのボトルネック確認
use_cache: true  # キャッシュ必須
```

### 品質が低い場合
- filter_channelsを768に戻す
- n_layersを6に戻す
- 学習率を1e-4に下げる