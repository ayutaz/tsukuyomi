# Tsukuyomi Features Summary

## Voice Morphing (音声モーフィング)

Yes, Tsukuyomi supports advanced voice morphing capabilities:

### Basic Voice Morphing
```python
# Blend two speakers
audio = tts.morph_voices(
    text="こんにちは、月読です",
    speaker_ids=[0, 5],
    speaker_weights=[0.7, 0.3]  # 70% speaker 0, 30% speaker 5
)
```

### Multi-Speaker Morphing
```python
# Blend multiple speakers
audio = tts.morph_voices(
    text="複数の声をミックスします",
    speaker_ids=[0, 5, 10, 15],
    speaker_weights=[0.4, 0.3, 0.2, 0.1]
)
```

### Emotion Morphing
```python
# Morph both speakers and emotions
audio = tts.morph_voices(
    text="感情豊かな音声です",
    speaker_ids=[0, 5],
    speaker_weights=[0.6, 0.4],
    emotion_ids=["happy", "excited"],
    emotion_weights=[0.7, 0.3]
)
```

### Gradient Morphing
```python
# Gradual transition between speakers
for i in range(11):
    weight = i / 10.0
    audio = tts.morph_voices(
        text="声が徐々に変化します",
        speaker_ids=[0, 5],
        speaker_weights=[1-weight, weight]
    )
```

## CUDA 12.1+ Optimizations

All models are optimized for CUDA 12.1+:

- **Flash Attention 2**: Up to 5x faster attention computation
- **CUDA Graphs**: 10-30% inference speedup for static shapes
- **Triton Kernels**: Custom optimized operations
- **BF16/TF32**: Enhanced tensor core utilization
- **torch.compile**: Dynamic shape optimization
- **cudaMallocAsync**: Improved memory management

## Language Support

- **English Documentation**: README.md
- **Japanese Documentation**: README_ja.md, train_ja.md
- **Bilingual Interface**: Easy language switching

## Key Features

1. **Ultimate Quality**: MOS 4.7+ target (human-level)
2. **500+ Speakers**: Game character voice support
3. **Voice Cloning**: Clone from reference audio
4. **Emotion Control**: 7 emotions × 10 speaking styles
5. **Real-time Performance**: RTF < 0.05
6. **Unity Integration**: ONNX export support
7. **Distributed Training**: Multi-GPU with FSDP
8. **Staged Training**: 100h → 1,000h → 10,000h approach