#!/usr/bin/env python3
"""
Test script for Tsukuyomi TTS synthesis
"""

import argparse
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.inference import TsukuyomiTTS


def main():
    parser = argparse.ArgumentParser(description="Test Tsukuyomi TTS synthesis")
    parser.add_argument("--text", type=str, default=None, help="Text to synthesize")
    parser.add_argument("--speaker_id", type=int, default=0, help="Speaker ID")
    parser.add_argument("--output", type=str, default="output.wav", help="Output file path")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed")
    parser.add_argument("--pitch_shift", type=float, default=0.0, help="Pitch shift in semitones")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    
    args = parser.parse_args()
    
    # Test texts in multiple languages
    test_texts = {
        "ja": [
            "こんにちは、月読です。高品質な音声合成システムです。",
            "今日は良い天気ですね。散歩に行きましょう。",
            "ゲームキャラクターの音声を、自然に合成できます。",
        ],
        "en": [
            "Hello, this is Tsukuyomi, a high-quality text-to-speech system.",
            "The weather is nice today. Let's go for a walk.",
        ],
        "zh": [
            "你好，我是月读，一个高质量的语音合成系统。",
            "今天天气很好，我们去散步吧。",
        ],
    }
    
    # Initialize TTS system
    print("Initializing Tsukuyomi TTS...")
    tts = TsukuyomiTTS(device=args.device)
    
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        tts.load_checkpoint(args.checkpoint)
    
    # Single text synthesis
    if args.text:
        print(f"\nSynthesizing: {args.text}")
        start_time = time.time()
        
        audio, sample_rate = tts.synthesize(
            args.text,
            speaker_id=args.speaker_id,
            speed=args.speed,
            pitch_shift=args.pitch_shift
        )
        
        synthesis_time = time.time() - start_time
        audio_duration = len(audio) / sample_rate
        rtf = synthesis_time / audio_duration
        
        print(f"Synthesis completed in {synthesis_time:.2f}s")
        print(f"Audio duration: {audio_duration:.2f}s")
        print(f"Real-time factor: {rtf:.2f}x")
        
        # Save audio
        sf.write(args.output, audio, sample_rate)
        print(f"Audio saved to {args.output}")
        
    # Benchmark mode
    elif args.benchmark:
        print("\nRunning performance benchmark...")
        results = tts.benchmark()
        
        print("\nBenchmark Results:")
        print(f"Average RTF: {results['avg_rtf']:.2f}x")
        print(f"Tokens/second: {results['tokens_per_second']:.1f}")
        print(f"Average latency: {results['avg_latency']:.3f}s")
        
        if torch.cuda.is_available():
            print(f"\nGPU Memory Usage:")
            print(f"Allocated: {results['gpu_memory_allocated']:.2f} GB")
            print(f"Reserved: {results['gpu_memory_reserved']:.2f} GB")
    
    # Multi-language test
    else:
        print("\nRunning multi-language synthesis test...")
        output_dir = Path("test_outputs")
        output_dir.mkdir(exist_ok=True)
        
        for lang, texts in test_texts.items():
            print(f"\n=== {lang.upper()} ===")
            for i, text in enumerate(texts):
                print(f"Synthesizing: {text}")
                
                # Synthesize
                start_time = time.time()
                audio, sample_rate = tts.synthesize(
                    text,
                    speaker_id=args.speaker_id,
                    language=lang
                )
                synthesis_time = time.time() - start_time
                
                # Save
                output_path = output_dir / f"{lang}_{i:02d}.wav"
                sf.write(output_path, audio, sample_rate)
                
                # Stats
                audio_duration = len(audio) / sample_rate
                rtf = synthesis_time / audio_duration
                print(f"  -> Saved to {output_path} (RTF: {rtf:.2f}x)")
        
        print(f"\nAll test outputs saved to {output_dir}/")
        
    # Test batch processing
    if not args.text and not args.benchmark:
        print("\n=== Batch Processing Test ===")
        batch_texts = [
            "バッチ処理のテストです。",
            "複数の文を同時に処理できます。",
            "効率的な音声合成が可能です。",
        ]
        
        print("Processing batch of 3 texts...")
        start_time = time.time()
        
        audio_list = tts.synthesize_batch(
            batch_texts,
            speaker_ids=[0, 1, 2],
            language="ja"
        )
        
        batch_time = time.time() - start_time
        print(f"Batch processing completed in {batch_time:.2f}s")
        
        # Save batch outputs
        for i, (audio, sr) in enumerate(audio_list):
            output_path = output_dir / f"batch_{i:02d}.wav"
            sf.write(output_path, audio, sr)
            print(f"  -> Saved to {output_path}")


if __name__ == "__main__":
    main()
