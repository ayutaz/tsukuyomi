#!/usr/bin/env python3
"""高度な機能のデモスクリプト

感情制御、スタイル転送、音声モーフィング、リアルタイムストリーミングのデモ
"""

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.inference.realtime_streaming import StreamingConfig
from src.models.advanced_tts import AdvancedTTS


def demo_emotion_control(model: AdvancedTTS, output_dir: Path):
    """感情制御のデモ"""
    print("\n=== 感情制御デモ ===")
    
    text = "今日はとても素晴らしい一日でした"
    text_tensor = torch.tensor([ord(c) % 256 for c in text]).unsqueeze(0)
    speaker_id = torch.tensor([0])
    
    emotions = [
        ("neutral", 0),
        ("happy", 1),
        ("sad", 2),
        ("angry", 3),
        ("excited", 7),
    ]
    
    for emotion_name, emotion_id in emotions:
        print(f"\n感情: {emotion_name}")
        
        # 感情IDを使用した合成
        outputs = model.synthesize(
            text_tensor,
            speaker_id,
            emotion_id=torch.tensor([emotion_id]),
            emotion_intensity=1.5,  # 感情を強調
        )
        
        audio = outputs['audio'].squeeze().cpu().numpy()
        
        # 保存
        output_path = output_dir / f"emotion_{emotion_name}.wav"
        sf.write(str(output_path), audio, 48000)
        print(f"保存: {output_path}")
        
    # 連続的な感情値（VAD）を使用
    print("\n連続感情値（VAD）でのデモ")
    vad_values = [
        ("positive_high_energy", [0.8, 0.9, 0.7]),   # Valence, Arousal, Dominance
        ("negative_low_energy", [-0.7, -0.5, -0.3]),
        ("neutral_calm", [0.0, -0.3, 0.0]),
    ]
    
    for name, vad in vad_values:
        outputs = model.synthesize(
            text_tensor,
            speaker_id,
            emotion_vad=torch.tensor([vad], dtype=torch.float),
        )
        
        audio = outputs['audio'].squeeze().cpu().numpy()
        output_path = output_dir / f"emotion_vad_{name}.wav"
        sf.write(str(output_path), audio, 48000)
        print(f"VAD {vad} -> {output_path}")


def demo_style_transfer(model: AdvancedTTS, output_dir: Path):
    """スタイル転送のデモ"""
    print("\n=== スタイル転送デモ ===")
    
    text = "スタイル転送のテストを行います"
    text_tensor = torch.tensor([ord(c) % 256 for c in text]).unsqueeze(0)
    speaker_id = torch.tensor([0])
    
    # 参照音声からスタイルを抽出（ダミー）
    # 実際には参照音声のメルスペクトログラムが必要
    reference_mel = torch.randn(1, 100, 80)  # ダミー
    
    print("\n参照音声からのスタイル転送")
    outputs = model.synthesize(
        text_tensor,
        speaker_id,
        reference_mel=reference_mel,
    )
    
    audio = outputs['audio'].squeeze().cpu().numpy()
    output_path = output_dir / "style_transfer_reference.wav"
    sf.write(str(output_path), audio, 48000)
    print(f"保存: {output_path}")
    
    # 事前定義されたスタイル
    print("\n事前定義スタイルの使用")
    style_ids = [0, 5, 10]  # 異なるスタイルID
    
    for style_id in style_ids:
        outputs = model.synthesize(
            text_tensor,
            speaker_id,
            style_id=style_id,
        )
        
        audio = outputs['audio'].squeeze().cpu().numpy()
        output_path = output_dir / f"style_preset_{style_id}.wav"
        sf.write(str(output_path), audio, 48000)
        print(f"スタイルID {style_id} -> {output_path}")


def demo_voice_morphing(model: AdvancedTTS, output_dir: Path):
    """音声モーフィングのデモ"""
    print("\n=== 音声モーフィングデモ ===")
    
    text = "音声モーフィングのデモンストレーション"
    text_tensor = torch.tensor([ord(c) % 256 for c in text]).unsqueeze(0)
    
    # 複数話者での合成
    speaker_ids = [0, 1, 2]
    source_mels = []
    
    print("\nソース音声の生成")
    for spk_id in speaker_ids:
        outputs = model.synthesize(
            text_tensor,
            torch.tensor([spk_id]),
        )
        source_mels.append(outputs['mel'])
        
        audio = outputs['audio'].squeeze().cpu().numpy()
        output_path = output_dir / f"morph_source_speaker_{spk_id}.wav"
        sf.write(str(output_path), audio, 48000)
        print(f"話者 {spk_id} -> {output_path}")
        
    # 2話者間のモーフィング
    print("\n2話者間のモーフィング")
    morph_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    for ratio in morph_ratios:
        weights = torch.tensor([[1 - ratio, ratio]])
        
        outputs = model.synthesize(
            text_tensor,
            torch.tensor([0]),  # ベース話者
            morph_targets=[source_mels[1]],  # ターゲット話者のメル
            morph_weights=weights,
        )
        
        audio = outputs['audio'].squeeze().cpu().numpy()
        output_path = output_dir / f"morph_2way_ratio_{int(ratio*100)}.wav"
        sf.write(str(output_path), audio, 48000)
        print(f"モーフィング率 {ratio:.1%} -> {output_path}")
        
    # 3話者のブレンド
    print("\n3話者のブレンド")
    blend_weights = [
        [1.0, 0.0, 0.0],
        [0.33, 0.33, 0.34],
        [0.5, 0.3, 0.2],
        [0.0, 0.5, 0.5],
    ]
    
    for i, weights in enumerate(blend_weights):
        weights_tensor = torch.tensor([weights])
        
        outputs = model.synthesize(
            text_tensor,
            torch.tensor([0]),
            morph_targets=source_mels[1:],
            morph_weights=weights_tensor,
        )
        
        audio = outputs['audio'].squeeze().cpu().numpy()
        output_path = output_dir / f"morph_3way_blend_{i}.wav"
        sf.write(str(output_path), audio, 48000)
        print(f"ブレンド {weights} -> {output_path}")


async def demo_streaming(model: AdvancedTTS, output_dir: Path):
    """リアルタイムストリーミングのデモ"""
    print("\n=== リアルタイムストリーミングデモ ===")
    
    if not model.enable_streaming:
        print("ストリーミングが有効化されていません")
        return
        
    # テキストストリームのシミュレーション
    async def text_generator():
        sentences = [
            "こんにちは。",
            "これはストリーミング音声合成のデモです。",
            "リアルタイムで音声が生成されます。",
            "低レイテンシでの処理が可能です。",
        ]
        
        for sentence in sentences:
            # 文字ごとにストリーミング
            for char in sentence:
                yield char
                await asyncio.sleep(0.05)  # タイピングをシミュレート
                
    # ストリーミング合成
    audio_chunks = []
    chunk_count = 0
    
    async for audio_chunk in model.stream_synthesis(text_generator()):
        audio_chunks.append(audio_chunk)
        chunk_count += 1
        print(f"チャンク {chunk_count} 受信 (サイズ: {len(audio_chunk)})")
        
    # 結合して保存
    full_audio = np.concatenate(audio_chunks)
    output_path = output_dir / "streaming_output.wav"
    sf.write(str(output_path), full_audio, 48000)
    print(f"\nストリーミング結果を保存: {output_path}")
    print(f"総チャンク数: {chunk_count}")
    print(f"総サンプル数: {len(full_audio)}")


def demo_combined_features(model: AdvancedTTS, output_dir: Path):
    """複合機能のデモ"""
    print("\n=== 複合機能デモ ===")
    
    text = "すべての機能を組み合わせたデモンストレーション"
    text_tensor = torch.tensor([ord(c) % 256 for c in text]).unsqueeze(0)
    
    # 感情 + スタイル + モーフィング
    print("\n感情制御 + スタイル転送 + 音声モーフィング")
    
    # ベース話者で異なる感情
    base_mels = []
    emotions = [1, 2]  # happy, sad
    
    for emotion_id in emotions:
        outputs = model.synthesize(
            text_tensor,
            torch.tensor([0]),
            emotion_id=torch.tensor([emotion_id]),
        )
        base_mels.append(outputs['mel'])
        
    # モーフィング + スタイル
    outputs = model.synthesize(
        text_tensor,
        torch.tensor([0]),
        emotion_id=torch.tensor([7]),  # excited
        style_id=5,  # 特定のスタイル
        morph_targets=[base_mels[0]],
        morph_weights=torch.tensor([[0.6, 0.4]]),
        pitch_shift=2.0,  # 2半音上げ
    )
    
    audio = outputs['audio'].squeeze().cpu().numpy()
    output_path = output_dir / "combined_all_features.wav"
    sf.write(str(output_path), audio, 48000)
    print(f"複合機能の結果: {output_path}")
    
    # 感情の遷移
    print("\n感情の動的遷移")
    # 実装は ContinuousMorphing を使用


def main():
    parser = argparse.ArgumentParser(description="高度な機能のデモ")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="学習済みモデルのパス",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="demo_outputs",
        help="出力ディレクトリ",
    )
    parser.add_argument(
        "--demo-type",
        type=str,
        choices=["emotion", "style", "morphing", "streaming", "combined", "all"],
        default="all",
        help="実行するデモの種類",
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # モデルの読み込み
    print("モデルを読み込み中...")
    checkpoint = torch.load(args.model_path, map_location="cpu")
    
    # モデル設定（実際の設定に合わせて調整）
    model = AdvancedTTS(
        n_speakers=100,
        n_emotions=10,
        enable_emotion_control=True,
        enable_style_transfer=True,
        enable_voice_morphing=True,
        enable_streaming=True,
    )
    
    # チェックポイントから重みを読み込み
    # model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # デモの実行
    if args.demo_type in ["emotion", "all"]:
        demo_emotion_control(model, output_dir)
        
    if args.demo_type in ["style", "all"]:
        demo_style_transfer(model, output_dir)
        
    if args.demo_type in ["morphing", "all"]:
        demo_voice_morphing(model, output_dir)
        
    if args.demo_type in ["streaming", "all"]:
        asyncio.run(demo_streaming(model, output_dir))
        
    if args.demo_type in ["combined", "all"]:
        demo_combined_features(model, output_dir)
        
    print(f"\n✅ デモが完了しました。出力: {output_dir}")


if __name__ == "__main__":
    main()
