#!/usr/bin/env python3
"""Tsukuyomi TTS CLI interface"""

import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Tsukuyomi TTS - Ultimate Japanese Text-to-Speech System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate speech from text
  tsukuyomi "こんにちは、月読です" -o output.wav
  
  # Use specific model and speaker
  tsukuyomi "Hello world" -m model.pt -s 0 -o output.wav
  
  # List available models
  tsukuyomi --list-models
  
  # Start API server
  tsukuyomi --serve --port 8080
        """,
    )

    # Main arguments
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to synthesize",
    )

    # Model options
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="default",
        help="Model checkpoint path or name",
    )
    parser.add_argument(
        "-s",
        "--speaker",
        type=int,
        default=0,
        help="Speaker ID",
    )
    parser.add_argument(
        "-e",
        "--emotion",
        type=str,
        default="neutral",
        help="Emotion (neutral, happy, sad, angry, etc.)",
    )

    # Output options
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output.wav",
        help="Output audio file path",
    )
    parser.add_argument(
        "-sr",
        "--sample-rate",
        type=int,
        default=48000,
        help="Output sample rate",
    )

    # Advanced options
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed multiplier",
    )
    parser.add_argument(
        "--pitch-shift",
        type=float,
        default=0.0,
        help="Pitch shift in semitones",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )

    # Server mode
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start API server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API server port",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="API server host",
    )

    # Utility commands
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models",
    )
    parser.add_argument(
        "--list-speakers",
        action="store_true",
        help="List available speakers in model",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version",
    )

    args = parser.parse_args()

    # Handle utility commands
    if args.version:
        print("Tsukuyomi TTS v0.1.0")
        return

    if args.list_models:
        print("Available models:")
        print("  - default: Base Japanese TTS model")
        print("  - vits: VITS-based model")
        print("  - matcha: Matcha-TTS model")
        return

    # Server mode
    if args.serve:
        print(f"Starting API server on {args.host}:{args.port}")
        import uvicorn

        from src.api.server import app

        uvicorn.run(app, host=args.host, port=args.port)
        return

    # Check if text is provided
    if not args.text:
        parser.print_help()
        sys.exit(1)

    # Import TTS system
    try:
        from src.tsukuyomi_tts import TsukuyomiTTS
    except ImportError:
        print("Error: Tsukuyomi TTS not properly installed.")
        print("Please run: pip install -e .")
        sys.exit(1)

    # Initialize TTS
    print(f"Loading model '{args.model}' on {args.device}...")
    try:
        tts = TsukuyomiTTS(device=args.device)
        if args.model != "default" and Path(args.model).exists():
            tts.load_checkpoint(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # List speakers if requested
    if args.list_speakers:
        print(f"Available speakers: 0-{tts.n_speakers-1}")
        return

    # Synthesize speech
    print(f"Synthesizing: {args.text}")
    try:
        audio = tts.synthesize(
            text=args.text,
            speaker_id=args.speaker,
            emotion=args.emotion,
            speed=args.speed,
            pitch_shift=args.pitch_shift,
        )

        # Save audio
        sf.write(args.output, audio, args.sample_rate)
        print(f"Audio saved to: {args.output}")

    except Exception as e:
        print(f"Error during synthesis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
