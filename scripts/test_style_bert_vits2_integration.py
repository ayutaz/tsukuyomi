#!/usr/bin/env python3
"""
Test script for Style-BERT-VITS2 integration
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

from src.integrations.style_bert_vits2 import create_style_bert_vits2_tts

logging.basicConfig(level=logging.INFO)


def test_integration():
    """Test Style-BERT-VITS2 integration."""
    print("=== Style-BERT-VITS2 Integration Test ===\n")
    
    # Create TTS instance
    print("1. Creating TTS instance...")
    tts = create_style_bert_vits2_tts()
    print("✅ TTS instance created\n")
    
    # Check installation
    print("2. Checking Style-BERT-VITS2 installation...")
    if tts.wrapper.check_installation():
        print("✅ Style-BERT-VITS2 is installed\n")
    else:
        print("❌ Style-BERT-VITS2 is not installed")
        print("Please follow instructions in docs/style-bert-vits2-setup.md\n")
        
        # Test with dummy mode
        print("3. Testing with placeholder implementation...")
        
    # Setup model (will use placeholder if not installed)
    print("3. Setting up model...")
    if tts.setup():
        print("✅ Model setup complete\n")
    else:
        print("⚠️  Using placeholder model\n")
    
    # Test synthesis
    print("4. Testing synthesis...")
    test_texts = [
        "月読は高品質な日本語音声合成システムです。",
        "本日は晴天なり。",
        "ゲームキャラクターの音声を自然に再現します。"
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\nTest {i+1}: {text}")
        
        output_path = Path(f"output/style_bert_vits2_test_{i+1}.wav")
        audio = tts.tts(
            text=text,
            speaker=0,
            style="Neutral",
            output_path=output_path
        )
        
        if audio is not None:
            print(f"✅ Generated {len(audio)} samples")
            if output_path.exists():
                print(f"✅ Saved to {output_path}")
        else:
            print("❌ Synthesis failed")
    
    # Test speaker list
    print("\n5. Getting available speakers...")
    speakers = tts.wrapper.get_speakers()
    print(f"Available speakers: {speakers}")
    
    # Test style list
    print("\n6. Getting available styles...")
    styles = tts.wrapper.get_styles()
    print(f"Available styles: {styles}")
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    test_integration()
