#!/usr/bin/env python3
"""
Basic TTS functionality test script.

This script tests the basic functionality of the Tsukuyomi TTS system
with the newly implemented components.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.frontend.japanese_g2p import create_japanese_g2p
from src.frontend.phonemizer import create_phonemizer
from src.models.acoustic_model import AcousticModel, AcousticModelConfig
from src.models.xphonebert import XPhoneBERTEncoder
from src.utils.audio import save_audio
from src.vocoder.vocoder_wrapper import create_vocoder


def test_g2p():
    """Test Japanese G2P functionality."""
    print("\n=== Testing Japanese G2P ===")
    
    try:
        g2p = create_japanese_g2p(backend="auto")
        
        test_texts = [
            "こんにちは",
            "今日はいい天気ですね。",
            "月読の音声合成システムです。"
        ]
        
        for text in test_texts:
            print(f"\nInput: {text}")
            
            # Get phonemes
            result = g2p.g2p(text, return_accent=True)
            
            if isinstance(result, tuple):
                phonemes, accent_phrases = result
                print(f"Phonemes: {phonemes}")
                print(f"Accent phrases: {len(accent_phrases)}")
            else:
                print(f"Phonemes: {result}")
                
        print("\n✅ G2P test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ G2P test failed: {e}")
        return False


def test_phoneme_encoder():
    """Test phoneme encoder functionality."""
    print("\n=== Testing Phoneme Encoder ===")
    
    try:
        # Initialize encoder
        encoder = XPhoneBERTEncoder(device="cpu", use_bf16=False)
        
        # Test phoneme sequences
        phoneme_seqs = [
            "k o N n i ch i w a",
            "t s u k u y o m i",
            "s a N p u r u"
        ]
        
        for seq in phoneme_seqs:
            print(f"\nPhoneme sequence: {seq}")
            
            # Encode
            embeddings = encoder.forward(seq, return_pooled=True)
            print(f"Embedding shape: {embeddings.shape}")
            print(f"Embedding mean: {embeddings.mean().item():.4f}")
            
        print("\n✅ Phoneme encoder test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Phoneme encoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_acoustic_model():
    """Test acoustic model functionality."""
    print("\n=== Testing Acoustic Model ===")
    
    try:
        # Initialize model
        config = AcousticModelConfig(
            hidden_dim=256,
            n_layers=2,
            n_heads=4,
            n_mel_channels=80,
            use_bf16=False
        )
        model = AcousticModel(config)
        model.eval()
        
        # Create dummy input
        batch_size = 1
        seq_len = 20
        phoneme_features = torch.randn(batch_size, seq_len, 768)  # XPhoneBERT dim
        phoneme_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                phoneme_embeddings=phoneme_features,
                phoneme_mask=phoneme_mask
            )
            
        print(f"\nOutput keys: {outputs.keys()}")
        print(f"Mel output shape: {outputs['mel_output'].shape}")
        print(f"Duration output shape: {outputs['duration_output'].shape}")
        
        print("\n✅ Acoustic model test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Acoustic model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vocoder():
    """Test vocoder functionality."""
    print("\n=== Testing Vocoder ===")
    
    try:
        # Initialize vocoder
        vocoder = create_vocoder(
            vocoder_type=None,  # Auto-select
            sample_rate=24000,
            n_mels=80,
            device="cpu"
        )
        
        print(f"Selected vocoder: {vocoder.vocoder_type}")
        
        # Create dummy mel-spectrogram
        mel_spec = torch.randn(80, 200)  # 80 mels, 200 frames
        
        # Generate waveform
        waveform = vocoder.inference(mel_spec, return_numpy=True)
        
        print(f"\nGenerated waveform shape: {waveform.shape}")
        print(f"Waveform duration: {len(waveform) / 24000:.2f} seconds")
        print(f"Waveform range: [{waveform.min():.4f}, {waveform.max():.4f}]")
        
        print("\n✅ Vocoder test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Vocoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_end_to_end():
    """Test end-to-end TTS pipeline."""
    print("\n=== Testing End-to-End Pipeline ===")
    
    try:
        # Initialize components
        g2p = create_japanese_g2p(backend="auto")
        encoder = XPhoneBERTEncoder(device="cpu", use_bf16=False)
        
        config = AcousticModelConfig(
            hidden_dim=256,
            n_layers=2,
            n_mel_channels=80,
            use_bf16=False
        )
        acoustic_model = AcousticModel(config)
        acoustic_model.eval()
        
        vocoder = create_vocoder(sample_rate=24000, n_mels=80, device="cpu")
        
        # Test text
        text = "こんにちは、月読です。"
        print(f"\nInput text: {text}")
        
        # Step 1: G2P
        phonemes = g2p.g2p(text, return_accent=False)
        print(f"Phonemes: {phonemes}")
        
        # Step 2: Encode phonemes
        phoneme_embeddings = encoder.forward(phonemes, return_pooled=False)[0]
        print(f"Phoneme embeddings shape: {phoneme_embeddings.shape}")
        
        # Step 3: Generate mel-spectrogram
        with torch.no_grad():
            outputs = acoustic_model(
                phoneme_embeddings=phoneme_embeddings,
                phoneme_mask=torch.ones(1, phoneme_embeddings.shape[1], dtype=torch.bool)
            )
        
        mel_output = outputs['mel_output']
        print(f"Mel-spectrogram shape: {mel_output.shape}")
        
        # Step 4: Generate waveform
        # Transpose mel for vocoder (expects [n_mels, time])
        mel_for_vocoder = mel_output.squeeze(0).transpose(0, 1)
        waveform = vocoder.inference(mel_for_vocoder, return_numpy=True)
        
        print(f"Generated waveform shape: {waveform.shape}")
        print(f"Duration: {len(waveform) / 24000:.2f} seconds")
        
        # Save output
        output_path = Path("output/test_tts.wav")
        output_path.parent.mkdir(exist_ok=True)
        
        # Normalize waveform
        waveform = np.clip(waveform, -1.0, 1.0)
        save_audio(str(output_path), waveform, sample_rate=24000)
        print(f"\nSaved audio to: {output_path}")
        
        print("\n✅ End-to-end test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logging.basicConfig(level=logging.INFO)
    
    print("=== Tsukuyomi TTS Basic Functionality Test ===")
    
    results = []
    
    # Run individual component tests
    results.append(("G2P", test_g2p()))
    results.append(("Phoneme Encoder", test_phoneme_encoder()))
    results.append(("Acoustic Model", test_acoustic_model()))
    results.append(("Vocoder", test_vocoder()))
    results.append(("End-to-End", test_end_to_end()))
    
    # Summary
    print("\n=== Test Summary ===")
    passed = 0
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name}: {status}")
        if result:
            passed += 1
            
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
