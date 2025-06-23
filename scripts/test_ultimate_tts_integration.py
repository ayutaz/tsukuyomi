#!/usr/bin/env python3
"""
Integration test script for the Ultimate TTS system

Tests the complete pipeline from text to audio synthesis.
"""

import sys
import os
from pathlib import Path
import time
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

# Import all components
from src.models.ultimate_g2p import create_ultimate_g2p
from src.models.f0_bert import create_f0_bert
from src.models.xphonebert_japanese import create_xphonebert_japanese
from src.models.ultimate_acoustic_model import create_ultimate_acoustic_model
from src.models.bigvgan_v2 import create_bigvgan_v2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_ultimate_g2p():
    """Test Ultimate G2P component."""
    print("\n=== Testing Ultimate G2P (97%+ accuracy target) ===")
    
    try:
        # Create model
        g2p_model = create_ultimate_g2p(device="cpu")
        logger.info("✅ Ultimate G2P model created successfully")
        
        # Test sentences
        test_texts = [
            "月読の音声合成システムです",
            "今日はいい天気ですね",
            "AIとMLの違いは何ですか"
        ]
        
        for text in test_texts:
            # Process text
            outputs = g2p_model.forward(
                text,
                return_accent=True,
                return_confidence=True
            )
            
            # Extract results
            phonemes = outputs['phonemes']
            accent_types = outputs['accent_types']
            confidence = outputs['confidence'].mean().item()
            
            print(f"\nText: {text}")
            print(f"Phoneme sequence length: {phonemes.shape[1]}")
            print(f"Average confidence: {confidence:.3f}")
            print(f"Accent types shape: {accent_types.shape}")
            
        return True, g2p_model
        
    except Exception as e:
        logger.error(f"❌ Ultimate G2P test failed: {e}")
        return False, None


def test_xphonebert_japanese():
    """Test XPhoneBERT Japanese component."""
    print("\n=== Testing XPhoneBERT Japanese ===")
    
    try:
        # Create model
        xphonebert = create_xphonebert_japanese(device="cpu")
        logger.info("✅ XPhoneBERT Japanese model created successfully")
        
        # Test phoneme sequences
        test_phonemes = [
            "k o N n i ch i w a",
            "a r i g a t o o g o z a i m a s u",
            "t s u k u y o m i"
        ]
        
        for phonemes in test_phonemes:
            # Encode phonemes
            embeddings = xphonebert.encode_japanese_phonemes(
                phonemes,
                accent_pattern="LHHHL",
                dialect="tokyo"
            )
            
            print(f"\nPhonemes: {phonemes}")
            print(f"Embeddings shape: {embeddings.shape}")
            print(f"Embedding norm: {embeddings.norm().item():.3f}")
            
        # Test LoRA adaptation
        if xphonebert.config.use_lora:
            print(f"\n✅ LoRA enabled with rank {xphonebert.config.lora_rank}")
            print(f"Number of LoRA layers: {len(xphonebert.lora_layers) if hasattr(xphonebert, 'lora_layers') else 0}")
            
        return True, xphonebert
        
    except Exception as e:
        logger.error(f"❌ XPhoneBERT Japanese test failed: {e}")
        return False, None


def test_f0_bert():
    """Test F0-BERT component."""
    print("\n=== Testing F0-BERT ===")
    
    try:
        # Create model
        f0_model = create_f0_bert(device="cpu")
        logger.info("✅ F0-BERT model created successfully")
        
        # Test F0 prediction
        batch_size = 2
        seq_len = 50
        
        # Create dummy inputs
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        # Test different emotions
        emotions = {
            0: "Neutral",
            1: "Happy",
            2: "Sad",
            3: "Angry"
        }
        
        for emotion_id, emotion_name in emotions.items():
            outputs = f0_model.inference(
                input_ids=input_ids,
                attention_mask=attention_mask,
                emotion_id=torch.tensor([emotion_id, emotion_id]),
                temperature=1.0
            )
            
            f0_mean = outputs['f0_final'].mean().item()
            f0_std = outputs['f0_final'].std().item()
            
            print(f"\nEmotion: {emotion_name}")
            print(f"F0 mean: {f0_mean:.1f} Hz")
            print(f"F0 std: {f0_std:.1f} Hz")
            
        return True, f0_model
        
    except Exception as e:
        logger.error(f"❌ F0-BERT test failed: {e}")
        return False, None


def test_ultimate_acoustic_model():
    """Test Ultimate Acoustic Model."""
    print("\n=== Testing Ultimate Acoustic Model ===")
    
    try:
        # Create model
        acoustic_model = create_ultimate_acoustic_model(device="cpu")
        logger.info("✅ Ultimate Acoustic Model created successfully")
        
        # Test synthesis
        batch_size = 2
        phoneme_length = 30
        
        # Create dummy inputs
        phoneme_embeddings = torch.randn(batch_size, 512, phoneme_length)
        phoneme_lengths = torch.tensor([30, 28])
        speaker_ids = torch.tensor([0, 1])
        emotion_ids = torch.tensor([1, 2])  # Happy, Sad
        
        # Inference
        mel = acoustic_model.inference(
            phoneme_embeddings=phoneme_embeddings,
            phoneme_lengths=phoneme_lengths,
            speaker_ids=speaker_ids,
            emotion_ids=emotion_ids,
            length_scale=1.0,
            temperature=1.0
        )
        
        print(f"\nGenerated mel-spectrogram shape: {mel.shape}")
        print(f"Mel range: [{mel.min().item():.3f}, {mel.max().item():.3f}]")
        print(f"Number of speakers supported: {acoustic_model.config.n_speakers}")
        print(f"Flow matching layers: {acoustic_model.config.n_flows}")
        
        return True, acoustic_model
        
    except Exception as e:
        logger.error(f"❌ Ultimate Acoustic Model test failed: {e}")
        return False, None


def test_bigvgan_v2():
    """Test BigVGAN-v2 vocoder."""
    print("\n=== Testing BigVGAN-v2 Vocoder ===")
    
    try:
        # Create model
        vocoder = create_bigvgan_v2(device="cpu")
        logger.info("✅ BigVGAN-v2 vocoder created successfully")
        
        # Test waveform generation
        batch_size = 2
        mel_frames = 100
        
        # Create dummy mel-spectrogram
        mel = torch.randn(batch_size, 128, mel_frames)
        
        # Generate audio
        start_time = time.time()
        audio = vocoder.inference(mel)
        generation_time = time.time() - start_time
        
        print(f"\nGenerated audio shape: {audio.shape}")
        print(f"Audio length: {audio.shape[2] / 48000:.2f} seconds")
        print(f"Generation time: {generation_time:.3f} seconds")
        print(f"RTF (Real-Time Factor): {generation_time / (audio.shape[2] / 48000):.3f}")
        print(f"Sample rate: {vocoder.config.sampling_rate} Hz")
        print(f"Using Snake-Beta activation: {vocoder.config.use_snake_activation}")
        
        return True, vocoder
        
    except Exception as e:
        logger.error(f"❌ BigVGAN-v2 test failed: {e}")
        return False, None


def test_end_to_end_pipeline():
    """Test complete end-to-end pipeline."""
    print("\n=== Testing End-to-End Pipeline ===")
    
    try:
        # Create all models
        print("\nLoading models...")
        g2p_model = create_ultimate_g2p(device="cpu")
        xphonebert = create_xphonebert_japanese(device="cpu")
        f0_model = create_f0_bert(device="cpu")
        acoustic_model = create_ultimate_acoustic_model(device="cpu")
        vocoder = create_bigvgan_v2(device="cpu")
        
        print("✅ All models loaded successfully")
        
        # Test text
        text = "月読は最高峰の音声合成システムです"
        print(f"\nInput text: {text}")
        
        # Step 1: G2P
        print("\n1. G2P conversion...")
        g2p_outputs = g2p_model.forward(text, return_accent=True)
        phoneme_ids = g2p_outputs['phonemes']
        accent_ids = g2p_outputs['accent_types']
        print(f"   Phoneme sequence length: {phoneme_ids.shape[1]}")
        
        # Step 2: Phoneme encoding
        print("\n2. Phoneme encoding...")
        phoneme_embeddings = xphonebert(
            phoneme_ids=phoneme_ids,
            accent_ids=accent_ids,
            dialect_ids=torch.tensor([0]),  # Tokyo
            return_dict=True
        )['hidden_states']
        print(f"   Phoneme embeddings shape: {phoneme_embeddings.shape}")
        
        # Step 3: F0 prediction
        print("\n3. F0 prediction...")
        f0_outputs = f0_model(
            input_ids=phoneme_ids,
            attention_mask=torch.ones_like(phoneme_ids),
            emotion_id=torch.tensor([0]),  # Neutral
            phoneme_embeddings=phoneme_embeddings
        )
        print(f"   F0 shape: {f0_outputs['f0_final'].shape}")
        print(f"   Average F0: {f0_outputs['f0_final'].mean().item():.1f} Hz")
        
        # Step 4: Acoustic modeling
        print("\n4. Acoustic modeling...")
        # Transpose for acoustic model
        phoneme_embeddings_t = phoneme_embeddings.transpose(1, 2)
        mel = acoustic_model.inference(
            phoneme_embeddings=phoneme_embeddings_t,
            phoneme_lengths=torch.tensor([phoneme_ids.shape[1]]),
            speaker_ids=torch.tensor([0]),
            emotion_ids=torch.tensor([0])
        )
        print(f"   Mel-spectrogram shape: {mel.shape}")
        
        # Step 5: Vocoder
        print("\n5. Waveform generation...")
        audio = vocoder.inference(mel)
        print(f"   Audio shape: {audio.shape}")
        print(f"   Duration: {audio.shape[2] / 48000:.2f} seconds")
        
        print("\n✅ End-to-end pipeline test completed successfully!")
        
        # Calculate RTF
        total_params = sum(p.numel() for p in [
            g2p_model.parameters(),
            xphonebert.parameters(),
            f0_model.parameters(),
            acoustic_model.parameters(),
            vocoder.parameters()
        ])
        print(f"\nTotal model parameters: {total_params / 1e6:.1f}M")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ End-to-end pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("Tsukuyomi Ultimate TTS Integration Tests")
    print("=" * 60)
    
    # Track results
    results = {}
    
    # Test individual components
    success_g2p, _ = test_ultimate_g2p()
    results['Ultimate G2P'] = success_g2p
    
    success_xphonebert, _ = test_xphonebert_japanese()
    results['XPhoneBERT Japanese'] = success_xphonebert
    
    success_f0, _ = test_f0_bert()
    results['F0-BERT'] = success_f0
    
    success_acoustic, _ = test_ultimate_acoustic_model()
    results['Ultimate Acoustic Model'] = success_acoustic
    
    success_vocoder, _ = test_bigvgan_v2()
    results['BigVGAN-v2'] = success_vocoder
    
    # Test end-to-end only if all components pass
    if all(results.values()):
        success_e2e = test_end_to_end_pipeline()
        results['End-to-End Pipeline'] = success_e2e
    else:
        results['End-to-End Pipeline'] = False
        print("\n⚠️  Skipping end-to-end test due to component failures")
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for component, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{component:.<40} {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! The Ultimate TTS system is ready!")
        print("Next steps:")
        print("1. Prepare 10,000 hours of training data")
        print("2. Set up H100 cluster for training")
        print("3. Begin stage-wise training process")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    return total_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)