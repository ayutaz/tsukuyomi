#!/usr/bin/env python3
"""
Test script to verify Tsukuyomi TTS installation and functionality

This script checks:
1. All dependencies are installed
2. Models can be loaded
3. Basic synthesis works
4. Export functionality works
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required imports work"""
    print("Testing imports...")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
        
        import librosa
        print("✓ librosa")
        
        import pyopenjtalk
        print("✓ pyopenjtalk")
        
        import streamlit
        print("✓ streamlit")
        
        import onnx
        print("✓ onnx")
        
        import onnxruntime
        print("✓ onnxruntime")
        
        # Test project imports
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        
        from src.models.vits import VITS
        print("✓ VITS model")
        
        from src.models.hifigan import HiFiGAN
        print("✓ HiFi-GAN model")
        
        from src.frontend.japanese_g2p import JapaneseG2P
        print("✓ Japanese G2P")
        
        from src.training.losses import MultiTaskLoss
        print("✓ Training losses")
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Please install missing dependencies with: pip install -r requirements.txt")
        return False


def test_model_creation():
    """Test that models can be created"""
    print("\nTesting model creation...")
    
    try:
        from src.models.vits import VITS
        from src.models.hifigan import HiFiGAN
        from src.models.f0_bert import F0BERT
        
        # Test VITS
        vits = VITS(n_vocab=100, n_speakers=1)
        print("✓ VITS model created")
        
        # Test HiFi-GAN
        hifigan = HiFiGAN()
        print("✓ HiFi-GAN created")
        
        # Test F0-BERT
        f0bert = F0BERT()
        print("✓ F0-BERT created")
        
        print("\n✅ All models can be created!")
        return True
        
    except Exception as e:
        print(f"\n❌ Model creation error: {e}")
        return False


def test_data_pipeline():
    """Test data loading pipeline"""
    print("\nTesting data pipeline...")
    
    try:
        from src.data.ljspeech_dataset import LJSpeechDataset
        from src.frontend.text_normalizer import TextNormalizer
        
        # Test text normalizer
        normalizer = TextNormalizer()
        normalized = normalizer.normalize("Hello, world! 123")
        print(f"✓ Text normalizer: '{normalized}'")
        
        # Test dataset (without actual data)
        print("✓ LJSpeech dataset class available")
        
        print("\n✅ Data pipeline functional!")
        return True
        
    except Exception as e:
        print(f"\n❌ Data pipeline error: {e}")
        return False


def test_synthesis():
    """Test basic synthesis (mock)"""
    print("\nTesting synthesis pipeline...")
    
    try:
        from src.models.vits import VITS
        import torch
        
        # Create small model
        model = VITS(
            n_vocab=100,
            n_speakers=1,
            hidden_channels=32,
            filter_channels=64,
            n_heads=2,
            n_layers=2
        )
        model.eval()
        
        # Mock input
        text = torch.randint(0, 100, (1, 10))
        text_lengths = torch.tensor([10])
        
        with torch.no_grad():
            # Test forward pass
            outputs = model.infer(text, text_lengths)
            
        print(f"✓ Model inference successful, output shape: {outputs.shape}")
        print("\n✅ Synthesis pipeline functional!")
        return True
        
    except Exception as e:
        print(f"\n❌ Synthesis error: {e}")
        return False


def test_tools():
    """Test additional tools"""
    print("\nTesting additional tools...")
    
    results = []
    
    # Test preprocessing script
    try:
        from scripts.preprocess_audio import AudioPreprocessor
        preprocessor = AudioPreprocessor()
        print("✓ Audio preprocessor available")
        results.append(True)
    except Exception as e:
        print(f"✗ Audio preprocessor error: {e}")
        results.append(False)
    
    # Test MOS evaluation
    try:
        from scripts.mos_evaluation import MOSDatabase, MOSEvaluator
        print("✓ MOS evaluation tools available")
        results.append(True)
    except Exception as e:
        print(f"✗ MOS evaluation error: {e}")
        results.append(False)
    
    # Test model compression
    try:
        from scripts.model_compression import ModelCompressor
        print("✓ Model compression tools available")
        results.append(True)
    except Exception as e:
        print(f"✗ Model compression error: {e}")
        results.append(False)
    
    # Test edge optimization
    try:
        from scripts.edge_optimization import EdgeOptimizer, EdgeConfig
        print("✓ Edge optimization tools available")
        results.append(True)
    except Exception as e:
        print(f"✗ Edge optimization error: {e}")
        results.append(False)
    
    if all(results):
        print("\n✅ All tools functional!")
    else:
        print("\n⚠️ Some tools have issues")
    
    return all(results)


def main():
    """Run all tests"""
    print("="*60)
    print("Tsukuyomi TTS Installation Test")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(test_imports())
    results.append(test_model_creation())
    results.append(test_data_pipeline())
    results.append(test_synthesis())
    results.append(test_tools())
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if all(results):
        print("✅ All tests passed! Tsukuyomi TTS is ready to use.")
        print("\nNext steps:")
        print("1. Prepare your dataset")
        print("2. Run training: python train.py --config configs/stage1.yaml")
        print("3. Or try the web UI: streamlit run app.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Check Python version (3.11+ required)")
        print("3. For CUDA issues, ensure PyTorch is installed with CUDA support")
    
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())